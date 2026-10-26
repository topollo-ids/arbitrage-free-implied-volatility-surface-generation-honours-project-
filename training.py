"""
The networks, the smoothness penalty of the generator loss, the training loop (Phi in or out of the loss),
the run folders and the training step. Imports common and preprocessing.

Contents
    1. Generator and discriminator (was networks.py)
    2. The smoothness penalty of the loss, not a data smoothing (was smoothness.py)
    3. The training loop (was training.py)
    4. Run folders (was runs.py)
    5. The training step (was pipeline.py)
"""

import copy
from dataclasses import dataclass, replace
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .common import (
    Config, INDEX_FILES, RATES_FILE, SEEDS, TRAINING_CONFIGS, _done, _plan, arbitrage_penalty_torch, check_inputs,
    file_hash, get_device, library_versions, load_checkpoint, log, relative_call_torch, run_path, save_checkpoint,
    set_seed, text_hash)
from .preprocessing import N_SCALARS, SOURCES, prepare_all, to_tensors


# =====================================================================================================================
# 1. Generator and discriminator (was networks.py)
# =====================================================================================================================

# Stage 5: the generator and the discriminator (paper section 3.4, Figures 1 and 2).
#
# Tracker rows: 5.1 (paper defaults, see config.py), 5.2 (R12: no unused sigmoid layer,
# no stored normalisation statistics, no LSTM docstring). 5.3 (A13 standardisation) is a
# later experiment, outside the control.
#
# The architecture and the order of the inputs are the released code's, so that a network trained
# by the original code and one built here hold the same weights in the same places:
#     generator input     = [noise, condition]
#     discriminator input = [condition, outcome]
# Weights start from PyTorch's default initialisation, as in the released code (the paper is silent).

class Generator(nn.Module):
    """
    G(a, z) = (R_hat, dg_hat): three dense layers of sizes H, 2H and out_dim,
    softplus after the first two and nothing after the last. The output is a return and
    changes in log-vol, both unbounded, so no final activation.
    """

    def __init__(self, noise_dim, cond_dim, hidden_dim, out_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.linear1 = nn.Linear(noise_dim + cond_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.linear3 = nn.Linear(2 * hidden_dim, out_dim)
        self.activation = nn.Softplus()

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=-1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return self.linear3(x)


class Discriminator(nn.Module):
    """
    D(a, outcome): one hidden layer of H softplus units, then a sigmoid, giving the
    probability that (condition, outcome) came from the data rather than the generator.
    """

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Softplus()

    def forward(self, condition_and_outcome):
        x = self.activation(self.linear1(condition_and_outcome))
        return torch.sigmoid(self.linear2(x))


def build_networks(cond_dim, out_dim, config, device):
    """Both networks, sized from the data (3 + K inputs, 1 + K outputs) and the config."""
    gen = Generator(config.noise_dim, cond_dim, config.hidden_dim, out_dim).to(device)
    disc = Discriminator(cond_dim + out_dim, config.hidden_dim).to(device)
    return gen, disc


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# =====================================================================================================================
# 2. The smoothness penalty of the loss, not a data smoothing (was smoothness.py)
# =====================================================================================================================

# Stage 6: the smoothness penalty.
#
# Tracker rows: 6.1 (F7: the paper's penalty in the control, the released one kept for the ablation),
# 6.2 (A2: vectorised, correct for any grid, no seams), 6.3 (tests), 6.4 (F8: one object, g).
#
# The paper (eqs 11-13), on a log-implied-vol surface g:
#     L_m(g)   = sum_{i,j} (g(m_i+1, tau_j) - g(m_i, tau_j))^2 / (m_i+1 - m_i)^2       moneyness neighbours
#     L_tau(g) = sum_{i,j} (g(m_i, tau_j+1) - g(m_i, tau_j))^2 / (tau_j+1 - tau_j)^2   maturity neighbours
# applied to the generated surface g_hat = g_t + dg_hat, and averaged over the batch in the loss.
# The formula is implemented as printed: there are no grid-cell weights (dm * dtau), so on an uneven
# grid it is a weighted sum of squared slopes rather than an exact L2 norm. Noted in the decision log.
#
# Layout (Stage 1): a surface is a vector v with v[j*lk + i] = g(m_i, tau_j), so v.reshape(lt, lk)[j, i].

def generated_log_surface(fake, past_log_surface):
    """g_hat = g_t + dg_hat: the condition-day log-vols plus the generated change (fake[:, 0] is the return)."""
    return past_log_surface + fake[:, 1:]


def penalised_surface(fake, past_log_surface, on="g"):
    """
    The object the penalty is computed on (F8). The paper uses g throughout ("g").
    The released code's gradient matching used sigma = exp(g) ("sigma"); its training used g.
    """
    g = generated_log_surface(fake, past_log_surface)
    if on == "g":
        return g
    if on == "sigma":
        return torch.exp(g)
    raise ValueError("on must be 'g' or 'sigma', got %r" % (on,))


class PaperSmoothness:
    """Paper eqs 11-12 for a batch of surfaces (B, lk*lt). Returns (L_m, L_tau), one value per surface."""

    def __init__(self, m, tau, device):
        m, tau = np.asarray(m, dtype=float), np.asarray(tau, dtype=float)
        self.lk, self.lt = len(m), len(tau)
        self.w_m = torch.tensor(1 / np.diff(m) ** 2, dtype=torch.float64, device=device)      # (lk-1,)
        self.w_tau = torch.tensor(1 / np.diff(tau) ** 2, dtype=torch.float64, device=device)  # (lt-1,)

    def __call__(self, v):
        G = v.reshape(v.shape[0], self.lt, self.lk)          # G[b, j, i] = g(m_i, tau_j)
        w_m, w_tau = self.w_m.to(v.dtype), self.w_tau.to(v.dtype)
        d_m = G[:, :, 1:] - G[:, :, :-1]                     # same maturity, next moneyness
        d_tau = G[:, 1:, :] - G[:, :-1, :]                   # same moneyness, next maturity
        L_m = (d_m ** 2 * w_m).sum(dim=(1, 2))               # w_m runs along the moneyness axis
        L_tau = (d_tau ** 2 * w_tau[:, None]).sum(dim=(1, 2))  # w_tau runs along the maturity axis
        return L_m, L_tau


# ---- The released construction, for the ablation only (F7). Same matrices as VolGAN.py lines 658-677,
# ---- built in float32 like the original; the per-sample loop of lines 732-738 becomes one batched product.

def released_m_block(m, lk, lt, device):
    """matrix_m: row k differences entries k and k+lk. m_seq: 1/dm^2 by moneyness gap, written in blocks of lk."""
    N = lk * lt
    matrix_m = torch.zeros((N - lk, N), dtype=torch.float, device=device)
    for k in range(N - lk):
        matrix_m[k, k], matrix_m[k, k + lk] = -1, 1
    m_t = torch.tensor(np.asarray(m, dtype=float), dtype=torch.float, device=device)
    m_seq = torch.zeros(lk * (lt - 1), dtype=torch.float, device=device)
    for i in range(len(m) - 1):
        m_seq[i * lk:(i + 1) * lk] = 1 / ((m_t[i + 1] - m_t[i]) ** 2)   # slicing past the end is silently dropped
    return matrix_m, m_seq


def released_t_block(tau, lk, lt, device):
    """matrix_t: row k differences entries k and k+1. tsq: 1/dtau^2 repeated lk times (cycling with period lt)."""
    N = lk * lt
    tau_t = torch.tensor(np.asarray(tau, dtype=float), dtype=torch.float, device=device)
    t_seq = torch.zeros(len(tau), dtype=torch.float, device=device)
    for j in range(len(tau) - 1):
        t_seq[j] = 1 / ((tau_t[j + 1] - tau_t[j]) ** 2)
    matrix_t = torch.zeros((N, N), dtype=torch.float, device=device)
    for k in range(N - 1):
        matrix_t[k, k], matrix_t[k, k + 1] = -1, 1
    return matrix_t, t_seq.repeat(lk)


class ReleasedSmoothness:
    """The released code's penalty, same numbers as VolGAN.py, with lk and lt taken from the grid (B14)."""

    def __init__(self, m, tau, device):
        lk, lt = len(m), len(tau)
        self.matrix_m, self.m_seq = released_m_block(m, lk, lt, device)
        self.matrix_t, self.tsq = released_t_block(tau, lk, lt, device)

    def __call__(self, v):
        L_m = ((v @ self.matrix_m.T) ** 2) @ self.m_seq
        L_tau = ((v @ self.matrix_t.T) ** 2) @ self.tsq
        return L_m, L_tau


def make_smoothness(kind, m, tau, device):
    """kind "paper" (control) or "released" (ablation)."""
    if kind == "paper":
        return PaperSmoothness(m, tau, device)
    if kind == "released":
        return ReleasedSmoothness(m, tau, device)
    raise ValueError("kind must be 'paper' or 'released', got %r" % (kind,))


# =====================================================================================================================
# 3. The training loop (was training.py)
# =====================================================================================================================

# Stage 7: training, written from the paper (section 3.4, eqs 13-14).
#
# Tracker rows: 7.1 (A3: one loop for both phases, so the two can no longer disagree as in F8),
# 7.3 (F9: restart switch), 7.4 (B10: no empty batch), 7.5 (B14: grid taken from the data),
# 7.6-7.9 (no duplicated loop, compact logs, no unused matrices or arguments). 7.10 (A12) stays out of the control.
#
# Released-code settings for the ablation (all in config.py): smoothness="released", match_on="sigma",
# restart=False, gen_bce_weight=1.0. With those, this loop reproduces the released GradientMatching and
# TrainLoopNoVal exactly (tests/test_stage7.py).
#
# Losses (paper eqs 13 and 14), with E the average over the minibatch:
#     discriminator: J_D = -1/2 E[log D(a, real)] - 1/2 E[log(1 - D(a, G(a, z)))]
#     generator:     J_G = -1/2 E[log D(a, G(a, z))] + alpha_m E[L_m(g_hat)] + alpha_tau E[L_tau(g_hat)]
#     with g_hat = g_t + dg_hat, the generated log-vol surface.
#
# Two phases (section 3.4):
#     "match": n_grad epochs of the classical GAN, the generator stepping on its BCE term only. At every
#              generator update, record the gradient norm of each of the three terms of eq 13 with respect
#              to the generator weights; then alpha_m = mean(|grad BCE| / |grad L_m|) and
#              alpha_tau = mean(|grad BCE| / |grad L_tau|).
#     "train": restart "from the same initialization used for the start of the gradient norm matching
#              procedure" (initial weights, fresh optimisers), then n_epochs with the full J_G.
# Each update is one discriminator step then one generator step on the same minibatch, each with fresh noise.
# The training set is reshuffled at the start of every epoch.
#
# Part 1, lever 2 (phi_in_loss): the arbitrage penalty enters eq 13,
#     J_G = BCE term + alpha_m L_m + alpha_tau L_tau + alpha_phi E[Phi(sigma_hat)],   sigma_hat = exp(g_t + dg_hat),
# the surface the smoothness terms see (they use its log). Phi is the torch copy of Stage 2's instrument, in float64,
# pricing each sample's surface with its predicted day's rates per maturity; the scalar is cast back to float32.
# alpha_phi is set by gradient-norm matching like alpha_m and alpha_tau, by a rule fixed before any result:
#     the mean over match-phase updates of |grad BCE| / |grad Phi|, over the updates with |grad Phi| > 1e-12;
#     if fewer than half the updates have such a gradient, the ratio of means over all updates instead.
# With phi_in_loss False no Phi is computed anywhere in the loss path, and fit() is unchanged bit for bit
# (tests/test_stage7_phi.py holds the weights fit() produced before Part 1).

PHI_GRAD_TOL = 1e-12          # an update "has a Phi gradient" when |grad Phi| exceeds this
PHI_MIN_SHARE = 0.5           # below this share of such updates, alpha_phi is the ratio of means instead


def batch_slices(n, batch_size):
    """B10: ceil(n / batch_size) minibatches covering every sample once; the last may be smaller, never empty."""
    return [slice(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]


def grad_norm(term, params):
    """Euclidean norm of d(term)/d(params), all generator weights stacked into one vector. Leaves .grad untouched."""
    grads = torch.autograd.grad(term, params, retain_graph=True)
    return torch.sqrt(sum((g ** 2).sum() for g in grads)).item()


class TorchPhi:
    """
    E[Phi(sigma_hat)] over a batch (Part 1, lever 2): sigma_hat = exp(g_t + dg_hat) in float64, as (B, lk, lt)
    matrices (Stage 1 layout -> arbitrage.py layout), priced with each sample's predicted-day rates (B, lt).
    """

    def __init__(self, m, tau, device):
        m, tau = np.asarray(m, dtype=float), np.asarray(tau, dtype=float)
        self.lk, self.lt = len(m), len(tau)
        ms, ts = np.meshgrid(m, tau, indexing="ij")
        self.m = torch.tensor(m, dtype=torch.float64, device=device)
        self.tau = torch.tensor(tau, dtype=torch.float64, device=device)
        self.ms = torch.tensor(ms, dtype=torch.float64, device=device)
        self.ts = torch.tensor(ts, dtype=torch.float64, device=device)

    def penalty(self, fake, past_log_surface, rates):
        """Phi of each generated surface, (B,) float64 tensor."""
        g_hat = generated_log_surface(fake.double(), past_log_surface.double())
        sigma = torch.exp(g_hat).reshape(-1, self.lt, self.lk).transpose(1, 2)        # (B, lk, lt)
        C = relative_call_torch(self.ms, self.ts, sigma, rates.double())
        return arbitrage_penalty_torch(C, self.m, self.tau).total

    def __call__(self, fake, past_log_surface, rates):
        return self.penalty(fake, past_log_surface, rates).mean().to(fake.dtype)


def match_alpha_phi(norm_bce, norm_phi):
    """alpha_phi by the fixed rule (module docstring); returns the value and how it was obtained, for run.json."""
    norm_bce, norm_phi = np.asarray(norm_bce, dtype=float), np.asarray(norm_phi, dtype=float)
    nonzero = norm_phi > PHI_GRAD_TOL
    out = {"alpha_phi_updates": int(len(norm_phi)), "alpha_phi_zero_share": float(1 - nonzero.mean())}
    if nonzero.mean() >= PHI_MIN_SHARE:
        out.update(alpha_phi=float(np.mean(norm_bce[nonzero] / norm_phi[nonzero])),
                   alpha_phi_rule="mean of ratios over updates with |grad Phi| > %g" % PHI_GRAD_TOL,
                   alpha_phi_n_used=int(nonzero.sum()))
    elif nonzero.any():
        out.update(alpha_phi=float(norm_bce.mean() / norm_phi.mean()),
                   alpha_phi_rule="ratio of means over all updates (fewer than %d%% had a Phi gradient)"
                   % (100 * PHI_MIN_SHARE), alpha_phi_n_used=int(len(norm_phi)))
    else:
        raise ValueError("no match-phase update had a Phi gradient above %g: alpha_phi cannot be matched" % PHI_GRAD_TOL)
    return out


def discriminator_step(gen, disc, disc_opt, bce, cond, real, noise_dim):
    """One step on J_D (eq 14). The generator is only evaluated, not updated."""
    noise = torch.randn(len(cond), noise_dim, device=cond.device)
    with torch.no_grad():
        fake = gen(noise, cond)
    p_real = disc(torch.cat([cond, real], dim=-1))
    p_fake = disc(torch.cat([cond, fake], dim=-1))
    loss = 0.5 * bce(p_real, torch.ones_like(p_real)) + 0.5 * bce(p_fake, torch.zeros_like(p_fake))
    disc_opt.zero_grad()
    loss.backward()
    disc_opt.step()
    return loss.item(), p_real.mean().item(), p_fake.mean().item()


def generator_step(gen, disc, gen_opt, bce, smoothness, cond, noise_dim, phase, on, alpha_m, alpha_tau,
                   bce_weight=0.5, phi=None, rates=None, alpha_phi=None):
    """
    One step on J_G (eq 13). Returns the terms' values and, in the match phase, their gradient norms.
    phase "match": step on the BCE term only. phase "train": step on the whole of eq 13.
    on: the surface the penalties are computed on ("g" in the paper; "sigma" only in the F8 ablation).
    bce_weight: eq 13's 1/2 (paper); 1.0 reproduces the released code.
    phi: None (the paper: Phi is out of the loss and not computed at all) or a TorchPhi (Part 1, lever 2), with
    rates (B, lt) the batch's predicted-day rates and alpha_phi its weight in the train phase.
    Returns (bce, L_m, L_tau, norms, phi value or nan); norms has a fourth entry, |grad Phi|, when phi is given.
    """
    noise = torch.randn(len(cond), noise_dim, device=cond.device)
    fake = gen(noise, cond)
    p_fake = disc(torch.cat([cond, fake], dim=-1))
    bce_term = bce_weight * bce(p_fake, torch.ones_like(p_fake))          # -1/2 E[log D(a, G(a, z))]
    L_m, L_tau = (x.mean() for x in smoothness(penalised_surface(fake, cond[:, N_SCALARS:], on)))
    phi_term = None if phi is None else phi(fake, cond[:, N_SCALARS:], rates)   # E[Phi(sigma_hat)]
    norms = None
    if phase == "match":
        params = list(gen.parameters())
        norms = (grad_norm(bce_term, params), grad_norm(L_m, params), grad_norm(L_tau, params))
        if phi_term is not None:
            norms += (grad_norm(phi_term, params),)
        loss = bce_term
    else:
        loss = bce_term + alpha_m * L_m + alpha_tau * L_tau
        if phi_term is not None:
            loss = loss + alpha_phi * phi_term
    gen_opt.zero_grad()
    loss.backward()
    gen_opt.step()
    return bce_term.item(), L_m.item(), L_tau.item(), norms, (np.nan if phi_term is None else phi_term.item())


def train(gen, disc, gen_opt, disc_opt, condition, target, smoothness, config, phase, n_epochs,
          alpha_m=None, alpha_tau=None, progress=False, rates=None, phi=None, alpha_phi=None):
    """
    The one training loop (A3). Returns a dict with per-epoch averages ("log"), and in the match phase
    the per-update gradient norms and alpha_m, alpha_tau (and alpha_phi with its rule, when phi is given).
    rates (n, lt): each sample's predicted-day rates, permuted with the SAME perm as condition and target every
    epoch so rows stay aligned; needed only with phi (Part 1, lever 2), whose per-epoch mean goes in the log.
    """
    if phase not in ("match", "train"):
        raise ValueError("phase must be 'match' or 'train', got %r" % (phase,))
    if phase == "train" and (alpha_m is None or alpha_tau is None):
        raise ValueError("the train phase needs alpha_m and alpha_tau from the match phase")
    if phi is not None and (rates is None or len(rates) != len(condition)):
        raise ValueError("Phi in the loss needs one row of rates per sample")
    if phi is not None and phase == "train" and alpha_phi is None:
        raise ValueError("the train phase with Phi in the loss needs alpha_phi from the match phase")
    on = config.match_on if phase == "match" else "g"                      # F8: training always on g
    bce = nn.BCELoss()
    n = len(condition)
    batches = batch_slices(n, config.batch_size)
    keys = ["d_loss", "d_real", "d_fake", "g_bce", "L_m", "L_tau"] + (["phi"] if phi is not None else [])
    log = {k: [] for k in keys}
    norms = []
    gen.train()
    disc.train()
    for _ in tqdm(range(n_epochs), desc=phase, disable=not progress):
        perm = torch.randperm(n).to(condition.device)                      # reshuffle every epoch
        condition, target = condition[perm], target[perm]
        if rates is not None:
            rates = rates[perm]                                            # the same perm: rows stay aligned
        sums = dict.fromkeys(keys, 0.0)
        for s in batches:
            cond, real = condition[s], target[s]
            d_loss, d_real, d_fake = discriminator_step(gen, disc, disc_opt, bce, cond, real, config.noise_dim)
            g_bce, L_m, L_tau, nrm, phi_value = generator_step(
                gen, disc, gen_opt, bce, smoothness, cond, config.noise_dim, phase, on, alpha_m, alpha_tau,
                config.gen_bce_weight, phi, None if rates is None else rates[s], alpha_phi)
            if nrm is not None:
                norms.append(nrm)
            w = len(cond) / n                                              # epoch averages weight by batch size
            for k, v in zip(keys, (d_loss, d_real, d_fake, g_bce, L_m, L_tau, phi_value)):
                sums[k] += w * v
        for k in keys:
            log[k].append(sums[k])
    out = {"log": {k: np.array(v) for k, v in log.items()}}
    if phase == "match":
        norms = np.array(norms)                                            # columns: BCE, L_m, L_tau (, Phi)
        out["norms"] = norms
        out["alpha_m"] = float(np.mean(norms[:, 0] / norms[:, 1]))
        out["alpha_tau"] = float(np.mean(norms[:, 0] / norms[:, 2]))
        if phi is not None:
            out.update(match_alpha_phi(norms[:, 0], norms[:, 3]))
    return out


def make_optimisers(gen, disc, config):
    return (torch.optim.RMSprop(gen.parameters(), lr=config.lr_g),
            torch.optim.RMSprop(disc.parameters(), lr=config.lr_d))


def check_config(config):
    if config.smoothness not in ("paper", "released"):
        raise ValueError("smoothness must be 'paper' or 'released'")
    if config.match_on not in ("g", "sigma"):
        raise ValueError("match_on must be 'g' or 'sigma'")
    if not isinstance(config.restart, bool):
        raise ValueError("restart must be True or False")
    if config.gen_bce_weight <= 0:
        raise ValueError("gen_bce_weight must be positive")
    if config.n_grad < 1:
        raise ValueError("n_grad must be at least 1: alpha_m and alpha_tau come from the matching epochs")
    if config.n_epochs < 0:
        raise ValueError("n_epochs cannot be negative")
    if not isinstance(config.phi_in_loss, bool):
        raise ValueError("phi_in_loss must be True or False")
    if config.rate_source not in SOURCES:
        raise ValueError("rate_source must be one of %s" % (SOURCES,))


@dataclass
class FitResult:
    gen: nn.Module
    disc: nn.Module
    alpha_m: float
    alpha_tau: float
    match: dict      # log, norms, alphas of the match phase
    trained: dict    # log of the train phase
    alpha_phi: float = None   # Part 1, lever 2: the weight of E[Phi] (None when Phi is out of the loss)
    phi_info: dict = None     # how alpha_phi was matched: rule, updates used, share with a zero Phi gradient


def fit(condition, target, m, tau, config, device, checkpoint_dir=None, progress=False, rates=None):
    """
    The whole of section 3.4: build, match, restart, train. condition and target are training tensors
    on the device (Stage 4); m and tau are the grid (Stage 1).
    rates: (n, lt) float64 tensor of each sample's predicted-day rates (to_tensors(..., with_rates=True)); needed
    only when config.phi_in_loss (Part 1, lever 2), and otherwise ignored.
    Checkpoints (A9), if checkpoint_dir is given: 0_initial.pt, 1_matched.pt, 2_trained.pt.
    """
    check_config(config)
    set_seed(config.seed)
    gen, disc = build_networks(condition.shape[1], target.shape[1], config, device)
    gen_opt, disc_opt = make_optimisers(gen, disc, config)
    smoothness = make_smoothness(config.smoothness, m, tau, device)
    initial = copy.deepcopy(gen.state_dict()), copy.deepcopy(disc.state_dict())
    phi = None
    if config.phi_in_loss:
        if rates is None:
            raise ValueError("phi_in_loss needs the training rates: fit(..., rates=...)")
        phi = TorchPhi(m, tau, device)
    else:
        rates = None                                     # the paper: Phi is out of the loss and never computed
    extra = {}

    def checkpoint(name, a_m, a_tau):
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(os.path.join(checkpoint_dir, name), gen, disc, gen_opt, disc_opt,
                            a_m, a_tau, config.seed, config.as_dict(), extra)

    checkpoint("0_initial.pt", None, None)
    match = train(gen, disc, gen_opt, disc_opt, condition, target, smoothness, config, "match",
                  config.n_grad, progress=progress, rates=rates, phi=phi)
    alpha_m, alpha_tau = match["alpha_m"], match["alpha_tau"]
    alpha_phi = match.get("alpha_phi")
    phi_info = {k: v for k, v in match.items() if k.startswith("alpha_phi")} if phi is not None else None
    extra.update(phi_info or {})
    checkpoint("1_matched.pt", alpha_m, alpha_tau)

    if config.restart:                                                     # F9: back to the initialisation
        gen.load_state_dict(initial[0])
        disc.load_state_dict(initial[1])
        gen_opt, disc_opt = make_optimisers(gen, disc, config)
    trained = train(gen, disc, gen_opt, disc_opt, condition, target, smoothness, config, "train",
                    config.n_epochs, alpha_m, alpha_tau, progress=progress, rates=rates, phi=phi, alpha_phi=alpha_phi)
    checkpoint("2_trained.pt", alpha_m, alpha_tau)
    return FitResult(gen, disc, alpha_m, alpha_tau, match, trained, alpha_phi, phi_info)


# =====================================================================================================================
# 4. Run folders (was runs.py)
# =====================================================================================================================

# Stage 8 (A9): a training run is a folder, so training and simulation are separate steps.
#
#     run_dir/0_initial.pt, 1_matched.pt, 2_trained.pt   checkpoints from fit() (Stage 7)
#     run_dir/run.json                                    index, data file, split, alphas, settings, time taken
#     run_dir/logs.npz                                    training logs (per-epoch averages, matching norms)
#
# load_run() rebuilds everything from the folder: the data and split from run.json, the networks from 2_trained.pt.
#
# Part 1:
#   - the data come from ONE call, data.load(index): surfaces and rates on their common days (G4: config.rate_source
#     "rates" uses them for Phi in the loss, the simulation and the evaluation; "zero" reads the surfaces only);
#     every Dataset carries its rates;
#   - the split is the three-way split by fixed dates (G1); runs trained before Part 1 (no dates in their config)
#     still load, with their split by calendar year;
#   - the networks train and condition on the raw vendor grid, which is also what everything is scored against;
#   - run.json also records the block boundaries and sizes, alpha_phi and its rule, sim_seed, the rate source, the
#     input-file hashes and the library versions (G8).

def split_record(split):
    """What run.json keeps about a split, to check it again at load time."""
    rec = {"counts": split.counts(), "bounds": split.bounds}
    rec["first_last"] = {b: [str(getattr(split, b).dates[i].date()) for i in (0, -1)]
                         for b in ("train", "val", "test") if getattr(split, b) is not None}
    return rec


def input_hashes(index, config):
    """Hashes of the input workbooks a run depends on (run.json; resumability of the matrix scripts)."""
    out = {"data_file": os.path.basename(INDEX_FILES[index]), "data_hash": file_hash(INDEX_FILES[index])}
    if config.rate_source == "rates":
        out.update(rates_file=os.path.basename(RATES_FILE), rates_hash=file_hash(RATES_FILE))
    return out


def train_run(index, config, run_dir, device, progress=True, extra_info=None):
    """Train on the index's training block and save the run folder. Returns fit()'s result."""
    P = prepare_all(index, config, progress=progress)
    cond, targ, rates = to_tensors(P.inputs.train, device, with_rates=True)
    t0 = time.time()
    res = fit(cond, targ, P.data.m, P.data.tau, config, device, checkpoint_dir=run_dir, progress=progress,
              rates=rates)
    info = {
        "index": index, **input_hashes(index, config),
        "rate_source": config.rate_source, "seed": config.seed, "sim_seed": config.sim_seed,
        "phi_in_loss": config.phi_in_loss,
        "split": split_record(P.inputs),
        "train_fraction": P.inputs.train_fraction,
        "n_train": len(P.inputs.train.dates), "n_val": len(P.inputs.val.dates), "n_test": len(P.inputs.test.dates),
        "alpha_m": res.alpha_m, "alpha_tau": res.alpha_tau, "alpha_phi": res.alpha_phi, **(res.phi_info or {}),
        "seconds": round(time.time() - t0, 1), "device": str(device), "versions": library_versions(),
        "config": config.as_dict(), **(extra_info or {}),
    }
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(info, f, indent=2)
    logs = {"match_" + k: v for k, v in res.match["log"].items()}
    logs.update({"train_" + k: v for k, v in res.trained["log"].items()})
    np.savez(os.path.join(run_dir, "logs.npz"), match_norms=res.match["norms"], **logs)
    return res


@dataclass
class Run:
    gen: object
    disc: object
    config: Config
    info: dict
    data: object     # SurfaceData (Stage 1), the raw vendor grid
    split: object    # Split (Stage 4) of the model's inputs; split.val (Part 1) or split.test is what simulate() needs
    market: object = None     # Split of the raw market (G9); the same object as split when nothing is processed
    prepared: object = None


# Settings that existed while the data lever (kernel smoothing of the vendor grid, and the projection onto the
# no-arbitrage set) was part of the code (removed 2026-10-14, DECISIONS.md section 6) with the values every run on the
# raw grid carries. They are no longer Config fields; they are put back inside signature() only, so the signatures
# (and hence the identity) of runs trained before the removal do not change, and dropped from a stored config
# before it is read back.
_REMOVED_FIELDS = {"data_smooth": False, "smooth_h1": 0.002, "smooth_h2": 0.046, "smooth_exclude_days": [270],
                   "data_repair": False, "repair_norm": "l2"}


def load_run(run_dir, device):
    with open(os.path.join(run_dir, "run.json")) as f:
        info = json.load(f)
    legacy = "train_end" not in info["config"]          # trained before Part 1: split by calendar year
    stored = dict(info["config"])
    if stored.get("data_smooth") or stored.get("data_repair"):
        raise ValueError("%s was trained on the kernel-smoothed or projected grid, which is no longer part of the "
                         "code (DECISIONS.md, section 6)" % run_dir)
    for key in _REMOVED_FIELDS:                         # runs trained before the removal carry these keys
        stored.pop(key, None)
    if legacy:
        stored.setdefault("rate_source", "zero")        # trained and evaluated with r = 0 before any rates existed
    config = Config(**stored)
    P = prepare_all(info["index"], config, legacy=legacy, surfaces=info.get("data_file"))   # the run's own workbook
    split = P.inputs
    if legacy:
        if split.test_start_year != info["test_start_year"] or len(split.test.dates) != info["n_test"]:
            raise ValueError("the data no longer gives the split this run was trained on")
    else:
        now, then = split_record(split), info["split"]
        if now["bounds"] != then["bounds"] or now["counts"] != then["counts"] or now["first_last"] != then["first_last"]:
            raise ValueError("the data no longer gives the split this run was trained on: %s, not %s" % (now, then))
    K = P.data.lk * P.data.lt
    gen, disc = build_networks(N_SCALARS + K, 1 + K, config, device)
    g_opt, d_opt = make_optimisers(gen, disc, config)
    load_checkpoint(os.path.join(run_dir, "2_trained.pt"), gen, disc, g_opt, d_opt, device)
    gen.eval()
    disc.eval()
    return Run(gen, disc, config, info, P.data, split, P.market, P)


# =====================================================================================================================
# 5. The training step (was pipeline.py)
# =====================================================================================================================

def signature(index, config):
    """
    Hash of everything a trained run depends on: the configuration and the input files.
    _REMOVED_FIELDS puts back the settings of the kernel smoothing, removed from Config on 2026-10-14, with
    the values every non-smoothed run carries: it exists only so the signatures of runs trained before the removal
    do not change (tests/fixtures/baseline.json).
    """
    parts = {"index": index, "config": {**config.as_dict(), **_REMOVED_FIELDS}, "inputs": input_hashes(index, config)}
    return text_hash(json.dumps(parts, sort_keys=True, default=str))


# ---- Step 2: training (Stage 7)

def time_one_run(index, config, device, epochs=100):
    """Seconds for the match phase plus `epochs` training epochs of one configuration, in a throwaway folder."""
    P = prepare_all(index, config)
    cond, targ, rates = to_tensors(P.inputs.train, device, with_rates=True)
    t0 = time.time()
    fit(cond, targ, P.data.m, P.data.tau, replace(config, n_epochs=epochs), device, rates=rates)   # nothing saved
    return time.time() - t0


def train_step(index="SPX", names=tuple(TRAINING_CONFIGS), seeds=SEEDS, overrides=None, device=None,
               time_first=True, progress=True):
    """
    Train every (training configuration, seed) not yet trained. Before the matrix, time one 100-epoch run per loss
    setting in the plan and print the projected total time.
    """
    overrides = overrides or {}
    device = device or get_device()
    todo = []
    for name, seed, config, folder in _plan(index, names, seeds, overrides):
        check_inputs(index, need_rates=config.rate_source == "rates")
        sig = signature(index, config)
        if _done(os.path.join(folder, "run.json"), "signature", sig, "run %s/%s/seed_%d" % (index, name, seed)):
            log("skip %s seed %d: already trained" % (name, seed))
            continue
        todo.append((name, seed, config, folder, sig))
    if not todo:
        return
    if time_first:
        total = 0.0
        for phi_in in sorted({c.phi_in_loss for _, _, c, _, _ in todo}):
            name, seed, config, _, _ = next(t for t in todo if t[2].phi_in_loss == phi_in)
            t = time_one_run(index, config, device)
            per_run = t * (config.n_grad + config.n_epochs) / (config.n_grad + 100)
            k = sum(c.phi_in_loss == phi_in for _, _, c, _, _ in todo)
            total += k * per_run
            log("timing (%s, Phi %s the loss): match + 100 epochs took %.0f s -> about %.1f h per run, %d run(s)"
                % (name, "in" if phi_in else "out of", t, per_run / 3600, k))
        log("projected total training time: about %.1f hours for %d run(s) on %s" % (total / 3600, len(todo), device))
    for name, seed, config, folder, sig in todo:
        log("train %s %s seed %d (%d epochs) -> %s" % (index, name, seed, config.n_epochs, folder))
        res = train_run(index, config, folder, device, progress=progress,
                        extra_info={"training_config": name, "signature": sig})
        log("done: alpha_m %.3e, alpha_tau %.3e%s" % (res.alpha_m, res.alpha_tau, "" if res.alpha_phi is None
                                                       else ", alpha_phi %.3e" % res.alpha_phi))


# ---- Step 3: simulation (Stage 8)

def _trained(index, names, seeds):
    for name in names:
        for seed in seeds:
            folder = run_path(index, name, seed)
            if os.path.exists(os.path.join(folder, "run.json")):
                yield name, seed, folder
            else:
                log("no trained run %s/%s/seed_%d: skipped" % (index, name, seed))
