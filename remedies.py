"""
Part 2, Part B: two remedies for the 21-day recursion, run through exactly the Part A experiment (PART2_PartB_BUILD_BRIEF).

The one principle: any difference between a remedied model and Part A is caused by the remedy and nothing else. The
networks, seeds, calendar-indexed noise, starts, validity rule, benchmark, margins and bootstrap are Part A's, and are
ASSERTED against the Part A files (check_fixed), not assumed.

Models (step 1 is always the control's step: network G0, no clipping):
    control       Part A, re-used (re-run only for the regression gate)
    bounded       the bounded step at k >= 2, network G0
    dad           no clip, network G1 = G0 fine-tuned on its own unclipped rollouts at k >= 2
    bounded_dad   clip at k >= 2, network G1b = G0 fine-tuned on its own clipped rollouts  (the primary model)

Remedy 1, the bounded step: the generator's log-vol increment is clipped node by node at the 0.1% and 99.9% quantiles
of the training block's daily log-vol changes (clip_bounds.json); returns are never clipped.
Remedy 2, Data as Demonstrator (Venkatraman, Hebert and Bagnell 2015): G0 rolls on its own output over the TRAINING
block (POOL_SEED noise, 4 paths per start, every start); each state that feeds steps 2..21 is paired with what the
market did next (the real return, and the real surface minus the synthetic one), and G0 is trained further for one
round on these pairs mixed 1:1 with the real ones, with the unchanged loss, optimisers and alphas.

Layout (VOLGAN_PARTB_DIR, default <code>/outputs/part2/partB):
    clip_bounds.json                 gates/regression.json            pilot.json, pilot_report.md
    pool/<variant>/seed_<k>.npz      networks/<variant>/seed_<k>/     ft_trained.pt, finetune.json, ft_logs.npz
    <model_id>/                      the Part A layout: work/, metrics_<k>.npz, results_all.npz, degradation.csv,
                                     survival.csv, tests.csv, run.json, identity_check.json, descriptive.npz
    comparison/                      classes.csv, delta.csv, attribution.csv, descriptive.csv, checks.json
    REPORT.md
Every step is resumable in the Part A style (signatures; a different configuration is refused, never mixed).
"""

from dataclasses import asdict, dataclass
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .common import (
    CODE_DIR, INDEX_FILES, RATES_FILE, RecursionConfig, SIM_SEED, check_inputs, file_hash, get_device,
    library_versions, load_checkpoint, log, save_checkpoint, set_seed, text_hash)
from .preprocessing import N_SCALARS, to_tensors
from .training import (batch_slices, build_networks, discriminator_step, generator_step, make_optimisers,
                       make_smoothness)
from .recursion import (
    FIXED_NOISE_DATES, Arena, NoiseTable, _rec_runs, bench_clouds, consolidate, identity_gate, is_done, load_npz,
    noise_hashes, parta_root, rollout, run_seed, run_start, save_npz, seed_signature, start_indices)
from .recursion_analysis import (
    NINE, PILLARS, analyse, block_indices, curve, load_margins, load_tables, save_results, split_runs, sufficient,
    write_tables)

REMEDY_VERSION = "partB-1"
DATA_HASH, RATES_HASH = "096285d9903266d9", "4afbe6a988420ed5"     # brief section 0: the Part A input files


@dataclass(frozen=True)
class RemedyConfig:
    """Brief section 8: fixed in advance, never changed after a run exists."""
    clip_q: tuple = (0.001, 0.999)   # per node, training-block daily log-vol changes
    pool_seed: int = 20261016        # the pool's noise; never used anywhere else
    pool_paths: int = 4              # per start
    ft_epochs: int = 2000
    ft_seed_base: int = 30000        # FT_SEED = 30000 + training seed
    rounds: int = 1

    def as_dict(self):
        return asdict(self)


RC = RemedyConfig()
VARIANTS = {"dad": False, "bounded_dad": True}                     # variant -> its pool and deployment are clipped
MODELS = {"bounded": (True, None), "dad": (False, "dad"), "bounded_dad": (True, "bounded_dad"),   # clip, late net
          "bounded_proj": (True, None)}                            # second remedy: clip + projection (projection.py)
PROJ_MODELS = ("bounded_proj",)                                     # models whose rollout projects at k >= 2
ORDER = ("control", "bounded", "dad", "bounded_dad")               # the first brief's 2 x 2
ORDER_ALL = ORDER + PROJ_MODELS


class GateError(RuntimeError):
    """A Part B gate failed: stop and report, do not change settings (brief section 0)."""


def partb_root():
    return os.path.abspath(os.environ.get("VOLGAN_PARTB_DIR") or os.path.join(CODE_DIR, "outputs", "part2", "partB"))


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


# =====================================================================================================================
# 1. The fixed items of brief section 0, asserted against the Part A files
# =====================================================================================================================

def check_fixed(cfg, runs, parta=None):
    """
    Assert that everything held fixed equals Part A: networks, data, settings, noise, margins. Returns (Part A run.json,
    Part A margins dict, deltas). runs: {seed: (folder, Run)}.
    """
    parta = parta or parta_root()
    with open(os.path.join(parta, "run.json")) as f:
        info = json.load(f)
    margins, deltas = load_margins(os.path.join(parta, "margins.json"))
    bad = []
    for s, (folder, _) in runs.items():
        if file_hash(os.path.join(folder, "run.json")) != info["network_run_json_hashes"][str(s)]:
            bad.append("network seed %d is not the Part A network" % s)
    for name, path, want in (("data", INDEX_FILES[cfg.index], DATA_HASH), ("rates", RATES_FILE, RATES_HASH)):
        h = file_hash(path)
        if h != want or h != margins[name + "_hash"]:
            bad.append("%s file hash %s, Part A %s, brief %s" % (name, h, margins[name + "_hash"], want))
    if tuple(cfg.seeds) != tuple(info["settings"]["seeds"]):
        bad.append("seeds %s, Part A %s" % (cfg.seeds, info["settings"]["seeds"]))
    for k in ("horizon", "n_paths", "stride", "start_offset", "sim_seed", "n_draw", "n_boot", "boot_seed",
              "block_len_starts", "vol_lo", "vol_hi", "index", "network_cfg", "delta_c", "delta_d", "delta_a_mult"):
        if getattr(cfg, k) != info["settings"][k]:
            bad.append("%s = %s, Part A %s" % (k, getattr(cfg, k), info["settings"][k]))
    if abs(deltas["A"] - 0.00029509366742844) > 1e-15 or deltas["C"] != 0.05 or deltas["D"] != 0.10:
        bad.append("margins %s are not Part A's" % deltas)
    if bad:
        raise GateError("fixed items differ from Part A:\n  " + "\n  ".join(bad))
    return info, margins, deltas


def check_noise(noise, ds, info):
    """The noise hashes of this run equal Part A's run.json noise_hashes (same dates)."""
    now = noise_hashes(noise, [ds.dates[0], ds.dates[len(ds.dates) // 2], ds.dates[-1]])
    now.update(noise_hashes(noise, FIXED_NOISE_DATES))
    if now != info["noise_hashes"]:
        raise GateError("noise hashes differ from Part A: %s vs %s" % (now, info["noise_hashes"]))
    return now


def parta_bench(arena, run, folder, seed, parta=None):
    """Part A's benchmark of this seed, after checking it is the UNCHANGED control's (its signature)."""
    parta = parta or parta_root()
    path = os.path.join(parta, "work", "control", "bench_seed_%d.npz" % seed)
    b = load_npz(path)
    want = seed_signature(run, folder, RecursionConfig(seeds=arena.cfg.seeds))
    if b["signature"] != want or len(b["days"]) != arena.n:
        raise GateError("the Part A benchmark of seed %d is not the control's (%s vs %s)" % (seed, b["signature"], want))
    return b


# =====================================================================================================================
# 2. Remedy 1: the bounded step
# =====================================================================================================================

def clip_bounds(run):
    """(lo, hi) per node: the 0.1% and 99.9% quantiles of the training block's daily log-vol changes, and a record."""
    dg = run.split.train.target[:, 1:]                                   # g(d) - g(c), (2389, 42)
    lo = np.quantile(dg, RC.clip_q[0], axis=0)
    hi = np.quantile(dg, RC.clip_q[1], axis=0)
    m, tau = run.data.m, run.data.tau
    nodes = ["m=%.2f %dd" % (m[i], round(tau[j] * 365)) for j in range(len(tau)) for i in range(len(m))]
    rec = {"lo": lo.tolist(), "hi": hi.tolist(), "nodes": nodes, "quantile_levels": list(RC.clip_q),
           "n_training_days": int(dg.shape[0]), "first": str(run.split.train.dates[0].date()),
           "last": str(run.split.train.dates[-1].date()), "data_hash": file_hash(INDEX_FILES[run.info["index"]]),
           "applies_to": "log-vol increments at steps k >= 2; returns never"}
    return lo, hi, rec


def clipbounds_step(root=None, device=None):
    root, device = root or partb_root(), device or get_device()
    cfg = RecursionConfig()
    runs = _rec_runs(cfg, device)
    check_fixed(cfg, runs)
    lo, hi, rec = clip_bounds(runs[cfg.seeds[0]][1])
    for s, (_, run) in runs.items():                                    # every seed sees the same training block
        lo2, hi2, _ = clip_bounds(run)
        if not (np.array_equal(lo, lo2) and np.array_equal(hi, hi2)):
            raise GateError("clip bounds differ between seeds (seed %d)" % s)
    path = os.path.join(root, "clip_bounds.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != json.loads(json.dumps(rec)):
            raise RuntimeError("clip_bounds.json exists with different numbers: %s" % path)
        log("clip bounds: unchanged (%s)" % path)
    else:
        _write_json(path, rec)
        log("clip bounds: lo in [%.4f, %.4f], hi in [%.4f, %.4f] over %d training days -> %s"
            % (lo.min(), lo.max(), hi.min(), hi.max(), rec["n_training_days"], path))
    return rec


def load_clip(root=None):
    path = os.path.join(root or partb_root(), "clip_bounds.json")
    if not os.path.exists(path):
        raise FileNotFoundError("no %s: run  python part2b.py clipbounds  first" % path)
    with open(path) as f:
        rec = json.load(f)
    return (np.array(rec["lo"], float), np.array(rec["hi"], float)), file_hash(path)


# =====================================================================================================================
# 3. Remedy 2: the pool of synthetic states and the fine-tuning (Data as Demonstrator)
# =====================================================================================================================

POOL_TOL = 1e-12        # cond_syn[3:] + target_syn[1:] == g_real(j): exact up to one float64 rounding


def pool_cfg():
    """The rollout settings of the pool: the evaluation's horizon and validity rule, 4 paths, the training block."""
    return RecursionConfig(block="train", n_paths=RC.pool_paths, sim_seed=RC.pool_seed)


def build_pool(run, clip=None, noise=None, log=print, every=500):
    """
    The pool of one network (brief 3.1): from every training start j0 (j0 + 20 <= n - 1), 4 paths, 21 steps, the same
    rollout (and clip) as the evaluation, noise from POOL_SEED. For k = 2..21 the condition FEEDING step k of every path
    still valid after step k - 1, with its target: the real return of the target sample j = j0 + k - 1 and the real
    log-vol surface of that day minus the synthetic one. Returns the arrays and a summary.
    """
    cfg = pool_cfg()
    ds, data = run.split.train, run.data
    noise = noise or NoiseTable(RC.pool_seed, RC.pool_paths, cfg.n_draw, run.config.noise_dim)
    n = len(ds.dates)
    starts = np.arange(0, n - cfg.horizon + 1)                          # j0 + 20 <= n - 1
    g_real = ds.past_log_surface + ds.target[:, 1:]                     # the real log-vol surface of the predicted day
    cond, tgt, rates, ks, j0s, js, paths = [], [], [], [], [], [], []
    dropped = np.zeros(cfg.horizon + 1, dtype=int)
    worst = 0.0
    t0 = time.time()

    def capture(k, c, r_hat, dg_hat, valid, buf):                       # read only: the condition is returned as is
        buf.append((k, c[valid].copy(), np.flatnonzero(valid)))
        dropped[k] += int((~valid).sum())
        return c, None

    for i, j0 in enumerate(starts):
        buf = []
        for _ in rollout(run.gen, ds, data, int(j0), noise, cfg, post_step=capture, ctx=buf, clip=clip):
            pass
        for k, c, idx in buf:
            j = int(j0) + k - 1
            t_surf = g_real[j] - c[:, N_SCALARS:]
            worst = max(worst, float(np.abs(c[:, N_SCALARS:] + t_surf - g_real[j]).max()) if len(c) else 0.0)
            cond.append(c)
            tgt.append(np.column_stack([np.full(len(c), ds.target[j, 0]), t_surf]))
            rates.append(np.repeat(ds.rates[j][None], len(c), axis=0))
            ks.append(np.full(len(c), k)), j0s.append(np.full(len(c), j0)), js.append(np.full(len(c), j))
            paths.append(idx)
        if every and (i + 1) % every == 0:
            log("  pool: %d/%d starts, %.0f s" % (i + 1, len(starts), time.time() - t0))
    if worst > POOL_TOL:
        raise GateError("pool targets do not rebuild the real surface: max error %.3e > %g" % (worst, POOL_TOL))
    out = {"cond": np.concatenate(cond), "target": np.concatenate(tgt), "rates": np.concatenate(rates),
           "k": np.concatenate(ks), "j0": np.concatenate(j0s), "j": np.concatenate(js), "path": np.concatenate(paths)}
    abs_t = np.abs(out["target"][:, 1:]).mean(axis=1)
    summary = {"n_pairs": int(len(out["k"])), "n_starts": int(len(starts)), "max_pairs": int(len(starts) * 20 * RC.pool_paths),
               "dropped_by_k": {int(k): int(dropped[k]) for k in range(2, cfg.horizon + 1)},
               "dropped_total": int(dropped.sum()), "identity_max_error": worst,
               "mean_abs_target_surface_by_k": {int(k): float(abs_t[out["k"] == k].mean()) for k in range(2, cfg.horizon + 1)},
               "first_target_date": str(ds.dates[out["j"].min()].date()), "last_target_date": str(ds.dates[out["j"].max()].date()),
               "seconds": round(time.time() - t0, 1)}
    return out, summary


def pool_path(root, variant, seed):
    return os.path.join(root, "pool", variant, "seed_%d.npz" % seed)


def pool_signature(folder, variant, clip_hash):
    parts = {"version": REMEDY_VERSION, "variant": variant, "network": file_hash(os.path.join(folder, "2_trained.pt")),
             "clip": clip_hash if VARIANTS[variant] else None, "remedy": RC.as_dict(),
             "pool_cfg": {k: getattr(pool_cfg(), k) for k in ("horizon", "n_paths", "n_draw", "vol_lo", "vol_hi", "block")},
             "data": [DATA_HASH, RATES_HASH]}
    return text_hash(json.dumps(parts, sort_keys=True, default=str))


def pool_step(variant, root=None, device=None, seeds=None):
    root, device = root or partb_root(), device or get_device()
    cfg = RecursionConfig()
    runs = _rec_runs(cfg, device)
    check_fixed(cfg, runs)
    clip, clip_hash = load_clip(root) if VARIANTS[variant] else (None, None)
    noise = None
    for s, (folder, run) in runs.items():
        if seeds and s not in seeds:
            continue
        path, sig = pool_path(root, variant, s), pool_signature(folder, variant, clip_hash)
        if is_done(path, sig, "pool %s seed %d" % (variant, s)):
            log("pool %s seed %d: done" % (variant, s))
            continue
        noise = noise or NoiseTable(RC.pool_seed, RC.pool_paths, cfg.n_draw, run.config.noise_dim)
        log("pool %s seed %d: building" % (variant, s))
        out, summ = build_pool(run, clip, noise, log)
        summ.update(variant=variant, seed=s, pool_seed=RC.pool_seed, signature=sig)
        save_npz(path, signature=sig, seed=s, variant=variant,
                 cond=out["cond"].astype(np.float32), target=out["target"].astype(np.float32),
                 rates=out["rates"].astype(np.float32), k=out["k"].astype(np.int16), j0=out["j0"].astype(np.int32),
                 j=out["j"].astype(np.int32), path_index=out["path"].astype(np.int16))
        _write_json(path.replace(".npz", "_summary.json"), summ)
        log("pool %s seed %d: %d pairs (of %d), %d states dropped (dead paths), %.0f s"
            % (variant, s, summ["n_pairs"], summ["max_pairs"], summ["dropped_total"], summ["seconds"]))


class FinetuneError(RuntimeError):
    """A loss became NaN or infinite during fine-tuning: stop and report (brief 3.2.5)."""


def net_dir(root, variant, seed):
    return os.path.join(root, "networks", variant, "seed_%d" % seed)


def finetune(run, folder, pool_file, variant, seed, out_dir, device, log=print, epochs=None):
    """
    One round of fine-tuning of G0 (brief 3.2): the control's networks, BOTH RMSprop states and alphas from 2_trained.pt;
    every epoch all real training pairs plus as many pool pairs, drawn without replacement (a fresh draw every epoch),
    concatenated, shuffled, batches of 100, the existing discriminator_step and generator_step (phase "train", on "g").
    """
    epochs = RC.ft_epochs if epochs is None else epochs
    config = run.config
    K = run.data.lk * run.data.lt
    gen, disc = build_networks(N_SCALARS + K, 1 + K, config, device)
    gen_opt, disc_opt = make_optimisers(gen, disc, config)
    alpha_m, alpha_tau, _, _ = load_checkpoint(os.path.join(folder, "2_trained.pt"), gen, disc, gen_opt, disc_opt, device)
    if alpha_m != run.info["alpha_m"] or alpha_tau != run.info["alpha_tau"]:
        raise GateError("checkpoint alphas (%r, %r) differ from run.json (%r, %r)"
                        % (alpha_m, alpha_tau, run.info["alpha_m"], run.info["alpha_tau"]))
    if config.phi_in_loss or config.smoothness != "paper" or config.gen_bce_weight != 0.5:
        raise GateError("the parent network is not the control configuration")
    cond_r, targ_r = to_tensors(run.split.train, device)
    with np.load(pool_file) as z:
        cond_s = torch.from_numpy(np.ascontiguousarray(z["cond"])).to(torch.float32).to(device)
        targ_s = torch.from_numpy(np.ascontiguousarray(z["target"])).to(torch.float32).to(device)
    n_real, n_pool = len(cond_r), len(cond_s)
    if n_pool < n_real:
        raise GateError("the pool (%d) is smaller than the real block (%d)" % (n_pool, n_real))
    smoothness = make_smoothness(config.smoothness, run.data.m, run.data.tau, device)
    ft_seed = RC.ft_seed_base + seed
    set_seed(ft_seed)                                                   # torch, numpy, random
    rng = np.random.default_rng(ft_seed)                                # the per-epoch pool draws
    bce = nn.BCELoss()
    keys = ("d_loss", "d_real", "d_fake", "g_bce", "L_m", "L_tau")
    logs = {k: np.empty(epochs) for k in keys}
    n = 2 * n_real
    batches = batch_slices(n, config.batch_size)
    gen.train()
    disc.train()
    t0 = time.time()
    for ep in range(epochs):
        pick = torch.from_numpy(rng.choice(n_pool, size=n_real, replace=False)).to(device)
        cond = torch.cat([cond_r, cond_s[pick]])
        targ = torch.cat([targ_r, targ_s[pick]])
        perm = torch.randperm(n).to(device)
        cond, targ = cond[perm], targ[perm]
        sums = dict.fromkeys(keys, 0.0)
        for s in batches:
            c, r = cond[s], targ[s]
            d_loss, d_real, d_fake = discriminator_step(gen, disc, disc_opt, bce, c, r, config.noise_dim)
            g_bce, L_m, L_tau, _, _ = generator_step(gen, disc, gen_opt, bce, smoothness, c, config.noise_dim, "train",
                                                     "g", alpha_m, alpha_tau, config.gen_bce_weight)
            w = len(c) / n
            for k, v in zip(keys, (d_loss, d_real, d_fake, g_bce, L_m, L_tau)):
                sums[k] += w * v
        for k in keys:
            logs[k][ep] = sums[k]
        if not all(np.isfinite(v) for v in sums.values()):
            raise FinetuneError("non-finite loss at epoch %d (%s seed %d): %s" % (ep + 1, variant, seed, sums))
        if (ep + 1) % 250 == 0:
            log("  finetune %s seed %d: epoch %d/%d, d_loss %.4f, g_bce %.4f, %.0f s"
                % (variant, seed, ep + 1, epochs, sums["d_loss"], sums["g_bce"], time.time() - t0))
    gen.eval()
    disc.eval()
    os.makedirs(out_dir, exist_ok=True)
    extra = {"variant": variant, "ft_seed": ft_seed, "parent": file_hash(os.path.join(folder, "2_trained.pt"))}
    save_checkpoint(os.path.join(out_dir, "ft_trained.pt"), gen, disc, gen_opt, disc_opt, alpha_m, alpha_tau, seed,
                    config.as_dict(), extra)
    np.savez(os.path.join(out_dir, "ft_logs.npz"), **logs)
    return {"seconds": round(time.time() - t0, 1), "final": {k: float(logs[k][-1]) for k in keys},
            "n_real": n_real, "n_pool": n_pool, "alpha_m": alpha_m, "alpha_tau": alpha_tau, "ft_seed": ft_seed,
            "rng": "numpy.random.default_rng(%d) for the per-epoch pool draws; set_seed(%d) for torch, numpy, random"
                   % (ft_seed, ft_seed)}


def finetune_signature(folder, pool_file, variant, seed):
    parts = {"version": REMEDY_VERSION, "variant": variant, "seed": seed, "remedy": RC.as_dict(),
             "parent": file_hash(os.path.join(folder, "2_trained.pt")), "parent_run_json": file_hash(os.path.join(folder, "run.json")),
             "pool": file_hash(pool_file)}
    return text_hash(json.dumps(parts, sort_keys=True))


def finetune_step(variant, root=None, device=None, seeds=None):
    root, device = root or partb_root(), device or get_device()
    cfg = RecursionConfig()
    runs = _rec_runs(cfg, device)
    check_fixed(cfg, runs)
    for s, (folder, run) in runs.items():
        if seeds and s not in seeds:
            continue
        pf = pool_path(root, variant, s)
        if not os.path.exists(pf):
            raise FileNotFoundError("no pool %s: run  python part2b.py pool --variant %s  first" % (pf, variant))
        out_dir, sig = net_dir(root, variant, s), finetune_signature(folder, pf, variant, s)
        meta = os.path.join(out_dir, "finetune.json")
        if os.path.exists(meta) and os.path.exists(os.path.join(out_dir, "ft_trained.pt")):
            with open(meta) as f:
                have = json.load(f).get("signature")
            if have == sig:
                log("finetune %s seed %d: done" % (variant, s))
                continue
            raise RuntimeError("%s exists with a different configuration (signature %s, now %s)" % (meta, have, sig))
        log("finetune %s seed %d: %d epochs" % (variant, s, RC.ft_epochs))
        res = finetune(run, folder, pf, variant, s, out_dir, device, log)
        _write_json(meta, {"signature": sig, "variant": variant, "seed": s,
                           "parent_network_hash": file_hash(os.path.join(folder, "2_trained.pt")),
                           "parent_run_json_hash": file_hash(os.path.join(folder, "run.json")),
                           "pool_file": pf, "pool_hash": file_hash(pf), "settings": RC.as_dict(),
                           "config": run.config.as_dict(), "mix": "each epoch: all real pairs + as many pool pairs, "
                           "uniform without replacement, fresh draw", "loss": "eq 13 unchanged, Phi not in the loss",
                           "optimiser": "RMSprop, continued from the 2_trained.pt states", **res,
                           "network_hash": file_hash(os.path.join(out_dir, "ft_trained.pt")),
                           "versions": library_versions(), "device": str(device)})
        log("finetune %s seed %d: done in %.0f s; final d_loss %.4f, g_bce %.4f"
            % (variant, s, res["seconds"], res["final"]["d_loss"], res["final"]["g_bce"]))


def load_late_network(run, variant, seed, root=None, device=None):
    """The fine-tuned generator G1 (or G1b) of one seed, in eval mode, and its file hash."""
    root, device = root or partb_root(), device or get_device()
    path = os.path.join(net_dir(root, variant, seed), "ft_trained.pt")
    if not os.path.exists(path):
        raise FileNotFoundError("no fine-tuned network %s: run  python part2b.py finetune --variant %s" % (path, variant))
    K = run.data.lk * run.data.lt
    gen, _ = build_networks(N_SCALARS + K, 1 + K, run.config, device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["gen"])
    return gen.eval(), file_hash(path)


# =====================================================================================================================
# 4. A remedied model, ready to roll
# =====================================================================================================================

def model_arenas(model_id, cfg, runs, noise, root, device, block_cfg=None):
    """{seed: (Arena, remedy dict or None)} for a model. block_cfg: the RecursionConfig to roll on (pilot: train)."""
    block_cfg = block_cfg or cfg
    out = {}
    if model_id == "control":
        return {s: (Arena(r, block_cfg, noise), None) for s, (_, r) in runs.items()}
    use_clip, variant = MODELS[model_id]
    clip, clip_hash = load_clip(root) if use_clip else (None, None)
    for s, (_, run) in runs.items():
        late, late_hash = load_late_network(run, variant, s, root, device) if variant else (None, None)
        remedy = {"model_id": model_id, "clip_hash": clip_hash, "late_hash": late_hash, "version": REMEDY_VERSION}
        repair = None
        if model_id in PROJ_MODELS:
            repair = projector_for(run, block_cfg)
            remedy["repair"] = repair.settings_hash()                     # only when the projection is active
        out[s] = (Arena(run, block_cfg, noise, gen_late=late, clip=clip, repair=repair), remedy)
    return out


def projector_for(run, block_cfg):
    """
    The Projector of a run, after the start-up checks of the second brief (2.1.2): the target-day rates the projection
    uses equal the ones cloud_metrics prices with, and the 50%-volatility witness satisfies every tightened constraint
    with slack > 1e-6 on every target day of the block.
    """
    from .projection import Projector, WITNESS_SLACK
    ds = run.split.block(block_cfg.block)
    market = run.market.block(block_cfg.block)
    if not np.array_equal(np.asarray(market.rates, float), np.asarray(ds.rates, float)):
        raise GateError("the projection's rates (inputs) differ from the scoring rates (market)")
    pr = Projector(run.data.m, run.data.tau)
    slack = min(pr.witness_slack(r) for r in np.unique(np.asarray(ds.rates, float), axis=0))
    if not slack > WITNESS_SLACK:
        raise GateError("the feasibility witness fails on the %s block: smallest slack %.3e" % (block_cfg.block, slack))
    pr.witness_min_slack = slack
    return pr


# =====================================================================================================================
# 5. The gates and the pilot (brief section 5)
# =====================================================================================================================

def _max_diff(a, b):
    """Largest absolute difference, NaN equal to NaN; inf if NaN-ness or shapes differ."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return np.inf
    na, nb = np.isnan(a), np.isnan(b)
    if (na != nb).any():
        return np.inf
    d = np.abs(np.where(na, 0.0, a) - np.where(nb, 0.0, b))
    return float(d.max()) if d.size else 0.0


def regression_gate(root=None, device=None, starts=tuple(range(0, 50, 5))):
    """
    Brief 5.1: the control through the NEW code, 10 starts x 5 seeds, against Part A's start files (every array but
    seconds and the signature) and Part A's benchmark for those days. Tolerance 0; at most 1e-12 is reported and
    accepted; anything larger stops.
    """
    root, device = root or partb_root(), device or get_device()
    cfg = RecursionConfig()
    runs = _rec_runs(cfg, device)
    info, _, _ = check_fixed(cfg, runs)
    noise = NoiseTable(cfg.sim_seed, cfg.n_paths, cfg.n_draw, next(iter(runs.values()))[1].config.noise_dim)
    parta = parta_root()
    worst, bench_worst, per = 0.0, 0.0, {}
    for s, (folder, run) in runs.items():
        arena = Arena(run, cfg, noise)
        check_noise(noise, arena.ds, info)
        sig = seed_signature(run, folder, cfg)
        w = 0.0
        for j0 in starts:
            ref = load_npz(os.path.join(parta, "work", "control", "seed_%d" % s, "start_%05d.npz" % j0))
            if ref["signature"] != sig:
                raise GateError("Part A start file of seed %d has signature %s, the control now %s" % (s, ref["signature"], sig))
            rec = run_start(arena, j0)
            for k, v in ref.items():
                if k in ("signature", "seconds", "j0"):
                    continue
                w = max(w, _max_diff(rec[k], v))
        days = np.arange(0, max(starts) + cfg.horizon)
        b, ref_b = bench_clouds(arena, days), parta_bench(arena, run, folder, s)
        bw = max(_max_diff(b[k], np.asarray(ref_b[k])[days]) for k in b if k != "days")
        per[s] = {"starts_max_abs_diff": w, "bench_max_abs_diff": bw}
        worst, bench_worst = max(worst, w), max(bench_worst, bw)
        log("regression gate seed %d: start files max diff %.3g, benchmark max diff %.3g" % (s, w, bw))
    res = {"passed": max(worst, bench_worst) <= 1e-12, "exact": max(worst, bench_worst) == 0.0,
           "starts": list(starts), "max_abs_diff": worst, "bench_max_abs_diff": bench_worst, "by_seed": per}
    _write_json(os.path.join(root, "gates", "regression.json"), res)
    if not res["passed"]:
        raise GateError("regression gate failed: max difference %.3g (start files), %.3g (benchmark)" % (worst, bench_worst))
    return res


PILOT_STARTS = 24


def pilot_step(root=None, device=None, n_starts=PILOT_STARTS):
    """
    Brief 5.3: training block only, 24 evenly spaced starts, all seeds, control and the three remedied models: D_P(h),
    S(h) and the clip activity. In-sample (these starts overlap the pool's), so it says nothing about validation.
    """
    root, device = root or partb_root(), device or get_device()
    cfg = RecursionConfig()
    runs = _rec_runs(cfg, device)
    check_fixed(cfg, runs)
    tcfg = RecursionConfig(block="train")
    noise = NoiseTable(cfg.sim_seed, cfg.n_paths, cfg.n_draw, next(iter(runs.values()))[1].config.noise_dim)
    n = len(runs[cfg.seeds[0]][1].split.train.dates)
    every = start_indices(n, tcfg)
    starts = every[np.linspace(0, len(every) - 1, n_starts).astype(int)]
    days = np.unique((starts[:, None] + np.arange(cfg.horizon)[None, :]).ravel())
    pos = np.searchsorted(days, starts[:, None] + np.arange(cfg.horizon)[None, :])
    bench = {}
    out = {"starts": starts.tolist(), "in_sample_note": "training block; these starts overlap the pool's starts, so "
           "this is in-sample and says nothing about validation performance", "models": {}}
    t0 = time.time()
    for model in ORDER:
        arenas = model_arenas(model, cfg, runs, noise, root, device, block_cfg=tcfg)
        F, B = {}, {}
        for s, (arena, _) in arenas.items():
            if s not in bench:
                bench[s] = bench_clouds(Arena(runs[s][1], tcfg, noise), days)
            recs = [run_start(arena, int(j0)) for j0 in starts]
            for k in ("n_valid", "phi_mean", "inside", "rho", "clip_share", "clip_any"):
                F.setdefault(k, []).append(np.stack([r[k] for r in recs]))
            for k in ("phi_mean", "inside", "rho"):
                B.setdefault(k, []).append(bench[s][k][pos])
        F = {k: np.stack(v) for k, v in F.items()}
        B = {k: np.stack(v) for k, v in B.items()}
        S = sufficient(F, B)
        out["models"][model] = {**{"D_" + P: curve(S, P).tolist() for P in PILLARS},
                                "S": (F["n_valid"] / cfg.n_paths).mean(axis=(0, 1)).tolist(),
                                "clip_share": np.nanmean(F["clip_share"], axis=(0, 1)).tolist(),
                                "clip_any": np.nanmean(F["clip_any"], axis=(0, 1)).tolist()}
        log("pilot %s: D_A(21) %.4g, D_C(21) %.4g, D_D(21) %.4g, S(21) %.4f"
            % (model, *(out["models"][model]["D_" + P][-1] for P in PILLARS), out["models"][model]["S"][-1]))
    out["seconds"] = round(time.time() - t0, 1)
    _write_json(os.path.join(root, "pilot.json"), out)
    L = ["# Part B pilot (training block, IN-SAMPLE)", "", "%d evenly spaced training starts, all five seeds. %s. Nothing "
         "here changes any setting." % (n_starts, out["in_sample_note"].capitalize()), ""]
    for h in (1, 2, 5, 10, 21):
        if h == 1:
            L += ["| model | h | D_A | D_C | D_D | S | clip share | clip any |", "|---|---|---|---|---|---|---|---|"]
        for model in ORDER:
            r = out["models"][model]
            L.append("| %s | %d | %.4g | %.4g | %.4g | %.4f | %.4f | %.4f |" % (
                model, h, r["D_A"][h - 1], r["D_C"][h - 1], r["D_D"][h - 1], r["S"][h - 1], r["clip_share"][h - 1],
                r["clip_any"][h - 1]))
    with open(os.path.join(root, "pilot_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return out


PROJ_RECORDS = ("proj_share", "proj_dvol_mean", "proj_dvol_max", "proj_nodes_mean", "proj_out_of_range", "proj_retries",
                "phi_post_max")


def pilot_proj_step(root=None, device=None, n_starts=PILOT_STARTS, models=("control", "bounded", "bounded_proj")):
    """
    Second brief 6.2: the pilot of bounded_proj (with control and bounded beside it, from the same code), training
    block, the same 24 evenly spaced starts, all seeds: D_P(h), S(h), the projection records by h, and the time per
    (seed, start) with the projected validation time. In-sample; changes no setting. Writes pilot_proj.json and
    pilot_proj_report.md (the first pilot's files are left as they are).
    """
    root, device = root or partb_root(), device or get_device()
    cfg = RecursionConfig()
    runs = _rec_runs(cfg, device)
    check_fixed(cfg, runs)
    tcfg = RecursionConfig(block="train")
    noise = NoiseTable(cfg.sim_seed, cfg.n_paths, cfg.n_draw, next(iter(runs.values()))[1].config.noise_dim)
    n = len(runs[cfg.seeds[0]][1].split.train.dates)
    every = start_indices(n, tcfg)
    starts = every[np.linspace(0, len(every) - 1, n_starts).astype(int)]
    days = np.unique((starts[:, None] + np.arange(cfg.horizon)[None, :]).ravel())
    pos = np.searchsorted(days, starts[:, None] + np.arange(cfg.horizon)[None, :])
    bench = {s: bench_clouds(Arena(r, tcfg, noise), days) for s, (_, r) in runs.items()}
    out = {"starts": starts.tolist(), "in_sample_note": "training block; in-sample, says nothing about validation "
           "performance", "models": {}}
    n_val = len(start_indices(len(runs[cfg.seeds[0]][1].split.val.dates), cfg))
    for model in models:
        arenas = model_arenas(model, cfg, runs, noise, root, device, block_cfg=tcfg)
        F, B, secs = {}, {}, []
        for s, (arena, _) in arenas.items():
            recs = []
            for j0 in starts:
                t0 = time.time()
                recs.append(run_start(arena, int(j0)))                     # raises IdentityError if h = 1 differs
                secs.append(time.time() - t0)
                if max(float(recs[-1][k]) for k in ("id_ret", "id_dg", "id_sigma")) != 0.0:
                    raise GateError("%s seed %d start %d: h = 1 differs from the benchmark" % (model, s, j0))
            keys = ("n_valid", "phi_mean", "inside", "rho") + (PROJ_RECORDS if model in PROJ_MODELS else ())
            for k in keys:
                F.setdefault(k, []).append(np.stack([r[k] for r in recs]))
            for k in ("phi_mean", "inside", "rho"):
                B.setdefault(k, []).append(bench[s][k][pos])
        F = {k: np.stack(v) for k, v in F.items()}
        B = {k: np.stack(v) for k, v in B.items()}
        S = sufficient(F, B)
        per = float(np.mean(secs))
        rec = {**{"D_" + P: curve(S, P).tolist() for P in PILLARS},
               "S": (F["n_valid"] / cfg.n_paths).mean(axis=(0, 1)).tolist(), "seconds_per_seed_start": per,
               "projected_validation_hours": per * n_val * len(runs) / 3600}
        if model in PROJ_MODELS:
            for k in PROJ_RECORDS:
                agg = np.nanmax if k in ("proj_dvol_max", "phi_post_max") else (np.nansum if k in ("proj_out_of_range", "proj_retries") else np.nanmean)
                rec[k] = agg(F[k], axis=(0, 1)).tolist()
        out["models"][model] = rec
        log("pilot (projection) %s: D_A(21) %.4g, D_C(21) %.4g, D_D(21) %.4g, S(21) %.4f, %.2f s per (seed, start) -> "
            "about %.1f h for the validation run" % (model, *(rec["D_" + P][-1] for P in PILLARS), rec["S"][-1], per,
                                                    rec["projected_validation_hours"]))
    _write_json(os.path.join(root, "pilot_proj.json"), out)
    L = ["# Part B pilot of the projection model (training block, IN-SAMPLE)", "",
         "%d evenly spaced training starts, all five seeds; in-sample, says nothing about validation performance; "
         "changes no setting. h = 1 identity checked on every start (difference 0)." % n_starts, "",
         "| model | h | D_A | D_C | D_D | S |", "|---|---|---|---|---|---|"]
    for h in (1, 2, 5, 10, 21):
        for model in models:
            r = out["models"][model]
            L.append("| %s | %d | %.4g | %.4g | %.4g | %.4f |" % (model, h, r["D_A"][h - 1], r["D_C"][h - 1],
                                                                    r["D_D"][h - 1], r["S"][h - 1]))
    L += ["", "Projection records of bounded_proj by h (share of alive paths projected; volatility change in vol points "
          "over projected paths: mean of the total, maximum single node; nodes changed; out-of-range paths, re-solves "
          "and max post-projection Phi over the pilot):", "",
          "| h | share | dvol mean | dvol max | nodes | out of range | retries | Phi post max |", "|---|---|---|---|---|---|---|---|"]
    for model in models:
        if model not in PROJ_MODELS:
            continue
        r = out["models"][model]
        for h in (1, 2, 5, 10, 21):
            L.append("| %d | %.4f | %.4g | %.4g | %.3g | %d | %d | %.3g |" % (
                h, r["proj_share"][h - 1], r["proj_dvol_mean"][h - 1], r["proj_dvol_max"][h - 1],
                r["proj_nodes_mean"][h - 1], r["proj_out_of_range"][h - 1], r["proj_retries"][h - 1], r["phi_post_max"][h - 1]))
    L += ["", "Timing: " + "; ".join("%s %.2f s per (seed, start), about %.1f h for the validation run"
                                     % (m, out["models"][m]["seconds_per_seed_start"], out["models"][m]["projected_validation_hours"])
                                     for m in models)]
    with open(os.path.join(root, "pilot_proj_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return out


# =====================================================================================================================
# 6. The validation run of one remedied model (the Part A experiment)
# =====================================================================================================================

def run_model(model_id, root=None, device=None, starts=None):
    """
    The Part A experiment for one remedied model on the validation block, all seeds and starts, resumable; the
    benchmark is Part A's (the unchanged control), loaded and checked; the h = 1 identity must hold with difference 0.
    """
    if model_id not in MODELS:
        raise ValueError("model must be one of %s" % list(MODELS))
    root, device = root or partb_root(), device or get_device()
    cfg = RecursionConfig(model_id=model_id)
    check_inputs(cfg.index)
    runs = _rec_runs(cfg, device)
    info, margins, deltas = check_fixed(cfg, runs)
    noise = NoiseTable(cfg.sim_seed, cfg.n_paths, cfg.n_draw, next(iter(runs.values()))[1].config.noise_dim)
    arenas = model_arenas(model_id, cfg, runs, noise, root, device)
    first = next(iter(arenas.values()))[0]
    hashes = check_noise(noise, first.ds, info)
    with np.load(os.path.join(parta_root(), "results_all.npz")) as z:
        parta_starts = z["starts"]
    starts = start_indices(first.n, cfg) if starts is None else np.asarray(starts)
    full = starts is None or np.array_equal(starts, start_indices(first.n, cfg))
    if full and not np.array_equal(starts, parta_starts):
        raise GateError("the starts are not Part A's")
    for s, (a, _) in arenas.items():                                    # the gate, before anything is written
        identity_gate(a, starts[:3], log=lambda m, s=s: log("%s seed %d: %s" % (model_id, s, m)))
    mroot = os.path.join(root, model_id)
    os.makedirs(mroot, exist_ok=True)
    _write_json(os.path.join(mroot, "margins.json"), margins)          # Part A's, copied (never recomputed)
    log("%s: %d starts on the %s block, %d seeds" % (model_id, len(starts), cfg.block, len(arenas)))
    ident, secs, remedies, guar = {}, {}, {}, None
    if model_id in PROJ_MODELS:
        guar = {"phi_post_max": 0.0, "proj_retries": 0, "proj_out_of_range": 0, "solver_failures": 0,
                "witness_min_slack": min(a.repair.witness_min_slack for a, _ in arenas.values())}
    for s, (folder, run) in runs.items():
        a, remedy = arenas[s]
        sig = seed_signature(run, folder, cfg, remedy)
        bench = parta_bench(a, run, folder, s)
        run_seed(a, mroot, s, sig, starts, log)
        out = consolidate(a, mroot, s, sig, starts, bench)
        ident[s] = {k: float(out[k].max()) for k in ("id_ret", "id_dg", "id_sigma")}
        if max(ident[s].values()) != 0.0:
            raise GateError("%s seed %d: h = 1 differs from the benchmark (%s)" % (model_id, s, ident[s]))
        secs[s], remedies[s] = float(out["seconds"].mean()), remedy
        if guar is not None:                                            # the guarantee, over every surface of the run
            guar["phi_post_max"] = max(guar["phi_post_max"], float(out["free_phi_post_max"].max()))
            guar["proj_retries"] += int(out["free_proj_retries"].sum())
            guar["proj_out_of_range"] += int(out["free_proj_out_of_range"].sum())
            if guar["phi_post_max"] != 0.0:
                raise GateError("%s seed %d: a projected surface has Phi %.3e after projection" % (model_id, s,
                                                                                                 guar["phi_post_max"]))
    _write_json(os.path.join(mroot, "identity_check.json"), {"passed": True, "max_abs_diff_by_seed": ident,
                "required": "exactly 0", "n_starts": int(len(starts))})
    _write_json(os.path.join(mroot, "run.json"), {
        "model_id": model_id, "settings": cfg.as_dict(), "remedy_settings": RC.as_dict(), "remedies": remedies,
        "margins": margins, "noise_hashes": hashes, "noise_hashes_equal_part_a": True,
        "benchmark": "Part A work/control/bench_seed_<k>.npz, signature-checked (the unchanged control)",
        "network_run_json_hashes": {str(s): file_hash(os.path.join(f, "run.json")) for s, (f, _) in runs.items()},
        "seconds_per_seed_start": secs, "versions": library_versions(), "device": str(device), "n_starts": int(len(starts)),
        **({"projection": {"settings": asdict(next(iter(arenas.values()))[0].repair.s),
                           "guarantee": guar}} if guar is not None else {})})
    if full:
        report_model(model_id, root)
    log("done: %s" % mroot)


def report_model(model_id, root=None):
    """Part A's tables (degradation, survival, tests, powerlaw), results_all.npz and descriptive.npz for one model."""
    root = root or partb_root()
    cfg = RecursionConfig(model_id=model_id)
    mroot = parta_root() if model_id == "control" else os.path.join(root, model_id)
    _, deltas = load_margins(os.path.join(parta_root(), "margins.json"))
    T = load_tables(mroot, list(cfg.seeds))
    res = analyse(T, cfg, deltas)
    with np.load(os.path.join(parta_root(), "results_all.npz")) as z:
        if not np.array_equal(res["idx"], z["boot_idx"]):
            raise GateError("the bootstrap index matrix is not Part A's")
    if model_id != "control":
        write_tables(mroot, res)
        save_results(mroot, res, T, deltas)
    desc = descriptive(T)
    if model_id != "control":
        np.savez_compressed(os.path.join(mroot, "descriptive.npz"), **desc)
    return T, res, deltas, desc


# =====================================================================================================================
# 7. The comparison (brief section 6)
# =====================================================================================================================

ATM_COL = 1             # (1.00, 30 days) among the nine calibration columns (maturity-major: 30d 0.90, 1.00, 1.10, ...)


def descriptive(T):
    """Brief 6.5, no verdicts: curves by horizon (and seed) for the thesis figures."""
    F, B = split_runs(T)
    n_paths = RecursionConfig().n_paths
    with np.errstate(all="ignore"):
        out = {
            "S_by_seed": (F["n_valid"] / n_paths).mean(axis=1),                                     # (S, H)
            "phi_pos_share_free": np.nanmean(F["viol_share"], axis=(0, 1)),
            "phi_pos_share_bench": np.nanmean(B["viol_share"], axis=(0, 1)),
            "mean_phi_free": np.nanmean(F["phi_mean"], axis=(0, 1)),
            "mean_phi_bench": np.nanmean(B["phi_mean"], axis=(0, 1)),
            "coverage_free": np.nanmean(F["inside"][..., NINE], axis=(0, 1, 3)),
            "below_free": np.nanmean(F["below"][..., NINE], axis=(0, 1, 3)),
            "above_free": np.nanmean(F["above"][..., NINE], axis=(0, 1, 3)),
            "coverage_bench": np.nanmean(B["inside"][..., NINE], axis=(0, 1, 3)),
            "below_bench": np.nanmean(B["below"][..., NINE], axis=(0, 1, 3)),
            "above_bench": np.nanmean(B["above"][..., NINE], axis=(0, 1, 3)),
            "median_minus_realised_atm1m_by_seed": np.nanmean(F["q"][..., 1, ATM_COL] - T["real_values"][None, :, :, ATM_COL], axis=1),
            "median_minus_realised_atm1m_bench_by_seed": np.nanmean(B["q"][..., 1, ATM_COL] - T["real_values"][None, :, :, ATM_COL], axis=1),
            "median_width_ratio": np.nanmedian((F["width"][..., NINE] / B["width"][..., NINE]).transpose(2, 0, 1, 3)
                                               .reshape(F["width"].shape[2], -1), axis=1),
            "mean_rho_free": np.nanmean(F["rho"], axis=(0, 1)),
            "mean_rho_bench": np.nanmean(B["rho"], axis=(0, 1)),
            "clip_share": np.nanmean(F["clip_share"], axis=(0, 1)) if "clip_share" in F else np.zeros(F["n_valid"].shape[2]),
            "clip_any": np.nanmean(F["clip_any"], axis=(0, 1)) if "clip_any" in F else np.zeros(F["n_valid"].shape[2]),
        }
        if "proj_share" in F:                                           # second remedy: the projection records by h
            first = pd.DatetimeIndex(T["target_dates"][:, 0])
            stress = np.asarray((first >= "2020-02-20") & (first <= "2020-06-30"))
            for k in PROJ_RECORDS:
                agg = np.nanmax if k in ("proj_dvol_max", "phi_post_max") else (
                    np.nansum if k in ("proj_out_of_range", "proj_retries") else np.nanmean)
                out[k] = agg(F[k], axis=(0, 1))
                if k in ("proj_share", "proj_dvol_mean", "proj_nodes_mean"):
                    out[k + "_feb_jun_2020"] = np.nanmean(F[k][:, stress], axis=(0, 1))
                    out[k + "_other"] = np.nanmean(F[k][:, ~stress], axis=(0, 1))
    return out


def klass(lo, hi, margin):
    if lo > margin:
        return "material degradation"
    if hi < margin:
        return "no material degradation detected"
    return "indeterminate at the margin"


def change_class(lo, hi):
    if hi < 0:
        return "improvement"
    if lo > 0:
        return "worsening"
    return "no clear change"


def interval(x):
    return tuple(float(v) for v in np.nanpercentile(x, [2.5, 97.5]))


def compare_step(root=None):
    root = root or partb_root()
    cfg = RecursionConfig()
    out_dir = os.path.join(root, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    models = ORDER + tuple(m for m in PROJ_MODELS if os.path.exists(os.path.join(root, m, "metrics_1.npz")))
    have = {m: report_model(m, root) for m in models}
    h21 = cfg.horizon - 1
    seeds = list(cfg.seeds)
    deltas = have["control"][2]
    S = {m: sufficient(*split_runs(have[m][0])) for m in models}
    star = {m: {P: have[m][1]["boot"][P][:, h21] for P in PILLARS} for m in models}
    point = {m: {P: float(curve(S[m], P)[h21]) for P in PILLARS} for m in models}
    per_seed = {m: {P: [float(curve(S[m], P, [r])[h21]) for r in range(len(seeds))] for P in PILLARS} for m in models}
    s21 = {m: (have[m][0]["free_n_valid"][:, :, h21] / cfg.n_paths) for m in models}
    classes, cls_at = [], {}
    for m in models:
        for P in PILLARS:
            lo, hi = interval(star[m][P])
            row = {"model_id": m, "pillar": P, "D21": point[m][P], "ci_lo": lo, "ci_hi": hi, "margin": deltas[P]}
            for name, mult in (("half_delta", 0.5), ("delta", 1.0), ("double_delta", 2.0)):
                row["class_" + name] = klass(lo, hi, mult * deltas[P])
            cls_at[(m, P)] = row["class_delta"]
            row.update({"D21_seed%d" % s: v for s, v in zip(seeds, per_seed[m][P])})
            row["S21"] = float(s21[m].mean())
            row.update({"S21_seed%d" % s: float(v) for s, v in zip(seeds, s21[m].mean(axis=1))})
            row["cause_reading"] = "instability" if row["S21"] < cfg.survival_tol else "drift"
            classes.append(row)
    delta_rows = []
    for m in models[1:]:
        for P in PILLARS:
            d_star = star[m][P] - star["control"][P]                    # the same replicate b in both models
            lo, hi = interval(d_star)
            ch = change_class(lo, hi)
            if cls_at[("control", P)] != "material degradation" and cls_at[(m, P)] == "material degradation":
                label = "new failure"
            elif hi < 0:
                label = "repaired" if cls_at[(m, P)] != "material degradation" else "improved but still degraded"
            elif lo > 0:
                label = "worsened"
            else:
                label = "no improvement"
            delta_rows.append({"model_id": m, "pillar": P, "Delta": point[m][P] - point["control"][P], "ci_lo": lo,
                               "ci_hi": hi, "change_class": ch, "class_at_delta": cls_at[(m, P)], "label": label,
                               **{"Delta_seed%d" % s: a - b for s, a, b in zip(seeds, per_seed[m][P], per_seed["control"][P])}})
    attr_rows = []
    for P in PILLARS:
        x = {m: (point[m][P], star[m][P], np.array(per_seed[m][P])) for m in ORDER}
        effects = {
            "bounded step": lambda c, b, d, bd: ((b - c) + (bd - d)) / 2,
            "rollout training": lambda c, b, d, bd: ((d - c) + (bd - b)) / 2,
            "interaction": lambda c, b, d, bd: (bd - d) - (b - c)}
        for name, f in effects.items():
            pt, rep, ps = (f(*(x[m][i] for m in ORDER)) for i in range(3))
            lo, hi = interval(rep)
            attr_rows.append({"pillar": P, "effect": name, "estimate": float(pt), "ci_lo": lo, "ci_hi": hi,
                              "change_class": change_class(lo, hi), **{"seed%d" % s: float(v) for s, v in zip(seeds, ps)}})
    tabs = {"classes": pd.DataFrame(classes), "delta": pd.DataFrame(delta_rows), "attribution": pd.DataFrame(attr_rows)}
    if "bounded_proj" in models:                                       # second brief 7.3: the two levers
        lev = []
        for name, (a, b) in (("rollout-training lever: bounded_dad - bounded", ("bounded_dad", "bounded")),
                             ("projection lever: bounded_proj - bounded", ("bounded_proj", "bounded")),
                             ("projection against rollout training: bounded_proj - bounded_dad",
                              ("bounded_proj", "bounded_dad"))):
            for P in PILLARS:
                lo, hi = interval(star[a][P] - star[b][P])
                lev.append({"lever": name, "pillar": P, "estimate": point[a][P] - point[b][P], "ci_lo": lo, "ci_hi": hi,
                            "change_class": change_class(lo, hi),
                            **{"seed%d" % s: x - y for s, x, y in zip(seeds, per_seed[a][P], per_seed[b][P])}})
        tabs["levers"] = pd.DataFrame(lev)
    for name, df in tabs.items():
        df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    long = []
    for m in models:
        for q, v in have[m][3].items():
            v = np.asarray(v)
            if v.ndim == 1:
                long += [{"model_id": m, "quantity": q, "seed": "all", "h": h + 1, "value": float(v[h])} for h in range(len(v))]
            else:
                long += [{"model_id": m, "quantity": q, "seed": seeds[r], "h": h + 1, "value": float(v[r, h])}
                         for r in range(v.shape[0]) for h in range(v.shape[1])]
    pd.DataFrame(long).to_csv(os.path.join(out_dir, "descriptive.csv"), index=False)
    _write_json(os.path.join(out_dir, "checks.json"), {"bootstrap_index_matrix_equal_part_a": True,
                "models": list(models), "boot_seed": cfg.boot_seed, "n_boot": cfg.n_boot})
    log("comparison written to %s" % out_dir)
    return tabs


# =====================================================================================================================
# 8. REPORT.md (brief section 7)
# =====================================================================================================================

def _md(df, floatfmt="%.4g"):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(floatfmt % v if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(rows)


def report_step(root=None, notes=()):
    root = root or partb_root()
    cmp_dir = os.path.join(root, "comparison")
    L = ["# Part 2, Part B: remedies for the 21-day recursion (report)", "",
         "Numbers only; no interpretation beyond what they say. Primary model: `bounded_dad`. All margins, seeds, noise, "
         "starts, benchmark and bootstrap are Part A's (asserted).", ""]
    reg = json.load(open(os.path.join(root, "gates", "regression.json")))
    L += ["## 1. Gates", "", "- Regression gate (control through the new code, starts %s, all seeds): start files max |diff| "
          "= %.3g, benchmark max |diff| = %.3g -> %s." % (reg["starts"], reg["max_abs_diff"], reg["bench_max_abs_diff"],
                                                         "PASS (exact)" if reg["exact"] else "PASS" if reg["passed"] else "FAIL")]
    for m in ORDER[1:]:
        p = os.path.join(root, m, "identity_check.json")
        if os.path.exists(p):
            d = json.load(open(p))["max_abs_diff_by_seed"]
            L.append("- Identity (h = 1) %s: max |diff| by seed %s." % (m, {s: max(v.values()) for s, v in d.items()}))
    L += ["- Noise hashes equal Part A, benchmark = Part A's (signature-checked), margins equal, bootstrap index matrix "
          "equal: asserted in every run (a failure stops the run).", ""]
    cb = json.load(open(os.path.join(root, "clip_bounds.json")))
    lo, hi = np.array(cb["lo"]), np.array(cb["hi"])
    wid = np.argsort(hi - lo)[::-1][:3]
    L += ["## 2. Clip bounds and activity", "",
          "Training block %s to %s (%d days), quantiles %s. lo in [%.4f, %.4f], hi in [%.4f, %.4f]. Widest ranges: %s." % (
              cb["first"], cb["last"], cb["n_training_days"], cb["quantile_levels"], lo.min(), lo.max(), hi.min(), hi.max(),
              ", ".join("%s (%.4f to %.4f)" % (cb["nodes"][i], lo[i], hi[i]) for i in wid)), ""]
    act = []
    for m in ("bounded", "bounded_dad"):
        p = os.path.join(root, m, "results_all.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        dates = pd.DatetimeIndex(z["target_dates"][:, 0])
        covid = (dates >= "2020-02-01") & (dates <= "2020-06-30")
        sh = z["free_clip_share"]
        for h in (2, 5, 10, 21):
            act.append({"model": m, "h": h, "clip share": float(np.nanmean(sh[:, :, h - 1])),
                        "clip any": float(np.nanmean(z["free_clip_any"][:, :, h - 1])),
                        "share, Feb-Jun 2020 starts": float(np.nanmean(sh[:, covid, h - 1])),
                        "share, other starts": float(np.nanmean(sh[:, ~covid, h - 1]))})
    if act:
        L += [_md(pd.DataFrame(act)), ""]
    L += ["## 3. Pool and fine-tuning", ""]
    prow, frow = [], []
    for v in VARIANTS:
        for s in RecursionConfig().seeds:
            p = os.path.join(root, "pool", v, "seed_%d_summary.json" % s)
            if os.path.exists(p):
                d = json.load(open(p))
                prow.append({"variant": v, "seed": s, "pairs": d["n_pairs"], "of": d["max_pairs"], "dropped": d["dropped_total"],
                             "mean abs target k=2": d["mean_abs_target_surface_by_k"]["2"],
                             "k=21": d["mean_abs_target_surface_by_k"]["21"], "identity max err": d["identity_max_error"],
                             "s": d["seconds"]})
            p = os.path.join(net_dir(root, v, s), "finetune.json")
            if os.path.exists(p):
                d = json.load(open(p))
                frow.append({"variant": v, "seed": s, **{k: d["final"][k] for k in ("d_loss", "d_real", "d_fake", "g_bce", "L_m", "L_tau")},
                             "s": d["seconds"]})
    if prow:
        L += [_md(pd.DataFrame(prow)), ""]
    if frow:
        L += ["Final-epoch losses of the fine-tuning (no NaN or infinite loss occurred; the run stops if one does):", "",
              _md(pd.DataFrame(frow)), ""]
    pp = os.path.join(root, "pilot_report.md")
    if os.path.exists(pp):
        L += ["## 4. Pilot (training block, IN-SAMPLE)", ""] + open(pp, encoding="utf-8").read().splitlines()[2:] + [""]
    if os.path.exists(os.path.join(cmp_dir, "classes.csv")):
        cl = pd.read_csv(os.path.join(cmp_dir, "classes.csv"))
        order = {m: i for i, m in enumerate(("bounded_dad", "bounded", "dad", "control"))}
        cl = cl.sort_values(by="model_id", key=lambda c: c.map(order), kind="stable")
        de = pd.read_csv(os.path.join(cmp_dir, "delta.csv")).sort_values(by="model_id", key=lambda c: c.map(order), kind="stable")
        at = pd.read_csv(os.path.join(cmp_dir, "attribution.csv"))
        L += ["## 5. Classes, changes and attribution (validation block, h = 21)", "", "### classes.csv", "",
              _md(cl[["model_id", "pillar", "D21", "ci_lo", "ci_hi", "margin", "class_half_delta", "class_delta",
                      "class_double_delta"] + ["D21_seed%d" % s for s in RecursionConfig().seeds]]), "",
              "### delta.csv (Delta = model - control, same bootstrap replicates)", "",
              _md(de), "", "### Attribution (2 x 2)", "", _md(at), "",
              "## 6. S(21) per model and seed", "",
              _md(cl[cl.pillar == "A"][["model_id", "S21"] + ["S21_seed%d" % s for s in RecursionConfig().seeds] + ["cause_reading"]]), ""]
    L += ["## 7. Run times, versions, device", ""]
    for m in ORDER[1:]:
        p = os.path.join(root, m, "run.json")
        if os.path.exists(p):
            d = json.load(open(p))
            L.append("- %s: %.2f s per (seed, start) on average; device %s." % (m, np.mean(list(d["seconds_per_seed_start"].values())), d["device"]))
    L += ["- Versions: %s" % json.dumps(library_versions()), "", "## 8. What did not go as specified", ""]
    L += ["- " + n for n in notes] if notes else ["- Nothing."]
    with open(os.path.join(root, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("report: %s" % os.path.join(root, "REPORT.md"))


# =====================================================================================================================
# 9. The second remedy's section of REPORT.md (second brief, section 8): appended, everything already there kept
# =====================================================================================================================

PROJ_SECTION = "## 9. Second remedy: in-loop projection onto the arbitrage-free set (bounded_proj)"


def report_proj_step(root=None, notes=()):
    from .projection import ProjSettings
    root = root or partb_root()
    m = "bounded_proj"
    mroot, cmp_dir = os.path.join(root, m), os.path.join(root, "comparison")
    info = json.load(open(os.path.join(mroot, "run.json")))
    ident = json.load(open(os.path.join(mroot, "identity_check.json")))["max_abs_diff_by_seed"]
    g = info["projection"]["guarantee"]
    L = [PROJ_SECTION, "", "Numbers only. Same networks, seeds, calendar noise, starts, benchmark, margins and bootstrap "
         "as every other model (asserted). Step 1 is never clipped or projected.", "",
         "### 9.1 Gates", "", "- Identity (h = 1) bounded_proj: max |diff| by seed %s." % {s: max(v.values()) for s, v in ident.items()},
         "- Noise hashes equal Part A; benchmark = Part A's (signature-checked); margins and bootstrap index matrix equal.",
         "- Feasibility witness (constant 50%% volatility) on every validation target day's rates: smallest slack %.3e "
         "(required > 1e-6)." % g["witness_min_slack"], "",
         "### 9.2 Constants and solver", ""]
    L += ["- %s: %s" % (k, v) for k, v in asdict(ProjSettings()).items()]
    L += ["", "### 9.3 The guarantee (over every projected surface of the validation run)", "",
          "| max post-projection Phi | re-solves (EPS_ARB x 100) | out-of-range paths | solver failures |", "|---|---|---|---|",
          "| %.3g | %d | %d | %d |" % (g["phi_post_max"], g["proj_retries"], g["proj_out_of_range"], g["solver_failures"]), "",
          "### 9.4 Projection activity and size, by h (validation)", "",
          "Share of alive paths projected; over projected paths: mean total |change| in implied volatility and largest "
          "single-node change (vol points), mean nodes changed by more than 1e-8; and the share projected for starts "
          "whose first target day is 20 Feb to 30 Jun 2020 against the rest.", "",
          "| h | share | dvol mean | dvol max | nodes | share Feb-Jun 2020 | share other |", "|---|---|---|---|---|---|---|"]
    d = dict(np.load(os.path.join(mroot, "descriptive.npz")))
    for h in (1, 2, 3, 5, 10, 15, 21):
        L.append("| %d | %.4f | %.4g | %.4g | %.3g | %.4f | %.4f |" % (h, d["proj_share"][h - 1], d["proj_dvol_mean"][h - 1],
                 d["proj_dvol_max"][h - 1], d["proj_nodes_mean"][h - 1], d["proj_share_feb_jun_2020"][h - 1],
                 d["proj_share_other"][h - 1]))
    pp = os.path.join(root, "pilot_proj_report.md")
    if os.path.exists(pp):
        L += ["", "### 9.5 Pilot (training block, IN-SAMPLE)", ""] + open(pp, encoding="utf-8").read().splitlines()[2:]
    cl = pd.read_csv(os.path.join(cmp_dir, "classes.csv"))
    de = pd.read_csv(os.path.join(cmp_dir, "delta.csv"))
    lv = pd.read_csv(os.path.join(cmp_dir, "levers.csv"))
    seeds = RecursionConfig().seeds
    L += ["", "### 9.6 classes.csv (bounded_proj; validation, h = 21)", "",
          _md(cl[cl.model_id == m][["model_id", "pillar", "D21", "ci_lo", "ci_hi", "margin", "class_half_delta", "class_delta",
                                    "class_double_delta"] + ["D21_seed%d" % s for s in seeds]]), "",
          "### 9.7 delta.csv (bounded_proj - control, same bootstrap replicates)", "",
          _md(de[de.model_id == m]), "", "### 9.8 levers.csv (each lever on top of the bounded step)", "", _md(lv), "",
          "### 9.9 S(21) per seed", "",
          _md(cl[(cl.model_id == m) & (cl.pillar == "A")][["model_id", "S21"] + ["S21_seed%d" % s for s in seeds] + ["cause_reading"]]), "",
          "### 9.10 Timings", "",
          "- bounded_proj: %.2f s per (seed, start) on average (%s), device %s; scipy %s." % (
              np.mean(list(info["seconds_per_seed_start"].values())),
              ", ".join("seed %s %.2f s" % kv for kv in info["seconds_per_seed_start"].items()), info["device"],
              info["versions"]["scipy"]), "",
          "### 9.11 What did not go as specified", ""]
    L += ["- " + n for n in notes] if notes else ["- Nothing."]
    path = os.path.join(root, "REPORT.md")
    old = open(path, encoding="utf-8").read()
    old = old.split(PROJ_SECTION)[0].rstrip() + "\n\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(old + "\n".join(L) + "\n")
    log("report: %s (section 9 written)" % path)
