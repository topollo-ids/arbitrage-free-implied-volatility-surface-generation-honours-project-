# Part 2, Part B: build and run brief (remedies for the 21-day recursion)

You are working in the VolGAN code folder (`code/`, the folder holding `main.py`, `part2.py` and `volgan_fixed/`).
Part 1 (one-step model) and Part 2, Part A (21-day free-running recursion, no remedies) are finished. Your job is to
implement two remedies, train what they need, run them through the same 21-day experiment as Part A, and report.

Read these first, in full: `volgan_fixed/recursion.py`, `volgan_fixed/recursion_analysis.py`, `volgan_fixed/training.py`,
`volgan_fixed/common.py` (`Config`, `RecursionConfig`, checkpoints), `volgan_fixed/postprocessing.py` (`day_noise`,
`day_key`, `step`, `generate_full`), `part2.py`, `PART2_PartA_RUN_GUIDE.md`, `DECISIONS.md`.

---

## 0. The one principle

**Any difference between a remedied model and Part A must be caused by the remedy and nothing else.** Every source of
randomness, every setting, every start day, every network, the benchmark and the bootstrap stay exactly as in Part A.
Concretely, all of the following are FIXED and must be asserted in code, not just assumed:

| Item | Value (must equal Part A) | Where it comes from |
|---|---|---|
| Networks | `runs/SPX/raw_phiout/seed_1..5/2_trained.pt` (the Part 1 control, cells A/B) | Part A `run.json` `network_run_json_hashes` |
| Training seeds | 1, 2, 3, 4, 5 | `RecursionConfig.seeds` |
| Simulation noise | calendar-indexed: `day_noise(sim_seed, day_key(target date), n_draw=10000, 32)[:1000]` via `NoiseTable` | `sim_seed = 20261012` |
| Noise check | `noise_hashes` of the remedied runs equal Part A's `run.json` `noise_hashes` | assert |
| Paths, horizon | 1,000 paths, 21 steps | `RecursionConfig` |
| Starts | validation block, `start_indices(n, cfg)`: stride 5, offset 0, 198 starts (j0 = 0 ... 985) | assert equal to Part A `results_all.npz["starts"]` |
| Validity | vols finite and in [1%, 300%], return finite; invalid is absorbing; nothing filtered | `vol_lo`, `vol_hi` |
| Benchmark | the UNCHANGED control network, one step, unclipped, from real history (cell B) | identical arrays to Part A `work/control/bench_seed_k.npz` (assert) |
| Margins | `delta_A = 0.000295093667...`, `delta_C = 0.05`, `delta_D = 0.10` | Part A `margins.json` (load and assert, do not recompute differently) |
| Bootstrap | moving blocks of 5 starts, B = 10,000, `boot_seed = 20261014`, ONE index matrix shared by every model | assert equal to Part A `results_all.npz["boot_idx"]` |
| Data | `spx_grid42_2007_2025_clean.xlsx` (hash `096285d9903266d9`), rates `4afbe6a988420ed5` | assert |
| Device and versions | CPU; record versions in `run.json` | as Part A |

The remedies add NO random numbers to the simulation. The bounded step is deterministic. The rollout-trained network
changes the weights used from step 2 onward, but it receives exactly the same noise vector for the same calendar day as
the control would. The only new randomness is in building the training pool and in fine-tuning, and both have their
own fixed seeds (section 3), on the training block only.

Never use the test block (2022 to 2025). Never modify Part 1 results, Part A outputs or the Part 1 networks.
Never tune any setting on validation results: every number in this brief is fixed in advance. If something fails,
stop and report; do not change settings to make it pass.

---

## 1. The models

| model_id | Bounded step | Network for steps 2 to 21 | Role |
|---|---|---|---|
| `control` | no | control G0 | Part A, already run; re-used, NOT re-run (except the regression check in 4.2) |
| `bounded` | yes | G0 | attribution: bounded step alone |
| `dad` | no | G1 = G0 fine-tuned on its own unclipped rollouts | attribution: rollout training alone |
| `bounded_dad` | yes | G1b = G0 fine-tuned on its own clipped rollouts | **the pre-specified adjusted model (primary)** |

The methodology's adjusted model is `bounded_dad` (the cause read from Part A is instability, S(21) = 0.987 < 0.99, and
the remedy grid's instability row is "bounded step + training on own rollouts" for every failing property). `bounded`
and `dad` are run so that the improvement can be attributed to each remedy (a 2 x 2 with `control`), exactly like the
Part 1 ablation. Report `bounded_dad` as the primary result and the other two as attribution.

In every model, **step 1 is the control's step, untouched**: network G0, no clipping. Remedies act only on what is
produced from fed-back inputs, i.e. steps k = 2 ... 21. This keeps the h = 1 identity exact and the benchmark unchanged.

---

## 2. Remedy 1: the bounded step

**Definition (fixed).** For each of the 42 grid nodes, take the daily change in log implied volatility on the TRAINING
block, `dg_real = split.train.target[:, 1:]` (that is g(d) - g(c), shape (2389, 42)), and its 0.1% and 99.9%
quantiles per node: `lo = np.quantile(dg_real, 0.001, axis=0)`, `hi = np.quantile(dg_real, 0.999, axis=0)`
(numpy default interpolation, float64). At every step k >= 2 the generator's log-vol increment is clipped node by node:
`dg = np.clip(dg, lo, hi)`. The return output is NOT clipped.

**Where it acts.** Inside `rollout`, right after `dg` is computed and BEFORE `sigma = exp(cond_g + dg)`, the validity
check, the yielded `StepOut` and the state update. So the scored surface, the validity check, the 1M ATM increment
used for rho, and the next input all use the clipped `dg`. Step k = 1 is never clipped.

Do NOT implement it through the existing `post_step` hook: that hook edits the condition before step k, so the surface
scored at step k would still be unclipped. Implement it as a new optional argument of `rollout` (section 4.1).

**Save** `outputs/part2/partB/clip_bounds.json` (lo, hi per node in grid order, quantile levels, number of training
days, first and last training date, data hash) and record its hash in every remedied run's signature.

**Record activity**: for every (seed, start, h) store `clip_share` = share of (valid path, node) entries at which the
clip was binding (0 for h = 1), and `clip_any` = share of valid paths with at least one node clipped.

---

## 3. Remedy 2: training on own rollouts (Data as Demonstrator)

Reference: Venkatraman, Hebert and Bagnell (2015), "Improving multi-step prediction of learned time series models",
AAAI. The network is shown the states it reaches when it runs on its own output, each paired with what the market did
next, and is trained further on them alongside the real data. One round only (no repeated rounds).

### 3.1 The pool of synthetic states (training block only)

For each training seed s (network G0_s) and each variant (`dad`: no clipping; `bounded_dad`: clipping at k >= 2 with
the bounds of section 2):

1. Inputs: `run = load_run(runs/SPX/raw_phiout/seed_s)`; `ds = run.split.train`; `data = run.data`.
2. Starts: every training sample j0 with j0 + 20 <= len(ds) - 1, i.e. j0 = 0 ... len(ds) - 21 (stride 1,
   2,369 starts). Every target day of every rollout lies inside the training block. Nothing from validation is used.
3. Paths: `POOL_PATHS = 4` per start, 21 steps, using the same `rollout` code (and, for `bounded_dad`, the same
   clipping) as the evaluation. Noise: a `NoiseTable` with `POOL_SEED = 20261016` (calendar-indexed exactly like the
   evaluation, `n_draw = 10000`, first 4 rows). This seed is never used anywhere else.
4. From each rollout keep, for k = 2 ... 21, the condition that FEEDS step k (built from k - 1 = 1 ... 20 simulated
   days) for every path still valid after step k - 1. Call it `cond_syn` (3 + 42 numbers, float64).
5. Its target is the REAL next day of the target sample `j = j0 + k - 1`, expressed relative to the synthetic state:
   - return slot: `ds.target[j, 0]` (the real daily return times `ret_scale`, exactly as in real training pairs);
   - surface slots: `g_real(j) - cond_syn[3:]`, where `g_real(j) = ds.past_log_surface[j] + ds.target[j, 1:]` is the
     real log-vol surface of the predicted day. By construction `cond_syn[3:] + target_syn[1:] == g_real(j)`; assert it.
   - the predicted day's rates `ds.rates[j]` (kept for completeness; Phi is not in the loss).
6. Also store `k`, `j0`, `j`, path index and seed.

Save per seed and variant: `outputs/part2/partB/pool/<variant>/seed_s.npz` (float32 is fine for conditions and
targets; keep float64 for the identity assert). Expected size: about 2,369 x 20 x 4 = 190,000 pairs per seed and
variant. Report the pool's size, the number of states dropped because a path died, and the mean |target_syn[1:]| by k.

### 3.2 Fine-tuning (one round)

For each seed s and variant v:

1. Load G0_s and its discriminator AND both RMSprop optimiser states from `2_trained.pt` (`load_checkpoint`), and the
   control's `alpha_m`, `alpha_tau` (from the checkpoint; check they equal `run.json`). No new gradient matching, no
   restart, same `Config` (lr 1e-4 both, batch 100, `gen_bce_weight = 0.5`, smoothness "paper" on g, Phi NOT in the
   loss).
2. Real pairs: the training block, `to_tensors(run.split.train, ...)`, exactly as Part 1 trained on (2,389 pairs).
3. `FT_EPOCHS = 2000`. One epoch = all 2,389 real pairs + 2,389 synthetic pairs drawn without replacement, uniformly,
   from the pool (a fresh draw every epoch), concatenated, shuffled, batches of 100, using the EXISTING
   `discriminator_step` and `generator_step` (phase "train", `on = "g"`). Do not write a new loss.
4. Seeds for this step: `set_seed(FT_SEED)` with `FT_SEED = 30000 + s` before fine-tuning (torch, numpy, random), and
   a `numpy.random.default_rng(FT_SEED)` for the per-epoch pool draws. Record both.
5. Log per epoch: d_loss, d_real, d_fake, g_bce, L_m, L_tau, separately for real and synthetic batches if easy (at
   least overall). If any loss becomes NaN or infinite, STOP and report; do not retry with other settings.
6. Save to `outputs/part2/partB/networks/<v>/seed_s/` (NOT under `runs/`): `ft_trained.pt` (same checkpoint format),
   `finetune.json` (parent network hash `file_hash(2_trained.pt)`, parent run.json hash, pool file hash, all settings
   above, seeds, epochs, time taken, library versions), `ft_logs.npz`.

The smoothness penalty acts on the generated surface `cond_g + dg`, so for a synthetic state it pushes the output
toward a smooth surface, in the same direction as the targets. That is intended.

### 3.3 Deployment

In models `dad` and `bounded_dad` the rollout uses G0_s at step 1 and the fine-tuned G1_s at steps 2 ... 21, with the
same noise vector per calendar day as G0 would have received. The benchmark always uses G0_s.

---

## 4. Code changes

### 4.1 `recursion.rollout` (minimal, backward compatible)

Add two optional keyword arguments, defaults `None`, so that `rollout(gen, ds, data, j0, noise, cfg)` behaves bit for
bit as now:

- `gen_late=None`: the network for steps k >= 2 (None: `gen` for every step). Put it in eval mode; same device.
- `clip=None`: a pair of float64 arrays `(lo, hi)`, shape (42,), applied as in section 2 for k >= 2 only.

Also return the clip activity for the step (e.g. in `StepOut.extra` or a new field defaulting to None). Thread both
arguments through `run_start`, `run_seed`, `Arena` (or a small wrapper) and `consolidate` (add `free_clip_share`,
`free_clip_any`, shape (n_starts, H); zeros for models without clipping). Keep `post_step` as it is.

`seed_signature` must include the remedy (model_id, clip-bounds hash, late-network hash) ONLY when a remedy is
active, so the control's signature, and therefore Part A's resume files, are unchanged.

### 4.2 New code

- `volgan_fixed/remedies.py`: `clip_bounds(run)`, `build_pool(run, variant, ...)`, `finetune(run, pool, variant, ...)`,
  `load_late_network(run, variant, seed)`.
- `part2b.py` (new entry point; leave `part2.py` untouched), subcommands:
  - `clipbounds` -> `outputs/part2/partB/clip_bounds.json`
  - `pool --variant {dad,bounded_dad}`
  - `finetune --variant {dad,bounded_dad}`
  - `pilot` (training block only, section 5)
  - `run --model {bounded,dad,bounded_dad}` (validation block, the Part A experiment for that model)
  - `compare` (section 6)
  All resumable in the Part A style (signature files, refuse to mix configurations).
- Output root `outputs/part2/partB/` with one subfolder per model (`outputs/part2/partB/<model_id>/`) using exactly the
  Part A layout and file schema (`work/`, `metrics_<k>.npz`, `results_all.npz`, `degradation.csv`, `survival.csv`,
  `run.json`, `identity_check.json`), plus the new keys `free_clip_share`, `free_clip_any`.
- Re-use `recursion_analysis` functions (`load_tables`, `sufficient`, `value`, `curve`, `block_indices`,
  `bootstrap_curves`, `survival`) instead of rewriting them.

### 4.3 Tests (add `tests/test_remedies.py`, run with `python tests/run_all.py`)

1. `rollout` with `gen_late=None, clip=None` gives identical arrays to the current code (same start, same noise).
2. `clip=(-inf, +inf)` gives identical arrays to no clip.
3. `gen_late=gen` gives identical arrays to `gen_late=None`.
4. Step 1 is never clipped and never uses `gen_late` (construct a `gen_late` that returns zeros and check h = 1).
5. Clipped increments lie inside [lo, hi] at every k >= 2; the state update and sigma use the clipped values.
6. Pool targets: `cond_syn[3:] + target_syn[1:] == g_real(j)` exactly; the return slot equals `ds.target[j, 0]`.
7. The pool uses only training-block dates; `POOL_SEED` differs from `sim_seed`.
8. The noise vector passed at step k is the same object/values for G0 and G1 (calendar-indexed).

---

## 5. Gates and the pilot (before any validation run)

1. **Regression gate.** Run model `control` through the NEW code for all five seeds on starts j0 = 0, 5, ..., 45
   (10 starts) and assert every metric array (everything except `seconds` and the signature) equals the Part A start
   files in `outputs/part2/partA/work/control/` exactly (tolerance 0; if exact equality fails only at the 1e-12
   level, report it and continue; anything larger: stop). Assert the benchmark
   arrays for those days equal Part A's `bench_seed_k.npz`.
2. **Identity gate.** For every remedied model and seed, the h = 1 outputs equal the benchmark exactly (the existing
   `identity_gate` / `check_identity`, tolerances unchanged). Must pass with maximum difference 0.
3. **Pilot (training block only, `RecursionConfig(block="train")`)**, 24 evenly spaced training starts, all seeds, all
   three remedied models plus `control`: report D_A(h), D_C(h), D_D(h), S(h) and the clip activity, as a sanity check
   that the code does what it should. Note in the report that these starts overlap the pool's starts, so this is
   in-sample and says nothing about validation performance. Nothing here may change any setting.

---

## 6. The validation runs and the comparison

Run `bounded`, `dad`, `bounded_dad` on the validation block with the fixed settings of section 0 (198 starts x 5 seeds
x 1,000 paths x 21 steps). Then `compare` writes `outputs/part2/partB/comparison/`:

1. `classes.csv`: for each model (including `control`) and property P in {A, C, D}: D_P(21), its 95% percentile
   interval from the shared bootstrap, the margin, and the class at delta/2, delta and 2 delta:
   - **material degradation** if the interval lies wholly above the margin;
   - **no material degradation detected** if it lies wholly below;
   - **indeterminate at the margin** if it straddles.
   Definitions of D_A, D_C, D_D exactly as in Part A (`sufficient` / `value`). Also the five per-seed D_P(21), and
   S(21) overall and per seed, and the cause reading (instability if S(21) < 0.99, else drift).
2. `delta.csv`: for each remedied model M and property P, Delta_P = D_P^M(21) - D_P^control(21), with a 95% interval
   from the SAME bootstrap index matrix applied to both models (for each replicate b, the difference of the two
   replicate values), so the pairing of starts holds. Per-seed Delta_P too. Class of Delta against zero: improvement
   (interval wholly below 0), worsening (wholly above 0), no clear change (straddles 0).
3. A label per (M, P), following the methodology, printed beside the class so the reader can see both:
   - **repaired**: Delta interval wholly below 0 and M's class at delta is not material degradation;
   - **improved but still degraded**: Delta interval wholly below 0 and M's class is material degradation;
   - **no improvement**: Delta interval straddles 0;
   - **worsened**: Delta interval wholly above 0;
   - **new failure**: control's class at delta was not material degradation and M's is (this overrides the others).
4. Attribution (2 x 2): for each P at h = 21, the effect of the bounded step = ((bounded - control) + (bounded_dad -
   dad)) / 2, of rollout training = ((dad - control) + (bounded_dad - bounded)) / 2, and their interaction =
   (bounded_dad - dad) - (bounded - control), each with a bootstrap interval from the shared index matrix and the
   per-seed values.
5. `results_all.npz` per model with the Part A schema (plus the clip arrays), so the same plots can be drawn for every
   model. Also save, per model, these descriptive curves (no verdicts): survival S(h) per seed; share of paths with
   Phi > 0 by h; mean Phi by h; pooled coverage of the nine nodes by h with the below / above split; median minus
   realised at 1M ATM by h and seed; median band-width ratio free / benchmark by h; mean rho free and benchmark by h;
   clip activity by h.

Use the same per-start NaN handling as Part A (clouds with no valid path drop out and are counted through S(h)).

---

## 7. What to report back (`outputs/part2/partB/REPORT.md`)

1. Gate results: regression (max difference), identity (max differences per model and seed), noise hashes equal,
   benchmark equal, margins equal, bootstrap matrix equal.
2. Clip bounds summary (smallest and largest lo, hi; nodes with the widest range) and clip activity by model and h,
   and by period (e.g. starts in February to June 2020 against the rest).
3. Pool summary per seed and variant, and fine-tuning summary (final losses, any instability, time).
4. The pilot table (training block, labelled in-sample).
5. `classes.csv`, `delta.csv` and the attribution table, as markdown tables, primary model first.
6. S(21) per model and seed.
7. Run times, library versions, device.
8. Anything that did not go as specified, stated plainly, with what was done instead (ideally: nothing was done
   instead; the step was stopped).

Do not write interpretation beyond what the numbers say. Do not draw thesis figures; the descriptive arrays in
section 6.5 are what the thesis figures will be drawn from.

---

## 8. Fixed constants (copy into `RecursionConfig` or a `RemedyConfig` dataclass; never change after a run exists)

| Constant | Value |
|---|---|
| `CLIP_Q` | (0.001, 0.999), per node, training-block daily log-vol changes |
| clip applies to | log-vol increments at steps k >= 2; returns never |
| `POOL_SEED` | 20261016 |
| `POOL_PATHS` | 4 per start |
| pool starts | every training sample with 20 real days after it (stride 1) |
| pool states | conditions feeding steps 2 ... 21 of each valid path |
| target | real return of the predicted day; real log-vol surface of the predicted day minus the synthetic surface |
| `FT_EPOCHS` | 2,000 |
| mix | each epoch: all 2,389 real pairs + 2,389 pool pairs (fresh uniform draw without replacement) |
| `FT_SEED` | 30000 + training seed |
| optimiser | RMSprop, lr 1e-4, continued from the `2_trained.pt` optimiser states |
| loss | unchanged eq 13 (BCE weight 0.5, smoothness on g with the control's alpha_m, alpha_tau), Phi not in the loss |
| rounds | one |
| deployment | G0 at step 1, fine-tuned network at steps 2 ... 21 |

Expected cost on the reference laptop (CPU): pools about 2 minutes per variant; fine-tuning about 6 minutes per seed
and variant (10 runs, about 1 hour); each validation model about 6 minutes plus the benchmark check; total under 2 hours.

## 9. Order of work

1. Read the code listed at the top. 2. Implement 4.1 and the tests 4.3; run all tests (Part A tests must still pass).
3. `clipbounds`. 4. Regression gate (5.1). 5. `pool` and `finetune` for both variants. 6. Identity gate and pilot
(5.2, 5.3). 7. `run` for `bounded`, `dad`, `bounded_dad`. 8. `compare`. 9. `REPORT.md`. Stop there.
