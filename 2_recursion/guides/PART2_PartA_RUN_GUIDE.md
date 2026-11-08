# Part 2 (multi-step recursion), Part A: how to run it on another computer and keep the results

Part 2 is the multi-step (recursive) simulation; Part A is its first stage: degradation of the five trained control
networks under 21-day recursion, with no remedies (those are Part 2, Part B). Part 1 is the one-step model.

This is exactly the procedure used for the reference run (SPX, 5 control networks, validation block).
Total time on the reference laptop (CPU): **training about 2 h** (about 15 min per seed, longer if the machine is busy),
**pilot 2 min, main run 6 min**. Everything is resumable.

## 1. What to copy to the other computer

| Item | Where it must end up | Notes |
|---|---|---|
| the whole `code/` folder | anywhere, e.g. `D:\volgan\code` | **Do not copy `.venv/`** (it does not move between machines). Copy `runs/` only if you want to reuse trained networks. |
| `spx_grid42_2007_2025_clean.xlsx` | `<code>/data/` or `<code>/../data/` | the implied-vol grid (Part A needs only SPX) |
| `rates_2008_2025_clean.xlsx` | same folder | rates |

The data folder is found automatically as `<code>/data`, else `<code>/../data` (the layout on the reference laptop:
`project/data` next to `project/code`). Anywhere else: pass `--data-dir <folder>` to every script, or set the
environment variable `VOLGAN_DATA_DIR`. File hashes of the reference run: SPX file `096285d9903266d9`,
rates file `4afbe6a988420ed5` (printed in `outputs/part2/partA/margins.json`; they must match if you want identical numbers).

## 2. Environment (once)

Code map: `part2.py` is the single entry point for Part 2 (subcommands `margins`, `pilot`, `run`, `report`);
`main.py` is Part 1 (used here only to train the networks). Part 2 code is in `volgan_fixed/recursion.py` (rollout,
metrics, run) and `volgan_fixed/recursion_analysis.py` (inference, pilot, tables, figures); its settings
(`RecursionConfig`) are in `volgan_fixed/common.py`. Options of every `part2.py` subcommand: `--index`, `--data-dir`,
`--rates-file`, `--runs-dir`, `--out-dir`, `--model-id`, `--seeds`.

Python 3.11. In the external IDE (PyCharm: Settings > Project > Python Interpreter > Add > New virtual environment),
then in its terminal, from the `code` folder:

```bash
python -m pip install -r requirements.txt
```

Reference versions: torch 2.14.1 (CPU is enough), numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, matplotlib 3.11.2.
Other versions usually work, but the noise and the training can then differ slightly, so check step 6.

## 3. Check the code (about 30 s)

```bash
python tests/run_all.py recursion
```

Must say PASS (`tests/test_recursion.py`: state update, noise, h = 1 identity, validity, metrics, bootstrap, Holm,
resume). It uses synthetic data, no workbook.

## 4. The overnight command (train, pilot, main run)

From the `code` folder, in the IDE terminal (or as three PyCharm Run Configurations, in this order):

```bash
python main.py --steps train --configs raw_phiout
python part2.py pilot --index SPX
python part2.py run --index SPX
```

One line, stops at the first failure and logs to a file:

```bash
python main.py --steps train --configs raw_phiout && python part2.py pilot && python part2.py run > partA.log 2>&1
```

What each does:

1. `main.py --steps train --configs raw_phiout`: trains the five control networks (Part 1 cell B networks), seeds 1 to 5,
   10,000 epochs each, into `runs/SPX/raw_phiout/seed_1 ... seed_5`. **Do not pass `--epochs` or `--tag`**: that would
   write somewhere Part A does not read. Settings are in the block at the top of `main.py` (leave at defaults).
2. `python part2.py pilot`: **training block only**. Checks the h = 1 identity, the noise floor, Monte Carlo precision,
   delta_A and timing; writes `outputs/part2/partA/pilot_report.md`. If the identity check fails it stops. A noise-floor
   FAIL is only written in the report (the chain continues): read it afterwards.
3. `python part2.py run`: the main run on the **validation block** (2018 to 2021): 198 starts (every 5th sample), 1,000
   paths, 21 days, all 5 seeds, the one-step benchmark, then tables, figures and `results_all.npz`.
   `--no-report` runs without tables; `--report-only` (or `python part2.py report`) redraws them from saved metrics.

Never use the test block (2022 to 2025); Part A refuses it.

## 5. Keeping it running

- Plug in, set the OS sleep to **Never** (a sleeping computer pauses the run; hibernate or shutdown kills it). A dark screen is fine.
- Training saves a network only when its 10,000 epochs finish. If the process is killed, finished seeds are kept and the
  **unfinished seed restarts from epoch 0**. Resume by rerunning the same command: finished seeds, starts and
  benchmarks are skipped.
- If you change any setting after output exists, the run **refuses to mix** and tells you to delete the output folder
  or point `VOLGAN_PARTA_DIR` elsewhere. Do not edit `RecursionConfig` midway.
- Output location: `<code>/outputs/part2/partA/` (change with `--out-dir <folder>`; networks with `--runs-dir <folder>`).

## 6. Check that it reproduces the reference run

`outputs/part2/partA/run.json` holds `noise_hashes`. On the reference machine:

| date | hash |
|---|---|
| 2018-01-02 | `6fb4decb627467bd` |
| 2019-07-15 | `5e247ab3cdf4b57e` |
| 2021-12-31 | `04c94274979b4839` |

If yours differ, the noise differs (another torch build or device): results are then not common-random-number
comparable with the reference. `tests/test_recursion.py` checks the same constants.
Reference `settings` also recorded there: sim_seed 20261012, boot_seed 20261014, horizon 21, 1,000 paths, stride 5,
10,000 bootstrap draws, block length 5 starts, validity bounds 1% to 300%, margins delta_C 0.05, delta_D 0.10,
delta_A = market mean Phi on the training block = 0.000295094.
Retrained networks will not be bit-identical to the reference ones across machines, so expect similar, not equal, numbers.

## 7. What you get (and what to save)

Copy the whole `outputs/part2/partA/` folder back (about 36 MB; `work/` can be dropped once the rest is there).
Keep `runs/SPX/raw_phiout/` (about 3 MB) too: the networks are needed to regenerate any path.

| File | Contents |
|---|---|
| `tests.csv` | **headline**: T per pillar, per-seed T, margin, raw and Holm p, 95% interval, sensitivity (delta/2, 2 delta), seed-sensitive flag, outcome |
| `degradation.csv`, `survival.csv`, `powerlaw.csv` | D(h) with bands, S(h), fit D ~ a h^b |
| `results_all.npz` | **everything for later analysis** (below) |
| `metrics_1.npz` ... `metrics_5.npz` | the same, one file per seed |
| `figures/` | `fig_timeseries`, `fig_degradation`, `fig_survival`, `fig_seed_dots`, `fig_coverage` (PDF and PNG) |
| `identity_check.json`, `run.json`, `margins.json`, `pilot_report.md` | gate result; settings, hashes, versions, timings; margins; pilot |

### Reading `results_all.npz`

```python
import numpy as np
z = np.load("outputs/part2/partA/results_all.npz")
print(z.files)
```

Axes: seed (5, in order 1 to 5), start (198), horizon (21, h = 1 to 21). `free_*` is the free-running run, `bench_*`
the one-step benchmark of the same target day.

| Key | Shape | Meaning |
|---|---|---|
| `free_n_valid`, `free_phi_mean`, `free_viol_share`, `free_rho` | (5, 198, 21) | valid paths (of 1,000), mean Phi, share of paths with Phi > 1e-12, cross-path correlation of return and 1M ATM log-vol change |
| `free_phi_parts` | (5, 198, 21, 3) | mean calendar, call-spread, butterfly terms |
| `free_inside`, `free_below`, `free_above`, `free_width` | (5, 198, 21, 10) | 95% band vs realised value, at the nine nodes (moneyness 0.9, 1.0, 1.1 x 30, 90, 360 days, in that order, maturity-major) then the daily return |
| `free_q` | (5, 198, 21, 3, 10) | 2.5%, 50%, 97.5% quantiles of the same ten variables |
| `free_viol_loc21`, `bench_viol_loc21` | (5, 198, 7, 6) | violation frequency per grid node at h = 21 (moneyness x maturity) |
| `bench_*` | same | the benchmark; `bench_n_oob` counts benchmark paths outside 1% to 300% (not masked) |
| `real_values` | (198, 21, 10) | the realised market values scored against |
| `var_names`, `starts`, `horizons`, `seeds`, `target_dates` | | labels; `target_dates` is (198, 21) |
| `deg_A/C/D`, `deg_*_lo`, `deg_*_hi` | (21,) | D_P(h) with the pointwise 95% band |
| `boot_A/C/D`, `boot_idx` | (10000, 21), (10000, 198) | bootstrap replicates and the resampled start indices |
| `id_ret`, `id_dg`, `id_sigma`, `seconds` | (5, 198) | h = 1 identity differences, seconds per start |

Simulated paths themselves are not stored (1,000 x 42 numbers per cloud). Any single start can be regenerated from
the saved network with `volgan_fixed.recursion.rollout` (same noise, same code).

Example, mean admissibility degradation by horizon and a seed-by-seed view at h = 21:

```python
d = np.nanmean(z["free_phi_mean"] - z["bench_phi_mean"], axis=(0, 1))     # (21,)
d21_by_seed = np.nanmean(z["free_phi_mean"][:, :, 20] - z["bench_phi_mean"][:, :, 20], axis=1)
```

## 8. Definitions (as implemented)

- **Admissibility** D_A(h) = mean over seeds and starts of (mean Phi free - mean Phi benchmark). Signed.
- **Calibration** D_C(h) = |pooled free coverage - 0.95| - |pooled benchmark coverage - 0.95|, coverage pooled over nine nodes, seeds and starts first.
- **Dynamics** D_D(h) = mean of |rho free - rho benchmark| over seeds and starts (ED section 6.4).
- **Test**: T_P = D_P(21); H0: mu_P <= delta_P. Moving-block bootstrap over the starts (blocks of 5 consecutive starts, drawn with replacement, no wrap-around, B = 10,000, one shared index matrix, seeds not resampled); p = (1 + #{T* <= delta}) / (B + 1); Holm across A, C, D.
- **Seed-sensitive**: the five per-seed T straddle the margin, or have mixed signs.
- **Instability** if the mean survival at h = 21 is below 0.99; otherwise "drift". A path is valid while all 42 vols are finite and within 1% to 300% and its return is finite; invalid is absorbing; nothing is repaired or filtered inside the recursion.

## 9. Reference results (to compare against)

Overall outcome: instability (S(21) = 0.987; by seed 0.995, 0.999, 1.000, 0.965, 0.977).

| Pillar | T at h = 21 | margin | 95% interval | Holm p |
|---|---|---|---|---|
| A admissibility | 0.372 | 0.000295 | [0.167, 0.735] | 0.0003 |
| C calibration | 0.137 | 0.05 | [0.102, 0.176] | 0.0003 (seed-sensitive) |
| D dynamics | 0.0989 | 0.10 | [0.086, 0.109] | 0.68 (seed-sensitive) |

The pilot's identity check passed (maximum difference 0). Its dynamics noise-floor check failed the "floor at most half
the margin" rule (0.065 against 0.05); that threshold is an assumption of this build, not of the brief.

## 10. Troubleshooting

- `missing input file(s)`: the data folder is not found: use `--data-dir` (see section 1).
- `no trained run ...`: the networks are not in `runs/SPX/raw_phiout/seed_k` (or use `--runs-dir`).
- `exists with a different configuration`: output from other settings is in the folder; delete `outputs/part2/partA/` or use `--out-dir`.
- `h = 1 identity fails`: stop and report; no tables are written.
- `The test block is used once...`: do not use the test block here.
- Very slow training: close other programs, set the power mode to maximum performance while plugged in.
