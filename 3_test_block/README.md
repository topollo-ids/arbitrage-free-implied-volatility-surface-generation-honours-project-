# The test block (2022 to 2025)

The scripts of the out-of-sample evaluation. The test block was opened once,
after the validation results were in, with the trained networks in `1_one_step/trained_networks/SPX` and every setting
frozen. No network was retrained and no file in `volgan_fixed`, `main.py` or `part2b.py` was changed: the scripts here
only call that code on the test block. (The folders were reorganised afterwards, on 2026-10-19: only the locations
written in the scripts changed, not what they do.)

## Scripts

Run from the top of the repository, in the project's virtual environment (or through `python reproduce.py test --confirm-final`).

| Command | What it does |
|---|---|
| `python 3_test_block/test_eval.py onestep` | `1_one_step/main.py --block test --confirm-final --steps simulate evaluate` for both training configurations: the one-day cells A to D on the 1,003 test days, 10,000 scenarios per day |
| `python 3_test_block/test_eval.py proj 1 2 3 4 5` | the 21-day bounded step + projection (Part B code) on 197 test starts, 1,000 paths each; runs the day-1 identity gate first and stops if the identity or any projected surface fails |
| `python 3_test_block/test_eval.py plots` | the recursion's degradations, bootstrap intervals and classes; the two figures; the numbers used in the write-up |
| `python 3_test_block/test_surface_day.py find` | candidate days: the market's highest penalty, volatility and return on the test block |
| `python 3_test_block/test_surface_day.py draws` | regenerates the four surfaces of 25 November 2022, the market's highest-penalty day (network 1) |
| `python 3_test_block/test_surface_day.py figure` | the 2 x 2 surface figure and its numbers |
| `python 3_test_block/test_surface_day.py checks` | where each drawn surface sits within its own ensemble |

`_paths.py` sets the locations: it reads the validation results from `1_one_step/results/SPX/val`, the margins from
`2_recursion/A_control_recursion`, and the clip bounds and validation projection results from
`2_recursion/B_interventions`. It writes the one-step test results next to the validation ones, the projection run
into `2_recursion/test_results`, and the figures and surface-day draws into this folder. `style.py` is the thesis
figure style.

## Outputs

- `1_one_step/results/SPX/test/`: per-network metrics and summaries of cells A to D, `market.npz`, and the raw one-day
  simulations in `sims/` (about 80 MB per network), written by `onestep`.
- `2_recursion/test_results/`: the projection run's per-network metrics, `results_all.npz`, the CSV tables, and the
  per-start checkpoints in `work/`, written by `proj`.
- `3_test_block/figures/`: `fig_test_phi.png`, `fig_test_pillars.png` (`plots`) and `fig_test_surfaces.png` (`figure`).
- `3_test_block/results/surface_day_2022-11-25.npz`: the four surfaces of the qualitative figure (`draws`).

## Reproducibility

`plots` run on the saved files prints the same numbers on every run, and `draws` reproduces
`surface_day_2022-11-25.npz` to the last digit. The simulation noise is calendar-indexed (sim_seed 20261012), so
`onestep` and `proj` draw the same noise on every run; on a different operating system or processor the noise, and so
the outputs, can differ in the last digits.
