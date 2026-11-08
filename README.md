# Arbitrage Free Volatility Surface Generation

Code for the honours project *Arbitrage Free Volatility Surface Generation* (Exaucé Mbuyi and Topollo Mokhatla,
University of Cape Town, 2026; supervised by Jake Stangroom).

The project implements VolGAN (Vuletić and Cont, *VolGAN: a generative model for arbitrage-free implied volatility
surfaces*, Applied Mathematical Finance 31(4), 2024) as the paper states it, on a 42-point Bloomberg grid for SPX and
NDX, and evaluates it in three parts:

1. **One day ahead** (`1_one_step/`): admissibility, coverage of the 95% prediction intervals and dynamics, with a
   2 x 2 ablation of the two arbitrage devices: the penalty Φ in the generator loss (out or in) and the reweighting of
   the scenarios (unweighted or reweighted). Cells A (control: penalty out, reweighted), B, C and D, five training
   seeds each.
2. **Twenty-one days ahead** (`2_recursion/`): the generator fed its own output for 21 trading days, judged against
   its own one-day forecasts (Part A), and three remedies: a bounded step, data as demonstrator and projection onto
   the arbitrage-free set (Part B).
3. **Test block** (`3_test_block/`): the one-time out-of-sample check on 2022 to 2025.

## Setup

Python 3.11. The results were produced on a CPU with exactly the versions in `requirements.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

### Data (not included)

The implied volatility and rate data are licensed from Bloomberg and are not in this repository. The code expects
three workbooks:

| File | Contents |
|---|---|
| `spx_grid42_2007_2025_clean.xlsx` | SPX implied volatilities, moneyness 0.80 to 1.20 x maturities 30 to 360 days (42 points) |
| `ndx_grid42_2010_2025_clean.xlsx` | the same for NDX |
| `rates_2008_2025_clean.xlsx` | US Treasury bill rates r_1M to r_1Y at the grid's maturities |

Put them in `data/` at the top of this repository (it is ignored by git), or pass `--data-dir <folder>` to
`reproduce.py` (or set the environment variable `VOLGAN_DATA_DIR`). The first 16 hex digits of the SHA-256 of the
files used in the thesis are `096285d9903266d9` (SPX) and `4afbe6a988420ed5` (rates); every `run.json` records them.

## Reproducing the results, in order

`python reproduce.py` lists the steps; `python reproduce.py all` runs every validation step in this order. Each step
is one command, which can also be run by hand from the top of the repository:

| # | Step | Command | Produces |
|---|---|---|---|
| 1 | `part1-spx` | `python 1_one_step/main.py --index SPX` | the ten networks, cells A to D on the validation block (2018 to 2021), tables and figures |
| 2 | `part1-ndx` | `python 1_one_step/main.py --index NDX` | the same for NDX |
| 3 | `part2a-margins` | `python 2_recursion/part2.py margins` | the admissibility anchor and the three margins |
| 4 | `part2a-pilot` | `python 2_recursion/part2.py pilot` | the training-block pilot and the day-1 identity check |
| 5 | `part2a-run` | `python 2_recursion/part2.py run` | the control's 21-day recursion: degradations, bootstrap intervals, classes, survival |
| 6 | `part2b-remedies` | `python 2_recursion/part2b.py all` | the bounded step and data as demonstrator (pools, fine-tuning, runs), comparison, report |
| 7 | `part2b-projection` | `python 2_recursion/part2b.py proj` | the bounded step with projection, comparison, report |
| 8 to 14 | `test` | `python reproduce.py test --confirm-final` | the test block (2022 to 2025): `3_test_block/test_eval.py onestep`, `proj 1 2 3 4 5`, `plots`, then `3_test_block/test_surface_day.py find`, `draws`, `figure`, `checks` |

The test block is opened once, after every validation result is in; `reproduce.py` and `main.py` refuse it unless
`--confirm-final` is given. Every step saves its output and skips finished work, so a run can be stopped and restarted. On a laptop CPU
training takes about 15 minutes per network (25 with the penalty in the loss), simulation about 2 minutes and
evaluation about 11 minutes per network; the recursion takes minutes per model.

Outputs are written next to the scripts: `1_one_step/trained_networks/` and `1_one_step/results/<INDEX>/<block>/`
(Part 1), `2_recursion/A_control_recursion/` and `2_recursion/B_interventions/` (Part 2),
`2_recursion/test_results/` and `3_test_block/figures/` (test block). `1_one_step/RESULTS_README.md` describes every
Part 1 output file and `3_test_block/README.md` the test-block scripts.

### Exactness

Every source of randomness is fixed: training seeds 1 to 5, deterministic PyTorch algorithms, simulation noise indexed
by calendar date (`sim_seed = 20261012`, so every model draws the same noise on the same day) and the bootstrap seed
(`20261014`). On the set-up that produced the thesis (Windows 11, CPU, Python 3.11 and the versions in
`requirements.txt`), a rerun reproduces the results exactly: networks retrained from the same seeds gave the same
results to the printed precision.

On another operating system or processor, PyTorch's random numbers and floating-point arithmetic can differ in the
last bits (the simulation noise differs by about 1e-6). Over 10,000 training epochs such differences grow, so retrained
networks, and every number built on them, will not match the thesis exactly there. Three unit tests check this
bit-for-bit agreement against values recorded on the original machine (`test_regression_guard` and
`test_alpha_phi_rule_and_record` in `tests/test_training.py`, `test_calendar_noise` in `tests/test_recursion.py`), and
they fail on such machines while the rest pass. Each `run.json` records the library versions, platform, data hashes and
settings of the run that made it.

## Layout

```
reproduce.py                 runs the steps above in order
requirements.txt
volgan_fixed/                the library shared by every part (each module imports only from those above it)
    common.py                  settings and Config, paths, seeds and determinism, the arbitrage penalty Φ
    preprocessing.py           load(index), rates, preprocessing, the train/validation/test split
    training.py                networks, smoothness penalty, training loop, run folders and signatures
    postprocessing.py          scenario simulation, reweighting, every one-day metric
    reporting.py               Part 1 tables and figures
    recursion.py               Part 2 A: rollout, per-path metrics, margins
    recursion_analysis.py      Part 2 A: degradations, bootstrap, classes, pilot, report
    projection.py              Part 2 B: projection onto the arbitrage-free set (linear programme, HiGHS)
    remedies.py                Part 2 B: bounded step, data as demonstrator, comparison and report
1_one_step/main.py           Part 1 entry point (settings at the top; --help lists every option)
1_one_step/DECISIONS.md      Part 1 decision log, fixed before any result was seen
2_recursion/part2.py         Part 2 A entry point
2_recursion/part2b.py        Part 2 B entry point
2_recursion/guides/          the Part 2 run guide and design briefs (written before the current folder names)
3_test_block/                the test-block scripts
tests/                       unit tests of volgan_fixed
```

The package keeps the name `volgan_fixed` because the saved checkpoints refer to it by name.

## Tests

```bash
python tests/run_all.py                       # every module: PASS / FAIL and time
python tests/run_all.py training recursion    # only these
```

Tests that need the workbooks are skipped, with a note, when the files are not found. On a machine other than the
original, the three bit-for-bit tests named under Exactness fail; the module runner stops at the first failure in a
module. `tests/original_reference.py`
holds verbatim copies of functions from the authors' released `VolGAN.py`, used only as a reference in the tests.

## What the scripts do not regenerate

The scripts produce the saved results: the networks, simulations and metrics, the Part 1 and Part 2 tables, and
figures of their own. The figures in the thesis, and some numbers in its tables and discussion, were made from these
saved results with separate plotting and analysis scripts that are not part of this repository.
