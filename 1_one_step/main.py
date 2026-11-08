"""
VolGAN Part 1: the one-step replication and the 2x2 arbitrage ablation (Loss x Weighting). THIS is the file to run.

    python main.py                          run the steps switched on in SETTINGS below
    python main.py --smoke                  end-to-end check with tiny settings (separate folders, a few minutes)
    python main.py --steps tables figures   only these steps this time (overrides STEPS)
    python main.py --index NDX --configs raw_phiout     the NDX robustness check: the control networks only
    python main.py --help                   every option

THE DATA CALL is one function, volgan_fixed/preprocessing.py: load(INDEX). It reads the index's implied-vol workbook
and the rates workbook (the three files named in SETTINGS), keeps the days both have, takes r at the grid's
maturities, checks everything, and returns one object (dates, index level, vols, grid, rates). Every step below uses
that same object (it is read once and cached).

The work is a chain of steps; each one saves its output, and the next one reads it, so the chain can be stopped and
restarted at any point (finished work is skipped). What each step calls, in volgan_fixed/:

    train     training.train_step -> training.train_run -> training.fit (gradient-norm matching, restart,
              10,000 epochs; training.TorchPhi adds Phi to the generator loss for the Loss lever)
    simulate  postprocessing.simulate_step -> simulate (10,000 scenarios per validation day, the same noise
              (sim_seed) for every model)
    evaluate  postprocessing.evaluate_step -> market_reference (the data side) + daily_metrics (weights, beta_t,
              weighted quantiles, rho_t, PCA): every metric unweighted AND re-weighted (the Weighting lever)
    tables    reporting.make_tables      CSV + LaTeX, from saved metrics only
    figures   reporting.make_figures     PDF + PNG, from saved metrics only
The settings below become a Config through common.training_config (the control plus one configuration's levers).

The four cells come from two TRAINING configurations, each evaluated twice (unweighted and re-weighted):
    raw_phiout -> A (re-weighted, the paper's control) and B (unweighted)
    raw_phiin  -> C (re-weighted) and D (unweighted)
The networks train on, and every cell is scored against, the raw vendor grid. See README.md.
"""

import argparse
import os
import sys
import time

# ======================================= SETTINGS: edit these, then run  python main.py ===========================

INDEX = "SPX"                   # "SPX" (the full study) or "NDX" (the robustness check: --configs raw_phiout)

STEPS = {                       # in this order; each step reads what the earlier ones saved
    "train": True,              # the networks: about 15 min per run on a laptop CPU (Phi-in runs about 25 min)
    "simulate": True,           # 10,000 scenarios per validation day, per run (about 2 minutes per run)
    "evaluate": True,           # all metrics, both weightings (about 11 minutes per run: a VIX per scenario)
    "tables": True,             # results/<INDEX>/val/tables/
    "figures": True,            # results/<INDEX>/val/figures/
}

TRAINING_CONFIGS = [            # which networks to train / simulate / evaluate (each gives two cells)
    "raw_phiout",               # cells A + B: the control networks (the paper). Keep this one always.
    "raw_phiin",                # cells C + D: Phi added to the generator loss
]
SEEDS = [1, 2, 3, 4, 5]         # training seeds, fixed in advance; never choose among them

BLOCK = "val"                   # Part 1 results come from the validation block (2018-2021)
CONFIRM_FINAL = False           # True only ONCE, at the very end, with BLOCK = "test" (2022-2025)

# Settings that change the model. Leave them at None / their defaults for the study; any change goes into its own
# output folders automatically (OUTPUT_TAG), so it can never mix with the main results.
N_EPOCHS = None                 # None = the paper's 10,000
N_SCENARIOS = None              # None = the paper's 10,000 per day
B_PCA = 1000                    # generated series for PCA and the correlation tables
RATE_SOURCE = "rates"           # "rates" (control: the rates workbook) or "zero" (the released code, r = 0)
RHO_OVERLAY_A = False           # fig_rho_time: also draw cell A (re-weighted) in ochre
OUTPUT_TAG = None               # None: runs/ and results/. A name such as "zero_rates": runs_<tag>/ and results_<tag>/

# ---- DATA: the only three files the model reads. A bare name is looked up in DATA_DIR; a full path works too.
DATA_DIR = None                 # None: <code>/data if it exists, otherwise ../data (project/data on this laptop)
SPX_FILE = "spx_grid42_2007_2025_clean.xlsx"   # implied vols, 42-point grid: moneyness 0.8-1.2 x 30-360 days
NDX_FILE = "ndx_grid42_2010_2025_clean.xlsx"   # the same for NDX
RATES_FILE = "rates_2008_2025_clean.xlsx"      # r_1M ... r_1Y (decimals), at exactly the grid's maturities

# ==================================================================================================================


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", choices=["SPX", "NDX"], help="the index (default: INDEX in SETTINGS)")
    p.add_argument("--steps", nargs="+", choices=list(STEPS), help="run only these steps")
    p.add_argument("--configs", nargs="+", choices=["raw_phiout", "raw_phiin"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--block", choices=["val", "test"])
    p.add_argument("--confirm-final", action="store_true", help="allow the test block (once, at the end)")
    p.add_argument("--data-dir")
    p.add_argument("--rates-file")
    p.add_argument("--tag", help="write to runs_<tag>/ and results_<tag>/")
    p.add_argument("--epochs", type=int)
    p.add_argument("--smoke", action="store_true",
                   help="end-to-end check: 1 seed, 2 matching + 30 epochs, 200 scenarios, into runs_smoke/results_smoke")
    return p.parse_args()


def main():
    a = parse()
    index = a.index or INDEX
    steps = [s for s in STEPS if (s in a.steps if a.steps else STEPS[s])]
    configs = a.configs or TRAINING_CONFIGS
    seeds = a.seeds or SEEDS
    block = a.block or BLOCK
    confirm = a.confirm_final or CONFIRM_FINAL
    overrides = {}
    if a.epochs or N_EPOCHS:
        overrides["n_epochs"] = a.epochs or N_EPOCHS
    if RATE_SOURCE != "rates":
        overrides["rate_source"] = RATE_SOURCE
    n_scenarios, b_pca, tag = N_SCENARIOS, B_PCA, a.tag or OUTPUT_TAG
    if a.smoke:
        overrides.update(n_epochs=30, n_grad=2)
        seeds, n_scenarios, b_pca, tag = (a.seeds or [1]), 200, 50, "smoke"
    if overrides and not tag:              # a changed model never writes into the main folders
        tag = "_".join("%s-%s" % kv for kv in sorted(overrides.items()))
    if block == "test" and not confirm:
        sys.exit("The test block is used once, at the very end: set CONFIRM_FINAL = True or pass --confirm-final.")

    # locations must be set before volgan_fixed is imported (volgan_fixed/common.py reads them once)
    here = os.path.dirname(os.path.abspath(__file__))       # 1_one_step/: this script, its networks and results
    code = os.path.dirname(here)                            # code/: volgan_fixed/ (shared with 2_recursion/)
    if a.data_dir or DATA_DIR:
        os.environ["VOLGAN_DATA_DIR"] = a.data_dir or DATA_DIR
    os.environ["VOLGAN_SPX_FILE"], os.environ["VOLGAN_NDX_FILE"] = SPX_FILE, NDX_FILE
    os.environ["VOLGAN_RATES_FILE"] = a.rates_file or RATES_FILE
    if tag:
        os.environ.setdefault("VOLGAN_RUNS_DIR", os.path.join(here, "runs_%s" % tag))
        os.environ.setdefault("VOLGAN_RESULTS_DIR", os.path.join(here, "results_%s" % tag))
    os.environ.setdefault("VOLGAN_RUNS_DIR", os.path.join(here, "trained_networks"))
    os.environ.setdefault("VOLGAN_RESULTS_DIR", os.path.join(here, "results"))
    sys.path.insert(0, code)
    from volgan_fixed import common, postprocessing, preprocessing, reporting, training

    print("VolGAN Part 1 | index %s | block %s | steps %s\n  configs %s | seeds %s | overrides %s\n  data %s\n  "
          "rates %s\n  runs %s\n  results %s" % (index, block, steps, configs, seeds, overrides or "none (the study)",
                                                 common.DATA_DIR, common.RATES_FILE, common.RUNS_DIR, common.RESULTS_DIR))
    if set(steps) - {"tables", "figures"}:          # tables and figures read saved results only
        use_rates = overrides.get("rate_source", "rates") == "rates"
        common.check_inputs(index, need_rates=use_rates)
        print("  loaded " + preprocessing.load(index, with_rates=use_rates).describe())      # THE data call
    t0 = time.time()
    if "train" in steps:
        training.train_step(index, configs, seeds, overrides, time_first=not a.smoke)
    if "simulate" in steps:
        postprocessing.simulate_step(index, block, configs, seeds, allow_test=confirm, n_scenarios=n_scenarios)
    if "evaluate" in steps:
        postprocessing.evaluate_step(index, block, configs, seeds, allow_test=confirm, b_pca=b_pca, overrides=overrides)
    if "tables" in steps:
        reporting.make_tables(index, block, seeds)
    if "figures" in steps:
        reporting.make_figures(index, block, seeds, rho_overlay_a=RHO_OVERLAY_A)
    print("Finished in %.1f minutes. Results: %s" % ((time.time() - t0) / 60, os.path.join(common.RESULTS_DIR, index)))


if __name__ == "__main__":
    main()
