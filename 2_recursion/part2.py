"""
VolGAN Part 2, Part A: how much the model degrades when it runs freely for 21 days. THE entry point for Part 2.

    python part2.py margins --index SPX      the admissibility anchor and the two fixed margins (market data only)
    python part2.py pilot   --index SPX      training block only: h = 1 identity, noise floor, precision, timing
    python part2.py run     --index SPX      the free-running experiment for one model, then the tables and figures
    python part2.py run --no-report          run only
    python part2.py run --report-only        tables and figures from the saved metrics
    python part2.py report  --index SPX      the same tables and figures, from the saved metrics only

Part 2 reads only the trained run folders of Part 1 (1_one_step/trained_networks/<INDEX>/raw_phiout/seed_<k>, the
control networks, cell B is their unweighted version), the data files and the settings in common.RecursionConfig. It never changes a Part 1
result and Part 1 never imports it. Each subcommand is resumable. The test block is never touched.

What each subcommand calls, in volgan_fixed/:
    margins   recursion.rec_margins_step      -> market_margins
    pilot     recursion_analysis.rec_pilot_step -> pilot
    run       recursion.rec_run_step          -> Arena, NoiseTable, run_seed, consolidate (then rec_report_step)
    report    recursion_analysis.rec_report_step -> analyse, write_tables, make_figures
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))               # 2_recursion/: this script and its outputs
CODE = os.path.dirname(HERE)                                     # code/: volgan_fixed/ and 1_one_step/
RUNS = os.path.join(CODE, "1_one_step", "trained_networks")      # Part 1's trained networks
PARTA = os.path.join(HERE, "A_control_recursion")                # Part A outputs


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["margins", "pilot", "run", "report"])
    p.add_argument("--index", choices=["SPX", "NDX"], default="SPX")
    p.add_argument("--data-dir", help="folder with the workbooks (default: <code>/data, else ../data)")
    p.add_argument("--rates-file", help="the rates workbook")
    p.add_argument("--runs-dir", help="folder holding <INDEX>/raw_phiout/seed_<k> (default 1_one_step/trained_networks)")
    p.add_argument("--out-dir", help="Part A output folder (default 2_recursion/A_control_recursion)")
    p.add_argument("--model-id", default="control", help="label of the model in the tables (default: control)")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--n-starts", type=int, default=12, help="pilot: starting days per seed (training block)")
    p.add_argument("--no-report", action="store_true", help="run: skip the tables and figures")
    p.add_argument("--report-only", action="store_true", help="run: only the tables and figures")
    return p.parse_args()


def main():
    a = parse()
    if a.data_dir:                                  # locations are read once, so set them before importing the package
        os.environ["VOLGAN_DATA_DIR"] = a.data_dir
    if a.rates_file:
        os.environ["VOLGAN_RATES_FILE"] = a.rates_file
    if a.runs_dir:
        os.environ["VOLGAN_RUNS_DIR"] = a.runs_dir
    if a.out_dir:
        os.environ["VOLGAN_PARTA_DIR"] = a.out_dir
    os.environ.setdefault("VOLGAN_RUNS_DIR", RUNS)              # the folder layout's defaults (flags and set variables win)
    os.environ.setdefault("VOLGAN_PARTA_DIR", PARTA)
    sys.path.insert(0, CODE)
    from volgan_fixed.common import RecursionConfig
    from volgan_fixed.recursion import rec_margins_step, rec_run_step
    from volgan_fixed.recursion_analysis import rec_pilot_step, rec_report_step

    cfg = RecursionConfig(index=a.index, seeds=tuple(a.seeds), model_id=a.model_id)
    if a.command == "margins":
        rec_margins_step(cfg)
    elif a.command == "pilot":
        rec_pilot_step(cfg, n_starts=a.n_starts)
    elif a.command == "report":
        rec_report_step(cfg)
    else:
        if not a.report_only:
            rec_run_step(cfg)
        if not a.no_report:
            rec_report_step(cfg)


if __name__ == "__main__":
    main()
