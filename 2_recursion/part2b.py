"""
VolGAN Part 2, Part B: two remedies for the 21-day recursion, run through the same experiment as Part A.
THE entry point for Part B (part2.py, Part A, is left untouched). In order:

    python part2b.py clipbounds                      the bounded step's per-node bounds (training block)
    python part2b.py gate                            regression gate: the control through the new code == Part A files
    python part2b.py pool --variant dad              the pool of own-rollout states (training block), all seeds
    python part2b.py pool --variant bounded_dad
    python part2b.py finetune --variant dad          one round of fine-tuning per seed (about 7 min per seed)
    python part2b.py finetune --variant bounded_dad
    python part2b.py pilot                           training block, in-sample sanity check of all four models
    python part2b.py run --model bounded             the validation experiment of one model (about 6 min)
    python part2b.py run --model dad
    python part2b.py run --model bounded_dad         (the primary, pre-specified adjusted model)
    python part2b.py compare                         classes.csv, delta.csv, attribution.csv, descriptive.csv
    python part2b.py report                          REPORT.md
    python part2b.py all                             every step above, in this order (resumable)

Second remedy (PART2_PartB projection brief): the bounded step plus in-loop projection onto the arbitrage-free set
    python part2b.py pilotproj                       training-block pilot of bounded_proj (with control and bounded)
    python part2b.py run --model bounded_proj        the validation experiment (projection at every step k >= 2)
    python part2b.py compare                         now also bounded_proj, and levers.csv
    python part2b.py reportproj                      appends section 9 to REPORT.md
    python part2b.py proj                            the four steps above, in this order

Everything held fixed (networks, seeds, noise, starts, benchmark, margins, bootstrap) is checked against the Part A
files in 2_recursion/A_control_recursion/; a failed check stops the step. Outputs go to 2_recursion/B_interventions/.
The test block is never used.

What each subcommand calls, in volgan_fixed/remedies.py:
    clipbounds  clipbounds_step -> clip_bounds          pool      pool_step -> build_pool (recursion.rollout)
    gate        regression_gate                          finetune  finetune_step -> finetune
    pilot       pilot_step                               run       run_model -> recursion.run_seed, consolidate
    compare     compare_step                             report    report_step
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))               # 2_recursion/: this script and its outputs
CODE = os.path.dirname(HERE)                                     # code/: volgan_fixed/ and 1_one_step/
RUNS = os.path.join(CODE, "1_one_step", "trained_networks")      # Part 1's trained networks
PARTA = os.path.join(HERE, "A_control_recursion")                # Part A outputs
PARTB = os.path.join(HERE, "B_interventions")                    # Part B outputs


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["clipbounds", "gate", "pool", "finetune", "pilot", "run", "compare", "report",
                                        "all", "pilotproj", "reportproj", "proj"])
    p.add_argument("--variant", choices=["dad", "bounded_dad"], help="pool / finetune: which pool and network")
    p.add_argument("--model", choices=["bounded", "dad", "bounded_dad", "bounded_proj"], help="run: which remedied model")
    p.add_argument("--seeds", nargs="+", type=int, help="pool / finetune: only these seeds (default: all five)")
    p.add_argument("--data-dir", help="folder with the workbooks (default: <code>/data, else ../data)")
    p.add_argument("--rates-file", help="the rates workbook")
    p.add_argument("--runs-dir", help="folder holding <INDEX>/raw_phiout/seed_<k> (default 1_one_step/trained_networks)")
    p.add_argument("--parta-dir", help="Part A output folder (default 2_recursion/A_control_recursion)")
    p.add_argument("--out-dir", help="Part B output folder (default 2_recursion/B_interventions)")
    a = p.parse_args()
    if a.command in ("pool", "finetune") and not a.variant:
        p.error("%s needs --variant dad or --variant bounded_dad" % a.command)
    if a.command == "run" and not a.model:
        p.error("run needs --model bounded, dad or bounded_dad")
    return a


def main():
    a = parse()
    for flag, env in (("data_dir", "VOLGAN_DATA_DIR"), ("rates_file", "VOLGAN_RATES_FILE"), ("runs_dir", "VOLGAN_RUNS_DIR"),
                      ("parta_dir", "VOLGAN_PARTA_DIR"), ("out_dir", "VOLGAN_PARTB_DIR")):
        if getattr(a, flag):                    # locations are read once, so set them before importing the package
            os.environ[env] = getattr(a, flag)
    for env, default in (("VOLGAN_RUNS_DIR", RUNS), ("VOLGAN_PARTA_DIR", PARTA), ("VOLGAN_PARTB_DIR", PARTB)):
        os.environ.setdefault(env, default)     # the folder layout's defaults (flags and set variables win)
    sys.path.insert(0, CODE)
    from volgan_fixed import remedies as R

    seeds = tuple(a.seeds) if a.seeds else None
    if a.command == "clipbounds":
        R.clipbounds_step()
    elif a.command == "gate":
        R.regression_gate()
    elif a.command == "pool":
        R.pool_step(a.variant, seeds=seeds)
    elif a.command == "finetune":
        R.finetune_step(a.variant, seeds=seeds)
    elif a.command == "pilot":
        R.pilot_step()
    elif a.command == "run":
        R.run_model(a.model)
    elif a.command == "compare":
        R.compare_step()
    elif a.command == "report":
        R.report_step()
    elif a.command == "pilotproj":
        R.pilot_proj_step()
    elif a.command == "reportproj":
        R.report_proj_step()
    elif a.command == "proj":
        R.pilot_proj_step()
        R.run_model("bounded_proj")
        R.compare_step()
        R.report_proj_step()
    else:
        R.clipbounds_step()
        R.regression_gate()
        for v in ("dad", "bounded_dad"):
            R.pool_step(v)
        for v in ("dad", "bounded_dad"):
            R.finetune_step(v)
        R.pilot_step()
        for m in ("bounded", "dad", "bounded_dad"):
            R.run_model(m)
        R.compare_step()
        R.report_step()


if __name__ == "__main__":
    main()
