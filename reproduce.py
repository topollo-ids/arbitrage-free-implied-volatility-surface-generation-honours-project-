"""
Runs the whole study in the order of the thesis, by calling the project's own entry points one after another.
Nothing here computes anything: each step is exactly the command listed beside it, so a step can also be run by hand.

    python reproduce.py                      list the steps, in order
    python reproduce.py all                  run every validation step (Part 1, Part 2 A and B), in order
    python reproduce.py part2a-run           run only this step (several names may be given)
    python reproduce.py --from part2a        run from this step to the end of the validation steps
    python reproduce.py test --confirm-final the test-block steps (2022-2025): once, at the very end
    python reproduce.py all --data-dir D:/data    the folder holding the three workbooks (default: ./data, else ../data)

Every step saves its output and skips work that is already finished, so a run can be stopped and restarted.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# (name, what it produces, command relative to ROOT, test block?)
STEPS = [
    ("part1-spx", "Part 1, SPX: train the networks (raw_phiout, raw_phiin, seeds 1 to 5), simulate and evaluate "
                  "cells A to D on the validation block, then tables and figures (one-day results)",
     ["1_one_step/main.py", "--index", "SPX"], False),
    ("part1-ndx", "Part 1, NDX: the same for the second index (one-day results, NDX)",
     ["1_one_step/main.py", "--index", "NDX"], False),
    ("part2a-margins", "Part 2 A: the admissibility anchor and the three margins",
     ["2_recursion/part2.py", "margins", "--index", "SPX"], False),
    ("part2a-pilot", "Part 2 A: training-block pilot, including the day-1 identity check",
     ["2_recursion/part2.py", "pilot", "--index", "SPX"], False),
    ("part2a-run", "Part 2 A: the 21-day recursion of the control, then its tables and figures (21-day results)",
     ["2_recursion/part2.py", "run", "--index", "SPX"], False),
    ("part2b-remedies", "Part 2 B: clip bounds, regression gate, pools, fine-tuning, pilot, the bounded step and "
                        "data as demonstrator, comparison and report (remedies)",
     ["2_recursion/part2b.py", "all"], False),
    ("part2b-projection", "Part 2 B: the bounded step with projection, comparison and report (remedies)",
     ["2_recursion/part2b.py", "proj"], False),
    ("test-onestep", "Test block: cells A to D one day ahead (out-of-sample check)",
     ["3_test_block/test_eval.py", "onestep"], True),
    ("test-projection", "Test block: the bounded step with projection over 21 days (out-of-sample check)",
     ["3_test_block/test_eval.py", "proj", "1", "2", "3", "4", "5"], True),
    ("test-plots", "Test block: degradations, bootstrap intervals, classes, fig_test_phi and fig_test_pillars",
     ["3_test_block/test_eval.py", "plots"], True),
    ("test-surface-find", "Test block: candidate days for the surface figure",
     ["3_test_block/test_surface_day.py", "find"], True),
    ("test-surface-draws", "Test block: the drawn surfaces of the chosen day",
     ["3_test_block/test_surface_day.py", "draws"], True),
    ("test-surface-figure", "Test block: fig_test_surfaces and its numbers",
     ["3_test_block/test_surface_day.py", "figure"], True),
    ("test-surface-checks", "Test block: where each drawn surface sits within its own ensemble",
     ["3_test_block/test_surface_day.py", "checks"], True),
]
NAMES = [s[0] for s in STEPS]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("steps", nargs="*", help="step names, 'all' (validation steps) or 'test' (test-block steps)")
    p.add_argument("--from", dest="start", help="run from the first step whose name starts with this, to the end "
                                                "of the validation steps")
    p.add_argument("--confirm-final", action="store_true", help="allow the test-block steps (once, at the very end)")
    p.add_argument("--data-dir", help="folder holding the three workbooks")
    return p.parse_args()


def chosen(a):
    if a.start:
        first = next((i for i, n in enumerate(NAMES) if n.startswith(a.start)), None)
        if first is None:
            sys.exit("no step starts with %r; steps: %s" % (a.start, ", ".join(NAMES)))
        return [s for s in STEPS[first:] if not s[3]]
    out = []
    for name in a.steps:
        if name == "all":
            out += [s for s in STEPS if not s[3]]
        elif name == "test":
            out += [s for s in STEPS if s[3]]
        elif name in NAMES:
            out.append(STEPS[NAMES.index(name)])
        else:
            sys.exit("unknown step %r; steps: %s" % (name, ", ".join(NAMES)))
    return out


def main():
    a = parse()
    if not a.steps and not a.start:
        for i, (name, what, cmd, test) in enumerate(STEPS, 1):
            print("%2d. %-20s python %s%s\n    %s" % (i, name, " ".join(cmd), "   [test block]" if test else "", what))
        print("\nRun with: python reproduce.py all   (then, once, at the very end: python reproduce.py test --confirm-final)")
        return
    steps = chosen(a)
    if any(s[3] for s in steps) and not a.confirm_final:
        sys.exit("The test block is used once, at the very end: add --confirm-final to run the test-block steps.")
    env = dict(os.environ)
    if a.data_dir:
        env["VOLGAN_DATA_DIR"] = os.path.abspath(a.data_dir)
    for name, what, cmd, _ in steps:
        print("\n==== %s: python %s\n     %s" % (name, " ".join(cmd), what), flush=True)
        t0 = time.time()
        done = subprocess.run([sys.executable, os.path.join(ROOT, cmd[0])] + cmd[1:], cwd=ROOT, env=env)
        if done.returncode != 0:
            sys.exit("==== %s failed (exit code %d); finished steps are kept, so rerun to resume"
                     % (name, done.returncode))
        print("==== %s finished in %.1f minutes" % (name, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
