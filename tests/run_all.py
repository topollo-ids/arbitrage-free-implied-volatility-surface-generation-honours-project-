"""
Run the test modules, one at a time, and report PASS / FAIL with the time each took:

    python tests/run_all.py                          all of them (a few minutes on a laptop CPU)
    python tests/run_all.py training recursion       only these (names: preprocessing training postprocessing recursion remedies projection)

Each module also runs on its own (python tests/test_training.py). Tests that need the market workbooks are skipped,
with a note, when the files are not found. Exit code is non-zero if any module fails.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["preprocessing", "training", "postprocessing", "recursion", "remedies", "projection"]

wanted = sys.argv[1:] or ORDER
failed = []
for name in wanted:
    path = os.path.join(HERE, "test_%s.py" % name)
    t0 = time.time()
    res = subprocess.run([sys.executable, path], capture_output=True, text=True)
    last = (res.stdout.strip().splitlines() or [""])[-1]
    ok = res.returncode == 0
    print("%-24s %-5s %6.0f s   %s" % (os.path.basename(path), "PASS" if ok else "FAIL", time.time() - t0, last if ok else ""),
          flush=True)
    if not ok:
        failed.append(name)
        print("\n".join((res.stdout + res.stderr).strip().splitlines()[-15:]), flush=True)
print("\nAll %d modules passed" % len(wanted) if not failed else "\nFAILED: %s" % ", ".join(failed))
sys.exit(1 if failed else 0)
