"""
Locations for the test-block scripts in this folder. Imported first by every script here.

Layout assumed (the project's own):
    <code>/                                  volgan_fixed/ (the shared library), tests/
    <code>/../data/                          the Bloomberg workbooks and the rates workbook (found by volgan_fixed itself)
    <code>/1_one_step/                       main.py; trained_networks/SPX/ (the networks)
    <code>/1_one_step/results/SPX/val/       the Part 1 validation results  (read only: the validation values)
    <code>/1_one_step/results/SPX/test/      the one-day cells A to D on the test block (written by onestep)
    <code>/2_recursion/A_control_recursion/  the Part A outputs, margins.json (read only: the margins)
    <code>/2_recursion/B_interventions/      the Part B outputs, clip_bounds.json and bounded_proj/   (read only)
    <code>/2_recursion/test_results/         the bounded step + projection on the test block (written by proj)
    <code>/3_test_block/                     this folder: the scripts, figures/, logs/ and results/ (the surface day)
Any VOLGAN_* variable already set in the environment wins over these defaults.
"""
import os
import sys

TEST = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(TEST)
ONE_STEP = os.path.join(CODE, "1_one_step")
RECURSION = os.path.join(CODE, "2_recursion")
RUNS = os.path.join(ONE_STEP, "trained_networks")         # the trained networks: RUNS/SPX/<config>/seed_<k>
RESULTS = os.path.join(ONE_STEP, "results")               # main.py writes the test block to RESULTS/SPX/test
VAL = os.path.join(RESULTS, "SPX", "val")                 # Part 1 validation results (cells A to D, market.npz)
PROJ_DIR = os.path.join(RECURSION, "test_results")        # the 21-day bounded step + projection on the test block
PARTA = os.path.join(RECURSION, "A_control_recursion")
PARTB = os.path.join(RECURSION, "B_interventions")
FIGURES = os.path.join(TEST, "figures")
OWN_RESULTS = os.path.join(TEST, "results")               # the surface-day draws

os.environ.setdefault("VOLGAN_RUNS_DIR", RUNS)
os.environ.setdefault("VOLGAN_RESULTS_DIR", RESULTS)
os.environ.setdefault("VOLGAN_PARTA_DIR", PARTA)
os.environ.setdefault("VOLGAN_PARTB_DIR", PARTB)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.makedirs(OWN_RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
if CODE not in sys.path:
    sys.path.insert(0, CODE)


def open_test_block():
    """
    The decision to use the test block, taken once, here and nowhere else. volgan_fixed refuses the test block unless
    asked (main.py: --confirm-final); for the scripts that call the library directly this lets Split.block return it.
    Nothing in volgan_fixed is edited: the patch lives only in the running process.
    """
    from volgan_fixed import preprocessing as PP
    if getattr(PP.Split.block, "_test_open", False):
        return
    orig = PP.Split.block

    def block(self, name, allow_test=False):
        return orig(self, name, True)

    block._test_open = True
    PP.Split.block = block
