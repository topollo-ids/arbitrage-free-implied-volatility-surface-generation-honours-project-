"""Figure style of the thesis: Latin Modern Roman when matplotlib can find it, serif otherwise; market red, control blue."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

_LM = [f for f in fm.findSystemFonts() if os.path.basename(f).lower().startswith("lmroman10-")]
for _f in _LM:
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Latin Modern Roman" if _LM else "serif", "mathtext.fontset": "cm",
                     "font.size": 10, "axes.linewidth": 0.6})

MKT, CTRL = "#c0392b", "#2166ac"           # market, control
PEN, PROJ = "#CC79A7", "#009E73"           # penalty in loss (cell C), bounded step + projection
