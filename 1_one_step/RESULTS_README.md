# Part 1 outputs: what each file is, and which step makes it

`<INDEX>` is SPX (or NDX for the robustness check), and `<block>` is `val` (Part 1) or `test` (once, with `--confirm-final`). Every step is run from `main.py` (`--steps`), and every step skips work that is already done.

**Inputs** (named in `main.py` SETTINGS, read by the one data call `volgan_fixed/preprocessing.py: load(INDEX)`):
- `data/spx_grid42_2007_2025_clean.xlsx` and `data/ndx_grid42_2010_2025_clean.xlsx`: implied vols on the 42-point grid (moneyness 0.80–1.20 × 30–360 days).
- `data/rates_2008_2025_clean.xlsx`: the risk-free curve r_1M … r_1Y, at exactly the grid's maturities. One curve serves both indices; the q columns are not used.
- The days used are those both workbooks have: SPX from 2008-06-04, NDX from 2010-04-01.

**Steps, in order:** `train` → `simulate` → `evaluate` → `tables` → `figures`.

| File | Contents | Made by |
|---|---|---|
| `trained_networks/<INDEX>/<config>/seed_<k>/0_initial.pt, 1_matched.pt, 2_trained.pt` | Checkpoints (networks, optimisers, α, config) | `train` |
| `trained_networks/<INDEX>/<config>/seed_<k>/run.json` | Index, input-file hashes, rate source, seed, sim_seed, block boundaries and sizes, α_m, α_τ, α_Φ with its rule, library versions, time taken, config, signature | `train` |
| `trained_networks/<INDEX>/<config>/seed_<k>/logs.npz` | Per-epoch losses and D outputs, match-phase gradient norms (BCE, L_m, L_τ, Φ), per-epoch Φ (Φ-in runs) | `train` |
| `results/<INDEX>/<block>/sims/<config>/seed_<k>.npz` | Φ and the daily return of every scenario (days × 10,000; about 110 MB), the rates used, sim_seed | `simulate` |
| `results/<INDEX>/<block>/market.npz` | The data side of every metric, from the raw vendor grid: Φ, its parts and violation nodes; realised calibration values; data PCA and correlations | `evaluate` |
| `results/<INDEX>/<block>/<cell>/seed_<k>/metrics.npz` | Per-day metrics of one cell and seed: admissibility, calibration quantiles and indicators, ρ_t and its band, per-series PCA and correlations, ESS / max weight / entropy | `evaluate` |
| `results/<INDEX>/<block>/<cell>/summary.csv` | Headline numbers per seed, then the mean, min and max over seeds | `evaluate` |
| `results/<INDEX>/<block>/tables/tab_*.csv, tab_*.tex` | tab_admissibility, tab_calibration (+ `_full_00..10` appendix), tab_dynamics, tab_pca, tab_reweighting, tab_effects | `tables` |
| `results/<INDEX>/<block>/figures/fig_*.pdf, .png` | Requested: fig_phi_time, fig_phi_location (arbitrage); fig_rho_time, fig_pc_loadings (dynamics); fig_bands_1m, _3m, _12m (calibration, 9 panels). Appendix: fig_grad_norms | `figures` |
| `results/<INDEX>/<block>/figures/captions.md` | A caption for each figure: seeds shown, log axes, n of the band | `figures` |

**Cells** (two lever columns in the tables: Loss, Weighting).
- A: Φ out of the loss, re-weighted. This is the control.
- B: the same networks as A, unweighted.
- C: Φ in the loss, re-weighted.
- D: the same networks as C, unweighted.

A and B come from `raw_phiout`, C and D from `raw_phiin`. Every cell is trained on, and scored against, the raw vendor grid.

Runs made with changed settings (`--tag`, `OUTPUT_TAG`, `--smoke`, `--epochs`) go to `runs_<tag>/` and `results_<tag>/` (inside `1_one_step/`) and never mix with the main results.

Part 2 writes under `../2_recursion/` (Part A in `A_control_recursion/`, Part B in `B_interventions/`). All paths in this file are relative to `1_one_step/`.
