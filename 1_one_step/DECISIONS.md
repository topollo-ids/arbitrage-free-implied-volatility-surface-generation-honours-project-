# Part 1 decision log (the 2×2 arbitrage ablation: Loss × Weighting)

These choices were fixed **before** any Part 1 result was seen. None may be tuned on validation or test results.

**What Part 1 is.** Part 1 studies the design choices Cont and Vuletić made for handling arbitrage, one device at a time, and their effect on the model's output. It does not try to fix anything. A small effect of a lever, or an effect that appears only together with another lever, is a finding and is reported as such.

## 0. The data (`volgan_fixed/preprocessing.py`: one call, `load(index)`)

- **Files** (2026-10-13): three workbooks in the data folder, named in `main.py` SETTINGS.
  - `spx_grid42_2007_2025_clean.xlsx` and `ndx_grid42_2010_2025_clean.xlsx`: Bloomberg `IVOL_MONEYNESS` (Listed Implied Volatility Engine), end of day, in percent to 2 decimals.
  - `rates_2008_2025_clean.xlsx`: r_1M … r_1Y in decimals, continuously compounded, at 30, 60, 90, 180, 270 and 360 days. These are exactly the grid's maturities, so r enters node for node with no interpolation.
- **Grid: 42 points.** Moneyness 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20 × maturities 30, 60, 90, 180, 270, 360 days, maturity-major. It is the grid of the workbook (7 × 6); there is no other reason for this choice. The 9M (270-day) row is interpolated by the vendor in total variance.
  - The 49-point workbooks (`*_ivol_surfaces_*`, 0.90–1.10 × 30 days–24 months) are **not used**. They cannot give this grid: they have no 80% or 120% strikes and no 270-day row. The 42-point workbooks' notes say they supersede them.
  - Their extra arbitrage came mainly from the 18M and 24M maturities, combined with the finer 97.5% and 102.5% strikes. That is a resolution effect: over a 2.5% strike step at long maturities, the true convexity of call prices is tiny, so small noise in the vols flips the butterfly sign.
- **One risk-free curve.** The rate is the US Treasury bill curve, the same for both indices. The rates workbook's SPX and NDX sheets hold identical r columns (difference exactly 0 on all 3,963 shared days), so `load` reads one sheet (`RATES_SHEET = "SPX"`), which covers every NDX day.
- **Dividends: not used** (the user's decision, 2026-10-13: rates only). The workbook's q columns are ignored, and Φ prices calls without dividends, as the paper does (F14).
- **Window.** Only days present in both the surfaces and the rates are used.
  - **SPX: 2008-06-04 to 2025-12-31**, 4,423 days. The 357 surface days before then have no rates: the US Treasury issued no 1-year bills from February 2001 to June 2008.
  - **NDX: 2010-04-01 to 2025-12-31**, 3,963 days.
  - The block dates are unchanged. SPX trains on 2,412 days from June 2008 (Lehman included); NDX trains on 1,952 days.
- **Arbitrage in the raw grid, with the real rates:**

  | | train: days with Φ > 0 | train: mean Φ | validation: days with Φ > 0 | validation: mean Φ |
  |---|---|---|---|---|
  | SPX | 4.6% (calendar 108, butterfly 4) | 2.9·10⁻⁴ | 1.2% (butterfly 8, calendar 4) | 1.1·10⁻³ |
  | NDX | 6.0% (calendar 85, butterfly 33) | 8.7·10⁻⁴ | 4.0% (butterfly 21, calendar 19) | 3.5·10⁻³ |

  - With the real rates, most violating training days are small calendar violations. The mean Φ still comes mostly from a few large butterfly days.
- **Flat 120% column** (reported, not acted on). Days with σ(1.20) = σ(1.10) exactly, at one or more maturities: SPX 455 of 2,412 training days and 109 of 1,008 validation days; NDX 758 of 1,952 and 118 of 1,008. Mostly 30-day.
- **Calibration points.** Moneyness 0.90, 1.00 and 1.10 at 30, 90 and 360 days: grid nodes, never interpolated. The 120% column is avoided because Bloomberg fills it flat; the 270-day row because it is interpolated.

## 1. The design: a 2 × 2 on the raw vendor grid

- **Levers.** Loss (Φ out of / in the generator loss) × Weighting (unweighted / re-weighted at evaluation time).
- **Two training configurations, five seeds each (10 networks).** Each is evaluated twice, which gives four cells:

  | Cell | Training configuration | Weighting | Role |
  |---|---|---|---|
  | A | `raw_phiout` | re-weighted | the control (the paper) |
  | B | `raw_phiout` | unweighted | same networks as A |
  | C | `raw_phiin` | re-weighted | |
  | D | `raw_phiin` | unweighted | same networks as C |

- Everything is trained and scored on the raw vendor grid. Unchanged from the earlier plan: block dates, rates, `sim_seed` 20261012, N = 10,000, the test block only with `--confirm-final`.
- The smoothness penalty in the generator loss (the paper's, `make_smoothness`) is part of VolGAN itself and is **not** a lever.

## 2. Loss lever: Φ in the generator loss (`volgan_fixed/training.py`)

- The loss is J_G = BCE term + α_m L_m + α_τ L_τ + α_Φ E[Φ(σ̂)], where σ̂ = exp(g_t + Δĝ). Φ is computed in float64 with each sample's predicted-day rates, and the scalar is cast back to float32.
- **α_Φ rule** (fixed in advance). α_Φ is the mean over match-phase updates of ‖∇BCE‖/‖∇Φ‖, taken over the updates with ‖∇Φ‖ > 10⁻¹². If fewer than 50% of updates qualify, it is the ratio of means over all updates instead. The rule used, the number of updates used and the share of zero gradients go into `run.json`.
- With `phi_in_loss=False`, nothing about Φ is computed in the loss path. `fit()` then reproduces the pre-Part-1 weights bit for bit (`tests/test_training.py`, `test_regression_guard`).
- The per-epoch mean Φ of the generated batch is logged for the Φ-in runs only (`train_phi` in `logs.npz`).

## 3. Weighting lever, scoring and metrics (`volgan_fixed/postprocessing.py`)

- Re-weighting is unchanged: `max_penalty` with c = 500 for the re-weighted cells, β = 0 (`rule="raw"`) for the unweighted ones.
- **Scoring reference: the raw vendor grid, in every cell.** Realised surfaces, returns, surface-VIX, increments and the market Φ reference all come from the raw grid (G9).
- **n for the constant-correlation band:** N = 10,000 for unweighted cells, and the **median daily ESS** for re-weighted cells. The paper does not state its n.
- **PCA series:** series b takes scenario b on days with uniform weights (β = 0). On other days it takes the scenario drawn from that day's weights, using an RNG seeded by (sim_seed, b). The choice is made day by day, so cell A at β = 0 is exactly cell B.
- The surface-VIX of a condition surface uses the condition day's 30-day rate (`Dataset.cond_rates`).
- **Seeds:** training seeds are 1–5. The simulation seed is 20261012 for every model. Results are summarised per seed, then as the mean and [min, max] across seeds, and never pooled. Single-network figures use seed 1.

## 4. Other

- **Split:** by predicted day. Train ≤ 2017-12-31 < validation ≤ 2021-12-31 < test ≤ 2025-12-31. Runs from before Part 1 still load with their old split by calendar year.
- **Rates:** `rate_source="rates"` is the control. `variant("released")` also sets `rate_source="zero"` (F4).
- **Determinism:** `set_seed` turns on `torch.use_deterministic_algorithms(True)` and sets `cudnn.benchmark = False`. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set on import. CPU and GPU runs are not bitwise equal, so train all runs on one kind of machine.
- **`fig_phi_time`:** shows the raw vendor grid (the market) and cells B and A. The y-axis is logarithmic when the positive values span more than two decades. The axis is cut at the 1st percentile of positive values, and at most six decades below the maximum. Zero days are gaps.
- **Configuration names:** `raw_phiout` (cells A, B) and `raw_phiin` (cells C, D).
- **Run signatures** identify a trained run (hash of the config, the index and the input files). The two removed data settings are kept inside the hash with their old values, so the signatures of runs trained before the removal are unchanged (`tests/fixtures/baseline.json`, checked by `test_signatures_match_the_baseline`).
- **Code layout (2026-10-14):** seven modules in `volgan_fixed/`, each importing only from the ones before it: `common` ← `preprocessing` ← `training` ← `postprocessing` ← `reporting` ← `recursion` ← `recursion_analysis`. Entry points: `main.py` (Part 1) and `part2.py` (Part 2, Part A).

## 5. Part 2 hook: the projection onto the no-arbitrage set

Not part of Part 1 and **not in the code**. It was built and tested for an earlier version, then removed (the user, 2026-10-14) because it belongs with the interventions in the multi-step section of Part 2, where it will be implemented. Nothing in Part 1 depends on it.

## 6. Data lever dropped (2026-10-14)

The earlier design was a 2 × 2 × 2 with a third lever, the data: the Bloomberg grid as loaded against the same grid passed through the paper's kernel estimator (cells E–H). **That lever is dropped and its code is removed**, on the user's decision, so the design returns to the 2 × 2 above. The reasons it was dropped:

- The paper smooths scattered raw option quotes. We only have the vendor's already-interpolated 42-point grid, so the kernel could only be a second-stage filter on it, which is not the paper's preprocessing and a weaker contrast than the paper's.
- The paper's bandwidth rule was degenerate on both of our indices (the seeded selection day had Φ = 0), so the bandwidth had to be set by hand to the paper's reported optimum (0.002, 0.046). That makes the size of the treatment a judgement call, not a result of the rule.
- On SPX the pass at that bandwidth moved the surface by about 1 vol point per node, removed all training-block arbitrage (days with Φ > 0 from 4.6% to 0%) and cut validation Φ by an order of magnitude. It was therefore a large treatment, applied by a filter whose scale we could not tie to the paper's.

What this changes: no run, cell, table, figure, signature, rate or seed of the 2 × 2 depends on it. Everything is trained and scored on the raw vendor grid. A run folder from the earlier design whose `run.json` says `data_smooth: true` is refused when loaded; one with it false loads normally.
