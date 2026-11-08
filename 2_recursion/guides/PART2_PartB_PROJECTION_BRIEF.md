# Part 2, Part B (second remedy): in-loop projection onto the arbitrage-free set

You built Part B from `PART2_PartB_BUILD_BRIEF.md` (models `bounded`, `dad`, `bounded_dad`). This brief adds ONE new
model, `bounded_proj`: the bounded step of that brief plus, after every simulated step from step 2 onward, a projection
of the simulated surface onto the arbitrage-free set. Everything in sections 0 and 5 to 7 of the first brief still
applies (same networks, seeds, calendar noise, starts, benchmark, margins, bootstrap; gates; resumability; never the
test block; never tune on validation results; stop and report rather than change a setting). Read the first brief
again before starting.

Do NOT run a model that combines rollout training and projection. The thesis now compares two levers, each on top of
the bounded step: rollout training (`bounded_dad`, a machine-learning remedy for exposure bias, already run) and
projection (`bounded_proj`, a mathematical-finance remedy for arbitrage, this brief).

---

## 1. The arbitrage-free set (exactly the one Phi measures)

Phi (`common.arbitrage_penalty`, paper eqs 3 to 5) is built from `common.penalty_terms(C, m, tau)`, which returns three
arrays of terms, each LINEAR in the relative call prices C with no constant term:

- calendar `(Nm, Nt-1)`: `tau_j (c(m_i, tau_j) - c(m_i, tau_j+1)) / (tau_j+1 - tau_j)`
- call spread `(Nm-1, Nt)`: `(c(m_i+1) - c(m_i)) / (m_i+1 - m_i)`
- butterfly `(Nm-2, Nt)`: left slope minus right slope at each interior node

A surface is arbitrage-free in the thesis's sense if and only if every term is <= 0, which is exactly Phi = 0. On the
7 x 6 grid that is 35 + 36 + 30 = 101 linear inequalities on 42 prices: `L c <= 0`.

**Build L from the code, never by hand.** Because the terms are linear with no constant, column k of L is
`penalty_terms` applied to the k-th unit price surface (flattened in the same order you use for c). Build L this way
once (it does not depend on rates or on the day) and assert, on 1,000 random C, that `L @ c` equals the stacked
`penalty_terms(C)` to 1e-13. This guarantees the projection uses the same definition of arbitrage as every result in
the thesis.

**Add the bounds every call price must satisfy for an implied volatility to exist and for the surface to be a price
surface** (these make the feasible set smaller, so every output is still inside Phi's arbitrage-free set):

- `intrinsic(m, tau, r) + DELTA_B <= c <= 1 - DELTA_B`, with `intrinsic = max(0, 1 - m exp(-r tau))`;
- call-spread lower bound: `(c(m_i+1) - c(m_i)) / (m_i+1 - m_i) >= -exp(-r tau_j)`.

Rates: the TARGET day's rates per maturity, exactly the ones `cloud_metrics` prices with (`arena.real.rates[j]`);
assert they equal `ds.rates[j]`.

## 2. The projection (Cohen, Reisinger and Wang, 2020)

Reference: S. N. Cohen, C. Reisinger and S. Wang (2020), "Detecting and repairing arbitrage in traded option prices",
Applied Mathematical Finance, doi:10.1080/1350486X.2020.1846573. Their repair is a linear programme that finds the
smallest change to the prices that satisfies the no-arbitrage constraints, measured in an l1 norm, which makes the
repair sparse (few nodes change). We use that formulation. They weight changes using bid-ask information; our data
have no quotes, so the weights convert price changes into approximate volatility changes instead.

For one surface with pre-projection prices `c0` (42,) and vegas `v` (42,):

```
minimise    sum_i  w_i (u_i + d_i)                    (u, d >= 0; the change is u - d)
subject to  L (c0 + u - d) <= -EPS_ARB                (all 101 Phi terms, tightened)
            slope lower bounds on c0 + u - d          (tightened by EPS_ARB)
            intrinsic + DELTA_B <= c0 + u - d <= 1 - DELTA_B
weights     w_i = 1 / max(v_i, V_FLOOR),   v_i = sqrt(tau_i) * phi(d1_i)  at the pre-projection volatility
```

`phi` is the standard normal density and `d1` as in `common.relative_call`. With these weights the objective is, to
first order, the total change in implied volatility, so the repair prefers to move nodes where a price change costs
few volatility points.

After solving: `c* = c0 + u - d`; invert each node to an implied volatility (section 2.2); re-price with
`common.relative_call`; and compute Phi with `common.arbitrage_penalty` on the re-priced surface.

### 2.1 The guarantee (all four must hold; assert them)

1. **Same set.** L is built from `penalty_terms` (section 1), so "feasible" means Phi = 0 by definition.
2. **Feasible.** The set is never empty: for any r >= 0, Black-Scholes prices with a single constant volatility
   satisfy every constraint strictly. The LP always has a solution. If the solver reports infeasible or a numerical
   failure, STOP and report (do not skip the surface).
3. **Robust to rounding.** Constraints are tightened by `EPS_ARB` and the bounds by `DELTA_B`, so the round trip
   price -> volatility -> price cannot push a term above 0.
4. **Checked.** For EVERY projected surface, Phi of the final re-priced surface must be exactly 0 (all three terms
   <= 0). If any surface fails, re-solve it once with `EPS_ARB x 100`; if it still fails, STOP the run and report.
   Record the maximum post-projection Phi over the whole run (it must be 0).

So either every surface the model outputs is in the arbitrage-free set, or the run stops.

### 2.2 Implied-volatility inversion

Vectorised bisection on log(sigma) over [1e-6, 50] for 200 iterations, then two Newton steps, on
`relative_call(m, tau, sigma, r) = c*`, node by node; price tolerance 1e-14. A node whose c* lies outside the prices
of sigma = 1e-6 and sigma = 50 cannot be represented: the path is marked invalid at this step (absorbing, as for any
invalid path) and counted separately as `proj_out_of_range`. This is not a failure of the guarantee (the price surface
is arbitrage-free) but it must be reported.

### 2.3 Which surfaces are projected

Only surfaces that are finite and have Phi > 0 (any term > 0) with the target day's rates. A surface with Phi = 0 is
already in the set and is returned UNCHANGED (bit for bit; test it). Non-finite surfaces are not projected; they are
invalid as before.

## 3. Where it acts in the recursion

Add an optional argument `repair=None` to `recursion.rollout` (default `None` reproduces the current code bit for
bit). For steps k >= 2 only, in this order:

1. `r_scaled, dg = step(gen, cond, z)` with the same calendar-day noise `z` as every other model;
2. `dg = clip(dg, lo, hi)` (the bounded step; same `clip_bounds.json`);
3. `sigma_pre = exp(cond_g + dg)`;
4. if finite and Phi(sigma_pre) > 0: `sigma = project(sigma_pre)`, else `sigma = sigma_pre`;
5. `dg_eff = log(sigma) - cond_g` (the step actually taken);
6. validity check on `sigma` (finite, 1% to 300%) and the return;
7. score and yield with `sigma` and `dg_eff` (rho uses `dg_eff` at the 1M ATM node);
8. state update with `r_scaled` and `dg_eff`.

Step 1 is never clipped or projected: the h = 1 identity must hold exactly (gate as before). The benchmark is never
touched. Projection adds no random numbers; the noise per calendar day is identical to every other model's.

Thread `repair` through `run_start`, `run_seed`, `consolidate`; include its settings hash in `seed_signature` only
when active, so earlier models' signatures and resume files are unchanged.

### Per-cloud records to add (shape (n_starts, H), zeros at h = 1 and for models without projection)

- `proj_share`: share of valid paths projected at this step;
- `proj_dvol_mean`, `proj_dvol_max`: mean and maximum over projected paths of the total absolute volatility change
  `sum_i |sigma_i - sigma_pre_i|` (volatility points), and the largest single-node change;
- `proj_nodes_mean`: mean number of nodes changed by more than 1e-8 in volatility (sparsity);
- `proj_out_of_range`: count of paths invalidated in 2.2;
- `proj_retries`: count of re-solves in 2.1.4;
- `phi_post_max`: maximum Phi after projection (must be 0).

## 4. Fixed constants (never change after a run exists)

| Constant | Value |
|---|---|
| `EPS_ARB` | 1e-10 (on every Phi term and every slope lower bound, in its own units) |
| `DELTA_B` | 1e-10 (price bounds) |
| `V_FLOOR` | 1e-6 (vega floor in the weights) |
| norm | weighted l1, weights 1 / max(vega, V_FLOOR) at the pre-projection volatility |
| solver | `scipy.optimize.linprog`, method `"highs"`, single thread, default tolerances; record the version |
| batching | the projected surfaces of one (seed, start, step) cloud, in path order, in fixed chunks of 200 surfaces per LP (block-diagonal; L as a sparse matrix, `scipy.sparse.kron`) |
| projected steps | k = 2 ... 21, after the clip |
| model_id | `bounded_proj` |

The l1 optimum can be non-unique. Fixed chunking, path order and a single thread keep the result deterministic;
assert that re-running two starts gives identical arrays.

## 5. Tests (`tests/test_projection.py`; all existing tests must still pass)

1. L from `penalty_terms` reproduces the terms on random surfaces (1e-13).
2. 1,000 perturbed Black-Scholes surfaces with injected violations (random bumps of up to 5 volatility points at
   random nodes): every projected surface has Phi exactly 0 after the round trip; inversion error < 1e-12 in price.
3. A surface with Phi = 0 is returned bit for bit.
4. `rollout` with `repair=None` equals the current code; `bounded_proj` with projection switched off equals `bounded`.
5. Step 1 is never projected (h = 1 identity).
6. Determinism: two runs of the same start give identical arrays.
7. Sparsity (report only): for a surface with a single butterfly violation, how many nodes change.

## 6. Gates and pilot

1. Identity gate for `bounded_proj` (max difference 0, all seeds).
2. Pilot on the training block (24 evenly spaced starts, all seeds, as before): D_A, D_C, D_D, S by h; the projection
   records of section 3; time per (seed, start). Labelled in-sample; changes no setting. If the time per (seed,
   start) projects the validation run beyond 10 hours, you may run the five seeds as parallel processes (results are
   unaffected); report the timing either way.

## 7. The validation run and the comparison

Run `bounded_proj` on the validation block with the fixed settings (198 starts x 5 seeds x 1,000 paths x 21 steps).
Then extend `compare` (do not re-run earlier models; load their results):

1. `classes.csv`: add `bounded_proj` (D_P(21), interval, classes at delta/2, delta, 2 delta, per-seed values, S(21)
   overall and per seed, cause reading). Expect D_A(21) to be NEGATIVE (projected surfaces carry no arbitrage, the
   benchmark carries some); report it as computed.
2. `delta.csv`: add `bounded_proj` minus `control`, with the label rules of the first brief.
3. `levers.csv` (new, replaces the 2 x 2 attribution for the thesis), for each property at h = 21, with intervals from
   the shared bootstrap index matrix and per-seed values:
   - rollout-training lever: `bounded_dad - bounded`;
   - projection lever: `bounded_proj - bounded`;
   - projection against rollout training: `bounded_proj - bounded_dad`.
   Class of each against zero: improvement / worsening / no clear change.
4. Descriptive curves for `bounded_proj` in `descriptive.csv` (same definitions as for the other models), plus the
   projection records of section 3 by h, overall and for starts in 20 February to 30 June 2020 against the rest.
5. `results_all.npz` for `bounded_proj` in the Part A schema, plus the new arrays.

## 8. Report

Append a section to `outputs/part2/partB/REPORT.md` (keep everything already there): gates; constant values and solver
version; the guarantee (maximum post-projection Phi over the run, retries, out-of-range paths, solver failures: all
expected 0); projection activity and size (share projected, volatility change, nodes changed) by h; the pilot table;
`classes`, `delta` and `levers` as markdown tables; S(21) per seed; timings. Numbers only. Anything not done as
specified, stated plainly. Do not draw thesis figures. Stop there. Do not touch the test block.

## 9. Order of work

1. Re-read the first brief and this one. 2. Build L and the projection; tests 5.1 to 5.3 and 5.7. 3. Add `repair` to
`rollout`; tests 5.4 to 5.6; run all tests. 4. Identity gate and pilot. 5. Validation run. 6. `compare`. 7. Report.
