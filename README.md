# Serverless-style vs Traditional VM Execution - simulation project

Event-driven simulation (pure Python, no SimPy needed) comparing a serverless (AWS Lambda-style)
platform with (a) a fixed pool of always-on VMs and (b) an optional autoscaling VM pool.

## Scope: what is and is NOT implemented (please read before citing this project)
* **Serverless platform model** (`serverless.py`): a **fixed idle timeout** (`IDLE_TIMEOUT_S`, the same for
  every instance) and **most-recently-used** reuse of warm instances; a fixed cold-start time; an optional
  concurrency cap. Nothing in the code adapts the timeout or chooses instances by lifecycle.
* **Adaptive warm-instance retention is not implemented.** **Lifecycle-aware scheduling is not implemented.**
  They are not part of this project; no result here says anything about them.
* **The analytical cold-start model (`analytical.py`) and the autoscaler (`vm_autoscale.py`) are custom,
  simplified models** written for this project. They draw on standard ideas (Little's law, M/G/infinity
  occupancy, an Erlang loss system, threshold autoscaling with a cooldown) but are **not reproductions of any
  published algorithm or of any cloud provider's policy**, and this README cites no paper. Do not describe
  them as such in a report.
* Numerical parameters are assumptions or approximate list prices, each tagged `[PRICE]` / `[ASSUMPTION]`
  in `config.py`.

## Setup
    pip install -r requirements.txt

## Run (the original commands are unchanged)
    python workload.py               # shows the 4 workloads + reproducibility check
    python tests_validation.py       # 31 original hand-calculated checks (unchanged)
    python tests_new_features.py     # tests of the first improvement pass (--fast skips the end-to-end run)
    python tests_corrections.py      # tests of the audit corrections (--fast skips the experiment tests)
    python run_experiments.py        # all experiments + validation -> results/*.csv  (about 1 minute)
    python plots.py                  # 25 graphs -> results/figures/
    python validation.py             # (also run by run_experiments.py) analytical model vs simulation
    python analytical.py             # prints the analytical estimate for the four default workloads

`run_experiments.py --no-autoscale` reproduces the original two-system comparison (and skips E11/E12);
`--only e8_exec_time e12_equal_capacity` re-runs selected experiments without truncating the combined CSVs.

## Files
| File | Purpose |
|---|---|
| config.py | every parameter and price, each tagged [PRICE] or [ASSUMPTION]; `validate_config()` checks them |
| workload.py | seeded generators: steady, bursty, variable, idle-to-burst (+ `profile_of`, the rate profile used by the analytical model) |
| common.py | shared SimResult object |
| serverless.py | instance pool, cold/warm starts, fixed idle timeout, MRU reuse, concurrency cap |
| vm.py | fixed VMs, CPU slots, FCFS queue, rejection when queue is full (+ `effective_slots`) |
| vm_autoscale.py | autoscaling VM pool (separate event loop; same SimResult) |
| analytical.py | analytical cold-start / rejection / cost estimate (Poisson and evenly-spaced arrivals) |
| validation.py | accuracy of the analytical model, with in-sample / audit-seen / development / held-out sets kept apart |
| cost.py, metrics.py | cost formulas, break-even; all reported metrics |
| tests_validation.py | original 31 baseline cases (unchanged) |
| tests_new_features.py, tests_corrections.py | tests of everything added later |
| run_experiments.py, plots.py | experiments E1-E12 -> CSV; figures 1-25 |

## Experiments
Original: E1 arrival-rate sweep | E2 four workloads | E3 cold-start time | E4 idle gap | E5 idle timeout | E6 burst size | E7 number of VMs
Added: **E8** execution time 50/200/500/1000 ms | **E9** function memory 128-4096 MB (4096 shows the VM RAM limit) |
**E10** serverless concurrency limit 10/25/50/100 (+1000 = default) | **E11** autoscaler provisioning delay |
**E12** equal-initial-capacity autoscaler comparison (below).
E1, E2, E4, E6 also contain `vm_autoscale` rows. The `serverless` and `vm` rows of E1-E7 are bit-identical to the
original results (1,340 rows x 25 original columns checked against the original CSV).
The only intentional change to E7 is the new column `utilization_ref_capacity` (see Utilization).

Output files: `all_raw_results.csv`, `summary_mean_std.csv`, `raw_e*.csv`, `analytical_vs_sim.csv` (in-sample
comparison, unchanged format), `analytical_validation.csv` + `analytical_validation_summary.csv` (all validation
groups), `equal_capacity_summary.csv` (E12 means/SDs and flapping).

## Metrics
`avg/median/p95/p99_response_s`, `avg_queue_s`, `avg_startup_s` describe **completed requests only**; always read
them with the reliability metrics, which cover **all offered requests**: `success_rate`, `rejection_rate`,
`sla_violation_rate` (rejected requests count as violations; SLA = `SLA_RESPONSE_S` = 1.0 s, an assumption),
`sla_violation_rate_completed`, `cost_per_successful_request_usd` (same value as the original `cost_per_request_usd`).

### Output columns added after the original 25 (all appended after `window_s`; none renamed or removed)
`success_rate`, `cost_per_successful_request_usd`, `sla_violations` (count of rejected + too-slow requests), `sla_violation_rate`,
`sla_violation_rate_completed`, `utilization_basis`, `avg_concurrency`, `avg_provisioned_slots`, `utilization_ref_capacity`,
`peak_instances` (serverless: most instances alive at once; VM systems: most VMs, running + booting), `scale_out_events` / `scale_in_events`
(VMs launched / terminated by the autoscaler), `vm_billed_seconds` (billed VM-seconds; NaN for serverless), `scale_cycles`, `utilization_paid_capacity`.
The system label `vm_autoscale_min2` (E12 only) is the autoscaler with minimum = fixed pool size.

### Other CSV files
* `analytical_vs_sim.csv` (in-sample, one row per serverless experiment point): `sim_p_cold`, `sim_p_cold_std` (mean / SD over seeds of cold starts per
  completed request), `model_p_cold`, `p_cold_abs_error` (model - sim, as a fraction), `sim_cold_starts` / `model_cold_starts` (counts per run),
  `sim_rejection_rate` / `model_rejection_rate`, `sim_avg_response_s` / `model_avg_response_s`, `sim_total_cost_usd` / `model_total_cost_usd`, `cost_rel_error`.
* `analytical_validation.csv` (all validation sets in one schema): `id`, `group` (in_sample, audit_seen, c1_dev, heldout_v1, heldout), `category`, `workload`,
  `params` (JSON of the configuration), `n_sim_seeds`, `sim_p_cold`, `sim_p_cold_sd`, `model_p_cold`, `p_cold_err_pp` (model - sim in percentage points),
  `sim_rejection_rate`, `model_rejection_rate`, `rejection_err_pp`, `sim_total_cost_usd`, `model_total_cost_usd`, `cost_rel_err`.
* `analytical_validation_summary.csv` (one row per group, plus per held-out category): `n_configs`, `p_cold_mae_pp` / `p_cold_median_ae_pp` / `p_cold_max_ae_pp`
  (absolute error statistics), `p_cold_bias_pp` (signed mean), `n_rel` and `rel_median_abs_pct` / `rel_mean_abs_pct` / `rel_max_abs_pct` (relative error, only where the
  simulated probability >= 1 %), `n_low_p` and `mae_low_p_pp` (points below 1 %, judged by absolute error), `n_nonzero_rejection`, `rej_mae_pp_all`, `rej_max_pp_all`,
  `rej_mae_pp_nonzero`, `rej_max_pp_nonzero` (rejection-rate error, all points / points with rejections), `cost_mape_pct`.
* `equal_capacity_summary.csv` (E12): for each workload and system the mean and SD (`*_mean`, `*_std`) of `success_rate`, `rejection_rate`, `p95_response_s`,
  `total_cost_usd`, `sla_violation_rate`, `scale_out_events`, `scale_in_events`, plus `runs`, `runs_with_scale_cycle` and `flapping_fraction`.

### Utilization
| Column | Meaning | Comparable across systems? |
|---|---|---|
| `utilization` (original) | executing time / capacity time. Serverless: alive-instance time; fixed VM: slots x window; autoscaler: **ready** slot time (VMs still booting excluded). `utilization_basis` names the denominator | No |
| `utilization_paid_capacity` | executing time / **paid** slot time = billed VM-seconds x slots (includes boot time and the 60 s per-VM minimum). VM systems only; NaN for serverless. Fixed VM: equal to `utilization`; autoscaler: lower | VM systems only |
| `avg_concurrency` | executing time / window = average requests running at once | Yes |
| `utilization_ref_capacity` | `avg_concurrency` / reference slots. The reference is the fixed-VM pool of that experiment: 4 slots by default, **the pool's real slot count in E7 (VM count) and E9 (memory limits slots per VM)**. Counts completed work only | Yes (same denominator within a row) |
| `avg_provisioned_slots` | capacity time / window | descriptive |

### Billing window
Each system is billed over its own effective window `window_s` = max(nominal horizon, time the last accepted
request finishes). The original code already worked this way. When one system's queue drains after the nominal
horizon its window is slightly longer: on idle-to-burst (nominal 920 s) the mean windows are serverless 920.5 s,
fixed VM 925.4 s, autoscaler 930.5 s. The effect on cost is below about 1 %; it is not corrected.

### Autoscaler flapping
`scale_cycles` = min(scale-outs, scale-ins) per run. `equal_capacity_summary.csv` reports `flapping_fraction`
(share of runs with at least one cycle). With the default thresholds flapping occurs (min-1 autoscaler: steady 30 %
of runs, variable 100 %, idle-to-burst 100 %; min-2: idle-to-burst 80 %, others 0 %).

## Analytical cold-start model (analytical.py)
The simulator is the reference. Inputs: arrival-rate profile, mean execution time E, cold-start time s, idle
timeout T, arrival pattern, optional concurrency cap. Occupancy E_eff = E + p_cold*s, offered load rho = lambda*E_eff.
Because the simulator reuses the most recently used instance, concurrency "level" j is used only when j-1 others
are busy; an arrival uses level j with probability pi_(j-1) (Poisson(rho) for Poisson arrivals, Poisson-binomial
for evenly spaced arrivals, truncated at the cap = Erlang loss). A request at level j is cold if it is the first use
of that level or the idle gap since the last use exceeds T. A **ramp-up rule** adds that in the first `s` seconds of a
phase every arrival finds all instances busy, so levels 1..lambda*s are certain to be used. The rule uses only
lambda, s and the phase length (no simulator output, no future information; the model reads the rate profile from
the past towards the future only). The ramp rule was developed while looking at simulation results of the experiment grids.

**Poisson arrivals** (bursty, variable, idle-to-burst): gaps between uses of a level are exponential (thinned Poisson),
E[#gaps > g] = lambda_j*max(L-g,0)*exp(-lambda_j*g), g = T + E_eff.

**Evenly spaced arrivals** (steady; corrected after an independent audit). The earlier version also used exponential gaps,
which is wrong for deterministic spacing and badly wrong when the gap is close to T (e.g. gap 5 s, T 10 s: 13.4 % predicted
vs 0.84 % simulated). Now the gap between two uses of a level is G*D (D = 1/lambda, +-`jitter` spacing randomness,
config `STEADY_JITTER`; G geometric because each arrival uses the level with probability pi_(j-1)). The previous use leaves the
instance busy for X (exponential, mean E) plus s if it was cold, so the next use is cold iff X < gap - T - s_prev.
For one gap, c(theta) = E[1 - exp(-(gap-theta)^+/E)] with theta = T (previous warm) or T + s (previous cold); this is
exact for a single spacing (uniform) and a normal approximation for G >= 2 spacings. A two-state warm/cold chain per level
gives the expected cold starts over M uses: M*pi + (p1-pi)(1-kappa^M)/(1-kappa), kappa = c_c - c_w, pi = c_w/(1-kappa).
Boundary conditions: gap == T is **warm** (the simulator destroys an instance only if idle > T); gap > T + a few E is
**cold**; with jitter the transition is smooth inside gap*(1+-jitter). The probability that a level is used at all is
1-(1-pi)^n for n evenly spaced arrivals. The Poisson branch is numerically unchanged (checked: 60/60 estimates identical to the previous version).

### How accurate is it? (python validation.py; results/analytical_validation_summary.csv)
Errors are model minus simulation in percentage points (pp) of cold-start probability, on the mean over simulation seeds.
Relative error is reported only where the simulated probability is >= 1 % (below that it would be misleading); the
other points are judged by absolute error.

| Set | What it is | n | MAE (pp) | median (pp) | max (pp) | rel. error median / max (n) |
|---|---|---|---|---|---|---|
| In-sample | the existing experiment grids (model developed on these) | 114 | 0.28 | 0.07 | 1.55 | 15.8 % / 57 % (57) |
| Audit-seen | 12 configurations tried by the independent audit | 12 | 0.62 | 0.20 | 2.57 | 11.2 % / 22 % (9) |
| C1 development | sparse regular + regular-with-cap cases that motivated the C1 fix | 16 | 0.50 | 0.01 | 7.15 | 0.9 % / 22 % (8) |
| Held-out v1 (now development data) | first random sample, seeds 100-119 | 42 | 0.65 | 0.11 | 5.60 | 3.5 % / 66 % (25) |
| **Held-out v2** | second random sample, sampler seed 20261123, simulation seeds 200-219 | 42 | **1.02** | 0.16 | **8.87** | 7.5 % / 82 % (29) |

**Only the held-out v2 row is independent evidence**: MAE about 1 pp, median 0.16 pp, worst case 8.9 pp; mean bias -0.57 pp (it
tends to under-estimate). 13 of the 42 v2 configurations have a simulated probability below 1 % (their MAE is 0.09 pp).
Cost (mean absolute percentage error): 1.4 % in-sample, 2.8 % held-out v2. Held-out v2 error by category (MAE / max, pp):
sparse regular **2.49 / 8.87**; dense regular 0.07 / 0.21; variable 0.36 / 0.88; medium bursty **2.00 / 4.54**; idle-to-burst 0.85 / 1.62;
capped bursty 0.02 / 0.04; capped idle-to-burst 0.06 / 0.18; capped regular 0.00 / 0.01.

**What the v2 results revealed (no model change was made afterwards).** The sparse-regular results are worse on v2 (all eight
errors negative) than on v1 (MAE 0.11 pp). The large misses share two features: (a) the cold-start time is as long as or longer
than the arrival gap (s = 1.4 s vs gaps of 1.05-1.6 s): a new arrival finds the previous instance still starting and creates another
cold instance, a cascade that the independent-level chain does not capture (worst case -8.9 pp); (b) the timeout lies inside the +-10 %
spacing band of the gap (T/gap = 1.02), where the result is hypersensitive (-2.6 pp). The model is reliable for sparse regular traffic only
when the cold-start time is clearly shorter than the gap and T is not within about 10 % of the gap.

**Held-out protocol and history (for transparency).**
1. The first held-out set (v1, sampler seed 20261122) was evaluated once.
2. A hand-calculation test (regular 25 req/s, s = 0.5 s: at least lambda*s = 12 cold starts) then exposed an implementation bug: the ramp-up
   rule's forced levels were weighted by the ordinary use-probability instead of 1 in the evenly-spaced branch. The bug was fixed. It did not
   change any v1 result at the reported precision, but by the protocol a set that was evaluated before a model change is no longer held-out,
   so v1 is reported as development data.
3. A new set (v2, new sampler seed, new simulation seeds) was drawn, printed before simulation, and evaluated once. The model was not
   changed after v2. If it is changed in future, v2 becomes development data and a new set must be drawn.
The in-sample row is **not** independent validation: the ramp-up rule was developed while looking at those grids.

**Rejection rate.** An earlier version of this README understated the maximum rejection error (it quoted 0.43 pp, which was an average for one experiment).
Over the 114 in-sample points the mean error is 0.06 pp and the **maximum is 1.78 pp**, but only 8 of those points have any simulated rejections
(9 if a nonzero model prediction is also counted). Held-out v2: 9 of 42 configurations have rejections; mean error 0.55 pp over all 42,
2.59 pp over the 9 nonzero ones, **maximum 13.05 pp**. Rejection accuracy rests on very few points and should not be described as strong.
Concurrency-capped **evenly spaced** traffic is over-estimated (v2: 4.5 pp mean, 13 pp max; development cases 10-15 pp) because the truncated
Poisson-binomial is not an exact loss system; for Poisson-like capped traffic the error is smaller (v2 capped bursty 1.6 pp mean, 2.5 max; capped idle-to-burst 1.7 / 4.4).

### Assumptions and limitations of the analytical model
* Arrivals are Poisson or evenly spaced (+-jitter) inside each constant-rate phase; clustering inside a phase is not modelled.
* Each concurrency level is treated as an independent stream and successive uses of a level as a two-state Markov chain. This breaks down when
  the cold-start time is comparable to the arrival gap (see above) and for **evenly spaced traffic with concurrency > 1 and a short timeout**:
  at 10 req/s, E = 0.5 s, T = 1 s the model over-estimates cold starts by about 49 % (a regression test bounds this at 60 %).
* Execution time enters through its mean; the evenly-spaced model assumes exponential execution times like the simulator.
* The ramp-up rule assumes the pool is empty at the start of a phase; it slightly over-estimates when many idle instances exist
  (cold start 2 s on bursty: 8.1 % vs 6.6 %) and under-estimates by ~25 % for 25 req/s steady with T = 600 s (12 vs 16.6 cold starts).
* **Medium-rate bursty traffic is also weak** (held-out v2 MAE 2.0 pp, errors in both directions).
* Serverless only: VM and autoscaler latency are not modelled analytically. The variable workload is approximated by 10 s bins.
* The model is an approximation for understanding, not a predictor to use in place of the simulation.

### How the tests were checked
Passing tests were not taken as proof. Seeded defects ("mutants") were inserted into scratch copies and `tests_corrections.py` was run on each:
15 mutants, 12 detected (ramp rule off, regular branch off, forced-level weighting, cooldown ignored, scale-in while booting, provisioning delay ignored,
min-VM floor, execution time ignored, previous-cold busy time ignored, jitter ignored, E9 reference slots, paid utilization). Two survivors are
**equivalent mutants** (cannot change behaviour): the scale-in "queue is empty" guard (an idle VM implies free slots, hence an empty queue) and
`a > theta` vs `a >= theta` (both give 0 at equality). One survivor exposed a test gap (scale-out step size capped by max_vms), now covered by a test that
detects the stronger mutant.

## Autoscaling VM model (vm_autoscale.py)
Optional third system; `vm.py` is unchanged and remains the baseline. Starts with `AUTOSCALE_MIN_VMS` ready VMs (2 slots
each); every 10 s it checks utilization over the interval. **Scale out** (+1 VM) if utilization >= 70 % or >= 10 requests wait,
running+booting < max, and 30 s have passed since the last scale-out; a new VM serves requests only after the 60 s provisioning
delay (billed from launch). **Scale in** (-1 idle VM) if utilization < 30 %, the queue is empty and no VM is booting, continuously for
120 s, and ready VMs > min. Billing: launch to termination, 60 s minimum per VM. With min = max it reproduces `simulate_vm()` exactly (tested).

### Fair comparison (E12, Fig 24): equal initial capacity
The fixed baseline starts with 2 VMs. The default autoscaler (min 1) starts with **half** that capacity, which made the first
comparison unfair. E12 keeps min 1 and adds **min 2** with every other parameter identical (60 s delay), against the fixed 2-VM pool
(means of 10 seeds; success % / rejection % / P95 completed-only s / cost USD / SLA-violation %):

| Workload | Fixed 2 VMs | Autoscaler min 1 | Autoscaler min 2 (equal start) |
|---|---|---|---|
| steady | 100 / 0 / 0.59 / 0.0139 / 0.6 | 100 / 0 / 0.62 / 0.0078 / 0.8 | 100 / 0 / 0.59 / 0.0139 / 0.6 (never scales) |
| variable | 100 / 0 / 0.60 / 0.0139 / 0.7 | 100 / 0 / 1.20 / 0.0177 / 5.7 | 100 / 0 / 0.60 / 0.0139 / 0.7 (never scales) |
| bursty | 80.1 / 19.9 / 5.40 / 0.0139 / 55.9 | 79.9 / 20.1 / 7.63 / 0.0170 / 55.2 | **89.7 / 10.4 / 4.66 / 0.0239 / 48.4** |
| idle-to-burst | 84.7 / 15.3 / 5.71 / 0.0214 / 85.7 | 58.5 / 41.5 / 11.2 / 0.0135 / 96.4 | 84.7 / 15.3 / 5.71 / 0.0237 / 85.7 |

What this does and does not show:
* With equal starting capacity the autoscaler is **not** worse than the fixed pool on bursts. On the bursty workload it halves rejections
  (19.9 % -> 10.4 %) for about 72 % more cost (bursts recur every 150 s, so VMs added after the first burst help later ones).
* On idle-to-burst (one 20 s burst) the 60 s boot delay means added VMs arrive after the burst: no reliability gain, about 11 % more cost.
* The earlier statement that autoscaling is worse on bursts was caused by starting with half the capacity (min 1), not by autoscaling itself.
* Min 1 is cheaper on calm load (steady: -44 %) but, with these thresholds, it flaps and lags on variable load (P95 doubles, cost +27 %).
* Results depend on the assumed thresholds, 10 s monitoring interval and 60 s boot delay (E11 sweeps the delay). They do not show that autoscaling is good or bad in general.

## Remaining limitations
* Memory affects only serverless billing and how many functions fit on a VM, not execution speed (real Lambda scales CPU with memory).
* The SLA threshold is an absolute 1.0 s, so for 1000 ms mean execution times many violations are inherent to the workload.
* Billing is idealised (exact seconds, no free tier, no data transfer / load balancer / storage); one function, one region, one VM type; no retries.
* Runs are 10 simulated minutes (idle-to-burst about 15 min) with 10 seeds; with T = 600 s the pool rarely shrinks, so cold starts are mostly warm-up.
* The held-out set has few capped and few rejection-producing configurations; rejection-rate accuracy is weakly evidenced.
* The autoscaler is a single simple policy; scale-in uses a fixed quiet period; flapping occurs with the default thresholds.

## Reproducibility
All randomness comes from numpy seeds (0-9 for experiments, 100-119 for the held-out set, 0-19 for the audit/development sets).
Same seed -> identical workload for every system. Outputs are byte-identical across independent runs and `PYTHONHASHSEED` values (checked).
