"""
run_experiments.py - Runs every experiment and writes CSV files to results/.

Run:  python run_experiments.py                (same command as before; runs everything)
      python run_experiments.py --no-autoscale (original two-system comparison only)
      python run_experiments.py --only e8_exec_time e10_concurrency   (subset)
Each experiment feeds the SAME seeded workload to every system.
Raw CSV = one row per (experiment, workload, parameter value, seed, system).
Mean/std over the seeds is computed in plots.py / summary files.

Systems: "serverless", "vm" (original fixed-capacity baseline), and - unless
--no-autoscale is given - "vm_autoscale". E1-E7 are the original experiments;
E8-E12 were added later (see README).
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import config as C
import workload as W
import analytical as AN
from serverless import simulate_serverless
from vm import simulate_vm, effective_slots
from vm_autoscale import simulate_vm_autoscaling
from metrics import summarize

OUT = "results"
os.makedirs(OUT, exist_ok=True)

STEADY_RATES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
COLD_STARTS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
IDLE_GAPS = [60, 200, 400, 600, 800, 1000, 1200]
TIMEOUTS = [5, 30, 60, 120, 300, 600, 1200]
BURST_SIZES = [50, 100, 200, 300, 500, 800]
VM_COUNTS = [2, 3, 4, 6, 8]
# ---- grids for the added experiments (E8-E12) ----
EXEC_TIMES_S = [0.05, 0.2, 0.5, 1.0]            # E8: mean execution time
MEMORIES_MB = [128, 512, 1024, 2048, 4096]      # E9: function memory (4096 added to show a VM memory limit)
CONCURRENCY_LIMITS = [10, 25, 50, 100, 1000]    # E10: serverless max concurrency (1000 = the default, effectively unlimited)
HEAVY_BURST_SIZE = 1000                         # E10: a burst heavy enough to hit limits of 50 and 100
PROVISION_DELAYS_S = [0, 15, 30, 60, 120, 180, 300]   # E12: autoscaler VM boot delay
INCLUDE_AUTOSCALE = True                        # set False by --no-autoscale


def row(exp, wl, pname, pval, seed, m):
    return {"experiment": exp, "workload": wl, "param": pname, "value": pval, "seed": seed, **m}


def both(exp, wl, pname, pval, seed, reqs, horizon, sl_kw=None, vm_kw=None,
         as_kw=None, memory_gb=None, ref_slots=None):
    """Run serverless, fixed VM and (optionally) autoscaling VM on the identical request list.
    sl_kw / vm_kw / as_kw are keyword arguments of the respective simulators; memory_gb only
    changes how serverless is BILLED (and, via vm_kw["slots"], how many functions fit on a VM).
    ref_slots sets the denominator of utilization_ref_capacity (default: the 4-slot baseline); pass
    the fixed pool's real slot count when it differs from the default (see metrics.py)."""
    sl = summarize(simulate_serverless(reqs, horizon, **(sl_kw or {})),
                   cold_start=(sl_kw or {}).get("cold_start"), memory_gb=memory_gb, ref_slots=ref_slots)
    vm = summarize(simulate_vm(reqs, horizon, **(vm_kw or {})), ref_slots=ref_slots)
    out = [row(exp, wl, pname, pval, seed, sl), row(exp, wl, pname, pval, seed, vm)]
    if INCLUDE_AUTOSCALE:
        a = summarize(simulate_vm_autoscaling(reqs, horizon, **(as_kw or {})), ref_slots=ref_slots)
        out.append(row(exp, wl, pname, pval, seed, a))
    return out


def e1_rate_sweep():
    rows = []
    for wl, key in (("steady", "rate"), ("variable", "mean_rate")):
        for rate in STEADY_RATES:
            for seed in C.SEEDS:
                reqs = W.GENERATORS[wl](seed=seed, **{key: float(rate)})
                rows += both("E1_rate_sweep", wl, "arrival_rate_rps", rate, seed, reqs, W.horizon_of(wl))
    return rows


def e2_workloads():
    rows = []
    for wl in W.GENERATORS:
        for seed in C.SEEDS:
            reqs = W.GENERATORS[wl](seed=seed)
            rows += both("E2_workloads", wl, "default", 0, seed, reqs, W.horizon_of(wl))
    return rows


def e3_cold_start():
    rows = []
    for wl in W.GENERATORS:
        for cs in COLD_STARTS:
            for seed in C.SEEDS:
                reqs = W.GENERATORS[wl](seed=seed)
                res = summarize(simulate_serverless(reqs, W.horizon_of(wl), cold_start=cs), cold_start=cs)
                rows.append(row("E3_cold_start", wl, "cold_start_s", cs, seed, res))
    return rows


def e4_idle_gap():
    rows = []
    for gap in IDLE_GAPS:
        for seed in C.SEEDS:
            reqs = W.idle_to_burst(seed=seed, idle=gap)
            rows += both("E4_idle_gap", "idle_to_burst", "idle_gap_s", gap, seed, reqs,
                         W.horizon_of("idle_to_burst", idle=gap))
    return rows


def e5_timeout():
    rows = []
    for wl in ("bursty", "variable"):
        for to in TIMEOUTS:
            for seed in C.SEEDS:
                reqs = W.GENERATORS[wl](seed=seed)
                res = summarize(simulate_serverless(reqs, W.horizon_of(wl), idle_timeout=to))
                rows.append(row("E5_timeout", wl, "idle_timeout_s", to, seed, res))
    return rows


def e6_burst_size():
    rows = []
    for size in BURST_SIZES:
        for seed in C.SEEDS:
            reqs = W.bursty(seed=seed, burst_size=size)
            rows += both("E6_burst_size", "bursty", "burst_size", size, seed, reqs, W.horizon_of("bursty"))
    return rows


def e7_vm_count():
    rows = []
    for wl in ("bursty", "idle_to_burst"):
        for n in VM_COUNTS:
            for seed in C.SEEDS:
                reqs = W.GENERATORS[wl](seed=seed)
                res = summarize(simulate_vm(reqs, W.horizon_of(wl), num_vms=n), ref_slots=n * C.SLOTS_PER_VM)
                rows.append(row("E7_vm_count", wl, "num_vms", n, seed, res))
    return rows


# ====================== added experiments (E8-E12) ======================
SENS_WORKLOADS = ("steady", "bursty")      # E8/E9: a calm and a spiky workload


def e8_exec_time():
    """E8: how the mean execution time changes cost and performance (all three systems).
    The workload keeps its ARRIVAL pattern; only request durations change."""
    rows = []
    for wl in SENS_WORKLOADS:
        for ex in EXEC_TIMES_S:
            for seed in C.SEEDS:
                reqs = W.GENERATORS[wl](seed=seed, mean_exec=ex)
                rows += both("E8_exec_time", wl, "mean_exec_s", ex, seed, reqs, W.horizon_of(wl))
    return rows


def e9_memory():
    """E9: function memory. Serverless: GB-second billing scales with memory. VMs: a VM can only
    run floor(VM RAM / function RAM) requests at once (capped by CPU slots), see vm.effective_slots.
    Memory does NOT change execution speed in this model (documented limitation)."""
    rows = []
    for wl in SENS_WORKLOADS:
        for mb in MEMORIES_MB:
            mem_gb = mb / 1024.0
            slots = effective_slots(function_memory_gb=mem_gb)
            for seed in C.SEEDS:
                reqs = W.GENERATORS[wl](seed=seed)
                rows += both("E9_memory", wl, "memory_mb", mb, seed, reqs, W.horizon_of(wl),
                             vm_kw={"slots": slots}, as_kw={"slots": slots}, memory_gb=mem_gb,
                             ref_slots=C.NUM_VMS * slots)
    return rows


def e10_concurrency():
    """E10: serverless max-concurrency limit. The VM systems do not depend on this limit, so their
    rows are identical at every x value and plot as flat reference lines."""
    rows = []
    cases = [("bursty", {}, "bursty"), ("bursty", {"burst_size": HEAVY_BURST_SIZE}, "bursty_heavy"),
             ("idle_to_burst", {}, "idle_to_burst")]
    for gen, kw, label in cases:
        for lim in CONCURRENCY_LIMITS:
            for seed in C.SEEDS:
                reqs = W.GENERATORS[gen](seed=seed, **kw)
                rows += both("E10_concurrency", label, "max_concurrency", lim, seed, reqs,
                             W.horizon_of(gen, **kw), sl_kw={"max_concurrency": lim})
    return rows


def e11_autoscale_delay():
    """E11: how the VM provisioning delay limits autoscaling. Workloads are chosen so the load
    actually exceeds one VM (steady 25 req/s; variable with mean 15 req/s peaking near 27 req/s)."""
    rows = []
    cases = [("steady", {"rate": 25.0}, "steady_25rps"), ("variable", {"mean_rate": 15.0}, "variable_15rps"),
             ("bursty", {}, "bursty")]
    for gen, kw, label in cases:
        for d in PROVISION_DELAYS_S:
            for seed in C.SEEDS:
                reqs = W.GENERATORS[gen](seed=seed, **kw)
                rows += both("E11_autoscale_delay", label, "provision_delay_s", d, seed, reqs,
                             W.horizon_of(gen, **kw), as_kw={"provision_delay": d})
    return rows


EQUAL_CAP_LABEL = f"vm_autoscale_min{C.NUM_VMS}"       # autoscaler that STARTS with the fixed pool's capacity


def e12_equal_capacity():
    """E12: fair autoscaler comparison. The fixed baseline starts with NUM_VMS (2) VMs. In E1-E11 the
    autoscaler starts with AUTOSCALE_MIN_VMS (1) VM, i.e. half the capacity. Here every workload is run
    with (a) serverless, (b) the fixed 2-VM baseline, (c) the autoscaler with min = 1 (default) and
    (d) the autoscaler with min = NUM_VMS (equal starting capacity). All other autoscaler parameters,
    including the 60 s provisioning delay, are identical in (c) and (d)."""
    rows = []
    for wl in W.GENERATORS:
        for seed in C.SEEDS:
            reqs = W.GENERATORS[wl](seed=seed)
            horizon = W.horizon_of(wl)
            rows += both("E12_equal_capacity", wl, "case", 0, seed, reqs, horizon)
            eq = summarize(simulate_vm_autoscaling(reqs, horizon, min_vms=C.NUM_VMS))
            eq["system"] = EQUAL_CAP_LABEL
            rows.append(row("E12_equal_capacity", wl, "case", 0, seed, eq))
    return rows


def equal_capacity_summary(df):
    """Mean and std over seeds (+ how often the autoscaler flapped) for E12 -> results/equal_capacity_summary.csv."""
    d = df[df.experiment == "E12_equal_capacity"]
    cols = ["success_rate", "rejection_rate", "p95_response_s", "total_cost_usd", "sla_violation_rate",
            "scale_out_events", "scale_in_events"]
    g = d.groupby(["workload", "system"])
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["runs"] = g.size()
    out["runs_with_scale_cycle"] = g.scale_cycles.apply(lambda x: int((x > 0).sum()))
    out["flapping_fraction"] = out.runs_with_scale_cycle / out.runs
    return out.reset_index()


# ------------- analytical model vs simulation (separate CSV; not a simulation experiment) -------------
def _model_args(exp, wl, v):
    """How each serverless experiment maps to analytical-model inputs: (generator, workload kwargs, model kwargs)."""
    if exp == "E1_rate_sweep":
        return wl, ({"rate": float(v)} if wl == "steady" else {"mean_rate": float(v)}), {}
    if exp == "E2_workloads":
        return wl, {}, {}
    if exp == "E3_cold_start":
        return wl, {}, {"cold_start": v}
    if exp == "E4_idle_gap":
        return "idle_to_burst", {"idle": v}, {}
    if exp == "E5_timeout":
        return wl, {}, {"idle_timeout": v}
    if exp == "E6_burst_size":
        return "bursty", {"burst_size": int(v)}, {}
    if exp == "E8_exec_time":
        return wl, {}, {"mean_exec": v}
    if exp == "E9_memory":
        return wl, {}, {"memory_gb": v / 1024.0}
    if exp == "E10_concurrency":
        gen, kw = ("bursty", {"burst_size": HEAVY_BURST_SIZE}) if wl == "bursty_heavy" else (wl, {})
        return gen, kw, {"max_concurrency": int(v)}
    return None


def analytical_vs_simulation(all_rows):
    """For every serverless experiment point: simulated mean (over seeds) vs analytical estimate."""
    df = pd.DataFrame(all_rows)
    df = df[df.system == "serverless"]
    out = []
    for (exp, wl, pname, v), g in df.groupby(["experiment", "workload", "param", "value"]):
        m = _model_args(exp, wl, v)
        if m is None:
            continue
        gen, wkw, mkw = m
        est = AN.estimate_for_workload(gen, **wkw, **mkw)
        sim_p = g.cold_start_rate.mean()
        out.append({"experiment": exp, "workload": wl, "param": pname, "value": v,
                    "sim_p_cold": sim_p, "sim_p_cold_std": g.cold_start_rate.std(),
                    "model_p_cold": est["p_cold"],
                    "p_cold_abs_error": est["p_cold"] - sim_p,
                    "sim_cold_starts": g.cold_starts.mean(), "model_cold_starts": est["expected_cold_starts"],
                    "sim_rejection_rate": g.rejection_rate.mean(), "model_rejection_rate": est["p_reject"],
                    "sim_avg_response_s": g.avg_response_s.mean(), "model_avg_response_s": est["mean_response_s"],
                    "sim_total_cost_usd": g.total_cost_usd.mean(), "model_total_cost_usd": est["cost_usd"],
                    "cost_rel_error": (est["cost_usd"] - g.total_cost_usd.mean()) / g.total_cost_usd.mean()
                    if g.total_cost_usd.mean() else np.nan})
    return pd.DataFrame(out)


EXPERIMENTS = [e1_rate_sweep, e2_workloads, e3_cold_start, e4_idle_gap,
               e5_timeout, e6_burst_size, e7_vm_count,
               e8_exec_time, e9_memory, e10_concurrency, e11_autoscale_delay, e12_equal_capacity]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the serverless-vs-VM experiments.")
    ap.add_argument("--no-autoscale", action="store_true",
                    help="skip the autoscaling VM (original two-system comparison; also skips E11 and E12)")
    ap.add_argument("--only", nargs="+", metavar="EXPERIMENT",
                    help="run only these experiment functions, e.g. e8_exec_time (others keep their saved CSV)")
    args = ap.parse_args()
    INCLUDE_AUTOSCALE = not args.no_autoscale

    todo = [fn for fn in EXPERIMENTS if not args.only or fn.__name__ in args.only]
    if args.only:
        unknown = set(args.only) - {fn.__name__ for fn in EXPERIMENTS}
        if unknown:
            raise SystemExit(f"unknown experiment(s): {sorted(unknown)}; choose from {[f.__name__ for f in EXPERIMENTS]}")
    if not INCLUDE_AUTOSCALE:
        todo = [fn for fn in todo if fn not in (e11_autoscale_delay, e12_equal_capacity)]
    for fn in todo:
        t0 = time.time()
        rows = fn()
        pd.DataFrame(rows).to_csv(f"{OUT}/raw_{fn.__name__}.csv", index=False)
        print(f"{fn.__name__:<22} {len(rows):>6} rows   {time.time()-t0:5.1f}s")

    # combined files are rebuilt from the per-experiment CSVs, so a partial run cannot truncate them
    # float_precision="round_trip": pandas' default parser can be off by 1 ULP, which would change old results
    parts = [pd.read_csv(f"{OUT}/raw_{fn.__name__}.csv", float_precision="round_trip") for fn in EXPERIMENTS
             if os.path.exists(f"{OUT}/raw_{fn.__name__}.csv")]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(f"{OUT}/all_raw_results.csv", index=False)
    # summary: mean and std over seeds
    keys = ["experiment", "workload", "system", "param", "value"]
    num = df.select_dtypes("number").columns.difference(["value", "seed"])
    summ = df.groupby(keys)[list(num)].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(f"{OUT}/summary_mean_std.csv", index=False)
    # analytical model vs simulation (serverless rows only)
    cmp_df = analytical_vs_simulation(df.to_dict("records"))
    cmp_df.to_csv(f"{OUT}/analytical_vs_sim.csv", index=False)
    # separated validation: in-sample / audit-seen / C1-development / held-out (see validation.py)
    import validation
    val_rows, val_summary = validation.run_and_save(OUT)
    if "E12_equal_capacity" in set(df.experiment):
        equal_capacity_summary(df).to_csv(f"{OUT}/equal_capacity_summary.csv", index=False)
    print(f"Saved {len(df)} rows -> {OUT}/all_raw_results.csv, summary_mean_std.csv; "
          f"{len(cmp_df)} model-vs-simulation points -> {OUT}/analytical_vs_sim.csv")
