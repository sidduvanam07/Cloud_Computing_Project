"""
validation.py - Honest accuracy assessment of the analytical cold-start model (analytical.py).

WHY THIS FILE EXISTS
  The analytical model (in particular its ramp-up rule) was developed while looking at the
  simulation results of the experiment grids E1-E6/E8-E10. Comparing the model with those same
  grids is therefore an IN-SAMPLE check, not independent validation. This module keeps the
  datasets strictly apart and reports each separately:

  group        what it is                                                    independent?
  -----------  ------------------------------------------------------------  ------------------------
  in_sample    the 114 points of the existing experiment grids               NO  (model developed on them)
  audit_seen   12 configurations an independent audit tried after the        NO  (seen before the C1
               model was built                                                   correction; see below)
  c1_dev       sparse evenly-spaced cases that exposed the C1 inconsistency   NO  (motivated the C1 fix)
               and the regular+cap rejection cases
  heldout_v1   first random sample (sampler seed 20261122, sim seeds 100-119).   NO - it was evaluated
               DEVELOPMENT DATA NOW: after it was evaluated, a bug in the        before a model bug fix
               evenly-spaced ramp-up term was fixed (found by a hand test, not   (see below)
               by looking at these errors), so by the rule below it is no
               longer held-out. Kept and re-evaluated to show the effect.
  heldout      second random sample (sampler seed 20261123, sim seeds 200-219),  YES, if the procedure
               42 configurations, stratified by category, drawn and printed       below is followed
               BEFORE it was evaluated, evaluated once

HELD-OUT PROTOCOL (do not break it)
  * The sampler below fixes the configurations. They were printed before any simulation was run on
    them, and the model was evaluated on them ONCE.
  * The model must not be tuned against this set. If the model is changed after looking at
    held-out results, those results become development data: write a NEW sampler seed
    (HELDOUT_V2) and report the old set as in-sample.

ERROR DEFINITIONS (all computed on the mean over simulation seeds of each configuration)
  p_cold_err_pp = 100 * (model p_cold - simulated p_cold)        signed, percentage points
  absolute error = |p_cold_err_pp|; reported as mean (MAE), median, maximum
  relative error = (model - sim) / sim, reported ONLY where sim p_cold >= REL_MIN (1 %). Below
                   that the denominator is tiny and a relative error would be misleading; those
                   points are reported through absolute error (n_low_p, mae_low_p_pp).
  rejection error: |model rejection rate - simulated| in pp, over all points AND over the points
                   where rejections actually occur (nonzero simulated or modelled rate)

Run:  python validation.py     (writes results/analytical_validation.csv and ..._summary.csv)
"""
import json
import os
import numpy as np
import pandas as pd
import config as C
import workload as W
import analytical as AN
from metrics import summarize
from serverless import simulate_serverless

OUT = "results"
HELDOUT_RNG_SEED = 20261123                  # fixes the CURRENT held-out configurations (see protocol above)
HELDOUT_SIM_SEEDS = list(range(200, 220))    # disjoint from C.SEEDS (0-9), the audit seeds (0-19) and v1 (100-119)
HELDOUT_V1_RNG_SEED = 20261122               # first sample: now development data
HELDOUT_V1_SIM_SEEDS = list(range(100, 120))
AUDIT_SIM_SEEDS = list(range(0, 20))         # the audit probes used 20 seeds, 0-19
REL_MIN = 0.01                               # relative error only where simulated p_cold >= 1 %


# ------------------------------------------------------------------ configuration helpers
def cfg(group, category, gen, wkw=None, idle_timeout=C.IDLE_TIMEOUT_S, cold_start=C.COLD_START_S,
        max_concurrency=None, mean_exec=C.MEAN_EXEC_TIME_S, cid=None):
    return {"id": cid, "group": group, "category": category, "gen": gen, "wkw": dict(wkw or {}),
            "idle_timeout": idle_timeout, "cold_start": cold_start, "max_concurrency": max_concurrency,
            "mean_exec": mean_exec}


def audit_configs():
    """The 12 configurations tried by the independent audit (reconstructed from its description).
    NOTE: the audit's 2nd case originally ran the simulator with the default 600 s timeout by mistake;
    here every case uses the intended parameters. These are SEEN configurations (not held-out)."""
    A = "audit_seen"
    return [
        cfg(A, "regular", "steady", {"rate": 7.0}, mean_exec=0.1),
        cfg(A, "regular", "steady", {"rate": 3.0}, idle_timeout=20, mean_exec=0.5),
        cfg(A, "regular_sparse", "steady", {"rate": 0.2}, idle_timeout=10),
        cfg(A, "regular_sparse", "steady", {"rate": 0.05}, idle_timeout=60),
        cfg(A, "variable", "variable", {"mean_rate": 8.0, "period": 60.0}, idle_timeout=15),
        cfg(A, "variable", "variable", {"mean_rate": 3.0, "amplitude": 0.95, "period": 100.0}, idle_timeout=30, cold_start=1.0),
        cfg(A, "bursty", "bursty", {"num_bursts": 6, "burst_size": 150, "base_rate": 0.3}, idle_timeout=60),
        cfg(A, "bursty", "bursty", {"num_bursts": 2, "burst_size": 500, "burst_window": 2.0}, cold_start=1.0),
        cfg(A, "bursty", "bursty", {"base_rate": 0.05, "burst_size": 200}, idle_timeout=100),
        cfg(A, "idle_to_burst", "idle_to_burst", {"warmup": 30.0, "burst_rate": 10.0}),
        cfg(A, "idle_to_burst", "idle_to_burst", {"idle": 500, "burst_rate": 60.0, "burst_duration": 10.0}, cold_start=1.0),
        cfg(A, "capped_regular", "steady", {"rate": 15.0}, max_concurrency=3),
    ]


def c1_dev_configs():
    """Cases that exposed the C1 inconsistency (sparse evenly spaced traffic) and the regular+cap
    rejection over-estimate. Development data for the C1 correction: reported, never counted as held-out."""
    D = "c1_dev"
    sparse = [(0.2, 10, .2), (0.2, 6, .2), (0.2, 3, .2), (0.2, 4.5, .2), (0.1, 5, .2), (0.1, 20, .2),
              (0.05, 60, .2), (1, 5, .2), (1, 0.5, .2), (3, 20, .5), (2, 30, .2), (0.5, 1, .2)]
    out = [cfg(D, "regular_sparse", "steady", {"rate": r}, idle_timeout=t, mean_exec=e) for r, t, e in sparse]
    out += [cfg(D, "capped_regular", "steady", {"rate": r}, max_concurrency=c) for r, c in
            [(15, 3), (15, 4), (30, 5), (10, 2)]]
    return out


def heldout_configs(version=2):
    """42 configurations from a seeded stratified random sampler (see protocol in the module docstring).
    version 2 = current held-out set (group 'heldout'); version 1 = first sample (group 'heldout_v1', development data)."""
    rng = np.random.default_rng(HELDOUT_RNG_SEED if version == 2 else HELDOUT_V1_RNG_SEED)
    sig = lambda x, n=3: float(f"{x:.{n}g}")
    lu = lambda a, b: sig(np.exp(rng.uniform(np.log(a), np.log(b))))       # log-uniform
    un = lambda a, b, nd=2: round(float(rng.uniform(a, b)), nd)            # uniform
    H = "heldout" if version == 2 else "heldout_v1"
    out = []
    for _ in range(8):      # sparse evenly spaced traffic: timeout from 0.4x to 3x the arrival gap
        rate = lu(0.05, 1.0)
        out.append(cfg(H, "regular_sparse", "steady", {"rate": rate}, idle_timeout=sig(un(0.4, 3.0) / rate),
                       cold_start=un(0.2, 1.5), mean_exec=lu(0.05, 1.0)))
    for _ in range(6):      # evenly spaced, medium/high rate, varied timeout and execution time
        out.append(cfg(H, "regular_dense", "steady", {"rate": lu(1, 15)}, idle_timeout=lu(5, 600),
                       cold_start=un(0.2, 1.5), mean_exec=lu(0.05, 0.8)))
    for _ in range(6):
        out.append(cfg(H, "variable", "variable", {"mean_rate": lu(0.5, 12), "amplitude": un(0.3, 0.95),
                                                    "period": un(40, 300, 0)}, idle_timeout=lu(10, 600),
                       cold_start=un(0.2, 1.5), mean_exec=lu(0.05, 0.6)))
    for _ in range(8):      # medium-rate bursty traffic
        out.append(cfg(H, "bursty_medium", "bursty", {"base_rate": lu(0.05, 2), "num_bursts": int(rng.integers(2, 7)),
                                                      "burst_size": int(rng.integers(50, 401)),
                                                      "burst_window": un(2, 10)}, idle_timeout=lu(10, 600),
                       cold_start=un(0.2, 1.5), mean_exec=lu(0.05, 0.6)))
    for _ in range(5):
        out.append(cfg(H, "idle_to_burst", "idle_to_burst", {"idle": un(100, 1200, 0), "burst_rate": lu(5, 40),
                                                              "burst_duration": un(10, 40, 0), "warmup": un(2, 20, 0)},
                       idle_timeout=lu(60, 900), cold_start=un(0.2, 1.5)))
    for _ in range(3):      # concurrency-limited bursty traffic
        out.append(cfg(H, "capped_bursty", "bursty", {"base_rate": lu(0.2, 2), "burst_size": int(rng.integers(100, 801))},
                       max_concurrency=int(rng.integers(5, 61)), cold_start=un(0.2, 1.5)))
    for _ in range(3):      # concurrency-limited idle-to-burst
        out.append(cfg(H, "capped_idle_to_burst", "idle_to_burst", {"burst_rate": lu(20, 60), "burst_duration": un(10, 40, 0)},
                       max_concurrency=int(rng.integers(3, 16)), cold_start=un(0.2, 1.5)))
    for _ in range(3):      # concurrency-limited evenly spaced traffic
        out.append(cfg(H, "capped_regular", "steady", {"rate": lu(5, 30)}, idle_timeout=lu(30, 600),
                       max_concurrency=int(rng.integers(3, 13)), cold_start=un(0.2, 1.5)))
    for i, c in enumerate(out):
        c["id"] = f"{'H' if version == 2 else 'V'}{i + 1:02d}"
    return out


# ------------------------------------------------------------------ evaluation of one configuration
def evaluate(c, seeds):
    """Simulate configuration c on `seeds`; return one result row with the analytical estimate beside it."""
    cap = c["max_concurrency"] if c["max_concurrency"] is not None else C.MAX_CONCURRENCY
    p, rej, cost, resp = [], [], [], []
    for s in seeds:
        reqs = W.GENERATORS[c["gen"]](seed=s, mean_exec=c["mean_exec"], **c["wkw"])
        res = simulate_serverless(reqs, W.horizon_of(c["gen"], **c["wkw"]), cold_start=c["cold_start"],
                                  idle_timeout=c["idle_timeout"], max_concurrency=cap)
        m = summarize(res, cold_start=c["cold_start"])
        p.append(res.n_cold / max(res.n_completed, 1))
        rej.append(res.n_rejected / max(res.n_total, 1))
        cost.append(m["total_cost_usd"]); resp.append(m["avg_response_s"])
    est = AN.estimate_for_workload(c["gen"], mean_exec=c["mean_exec"], cold_start=c["cold_start"],
                                   idle_timeout=c["idle_timeout"], max_concurrency=c["max_concurrency"], **c["wkw"])
    sim_p, sim_rej, sim_cost = float(np.mean(p)), float(np.mean(rej)), float(np.mean(cost))
    return {"id": c["id"], "group": c["group"], "category": c["category"], "workload": c["gen"],
            "params": json.dumps({**c["wkw"], "idle_timeout": c["idle_timeout"], "cold_start": c["cold_start"],
                                  "mean_exec": c["mean_exec"], "max_concurrency": c["max_concurrency"]}),
            "n_sim_seeds": len(list(seeds)),
            "sim_p_cold": sim_p, "sim_p_cold_sd": float(np.std(p, ddof=1)) if len(p) > 1 else np.nan,
            "model_p_cold": est["p_cold"], "p_cold_err_pp": 100 * (est["p_cold"] - sim_p),
            "sim_rejection_rate": sim_rej, "model_rejection_rate": est["p_reject"],
            "rejection_err_pp": 100 * (est["p_reject"] - sim_rej),
            "sim_total_cost_usd": sim_cost, "model_total_cost_usd": est["cost_usd"],
            "cost_rel_err": (est["cost_usd"] - sim_cost) / sim_cost if sim_cost else np.nan}


def in_sample_rows(path=f"{OUT}/analytical_vs_sim.csv"):
    """The existing-grid comparison (written by run_experiments.py), mapped to the common schema."""
    a = pd.read_csv(path)
    return pd.DataFrame({
        "id": [f"{e.split('_')[0]}-{i}" for i, e in enumerate(a.experiment)], "group": "in_sample",
        "category": a.experiment, "workload": a.workload,
        "params": [json.dumps({"param": p, "value": v}) for p, v in zip(a.param, a.value)],
        "n_sim_seeds": len(C.SEEDS), "sim_p_cold": a.sim_p_cold, "sim_p_cold_sd": a.sim_p_cold_std,
        "model_p_cold": a.model_p_cold, "p_cold_err_pp": 100 * (a.model_p_cold - a.sim_p_cold),
        "sim_rejection_rate": a.sim_rejection_rate, "model_rejection_rate": a.model_rejection_rate,
        "rejection_err_pp": 100 * (a.model_rejection_rate - a.sim_rejection_rate),
        "sim_total_cost_usd": a.sim_total_cost_usd, "model_total_cost_usd": a.model_total_cost_usd,
        "cost_rel_err": a.cost_rel_error})


# ------------------------------------------------------------------ error statistics
def accuracy_summary(d):
    """Accuracy statistics of the rows in `d` (see 'ERROR DEFINITIONS' in the module docstring)."""
    d = pd.DataFrame(d)
    if len(d) == 0:
        return {"n_configs": 0}
    ae = d.p_cold_err_pp.abs()
    rel_ok = d.sim_p_cold >= REL_MIN
    rel = (d.p_cold_err_pp[rel_ok] / 100.0) / d.sim_p_cold[rel_ok]
    low = d.sim_p_cold < REL_MIN
    rej_ae = d.rejection_err_pp.abs()
    rej_nz = (d.sim_rejection_rate > 0) | (d.model_rejection_rate > 0)
    return {
        "n_configs": len(d),
        "p_cold_mae_pp": ae.mean(), "p_cold_median_ae_pp": ae.median(), "p_cold_max_ae_pp": ae.max(),
        "p_cold_bias_pp": d.p_cold_err_pp.mean(),
        "n_rel": int(rel_ok.sum()), "rel_median_abs_pct": 100 * rel.abs().median() if len(rel) else np.nan,
        "rel_mean_abs_pct": 100 * rel.abs().mean() if len(rel) else np.nan,
        "rel_max_abs_pct": 100 * rel.abs().max() if len(rel) else np.nan,
        "n_low_p": int(low.sum()), "mae_low_p_pp": ae[low].mean() if low.any() else np.nan,
        "n_nonzero_rejection": int(rej_nz.sum()),
        "rej_mae_pp_all": rej_ae.mean(), "rej_max_pp_all": rej_ae.max(),
        "rej_mae_pp_nonzero": rej_ae[rej_nz].mean() if rej_nz.any() else np.nan,
        "rej_max_pp_nonzero": rej_ae[rej_nz].max() if rej_nz.any() else np.nan,
        "cost_mape_pct": 100 * d.cost_rel_err.abs().mean(),
    }


def summary_table(allrows):
    """One row per group, plus one per held-out category."""
    rows = []
    for g, d in allrows.groupby("group", sort=False):
        rows.append({"set": g, **accuracy_summary(d)})
    for cat, d in allrows[allrows.group == "heldout"].groupby("category", sort=False):
        rows.append({"set": f"heldout/{cat}", **accuracy_summary(d)})
    return pd.DataFrame(rows)


def run_and_save(out=OUT, heldout_seeds=HELDOUT_SIM_SEEDS):
    """Evaluate audit_seen, c1_dev and held-out configurations, merge with the in-sample comparison, save CSVs."""
    parts = [in_sample_rows(f"{out}/analytical_vs_sim.csv")]
    parts.append(pd.DataFrame([evaluate(c, AUDIT_SIM_SEEDS) for c in audit_configs()]))
    parts.append(pd.DataFrame([evaluate(c, AUDIT_SIM_SEEDS) for c in c1_dev_configs()]))
    parts.append(pd.DataFrame([evaluate(c, HELDOUT_V1_SIM_SEEDS) for c in heldout_configs(version=1)]))
    parts.append(pd.DataFrame([evaluate(c, heldout_seeds) for c in heldout_configs(version=2)]))
    allrows = pd.concat(parts, ignore_index=True)
    allrows.to_csv(f"{out}/analytical_validation.csv", index=False)
    summ = summary_table(allrows)
    summ.to_csv(f"{out}/analytical_validation_summary.csv", index=False)
    return allrows, summ


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--list-heldout" in os.sys.argv:                       # print the frozen configurations without simulating
        for c in heldout_configs(version=2 if "--v1" not in os.sys.argv else 1):
            print(c["id"], c["category"], c["gen"], c["wkw"], "T=", c["idle_timeout"], "s=", c["cold_start"],
                  "E=", c["mean_exec"], "cap=", c["max_concurrency"])
        raise SystemExit
    allrows, summ = run_and_save()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ.round(3).to_string(index=False))
