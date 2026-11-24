"""
plots.py - Builds every graph from results/all_raw_results.csv.

Run:  python plots.py        (run run_experiments.py first)
Shaded areas / error bars = +-1 standard deviation across the 10 seeds.
Colours are the same in every figure: blue = serverless, orange = fixed VM, green = autoscaling VM.
Figures 1-15 are the original ones; 16-23 were added (see README).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import workload as W
import cost as K

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)
EQ = f"vm_autoscale_min{C.NUM_VMS}"                      # autoscaler that starts with the fixed pool's capacity (E12)
COL = {"serverless": "#1f77b4", "vm": "#ff7f0e", "vm_autoscale": "#2ca02c", EQ: "#006d2c"}
LAB = {"serverless": "Serverless", "vm": f"Fixed VM ({C.NUM_VMS} x t3.medium)",
       "vm_autoscale": f"Autoscaling VM ({C.AUTOSCALE_MIN_VMS}-{C.AUTOSCALE_MAX_VMS} x t3.medium)"}
LAB[EQ] = f"Autoscaling VM, min {C.NUM_VMS} (equal start)"
ALL3 = ("serverless", "vm", "vm_autoscale")
ALL4 = ("serverless", "vm", "vm_autoscale", EQ)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})

df = pd.read_csv("results/all_raw_results.csv")


def agg(exp, workload, system, metric):
    d = df[(df.experiment == exp) & (df.workload == workload) & (df.system == system)]
    g = d.groupby("value")[metric].agg(["mean", "std"]).reset_index()
    return g["value"].values, g["mean"].values, g["std"].fillna(0).values


def line(ax, exp, wl, metric, systems=("serverless", "vm"), scale=1.0, **kw):
    for s in systems:
        x, m, sd = agg(exp, wl, s, metric)
        ax.plot(x, m * scale, marker="o", ms=4, color=COL[s], label=LAB[s], **kw)
        ax.fill_between(x, (m - sd) * scale, (m + sd) * scale, color=COL[s], alpha=0.2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png")
    plt.close(fig)
    print("saved", name)


E1 = "E1_rate_sweep"
CAP = C.NUM_VMS * C.SLOTS_PER_VM / C.MEAN_EXEC_TIME_S      # VM capacity in req/s (= 20)
BE = K.breakeven_rate()


def rate_axis(ax, wl_note="steady workload"):
    ax.set_xlabel("Request arrival rate (requests/second)")
    ax.axvline(CAP, color="grey", ls=":", lw=1)
    ax.text(CAP, ax.get_ylim()[1] * 0.97, f" VM capacity\n ({CAP:g} req/s)", va="top", fontsize=8, color="grey")


# ---- Required graphs 1-8 (steady workload) ------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
line(ax, E1, "steady", "avg_response_s"); rate_axis(ax)
ax.set_ylabel("Average response time (s)"); ax.set_title("Fig 1. Average response time vs arrival rate (steady)")
ax.legend(); save(fig, "fig01_avg_response_vs_rate")

fig, ax = plt.subplots(figsize=(7, 4.2))
line(ax, E1, "steady", "p95_response_s"); rate_axis(ax)
ax.set_ylabel("P95 response time (s)"); ax.set_title("Fig 2. P95 response time vs arrival rate (steady)")
ax.legend(); save(fig, "fig02_p95_response_vs_rate")

fig, ax = plt.subplots(figsize=(7, 4.2))
for wl, ls in (("steady", "-"), ("variable", "--")):
    x, m, sd = agg(E1, wl, "serverless", "cold_start_rate")
    ax.plot(x, m * 100, marker="o", ms=4, ls=ls, color=COL["serverless"], label=f"Serverless, {wl} workload")
    ax.fill_between(x, (m - sd) * 100, (m + sd) * 100, color=COL["serverless"], alpha=0.15)
ax.set_xlabel("Request arrival rate (requests/second; mean rate for variable)")
ax.set_ylabel("Cold-start frequency (% of requests)")
ax.set_title("Fig 3. Cold-start frequency vs workload intensity"); ax.legend()
save(fig, "fig03_cold_start_vs_rate")

fig, ax = plt.subplots(figsize=(7, 4.2))
for s in ("serverless", "vm"):
    d = df[(df.experiment == E1) & (df.workload == "steady") & (df.system == s)]
    g = d.groupby("value")[["n_total", "total_cost_usd"]].mean()
    ax.plot(g.n_total, g.total_cost_usd, marker="o", ms=4, color=COL[s], label=LAB[s])
ax.axvline(BE * C.SIM_DURATION_S, color="grey", ls="--", lw=1)
ax.text(BE * C.SIM_DURATION_S, ax.get_ylim()[1] * 0.5,
        f" analytical break-even\n ~{BE * C.SIM_DURATION_S:,.0f} requests / {C.SIM_DURATION_S:.0f} s\n ({BE:.1f} req/s)", fontsize=8, color="grey")
ax.set_xlabel(f"Total number of requests in a {C.SIM_DURATION_S:.0f} s run"); ax.set_ylabel("Total cost (USD)")
ax.set_title("Fig 4. Total cost vs number of requests (steady)"); ax.legend()
save(fig, "fig04_cost_vs_requests")

fig, ax = plt.subplots(figsize=(7, 4.2))
for s in ("serverless", "vm"):
    d = df[(df.experiment == E1) & (df.workload == "steady") & (df.system == s)]
    g = d.groupby("value")[["avg_response_s", "total_cost_usd"]].mean().reset_index()
    ax.plot(g.avg_response_s, g.total_cost_usd, marker="o", ms=4, color=COL[s], label=LAB[s])
    for _, r in g.iterrows():
        if r.value in (1, 10, 20, 25):
            ax.annotate(f"{int(r.value)} req/s", (r.avg_response_s, r.total_cost_usd), fontsize=7,
                        xytext=(4, 4), textcoords="offset points", color=COL[s])
ax.set_xlabel("Average response time (s)"); ax.set_ylabel(f"Total cost per {C.SIM_DURATION_S:.0f} s run (USD)")
ax.set_title("Fig 5. Cost vs average response time (steady; each point = one arrival rate)"); ax.legend()
save(fig, "fig05_cost_vs_response")

fig, ax = plt.subplots(figsize=(7, 4.2))
line(ax, E1, "steady", "throughput_rps")
x = np.array(sorted(df[df.experiment == E1].value.unique()))
ax.plot(x, x, ls=":", color="k", lw=1, label="Offered load (ideal)")
rate_axis(ax); ax.set_ylabel("Throughput (completed requests/second)")
ax.set_title("Fig 6. Throughput vs workload intensity (steady)"); ax.legend()
save(fig, "fig06_throughput_vs_rate")

fig, ax = plt.subplots(figsize=(7, 4.2))
line(ax, E1, "steady", "utilization", systems=("vm",), scale=100); rate_axis(ax)
ax.set_ylabel(f"Fixed-VM slot utilization (%)\n(executing time / ({C.NUM_VMS * C.SLOTS_PER_VM} slots x window))"); ax.set_title("Fig 7. Fixed-VM utilization vs workload intensity (steady)")
ax.legend(); save(fig, "fig07_vm_utilization_vs_rate")

fig, ax = plt.subplots(figsize=(7, 4.2))
line(ax, E1, "steady", "utilization", systems=("serverless",), scale=100)
ax.set_xlabel("Request arrival rate (requests/second)")
ax.set_ylabel("Serverless instance utilization (%)\n(executing time / instance alive time)")
ax.set_title("Fig 8. Serverless instance utilization vs workload intensity (steady)")
ax.legend(); save(fig, "fig08_serverless_utilization_vs_rate")

# ---- Extra graphs -------------------------------------------------------
# Fig 9: the four workloads side by side
E2 = "E2_workloads"
wls = ["steady", "variable", "bursty", "idle_to_burst"]
names = ["Steady", "Variable", "Bursty", "Idle-to-burst"]
panels = [("avg_response_s", "Average response time (s)", 1), ("p95_response_s", "P95 response time (s)", 1),
          ("total_cost_usd", "Total cost per run (USD)", 1), ("rejection_rate", "Rejected requests (%)", 100)]
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
for ax, (met, lab, sc) in zip(axes.ravel(), panels):
    for i, s in enumerate(("serverless", "vm")):
        vals = [df[(df.experiment == E2) & (df.workload == w) & (df.system == s)][met].mean() * sc for w in wls]
        sds = [df[(df.experiment == E2) & (df.workload == w) & (df.system == s)][met].std() * sc for w in wls]
        ax.bar(np.arange(4) + (i - 0.5) * 0.38, vals, 0.38, yerr=sds, capsize=3, color=COL[s], label=LAB[s])
    ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_ylabel(lab)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Fig 9. Serverless vs fixed VM across the four workload patterns (default parameters)")
save(fig, "fig09_workload_comparison")

# Fig 10: cold-start sensitivity
E3 = "E3_cold_start"
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for wl, n in zip(wls, names):
    for ax, met in zip(axes, ("avg_response_s", "p95_response_s")):
        x, m, sd = agg(E3, wl, "serverless", met)
        ax.plot(x, m, marker="o", ms=4, label=n)
for ax, lab in zip(axes, ("Average response time (s)", "P95 response time (s)")):
    ax.set_xlabel("Cold-start time (s)"); ax.set_ylabel(lab)
axes[0].legend(); fig.suptitle("Fig 10. Sensitivity of serverless latency to cold-start time")
save(fig, "fig10_cold_start_sensitivity")

# Fig 11: idle gap
E4 = "E4_idle_gap"
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
x, m, sd = agg(E4, "idle_to_burst", "serverless", "cold_start_rate")
axes[0].plot(x, m * 100, marker="o", color=COL["serverless"], label=LAB["serverless"])
axes[0].fill_between(x, (m - sd) * 100, (m + sd) * 100, color=COL["serverless"], alpha=0.2)
axes[0].axvline(C.IDLE_TIMEOUT_S, color="grey", ls="--"); axes[0].text(C.IDLE_TIMEOUT_S, m.max() * 100 * 0.5, " idle timeout", color="grey")
axes[0].set_xlabel("Idle gap before the burst (s)"); axes[0].set_ylabel("Cold-start frequency (% of requests)")
line(axes[1], E4, "idle_to_burst", "total_cost_usd")
axes[1].set_xlabel("Idle gap before the burst (s)"); axes[1].set_ylabel("Total cost (USD)"); axes[1].legend()
fig.suptitle("Fig 11. Idle-to-burst: effect of the idle gap")
save(fig, "fig11_idle_gap")

# Fig 12: idle timeout trade-off
E5 = "E5_timeout"
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for wl in ("bursty", "variable"):
    x, m, _ = agg(E5, wl, "serverless", "cold_start_rate"); axes[0].plot(x, m * 100, marker="o", ms=4, label=wl.capitalize())
    x, m, _ = agg(E5, wl, "serverless", "utilization"); axes[1].plot(x, m * 100, marker="o", ms=4, label=wl.capitalize())
for ax, lab in zip(axes, ("Cold-start frequency (% of requests)", "Serverless instance utilization (%)")):
    ax.set_xscale("log"); ax.set_xlabel("Idle timeout (s, log scale)"); ax.set_ylabel(lab); ax.legend()
fig.suptitle("Fig 12. Idle-timeout trade-off: fewer cold starts vs more idle instance time")
save(fig, "fig12_timeout_tradeoff")

# Fig 13: burst size
E6 = "E6_burst_size"
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
line(axes[0], E6, "bursty", "avg_response_s"); axes[0].set_ylabel("Average response time (s)")
line(axes[1], E6, "bursty", "p95_response_s"); axes[1].set_ylabel("P95 response time (s)")
line(axes[2], E6, "bursty", "rejection_rate", scale=100); axes[2].set_ylabel("Rejected requests (%)")
for ax in axes: ax.set_xlabel("Burst size (requests per burst)")
axes[0].legend(); fig.suptitle("Fig 13. Bursty workload: growing burst size")
save(fig, "fig13_burst_size")

# Fig 14: VM count vs serverless
E7 = "E7_vm_count"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, wl, n in zip(axes, ("bursty", "idle_to_burst"), ("Bursty", "Idle-to-burst")):
    d = df[(df.experiment == E7) & (df.workload == wl)].groupby("value")[["total_cost_usd", "p95_response_s"]].mean().reset_index()
    ax.plot(d.total_cost_usd, d.p95_response_s, marker="o", color=COL["vm"], label="Fixed VMs (label = number of VMs)")
    for _, r in d.iterrows():
        ax.annotate(f"{int(r.value)} VMs", (r.total_cost_usd, r.p95_response_s), xytext=(4, 4), textcoords="offset points", fontsize=8)
    sv = df[(df.experiment == E2) & (df.workload == wl) & (df.system == "serverless")][["total_cost_usd", "p95_response_s"]].mean()
    ax.scatter(sv.total_cost_usd, sv.p95_response_s, marker="*", s=200, color=COL["serverless"], label="Serverless", zorder=5)
    ax.set_xlabel("Total cost per run (USD)"); ax.set_ylabel("P95 response time (s)"); ax.set_title(n); ax.legend(fontsize=8)
fig.suptitle("Fig 14. Cost-latency trade-off: how many VMs are needed to match serverless?")
save(fig, "fig14_vm_count_tradeoff")

# Fig 15: what the four workloads look like
fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharey=True)
for ax, wl, n in zip(axes.ravel(), wls, names):
    reqs = W.GENERATORS[wl](seed=0)
    a = np.array([r.arrival_time for r in reqs]); h = W.horizon_of(wl)
    bins = np.arange(0, h + 10, 10)
    cnt, _ = np.histogram(a, bins)
    ax.bar(bins[:-1], cnt / 10, width=10, align="edge", color="#555")
    ax.axhline(CAP, color="red", ls=":", lw=1); ax.set_title(n)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Arrival rate (req/s, 10 s bins)")
axes[0, 0].text(5, CAP + 1, f"VM capacity ({CAP:g} req/s)", color="red", fontsize=8)
fig.suptitle("Fig 15. The four workload patterns (seed 0)")
save(fig, "fig15_workload_profiles")


# =====================================================================
# Added figures 16-23 (new experiments / systems). Each is skipped if its data is absent.
# =====================================================================
have = set(df.experiment.unique())
systems_present = set(df.system.unique())


def agg_all(exp, wl, system, metric):
    return agg(exp, wl, system, metric)


# Fig 16: three-way comparison with reliability metrics (E2)
if E2 in have and "vm_autoscale" in systems_present:
    panels = [("success_rate", "Success rate (%)", 100), ("sla_violation_rate", "SLA violations (% of ALL offered requests)", 100),
              ("p95_response_s", "P95 response time, completed requests only (s)", 1),
              ("cost_per_successful_request_usd", "Cost per successful request (USD)", 1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    for ax, (met, lab, sc) in zip(axes.ravel(), panels):
        for i, sname in enumerate(ALL3):
            g = [df[(df.experiment == E2) & (df.workload == w) & (df.system == sname)][met] for w in wls]
            ax.bar(np.arange(4) + (i - 1) * 0.27, [x.mean() * sc for x in g], 0.27, yerr=[x.std() * sc for x in g],
                   capsize=2, color=COL[sname], label=LAB[sname])
        ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_ylabel(lab, fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Fig 16. Three-way comparison (default parameters, SLA = {C.SLA_RESPONSE_S:g} s). Autoscaler starts with "
                 f"{C.AUTOSCALE_MIN_VMS} VM vs {C.NUM_VMS} fixed: see Fig 24 for equal capacity", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{FIG}/fig16_three_way_comparison.png"); plt.close(fig); print("saved fig16_three_way_comparison")

# Fig 17: execution-time sensitivity (E8)
E8 = "E8_exec_time"
if E8 in have:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for r_, wl in enumerate(("steady", "bursty")):
        line(axes[r_, 0], E8, wl, "total_cost_usd", systems=ALL3); axes[r_, 0].set_ylabel(f"Total cost per run (USD)\n[{wl}]")
        line(axes[r_, 1], E8, wl, "success_rate", systems=ALL3, scale=100); axes[r_, 1].set_ylabel("Success rate (%)")
        line(axes[r_, 2], E8, wl, "p95_response_s", systems=ALL3); axes[r_, 2].set_ylabel("P95 response, completed only (s)")
        for ax in axes[r_]:
            ax.set_xlabel("Mean execution time (s)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fig 17. Execution-time sensitivity: serverless cost grows with duration; the VM pool saturates", fontsize=10)
    save(fig, "fig17_exec_time_sensitivity")

# Fig 18: memory sensitivity (E9) + analytical break-even
E9 = "E9_memory"
if E9 in have:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    line(axes[0], E9, "steady", "total_cost_usd", systems=ALL3)
    axes[0].set_ylabel("Total cost per run (USD, steady 5 req/s)"); axes[0].legend(fontsize=7)
    line(axes[1], E9, "bursty", "success_rate", systems=ALL3, scale=100); axes[1].set_ylabel("Success rate (%, bursty)")
    mems = np.array(sorted(df[df.experiment == E9].value.unique()))
    axes[2].plot(mems, [K.breakeven_rate(memory_gb=m / 1024) for m in mems], marker="o", color=COL["serverless"])
    axes[2].axhline(C.STEADY_RATE, color="grey", ls=":", lw=1); axes[2].text(mems[0], C.STEADY_RATE * 1.04, f" default steady rate {C.STEADY_RATE:g} req/s", fontsize=8, color="grey")
    axes[2].set_ylabel("Analytical break-even rate vs fixed VMs (req/s)"); axes[2].set_title("Serverless cheaper BELOW the line", fontsize=9)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xlabel("Function memory (MB)")
    fig.suptitle("Fig 18. Memory sensitivity: serverless bill scales with memory; VM cost is flat until RAM limits slots (4096 MB)", fontsize=10)
    save(fig, "fig18_memory_sensitivity")

# Fig 19: concurrency limits (E10)
E10 = "E10_concurrency"
if E10 in have:
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for r_, wl in enumerate(("bursty_heavy", "idle_to_burst")):
        for ax, (met, lab, sc) in zip(axes[r_], [("success_rate", "Success rate (%)", 100), ("p95_response_s", "P95 response, completed only (s)", 1),
                                                    ("total_cost_usd", "Total cost per run (USD)", 1), ("throughput_rps", "Throughput (completed req/s)", 1)]):
            line(ax, E10, wl, met, systems=ALL3 if "vm_autoscale" in systems_present else ("serverless", "vm"), scale=sc)
            ax.set_xscale("log"); ax.set_xlabel("Serverless max concurrency (VM lines are flat references)")
            ax.set_ylabel(lab + f"\n[{wl}]", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fig 19. Serverless concurrency limit: tighter limits trade rejections for lower cost; 1000 = default (unlimited)", fontsize=10)
    save(fig, "fig19_concurrency_limits")

# Fig 20: analytical model vs simulation - IN-SAMPLE vs HELD-OUT shown separately
if os.path.exists("results/analytical_validation.csv"):
    av = pd.read_csv("results/analytical_validation.csv")
    sm = pd.read_csv("results/analytical_validation_summary.csv").set_index("set")
    style = {"in_sample": ("o", "#1f77b4", "In-sample (existing grids; model developed on them)"),
             "audit_seen": ("s", "#999999", "Audit-seen (already examined)"),
             "c1_dev": ("^", "#bbbbbb", "C1 development (sparse regular, regular+cap)"),
             "heldout_v1": ("d", "#e8a0a0", "Held-out v1 (re-used after a bug fix: development data)"),
             "heldout": ("D", "#d62728", "HELD-OUT v2 (random, seeds 200-219; evaluated once)")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for g, (mk, col, lab) in style.items():
        d = av[av.group == g]
        n = int(sm.loc[g, "n_configs"]); mae = sm.loc[g, "p_cold_mae_pp"]
        axes[0].scatter(d.sim_p_cold * 100, d.model_p_cold * 100, marker=mk, s=26 if g != "heldout" else 34, color=col,
                        alpha=0.65 if g != "heldout" else 0.95, label=f"{lab}: n={n}, MAE {mae:.2f} pp")
        axes[2].scatter(d.sim_rejection_rate * 100, d.model_rejection_rate * 100, marker=mk, s=26 if g != "heldout" else 34,
                        color=col, alpha=0.65 if g != "heldout" else 0.95)
    lim = max(av.sim_p_cold.max(), av.model_p_cold.max()) * 100 * 1.05
    axes[0].plot([0, lim], [0, lim], "k:", lw=1); axes[0].set_xlabel("Simulated cold-start probability (%)"); axes[0].set_ylabel("Analytical estimate (%)")
    axes[0].legend(fontsize=6.5, loc="upper left"); axes[0].set_title("Cold-start probability", fontsize=9)
    lim2 = max(av.sim_rejection_rate.max(), av.model_rejection_rate.max()) * 100 * 1.05
    axes[2].plot([0, lim2], [0, lim2], "k:", lw=1); axes[2].set_xlabel("Simulated rejection rate (%)"); axes[2].set_ylabel("Analytical estimate (%)")
    nz = int(sm.loc["heldout", "n_nonzero_rejection"])
    axes[2].set_title(f"Rejection rate (only {nz} held-out points have rejections;\nregular+cap is over-estimated)", fontsize=9)
    h = av[av.group == "heldout"].copy(); h["ae"] = h.p_cold_err_pp.abs()
    cats = list(h.category.unique())
    axes[1].boxplot([h[h.category == c].ae for c in cats], showmeans=True)
    axes[1].set_xticklabels([c.replace("_", "\n") for c in cats])
    axes[1].set_ylabel("Absolute cold-start error (percentage points)"); axes[1].tick_params(axis="x", labelsize=6.5)
    axes[1].set_title("Held-out error by category", fontsize=9)
    fig.suptitle("Fig 20. Analytical approximation vs simulation: in-sample and held-out results are reported separately", fontsize=10)
    save(fig, "fig20_analytical_vs_simulation")

# Fig 21: utilization - the legacy metric vs common metrics (E1 steady)
if "avg_concurrency" in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    sysl = ALL3 if "vm_autoscale" in systems_present else ("serverless", "vm")
    line(axes[0], E1, "steady", "utilization", systems=sysl, scale=100)
    axes[0].set_ylabel("Legacy utilization (%) - DIFFERENT denominators"); axes[0].set_title("executing / (alive instance time | slot time | ready slot time)", fontsize=8)
    line(axes[1], E1, "steady", "avg_concurrency", systems=sysl)
    axes[1].set_ylabel("Average requests executing at once"); axes[1].set_title("No capacity in the denominator: comparable", fontsize=8)
    line(axes[2], E1, "steady", "utilization_ref_capacity", systems=sysl, scale=100)
    axes[2].set_ylabel(f"Share of reference capacity ({C.REF_CAPACITY_SLOTS} slots) used (%)"); axes[2].set_title("Common denominator: comparable", fontsize=8)
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("Request arrival rate (requests/second, steady)")
    fig.suptitle("Fig 21. Utilization: system-specific metric (left) vs. directly comparable metrics (middle, right)", fontsize=10)
    save(fig, "fig21_utilization_definitions")

# Fig 22: what the autoscaler actually does (one run, seed 0)
if "vm_autoscale" in systems_present:
    from vm_autoscale import simulate_vm_autoscaling
    wkw = {"mean_rate": 15.0}
    reqs = W.variable(seed=0, **wkw)
    res = simulate_vm_autoscaling(reqs, W.horizon_of("variable"))
    tl = np.array([(t, a, b) for t, a, b in res.timeline] + [(res.horizon, res.timeline[-1][1], res.timeline[-1][2])])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.step(tl[:, 0], tl[:, 1], where="post", color=COL["vm_autoscale"], lw=2, label="VMs ready (serving)")
    ax.step(tl[:, 0], tl[:, 1] + tl[:, 2], where="post", color=COL["vm_autoscale"], lw=1, ls="--", label="ready + booting (billed)")
    ax.axhline(C.AUTOSCALE_MAX_VMS, color="grey", ls=":", lw=1); ax.axhline(C.AUTOSCALE_MIN_VMS, color="grey", ls=":", lw=1)
    ax.text(595, C.AUTOSCALE_MAX_VMS + 0.1, "max VMs", ha="right", fontsize=8, color="grey")
    ax.text(595, C.AUTOSCALE_MIN_VMS + 0.1, "min VMs", ha="right", fontsize=8, color="grey")
    ax.set_ylabel("Number of VMs"); ax.set_xlabel("Time (s)"); ax.set_ylim(0, C.AUTOSCALE_MAX_VMS + 1)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    a = np.array([r.arrival_time for r in reqs]); bins = np.arange(0, 610, 10)
    ax2.plot(bins[:-1] + 5, np.histogram(a, bins)[0] / 10, color="#555", lw=1, alpha=0.7); ax2.set_ylabel("Arrival rate (req/s)", color="#555")
    ax2.axhline(CAP, color="#ff7f0e", ls=":", lw=1)
    ax2.text(5, C.NUM_VMS * C.SLOTS_PER_VM / C.MEAN_EXEC_TIME_S + 0.4, f"fixed-VM capacity ({CAP:g} req/s)", color="#ff7f0e", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Fig 22. Autoscaler behaviour, variable workload (mean 15 req/s, seed 0): {res.n_scale_out} scale-outs, {res.n_scale_in} scale-ins, "
                 f"{C.AUTOSCALE_PROVISION_DELAY_S:.0f} s boot delay", fontsize=9)
    save(fig, "fig22_autoscaler_timeline")

# Fig 23: provisioning delay (E11)
E11 = "E11_autoscale_delay"
if E11 in have:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    cases = [("steady_25rps", "Steady 25 req/s"), ("variable_15rps", "Variable, mean 15 req/s"), ("bursty", "Bursty")]
    for ax, met, lab, sc in zip(axes, ("success_rate", "total_cost_usd", "p95_response_s"),
                                ("Success rate (%)", "Total cost per run (USD)", "P95 response, completed only (s)"), (100, 1, 1)):
        for wl, nm in cases:
            x, m, sd = agg(E11, wl, "vm_autoscale", met)
            l, = ax.plot(x, m * sc, marker="o", ms=4, label=f"Autoscaling: {nm}")
            xf, mf, _ = agg(E11, wl, "vm", met)
            ax.axhline(mf.mean() * sc, color=l.get_color(), ls=":", lw=1)
        ax.set_xlabel("VM provisioning delay (s)"); ax.set_ylabel(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fig 23. Autoscaling is only as good as it is fast (dotted = fixed-VM baseline for the same workload)", fontsize=10)
    save(fig, "fig23_autoscale_provisioning_delay")


# Fig 24: equal-capacity autoscaler comparison (E12)
E12 = "E12_equal_capacity"
if E12 in have:
    panels = [("success_rate", "Success rate (%)", 100), ("rejection_rate", "Rejection rate (%)", 100),
              ("sla_violation_rate", "SLA violations (% of offered)", 100),
              ("p95_response_s", "P95 response, completed only (s)", 1), ("total_cost_usd", "Total cost per run (USD)", 1)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    wl12 = list(W.GENERATORS)
    for ax, (met, lab, sc) in zip(axes.ravel(), panels):
        for i, sname in enumerate(ALL4):
            g = [df[(df.experiment == E12) & (df.workload == w) & (df.system == sname)][met] for w in wl12]
            ax.bar(np.arange(len(wl12)) + (i - 1.5) * 0.2, [x.mean() * sc for x in g], 0.2, yerr=[x.std() * sc for x in g],
                   capsize=2, color=COL[sname], label=LAB[sname])
        ax.set_xticks(range(len(wl12))); ax.set_xticklabels([w.replace("_", "\n") for w in wl12]); ax.set_ylabel(lab, fontsize=9)
    ax = axes[1, 2]; ax.axis("off")
    ax.legend(*axes[0, 0].get_legend_handles_labels(), loc="center", fontsize=9, frameon=False)
    ax.text(0.0, 0.08, f"Same {C.AUTOSCALE_PROVISION_DELAY_S:.0f} s boot delay and thresholds in both autoscalers.\n"
            f"Min {C.AUTOSCALE_MIN_VMS} starts with half the fixed pool's capacity; min {C.NUM_VMS} starts equal.", transform=ax.transAxes, fontsize=8)
    fig.suptitle("Fig 24. Equal-initial-capacity autoscaler comparison (E12, mean +- SD over seeds)", fontsize=10)
    save(fig, "fig24_equal_capacity_autoscaler")

# Fig 25: regular arrivals - cold probability vs idle timeout is a STEP at the arrival gap (C1 correction)
if True:
    import analytical as AN
    from serverless import simulate_serverless
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, rate in zip(axes, (0.2, 0.1)):
        gap = 1.0 / rate
        Ts = np.arange(0.5 * gap, 2.5 * gap + 1e-9, 0.25 * gap / 2)
        sim = [np.mean([(lambda r: r.n_cold / max(r.n_completed, 1))(simulate_serverless(W.steady(seed=s, rate=rate), 600.0, idle_timeout=T))
                        for s in C.SEEDS]) * 100 for T in Ts]
        mod = [AN.estimate_for_workload("steady", rate=rate, idle_timeout=T)["p_cold"] * 100 for T in Ts]
        ax.plot(Ts, sim, "o-", ms=4, color=COL["serverless"], label="Simulation (mean of seeds)")
        ax.plot(Ts, mod, "x--", ms=6, color="#d62728", label="Analytical (regular-arrival model)")
        ax.axvspan(gap * (1 - C.STEADY_JITTER), gap * (1 + C.STEADY_JITTER), color="grey", alpha=0.15, label=f"gap +-{C.STEADY_JITTER:.0%} (jitter)")
        ax.set_xlabel("Idle timeout T (s)"); ax.set_ylabel("Cold-start probability (%)")
        ax.set_title(f"Evenly spaced arrivals, gap = {gap:g} s ({rate:g} req/s)", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fig 25. Regular arrivals: warm if the gap fits inside the timeout, cold otherwise (development sweep, not held-out)", fontsize=10)
    save(fig, "fig25_regular_gap_step")
