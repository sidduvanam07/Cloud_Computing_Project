"""
tests_corrections.py - Tests added with the audit corrections (C1-C4, minor items).

Run:  python tests_corrections.py          (about 1 minute)
      python tests_corrections.py --fast   (skips the experiment-pipeline tests)
tests_validation.py (original) and tests_new_features.py are NOT modified and still run separately.

Style: expected values are hand arithmetic from stated formulas; the simulator is used as an
ORACLE in tests, never as an input to the analytical model.

Hand calculations used below (evenly spaced arrivals, jitter = 0, spacing D = 1/rate,
execution X ~ Exp(E), cold start s; the next use is cold iff gap - s_prev - X > T):
   c_w = 1 - exp(-(D - T)^+ / E)            previous use warm
   c_c = 1 - exp(-(D - T - s)^+ / E)        previous use cold
   kappa = c_c - c_w ;  pi = c_w / (1 - kappa) ;  expected colds over M uses (first cold) =
   M*pi + (1 - pi)(1 - kappa^M)/(1 - kappa)
"""
import math
import sys
import numpy as np
import pandas as pd
import config as C
import workload as W
import analytical as AN
import metrics
import validation as V
from workload import Request
from serverless import simulate_serverless
from vm import simulate_vm
from vm_autoscale import simulate_vm_autoscaling as AUTO
from common import SimResult

FAST = "--fast" in sys.argv
PASS = FAIL = 0


def ok(name, cond, detail=""):
    global PASS, FAIL
    PASS += bool(cond); FAIL += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def near(name, got, expected, rel=1e-9, abs_=1e-12):
    ok(name, math.isclose(got, expected, rel_tol=rel, abs_tol=abs_), f"got={got:.6g} expected={expected:.6g}")


def reqs(arrivals, exec_time):
    return [Request(i, float(a), exec_time) for i, a in enumerate(arrivals)]


def regular(rate, T, E=0.2, s=0.5, jitter=0.0, L=600.0, cap=None):
    """Analytical estimate for ONE evenly spaced phase (the estimator is used directly, no simulator input)."""
    return AN.estimate_cold_starts([(0.0, L, rate)], "regular", mean_exec=E, cold_start=s, idle_timeout=T,
                                   max_concurrency=cap, jitter=jitter)


def hand_regular_p(D, T, E, s, M, jitter=0.0):
    """Independent hand/numeric evaluation of the chain formula (no code shared with analytical.py):
    c(theta) = mean over the gap ~ U(D(1-j), D(1+j)) of 1 - exp(-(gap - theta)^+ / E)."""
    gaps = np.linspace(D * (1 - jitter), D * (1 + jitter), 200001) if jitter else np.array([D])
    c = lambda th: float(np.mean(1 - np.exp(-np.maximum(gaps - th, 0) / E)))
    cw, cc = c(T), c(T + s)
    kap = cc - cw
    pi = cw / (1 - kap)
    return (M * pi + (1 - pi) * (1 - kap ** M) / (1 - kap)) / M


def sim_cold(gen, seeds=range(20), **kw):
    """Mean simulated cold-start probability and mean cold-start COUNT."""
    sk = {k: kw.pop(k) for k in ("idle_timeout", "cold_start", "max_concurrency") if k in kw}
    rs = [simulate_serverless(W.GENERATORS[gen](seed=s, **kw), W.horizon_of(gen, **{k: v for k, v in kw.items() if k != "mean_exec"}), **sk)
          for s in seeds]
    return np.mean([r.n_cold / max(r.n_completed, 1) for r in rs]), np.mean([r.n_cold for r in rs])


# =====================================================================================
print("=== C1. Regular-arrival analytical model: gap vs idle timeout ===")
D = 5.0                                              # rate 0.2 /s -> 120 arrivals in 600 s
e = regular(0.2, T=10.0)
ok("gap (5 s) < timeout (10 s): only the very first request is cold (expected cold starts ~ 1)",
   0.99 <= e["expected_cold_starts"] <= 1.02, f"{e['expected_cold_starts']:.4f}")
near("   ... p_cold = 1/120", e["p_cold"], 1 / 120, rel=0.02)
e = regular(0.2, T=5.0)
ok("gap == timeout: still WARM (simulator destroys only if idle > T, strictly)",
   0.99 <= e["expected_cold_starts"] <= 1.02, f"{e['expected_cold_starts']:.4f}")
e = regular(0.2, T=3.0)
ok("gap (5 s) > timeout (3 s) by 10 execution times: every request cold", e["expected_cold_starts"] >= 119.5,
   f"{e['expected_cold_starts']:.2f}")
near("gap 5 s, T = 4.5 s, E = 0.2: hand-calculated chain value 0.4809 (c_w = 0.9179, c_c = 0)",
     regular(0.2, 4.5, E=0.2)["p_cold"], hand_regular_p(5.0, 4.5, 0.2, 0.5, 120), rel=0.01)
near("   ... same gap, E = 0.5 (longer execution leaves less idle time): hand value 0.3904",
     regular(0.2, 4.5, E=0.5)["p_cold"], hand_regular_p(5.0, 4.5, 0.5, 0.5, 120), rel=0.015)
ok("longer execution durations lower the cold probability near the boundary (E 0.1 > 0.2 > 0.5 > 1.0)",
   regular(0.2, 4.7, E=0.1)["p_cold"] > regular(0.2, 4.7, E=0.2)["p_cold"] > regular(0.2, 4.7, E=0.5)["p_cold"]
   > regular(0.2, 4.7, E=1.0)["p_cold"])
ts = [2.5, 3.5, 4.5, 4.9, 5.0, 5.1, 5.5, 6.0, 8.0, 12.0]
ps = [regular(0.2, t)["p_cold"] for t in ts]
ok("cold probability never increases with the timeout", all(a >= b - 1e-12 for a, b in zip(ps, ps[1:])))
near("T = 0.8 gap (4.0 s): hand chain c_w = 0.9933, c_c = 0.9179 -> p = 0.9238 (almost all cold, NOT the old 28 %)",
     regular(0.2, 4.0)["p_cold"], hand_regular_p(5.0, 4.0, 0.2, 0.5, 120), rel=0.01)
ok("step-like, not exponential: p_cold(T = 0.8 gap) > 0.9 and p_cold(T = 1.2 gap) < 0.02",
   regular(0.2, 4.0)["p_cold"] > 0.9 and regular(0.2, 6.0)["p_cold"] < 0.02)
print("-- spacing jitter (+-10 %: gaps 4.5 ... 5.5 s)")
near("T below the whole jitter band (4.3 s): independent numeric chain value (~0.65: part of the band is shorter than s)",
     regular(0.2, 4.3, jitter=0.1)["p_cold"], hand_regular_p(5.0, 4.3, 0.2, 0.5, 120, jitter=0.1), rel=0.015)
ok("T above the whole jitter band (6.0 s): only the first request is cold",
   regular(0.2, 6.0, jitter=0.1)["expected_cold_starts"] < 1.05)
near("T at the lower edge of the band (4.5 s, jitter 0.1, gaps uniform 4.5-5.5 s): independent numeric chain value",
     regular(0.2, 4.5, jitter=0.1)["p_cold"], hand_regular_p(5.0, 4.5, 0.2, 0.5, 120, jitter=0.1), rel=0.01)
near("T in the middle of the band (5.0 s)", regular(0.2, 5.0, jitter=0.1)["p_cold"],
     hand_regular_p(5.0, 5.0, 0.2, 0.5, 120, jitter=0.1), rel=0.02)
print("-- sparse regular arrivals")
e = regular(0.05, T=60.0)                           # one request every 20 s, timeout 60 s
ok("0.05 req/s, T = 60 s: 1 cold start in 30 requests (3.3 %)", 0.99 <= e["expected_cold_starts"] <= 1.02
   and abs(e["p_cold"] - 1 / 30) < 0.001, f"{e['p_cold']:.4f}")
ok("0.05 req/s, T = 15 s (< gap): every request cold", regular(0.05, 15.0)["p_cold"] > 0.99)
print("-- the simulator as an oracle (20 seeds; NOT an input to the model)")
for rate, T, tol in [(0.2, 10, 0.5), (0.2, 6, 0.5), (0.2, 3, 0.5), (0.2, 4.5, 5.0), (0.1, 5, 0.5), (0.1, 20, 0.5), (0.05, 60, 0.5)]:
    sp, _ = sim_cold("steady", rate=rate, idle_timeout=T)
    mp = AN.estimate_for_workload("steady", rate=rate, idle_timeout=T)["p_cold"]
    ok(f"rate {rate}/s, T = {T}: simulation {sp*100:6.2f} %  model {mp*100:6.2f} %  (|error| <= {tol} pp)",
       abs(mp - sp) * 100 <= tol, f"error {abs(mp-sp)*100:.2f} pp")
print("-- different execution durations vs simulator")
for E in (0.1, 0.5, 1.0):
    sp, _ = sim_cold("steady", rate=0.2, idle_timeout=4.5, mean_exec=E)
    mp = AN.estimate_for_workload("steady", rate=0.2, idle_timeout=4.5, mean_exec=E)["p_cold"]
    ok(f"gap 5 s, T 4.5 s, E = {E}: simulation {sp*100:5.1f} %  model {mp*100:5.1f} %", abs(mp - sp) * 100 <= 5.0)
print("-- regular arrivals with concurrency > 1")
sp, sc = sim_cold("steady", rate=10.0, mean_exec=0.5, idle_timeout=600)
m = AN.estimate_for_workload("steady", rate=10.0, mean_exec=0.5, idle_timeout=600)
ok("10 req/s, E = 0.5 (about 5 busy instances), T = 600: cold starts = instances created; within 15 %",
   abs(m["expected_cold_starts"] / sc - 1) <= 0.15, f"sim {sc:.1f} model {m['expected_cold_starts']:.1f}")
ok("   ... at least lambda*s = 5 cold starts (ramp-up) and more than the mean concurrency 5", m["expected_cold_starts"] >= 5.0 + 5.0)
sp, sc = sim_cold("steady", rate=25.0, idle_timeout=1.0)
m = AN.estimate_for_workload("steady", rate=25.0, idle_timeout=1.0)
ok("25 req/s, T = 1 s (concurrency ~ 5): within 15 %", abs(m["expected_cold_starts"] / sc - 1) <= 0.15,
   f"sim {sc:.1f} model {m['expected_cold_starts']:.1f}")
sp, sc = sim_cold("steady", rate=10.0, mean_exec=0.5, idle_timeout=1.0)
m = AN.estimate_for_workload("steady", rate=10.0, mean_exec=0.5, idle_timeout=1.0)
ok("KNOWN LIMITATION guard: 10 req/s, E = 0.5, T = 1 s: model over-estimates (measured +49 %); must stay within 60 %",
   0 < m["expected_cold_starts"] / sc - 1 <= 0.60, f"ratio {m['expected_cold_starts']/sc:.2f}")
print("-- Poisson branch untouched, validation")
before = AN.estimate_for_workload("bursty")
ok("Poisson pattern ignores the jitter argument", AN.estimate_cold_starts(*W.profile_of("bursty"), jitter=0.4)["p_cold"] == before["p_cold"])
for bad in (-0.1, 1.0):
    try:
        regular(0.2, 5.0, jitter=bad); ok(f"jitter {bad} rejected", False)
    except ValueError:
        ok(f"jitter {bad} rejected", True)

# =====================================================================================
print("\n=== Analytical model: ramp-up rule, low-probability cases, each pattern independently ===")
# hand argument: with an empty pool, every arrival in the first s seconds is cold  =>  >= lambda*s cold starts
e = AN.estimate_cold_starts([(0.0, 5.0, 60.0)], "poisson", cold_start=2.0, idle_timeout=600.0)
ok("burst of 60 req/s, s = 2 s: at least lambda*s = 120 expected cold starts (ramp-up rule)",
   e["expected_cold_starts"] >= 120.0, f"{e['expected_cold_starts']:.1f}")
sp, sc = sim_cold("bursty", num_bursts=1, burst_size=300, base_rate=0.05, cold_start=2.0)
m = AN.estimate_for_workload("bursty", num_bursts=1, burst_size=300, base_rate=0.05, cold_start=2.0)
ok("single 300-request burst, s = 2 s: model within 10 % of the simulated count", abs(m["expected_cold_starts"] / sc - 1) <= 0.10,
   f"sim {sc:.1f} model {m['expected_cold_starts']:.1f}")
e = AN.estimate_cold_starts([(0.0, 5.0, 25.0)], "regular", cold_start=0.5, idle_timeout=600.0)
ok("regular 25 req/s, s = 0.5: at least lambda*s = 12 cold starts", e["expected_cold_starts"] >= 12.0, f"{e['expected_cold_starts']:.1f}")
e = AN.estimate_cold_starts([(0.0, 5.0, 25.0)], "regular", cold_start=0.5, idle_timeout=600.0)
ok("regular ramp: forced levels are used with certainty (regression test for a q-weighting bug): colds in [12, 12.6]",
   12.0 <= e["expected_cold_starts"] <= 12.6, f"{e['expected_cold_starts']:.2f}")
ok("ramp-up grows with the cold-start time (s = 0.5 -> 2.0)",
   AN.estimate_cold_starts([(0, 5, 60.0)], "poisson", cold_start=2.0)["expected_cold_starts"]
   > AN.estimate_cold_starts([(0, 5, 60.0)], "poisson", cold_start=0.5)["expected_cold_starts"] + 50)
print("-- relative error at LOW cold-start probabilities (an absolute tolerance would hide a model returning ~0)")
for wl in W.GENERATORS:
    sp, _ = sim_cold(wl)
    mp = AN.estimate_for_workload(wl)["p_cold"]
    ok(f"{wl:14s} default: simulation {sp*100:6.3f} %  model {mp*100:6.3f} %  relative error {100*(mp/sp-1):+6.1f} % (|.| <= 30 %)",
       abs(mp / sp - 1) <= 0.30)
ok("a degenerate 'return 0' model would fail this: model p_cold for steady > 0.05 %", AN.estimate_for_workload("steady")["p_cold"] > 0.0005)

# =====================================================================================
print("\n=== C2/autoscaler: cooldown, booting guard, limits, provisioning delay ===")


def launch_times(res):
    out, prev = [], res.timeline[0]
    for t, ready, boot in res.timeline[1:]:
        if boot > prev[2]:
            out.append(t)
        prev = (t, ready, boot)
    return out


over = reqs(np.arange(0, 100, 0.05), 1.0)           # 20 req/s of 1 s jobs: permanent overload of 1 VM (2 slots)
common = dict(min_vms=1, max_vms=8, provision_delay=60.0, queue_limit=10 ** 6, eval_interval=10.0)
r = AUTO(over, 100.0, out_cooldown=30.0, **common)
ok("cooldown 30 s, tick 10 s: scale-outs at exactly t = 10, 40, 70, 100", launch_times(r)[:4] == [10.0, 40.0, 70.0, 100.0],
   str(launch_times(r)[:5]))
r0 = AUTO(over, 100.0, out_cooldown=0.0, **common)
ok("cooldown 0: a scale-out at every tick (t = 10, 20, 30, 40, 50)", launch_times(r0)[:5] == [10.0, 20.0, 30.0, 40.0, 50.0],
   str(launch_times(r0)[:6]))
ok("with the cooldown, no two scale-outs are closer than 30 s",
   all(b - a >= 30.0 - 1e-9 for a, b in zip(launch_times(r), launch_times(r)[1:])))

print("-- scale-out step size is capped by max_vms")
def launch_sizes(res):
    out, prev = [], res.timeline[0]
    for tt, ready, boot in res.timeline[1:]:
        if boot > prev[2]:
            out.append((tt, boot - prev[2]))
        prev = (tt, ready, boot)
    return out
r = AUTO(over, 100.0, min_vms=1, max_vms=5, step=3, provision_delay=60.0, eval_interval=10.0, out_cooldown=30.0, queue_limit=10 ** 6)
ok("step 3, max 5: first scale-out adds 3 VMs (t = 10), the second only 1 (t = 40) because max_vms = 5 caps it",
   launch_sizes(r)[:2] == [(10.0, 3), (40.0, 1)], str(launch_sizes(r)[:3]))
ok("   ... never more than 5 VMs (running + booting) and no further scale-outs", max(a + b for _, a, b in r.timeline) == 5 and r.n_scale_out == 4,
   f"max {max(a + b for _, a, b in r.timeline)}, scale-outs {r.n_scale_out}")

print("-- scale-in must not remove capacity while a VM is still booting")
w = reqs(np.arange(0, 45, 1 / 3), 1.0)              # 3 req/s of 1 s jobs for 45 s, then nothing
r = AUTO(w, 400.0, min_vms=1, max_vms=3, provision_delay=40.0, eval_interval=10.0, out_cooldown=30.0,
         scale_in_after=0.0, queue_limit=10 ** 6)
si = [(t1, a1, b1) for (t0, a0, b0), (t1, a1, b1) in zip(r.timeline, r.timeline[1:]) if a1 < a0]
ok("scenario works: scaled out twice (launches at 10 and 40, ready at 50 and 80) and later scaled in",
   launch_times(r)[:2] == [10.0, 40.0] and len(si) > 0, f"launches {launch_times(r)[:3]}, scale-ins {si[:2]}")
ok("every scale-in happened with NO VM booting", all(b == 0 for _, _, b in si), str(si[:3]))
ok("the first scale-in is not before the last boot finished (t = 80)", min(t for t, _, _ in si) >= 80.0, f"t={min(t for t,_,_ in si)}")
print("-- scale-in respects the quiet period and the minimum")
w = reqs(np.arange(0, 60, 0.05), 1.0)
r = AUTO(w, 900.0, min_vms=2, max_vms=6, provision_delay=10.0, scale_in_after=100.0)
si_t = [t1 for (t0, a0, b0), (t1, a1, b1) in zip(r.timeline, r.timeline[1:]) if a1 < a0]
ok("scaled in only after >= 100 s of low utilization (load ends at t = 60)", len(si_t) > 0 and min(si_t) >= 160.0, f"first {min(si_t, default=None)}")
ok("successive scale-ins are >= 100 s apart", all(b - a >= 100.0 - 1e-9 for a, b in zip(si_t, si_t[1:])))
ok("ends at exactly min_vms", r.timeline[-1][1] == 2)
# NOTE: the autoscaler's 'queue is empty' scale-in guard is redundant by construction: a request only waits when no
# slot is free, so an idle VM (the only kind that can be removed) implies an empty queue. Removing the guard is an
# EQUIVALENT mutant and cannot be detected by any test; the observable invariants are tested instead:
rng = np.random.default_rng(7)
bad_lim = bad_acc = 0
for trial in range(30):
    gen = ("steady", "variable", "bursty", "idle_to_burst")[trial % 4]
    mn = int(rng.integers(1, 4)); mx = mn + int(rng.integers(0, 5))
    kw = dict(min_vms=mn, max_vms=mx, provision_delay=float(rng.choice([0, 15, 60])), eval_interval=float(rng.choice([5, 10, 20])),
              out_cooldown=float(rng.choice([0, 30])), scale_in_after=float(rng.choice([0, 60, 120])),
              queue_limit=int(rng.choice([10, 100])))
    wl = W.GENERATORS[gen](seed=trial) if gen != "steady" else W.steady(seed=trial, rate=float(rng.choice([5, 25, 40])))
    res = AUTO(wl, W.horizon_of(gen), **kw)
    bad_lim += not all(a >= mn and a + b <= mx and a >= 1 for _, a, b in res.timeline) or res.peak_instances > mx
    bad_acc += (res.n_completed + res.n_rejected != res.n_total) or len(res.queue) != res.n_completed
ok("30 random configurations: ready >= min_vms and ready + booting <= max_vms at every timeline point", bad_lim == 0, f"violations {bad_lim}")
ok("30 random configurations: completed + rejected = offered, nothing lost", bad_acc == 0, f"violations {bad_acc}")

print("-- provisioning delay really delays capacity (hand-counted request starts)")
burst = reqs([0.0] * 2000, 1.0)                     # 2000 one-second jobs all at t = 0; 1 VM = 2 slots
starts = lambda res: np.asarray(res.queue)          # all arrive at t = 0, so start time = queue wait
r = AUTO(burst, 100.0, min_vms=1, max_vms=2, provision_delay=60.0, queue_limit=10 ** 6)
near("delay 60 s: VM#2 launched at t = 10, usable at t = 70. Starts before t = 80: 2/s x 70 s + 4/s x 10 s = 180",
     float(np.sum(starts(r) < 80.0)), 180.0)
near("   ... and before the VM is usable (t < 70) only 2 slots work: 140 starts", float(np.sum(starts(r) < 70.0)), 140.0)
r = AUTO(burst, 100.0, min_vms=1, max_vms=2, provision_delay=0.0, queue_limit=10 ** 6)
near("delay 0: VM#2 usable at t = 10. Starts before t = 20: 2/s x 10 s + 4/s x 10 s = 60", float(np.sum(starts(r) < 20.0)), 60.0)
r = AUTO(burst, 100.0, min_vms=1, max_vms=2, provision_delay=200.0, queue_limit=10 ** 6)
near("delay 200 s: still one VM at t = 100 (starts before 100 = 200)", float(np.sum(starts(r) < 100.0)), 200.0)

print("-- flapping indicator")
ok("scale_cycles = min(scale-outs, scale-ins)", metrics.summarize(SimResult(system="vm_autoscale", n_total=0, n_scale_out=3, n_scale_in=1))["scale_cycles"] == 1)
cyc = [metrics.summarize(AUTO(W.steady(seed=s), 600.0))["scale_cycles"] for s in range(5)]
ok("calm 5 req/s steady workload: the default thresholds make the autoscaler add then drop a VM in some runs", max(cyc) >= 1, str(cyc))
ok("fixed VM and serverless report 0 scale cycles", metrics.summarize(simulate_vm(W.steady(seed=0), 600.0))["scale_cycles"] == 0)

# =====================================================================================
print("\n=== Utilization: reference slots and ready vs paid capacity ===")
v = simulate_vm(reqs([0.0, 0.0], 1.0), 1.0, slots=1)            # 2 VMs x 1 slot, both busy for 1 s
near("default reference (4 slots) would say 50 % for a fully busy 2-slot pool", metrics.summarize(v)["utilization_ref_capacity"], 0.5)
near("with the pool's real slots (ref_slots = 2) it says 100 %", metrics.summarize(v, ref_slots=2)["utilization_ref_capacity"], 1.0)
a = AUTO(reqs([0.0] * 4, 1.0), 100.0, min_vms=1, max_vms=1, min_billed_s=0)   # 1 VM, 2 slots, 100 s window, 4 s of work x 1 slot
m = metrics.summarize(a)
near("autoscaler, no boot: paid utilization = ready utilization = 4 / (2 x 100) = 2 %", m["utilization_paid_capacity"], 0.02)
over2 = reqs(np.arange(0, 100, 0.05), 1.0)
m = metrics.summarize(AUTO(over2, 100.0, min_vms=1, max_vms=2, provision_delay=60.0, queue_limit=10 ** 6))
ok("with a booting VM billed from launch, paid utilization < ready utilization", m["utilization_paid_capacity"] < m["utilization"] - 1e-6,
   f"paid {m['utilization_paid_capacity']:.4f} ready {m['utilization']:.4f}")
mv = metrics.summarize(simulate_vm(W.steady(seed=0), 600.0))
near("fixed VM: paid = ready = reference utilization", mv["utilization_paid_capacity"], mv["utilization"])
ok("serverless: paid utilization is NaN (billed per execution, not for capacity)",
   math.isnan(metrics.summarize(simulate_serverless(W.steady(seed=0), 600.0))["utilization_paid_capacity"]))

# =====================================================================================
print("\n=== Validation module: separated datasets and honest statistics ===")
h1, h2 = V.heldout_configs(), V.heldout_configs()
ok("held-out configurations are deterministic (fixed sampler seed)", h1 == h2)
ok("42 held-out configurations with unique ids", len(h1) == 42 and len({c["id"] for c in h1}) == 42)
cats = {c["category"] for c in h1}
ok("held-out strata cover sparse regular, medium bursty, varied timeout/exec time and concurrency limits",
   {"regular_sparse", "regular_dense", "variable", "bursty_medium", "idle_to_burst", "capped_bursty", "capped_regular"} <= cats)
ok("held-out simulation seeds are disjoint from the experiment seeds, the audit seeds and held-out v1",
   set(V.HELDOUT_SIM_SEEDS).isdisjoint(set(C.SEEDS) | set(V.AUDIT_SIM_SEEDS) | set(V.HELDOUT_V1_SIM_SEEDS)))
ok("held-out v2 and v1 use different sampler seeds and different configurations",
   V.HELDOUT_RNG_SEED != V.HELDOUT_V1_RNG_SEED and V.heldout_configs(version=1) != h1)
ok("v1 is labelled as development data (group heldout_v1); v2 is the current held-out group",
   {c["group"] for c in V.heldout_configs(version=1)} == {"heldout_v1"} and {c["group"] for c in h1} == {"heldout"})
key = lambda c: (c["gen"], tuple(sorted(c["wkw"].items())), c["idle_timeout"], c["cold_start"], c["max_concurrency"], c["mean_exec"])
seen = {key(c) for c in V.audit_configs() + V.c1_dev_configs() + V.heldout_configs(version=1)}
ok("no held-out v2 configuration duplicates an audit-seen, C1-development or v1 configuration", not any(key(c) in seen for c in h1))
ok("held-out set varies timeouts (> 10 distinct), execution times (> 10 distinct) and caps (>= 5 distinct)",
   len({c["idle_timeout"] for c in h1}) > 10 and len({c["mean_exec"] for c in h1}) > 10
   and len({c["max_concurrency"] for c in h1 if c["max_concurrency"]}) >= 5)
ok("audit set has 12 configurations", len(V.audit_configs()) == 12)
syn = pd.DataFrame({"sim_p_cold": [0.5, 0.2, 0.0001, 0.0], "model_p_cold": [0.45, 0.3, 0.0003, 0.0],
                    "sim_rejection_rate": [0.0, 0.1, 0.0, 0.0], "model_rejection_rate": [0.0, 0.0, 0.0, 0.0],
                    "cost_rel_err": [0.1, -0.1, 0.0, 0.0]})
syn["p_cold_err_pp"] = 100 * (syn.model_p_cold - syn.sim_p_cold); syn["rejection_err_pp"] = 100 * (syn.model_rejection_rate - syn.sim_rejection_rate)
s = V.accuracy_summary(syn)
near("MAE of errors (-5, +10, +0.02, 0) pp = 3.755", s["p_cold_mae_pp"], (5 + 10 + 0.02 + 0) / 4)
near("median absolute error = (0.02 + 5)/2 = 2.51", s["p_cold_median_ae_pp"], 2.51)
near("maximum absolute error = 10", s["p_cold_max_ae_pp"], 10.0)
ok("near-zero simulated probabilities (0.01 %, 0 %) are EXCLUDED from relative error and counted separately", s["n_rel"] == 2 and s["n_low_p"] == 2)
near("relative error uses only the two points >= 1 %: median |(-10 %, +50 %)| = 30 %", s["rel_median_abs_pct"], 30.0)
ok("rejection statistics: 1 nonzero point, its error is 10 pp", s["n_nonzero_rejection"] == 1 and abs(s["rej_max_pp_nonzero"] - 10.0) < 1e-9)

# =====================================================================================
print("\n=== Documentation / wording checks ===")
readme = open("README.md").read()
ok("README does not call anything 'textbook'", "textbook" not in readme.lower())
ok("analytical.py does not call anything 'textbook'", "textbook" not in open("analytical.py").read().lower())
ok("README states that adaptive warm-instance retention is NOT implemented", "adaptive warm-instance retention" in readme.lower() and "not implemented" in readme.lower())
ok("README states that lifecycle-aware scheduling is NOT implemented", "lifecycle-aware scheduling" in readme.lower())
ok("README describes the fixed idle timeout and most-recently-used reuse", "most-recently-used" in readme.lower() and "fixed idle timeout" in readme.lower())
ok("README does not keep the incorrect rejection claim", "rejection-rate error <= 0.43" not in readme and "<= 0.43" not in readme)
ok("README reports the true maximum rejection error (1.78 pp in-sample)", "1.78" in readme)
src = open("plots.py").read()
ok("plots.py has no hard-coded '20 req/s', '4 slots' or 5.0 req/s labels", "20 req/s" not in src and "(4 slots" not in src and "axhline(5.0" not in src)

# =====================================================================================
if not FAST:
    import run_experiments as RE
    print("\n=== Experiments: E12 equal capacity; E7/E9 reference slots ===")
    saved = C.SEEDS
    C.SEEDS = [0, 1]
    try:
        RE.INCLUDE_AUTOSCALE = True
        d = pd.DataFrame(RE.e12_equal_capacity())
        ok("E12 has the four systems on all four workloads",
           set(d.system) == {"serverless", "vm", "vm_autoscale", "vm_autoscale_min2"} and set(d.workload) == set(W.GENERATORS))
        ok("E12 keeps the min-1 autoscaler AND adds min-2 with identical other parameters",
           (d.groupby(["workload", "seed"]).system.nunique() == 4).all())
        m = d.groupby(["workload", "system"])[["success_rate", "rejection_rate", "p95_response_s", "total_cost_usd", "sla_violation_rate"]].mean()
        ok("E12 reports success, rejection, p95, cost and SLA violations", m.notna().all().all())
        ok("equal-capacity autoscaler equals the fixed 2-VM pool on idle-to-burst (the 60 s boot never helps a 20 s burst)",
           abs(m.loc[("idle_to_burst", "vm_autoscale_min2"), "success_rate"] - m.loc[("idle_to_burst", "vm"), "success_rate"]) < 1e-9)
        ok("equal-capacity autoscaler is never less reliable than the fixed pool (same start, only adds capacity)",
           all(m.loc[(w, "vm_autoscale_min2"), "success_rate"] >= m.loc[(w, "vm"), "success_rate"] - 1e-9 for w in W.GENERATORS))
        ok("equal-capacity autoscaler never costs less than the fixed pool", all(
            m.loc[(w, "vm_autoscale_min2"), "total_cost_usd"] >= m.loc[(w, "vm"), "total_cost_usd"] - 1e-9 for w in W.GENERATORS))
        summ = RE.equal_capacity_summary(d)
        ok("E12 summary has mean/std and a flapping fraction per workload and system", len(summ) == 16 and "flapping_fraction" in summ)
        d7 = pd.DataFrame(RE.e7_vm_count())
        ok("E7: fixed-VM utilization equals the reference-capacity metric for every VM count (reference = real pool size)",
           np.allclose(d7.utilization, d7.utilization_ref_capacity))
        d9 = pd.DataFrame(RE.e9_memory())
        v9 = d9[d9.system == "vm"]
        ok("E9: fixed-VM utilization equals the reference-capacity metric at every memory size, incl. 4096 MB (1 slot/VM)",
           np.allclose(v9.utilization, v9.utilization_ref_capacity))
        s9 = d9[(d9.system == "serverless") & (d9.value == 4096)]
        ok("E9 at 4096 MB: serverless rows use the same 2-slot reference (ref utilization x 2 = average concurrency)",
           np.allclose(s9.utilization_ref_capacity * 2, s9.avg_concurrency))
        RE.INCLUDE_AUTOSCALE = False
        d2 = pd.DataFrame(RE.e2_workloads())
        ok("--no-autoscale still produces only the original two systems", set(d2.system) == {"serverless", "vm"})
        RE.INCLUDE_AUTOSCALE = True
    finally:
        C.SEEDS = saved

print(f"\nRESULT: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
