"""
tests_new_features.py - Tests for everything added in the improvement pass.

Run:  python tests_new_features.py          (about 1 minute; includes an end-to-end run)
      python tests_new_features.py --fast   (skips the end-to-end run of run_experiments.py + plots.py)
The ORIGINAL tests stay in tests_validation.py and are not modified.

Same style as tests_validation.py: tiny hand-made workloads, expected values written
as plain arithmetic from config constants, not copied from the simulator.
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import config as C
import workload as W
import cost as K
import metrics
import analytical as AN
import run_experiments as RE
from workload import Request, steady, bursty, variable, idle_to_burst
from serverless import simulate_serverless
from vm import simulate_vm, effective_slots
from vm_autoscale import simulate_vm_autoscaling as AUTO

FAST = "--fast" in sys.argv
PASS = FAIL = 0


def check(name, got, expected, tol=1e-9):
    global PASS, FAIL
    ok = math.isclose(got, expected, rel_tol=tol, abs_tol=1e-12)
    PASS += ok; FAIL += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got:.8g}  expected={expected:.8g}")


def ok(name, cond, detail=""):
    global PASS, FAIL
    PASS += bool(cond); FAIL += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def raises(name, fn, exc=ValueError):
    try:
        fn()
    except exc:
        ok(name, True)
    except Exception as e:                                   # wrong exception type
        ok(name, False, f"raised {type(e).__name__}")
    else:
        ok(name, False, "no exception")


def reqs(arrivals, exec_time):
    return [Request(i, a, exec_time) for i, a in enumerate(arrivals)]


P_REQ, P_GBS, MEM, VMP = C.LAMBDA_PRICE_PER_REQUEST, C.LAMBDA_PRICE_PER_GB_SECOND, C.FUNCTION_MEMORY_GB, C.VM_PRICE_PER_HOUR

# =====================================================================================
print("=== B. Metrics: totals, success/rejection, SLA, cost per successful request ===")
# 200 one-second jobs at t=0 on the default fixed VM (4 slots, queue 100): 104 accepted, 96 rejected.
v = simulate_vm(reqs([0.0] * 200, 1.0), horizon=1.0)
m = metrics.summarize(v)
check("n_total", m["n_total"], 200)
check("n_completed (4 running + 100 queued)", m["n_completed"], 104)
check("n_rejected", m["n_rejected"], 96)
check("success_rate = 104/200", m["success_rate"], 104 / 200)
check("rejection_rate = 96/200", m["rejection_rate"], 96 / 200)
check("success + rejection = 1", m["success_rate"] + m["rejection_rate"], 1.0)
# responses are 1,2,...,26 s, four requests each. Only the four 1.0 s ones meet SLA=1.0 s (<=), 100 are slower.
check("SLA violations = 100 slow + 96 rejected", m["sla_violations"], 196)
check("sla_violation_rate over OFFERED requests = 196/200", m["sla_violation_rate"], 196 / 200)
check("sla_violation_rate_completed = 100/104", m["sla_violation_rate_completed"], 100 / 104)
vm_total = 2 * VMP * (v.horizon / 3600)
check("VM total cost = 2 VMs x price x window", m["total_cost_usd"], vm_total)
check("cost per successful request = cost / 104", m["cost_per_successful_request_usd"], vm_total / 104)
check("legacy cost_per_request_usd unchanged (same value)", m["cost_per_request_usd"], vm_total / 104)
ok("latency percentiles still describe COMPLETED requests only", math.isclose(m["avg_response_s"], 13.5))
m2 = metrics.summarize(v, sla_s=30.0)
check("SLA threshold is configurable (30 s -> only rejections violate)", m2["sla_violation_rate"], 96 / 200)

# serverless: throttled requests are rejected, not billed, and not hidden
s = simulate_serverless(reqs([0.0] * 5, 1.0), horizon=1.0, max_concurrency=2)
ms = metrics.summarize(s)
check("serverless cap 2, 5 simultaneous: completed", ms["n_completed"], 2)
check("serverless rejected", ms["n_rejected"], 3)
check("serverless success_rate", ms["success_rate"], 0.4)
check("cost counts ONLY completed requests", ms["total_cost_usd"],
      2 * P_REQ + 2 * (1.0 + C.COLD_START_S) * MEM * P_GBS)
check("cost per successful request", ms["cost_per_successful_request_usd"], ms["total_cost_usd"] / 2)

# empty workload must not crash or divide by zero
e = metrics.summarize(simulate_serverless([], 10.0))
check("no requests: success_rate defined (1.0)", e["success_rate"], 1.0)
check("no requests: SLA violation rate 0", e["sla_violation_rate"], 0.0)
ok("no requests: per-request cost is NaN, not an error", math.isnan(e["cost_per_successful_request_usd"]))

# backward compatibility of the original schema
legacy = ["system", "n_total", "n_completed", "n_rejected", "rejection_rate", "avg_response_s", "median_response_s",
          "p95_response_s", "p99_response_s", "avg_queue_s", "avg_startup_s", "cold_starts", "cold_start_rate",
          "total_cost_usd", "cost_per_request_usd", "cost_per_hour_usd", "throughput_rps", "utilization",
          "instances_created", "window_s"]
ok("all 20 original metric keys are still present, in the original order", list(m.keys())[:20] == legacy)

# =====================================================================================
print("\n=== F. Utilization definitions ===")
# one 1 s request, 10 s timeout, window 100 s. Instance is alive from t=0 until free_at(1.5) + 10.
s = simulate_serverless(reqs([0.0], 1.0), horizon=100.0, idle_timeout=10.0)
ms = metrics.summarize(s)
alive = (C.COLD_START_S + 1.0) + 10.0
check("legacy serverless utilization = exec / alive time", ms["utilization"], 1.0 / alive)
check("avg_concurrency = exec seconds / window", ms["avg_concurrency"], 1.0 / 100.0)
check("utilization_ref_capacity = avg_concurrency / 4 reference slots", ms["utilization_ref_capacity"], 0.01 / 4)
check("avg_provisioned_slots = alive time / window", ms["avg_provisioned_slots"], alive / 100.0)
ok("serverless basis label names alive instance time", "alive_instance_time" in ms["utilization_basis"])
v = simulate_vm(reqs([0.0] * 4, 1.0), horizon=1.0)
mv = metrics.summarize(v)
check("fixed VM: legacy utilization == common reference-capacity metric", mv["utilization"], mv["utilization_ref_capacity"])
check("fixed VM: avg_provisioned_slots = 4", mv["avg_provisioned_slots"], 4.0)
ok("the two systems' legacy utilization use DIFFERENT labelled bases", ms["utilization_basis"] != mv["utilization_basis"])
# same workload, different systems -> the common metric uses the same denominator
w = steady(seed=3, rate=5.0)
a = metrics.summarize(simulate_serverless(w, 600.0)); b = metrics.summarize(simulate_vm(w, 600.0))
check("common metric: same offered work -> same value for serverless and VM (no rejections)",
      a["utilization_ref_capacity"], b["utilization_ref_capacity"], tol=1e-9)

# =====================================================================================
print("\n=== E. Serverless concurrency limit ===")
s = simulate_serverless(reqs([0.0] * 30, 1.0), horizon=1.0, max_concurrency=10)
check("cap 10, 30 simultaneous: completed", s.n_completed, 10)
check("rejected", s.n_rejected, 20)
check("peak instances never exceed the cap", s.peak_instances, 10)
check("completed + rejected = offered", s.n_completed + s.n_rejected, 30)
s = simulate_serverless(reqs([0.0] * 30, 1.0), horizon=1.0, max_concurrency=100)
check("cap 100: nothing rejected", s.n_rejected, 0)
raises("max_concurrency=0 rejected", lambda: simulate_serverless([], 1.0, max_concurrency=0))
raises("max_concurrency=2.5 rejected", lambda: simulate_serverless([], 1.0, max_concurrency=2.5))
raises("negative cold start rejected", lambda: simulate_serverless([], 1.0, cold_start=-1))
heavy = bursty(seed=0, burst_size=1000)
prev_done, prev_cost, accounted = -1, -1.0, True
for cap in (10, 25, 50, 100):
    r = simulate_serverless(heavy, 600.0, max_concurrency=cap)
    accounted &= (r.n_completed + r.n_rejected == r.n_total)
    mm = metrics.summarize(r)
    ok(f"cap {cap}: more capacity -> at least as many completed and as much cost",
       r.n_completed >= prev_done and mm["total_cost_usd"] >= prev_cost, f"completed={r.n_completed}")
    prev_done, prev_cost = r.n_completed, mm["total_cost_usd"]
ok("completed + rejected = offered for every cap", accounted)
check("default cap 1000 is unchanged by the new code (no rejections on bursty)",
      simulate_serverless(bursty(seed=0), 600.0).n_rejected, 0)

# =====================================================================================
print("\n=== D. Execution-time and memory sensitivity ===")
# cost of one cold request with a 1 s run at 2 GB:  request fee + (exec + init) * GB * price
r1 = simulate_serverless(reqs([0.0], 1.0), horizon=10)
expected = P_REQ + (1.0 + C.COLD_START_S) * 2.0 * P_GBS
check("serverless cost at 2 GB", K.serverless_cost(r1, memory_gb=2.0), expected)
fee = P_REQ
c05, c10 = K.serverless_cost(r1, memory_gb=0.5) - fee, K.serverless_cost(r1, memory_gb=1.0) - fee
check("duration cost is linear in memory (1 GB = 2 x 0.5 GB)", c10, 2 * c05)
check("metrics.summarize(memory_gb=...) uses it", metrics.summarize(r1, memory_gb=2.0)["total_cost_usd"], expected)
check("default memory still 0.5 GB", metrics.summarize(r1)["total_cost_usd"], P_REQ + (1.0 + C.COLD_START_S) * 0.5 * P_GBS)
check("effective_slots default (0.5 GB fn on 4 GB VM) = CPU slots", effective_slots(), C.SLOTS_PER_VM)
check("effective_slots 2 GB fn = 2 (fits twice)", effective_slots(function_memory_gb=2.0), 2)
check("effective_slots 4 GB fn = 1 (memory-bound)", effective_slots(function_memory_gb=4.0), 1)
raises("function larger than the VM is rejected", lambda: effective_slots(function_memory_gb=8.0))
# memory-bound VM pool: 2 VMs x 1 slot, eight 1 s jobs at t=0 -> finish at 1,1,2,2,3,3,4,4
v = simulate_vm(reqs([0.0] * 8, 1.0), horizon=4.0, slots=effective_slots(function_memory_gb=4.0))
check("1 slot per VM: mean response = 2.5", v.response.mean(), 2.5)
# longer mean execution time on the same arrivals raises serverless cost ~ linearly
a50, a500 = steady(seed=1, rate=5.0, mean_exec=0.05), steady(seed=1, rate=5.0, mean_exec=0.5)
c_a = K.serverless_cost(simulate_serverless(a50, 600.0)); c_b = K.serverless_cost(simulate_serverless(a500, 600.0))
ok("serverless cost grows with execution time (50 ms -> 500 ms)", c_b > 3 * c_a, f"{c_a:.5f} -> {c_b:.5f}")
ok("generator honours mean_exec (sample mean within 10 %)",
   abs(np.mean([x.exec_time for x in a500]) / 0.5 - 1) < 0.10)
# overloaded VM: 1000 ms jobs at 5 req/s need 5 slots but only 4 exist -> rejections appear
o = metrics.summarize(simulate_vm(steady(seed=1, rate=5.0, mean_exec=1.0), 600.0))
ok("1000 ms jobs at 5 req/s overload the 4-slot fixed VM (rejections > 0)", o["n_rejected"] > 0, f"rejected={o['n_rejected']}")

# =====================================================================================
print("\n=== C. Autoscaling VM ===")
print("-- consistency with the fixed-VM simulator")
same = True
for wl, gen in W.GENERATORS.items():
    for seed in range(3):
        w = gen(seed=seed); h = W.horizon_of(wl)
        for n in (2, 3):
            f = simulate_vm(w, h, num_vms=n)
            a = AUTO(w, h, min_vms=n, max_vms=n, min_billed_s=0)
            same &= (f.n_rejected == a.n_rejected and np.array_equal(f.queue, a.queue)
                     and np.array_equal(f.execution, a.execution) and math.isclose(f.horizon, a.horizon)
                     and math.isclose(f.capacity_time, a.capacity_time)
                     and math.isclose(a.vm_seconds, n * f.horizon) and a.n_scale_out == 0 and a.n_scale_in == 0)
ok("min_vms == max_vms reproduces simulate_vm() exactly (4 workloads x 3 seeds x 2 sizes)", same)
check("autoscaling with min=max=2 costs the same as the fixed 2-VM baseline",
      K.vm_cost(AUTO(steady(seed=0), 600.0, min_vms=2, max_vms=2)), K.vm_cost(simulate_vm(steady(seed=0), 600.0)))

print("-- cost accounting")
r = AUTO([], 3600.0, min_vms=2, max_vms=2)
check("idle, 2 VMs, 1 hour: cost = 2 x hourly price", K.vm_cost(r), 2 * VMP)
r = AUTO([], 30.0, min_vms=1, max_vms=1)
check("minimum billing: a 30 s window is billed 60 s", r.vm_seconds, 60.0)
# constant overload: 20 req/s x 1 s jobs from t=0..100 on 1 VM (2 slots). Util hits 100 % at the first tick (t=10).
over = reqs(np.arange(0, 100, 0.05), 1.0)
r = AUTO(over, 100.0, min_vms=1, max_vms=2, provision_delay=60.0, queue_limit=10**6)
t_end = r.horizon
check("exactly one scale-out (max 2)", r.n_scale_out, 1)
check("billing = VM#1 whole window + VM#2 from LAUNCH (t=10)", r.vm_seconds, t_end + (t_end - 10.0))
check("cost = billed seconds x price / 3600", K.vm_cost(r), (t_end + t_end - 10.0) * VMP / 3600)
r2 = AUTO(over, 100.0, min_vms=1, max_vms=2, provision_delay=60.0, queue_limit=10**6, bill_provisioning=False)
check("bill_provisioning=False: VM#2 billed only from READY (t=70)", r2.vm_seconds, r2.horizon + (r2.horizon - 70.0))

print("-- provisioning delay")
tl = r.timeline
ok("VM is launched at the first monitoring tick (t=10): ready=1, booting=1", (10.0, 1, 1) in tl, str(tl[:3]))
ok("...and becomes usable exactly provision_delay later (t=70): ready=2, booting=0", (70.0, 2, 0) in tl)
# capacity really is missing during the boot: slot-time before t=70 is 1 VM, after is 2 VMs
check("capacity_time = ready slots integrated over time (1 VM until t=70, then 2 VMs)",
      r.capacity_time, C.SLOTS_PER_VM * (70.0 + 2 * (r.horizon - 70.0)))
slow = AUTO(over, 100.0, min_vms=1, max_vms=4, provision_delay=60.0, queue_limit=10**6)
fast = AUTO(over, 100.0, min_vms=1, max_vms=4, provision_delay=0.0, queue_limit=10**6)
ok("longer provisioning delay -> longer queueing (avg wait)", slow.queue.mean() > fast.queue.mean(),
   f"{slow.queue.mean():.1f}s vs {fast.queue.mean():.1f}s")
ok("a larger provisioning delay never improves success on a 25 req/s steady load",
   all(metrics.summarize(AUTO(steady(seed=2, rate=25.0), 600.0, provision_delay=d))["success_rate"]
       >= metrics.summarize(AUTO(steady(seed=2, rate=25.0), 600.0, provision_delay=d + 120))["success_rate"] - 1e-9
       for d in (0, 60)))

print("-- min / max limits")
w = steady(seed=0, rate=40.0)                 # needs 8 slots = 4 VMs, but we allow only 3
r = AUTO(w, 600.0, min_vms=1, max_vms=3, provision_delay=10.0)
ok("never more than max_vms (running + booting) at any time", max(a + b for _, a, b in r.timeline) <= 3)
check("peak_instances == max_vms when demand exceeds it", r.peak_instances, 3)
# burst then silence: must scale in, but never below the minimum
w = reqs(np.arange(0, 60, 0.05), 1.0)         # 20 req/s for 60 s, then nothing for ~540 s
r = AUTO(w, 600.0, min_vms=2, max_vms=6, provision_delay=10.0, scale_in_after=60.0)
ok("scaled out during the burst", r.n_scale_out > 0)
ok("scaled back in after it", r.n_scale_in > 0)
ok("ready VMs never drop below min_vms", min(a for _, a, _b in r.timeline) >= 2)
check("ends at exactly min_vms", r.timeline[-1][1], 2)
scale_in_times = [t1 for (t0, a0, b0), (t1, a1, b1) in zip(r.timeline, r.timeline[1:]) if a1 < a0]
ok("scale-in only after the low-utilization period (load ends at t=60, so not before t=120)",
   len(scale_in_times) > 0 and min(scale_in_times) >= 60.0 + 60.0 - 1e-9, f"first scale-in at t={min(scale_in_times, default=None)}")
check("scale-in count + final VMs consistent: launched - terminated = ready at end",
      r.n_instances - r.n_scale_in, r.timeline[-1][1])
ok("timeline is time-ordered", all(x[0] <= y[0] for x, y in zip(r.timeline, r.timeline[1:])))

print("-- rejected requests and accounting")
w = steady(seed=0, rate=40.0)
r = AUTO(w, 600.0, min_vms=1, max_vms=1, queue_limit=20)
ok("overloaded + small queue -> some rejections", r.n_rejected > 0)
check("completed + rejected = offered", r.n_completed + r.n_rejected, r.n_total)
mm = metrics.summarize(r)
check("success_rate + rejection_rate = 1", mm["success_rate"] + mm["rejection_rate"], 1.0)
r = AUTO(reqs([0.0] * 100, 1.0), 1.0, min_vms=1, max_vms=1, queue_limit=20)
check("queue limit respected: 1 VM (2 slots) + 20 queued = 22 accepted of 100 simultaneous", r.n_completed, 22)
check("...so 78 are rejected", r.n_rejected, 78)
r = AUTO(reqs([0.0] * 200, 1.0), 1.0, min_vms=2, max_vms=2, provision_delay=0.0)
check("200 simultaneous 1 s jobs, 2 fixed VMs: 4 running + 100 queued accepted", r.n_completed, 104)
check("...and 96 rejected (identical to the fixed model)", r.n_rejected, 96)

print("-- determinism and validation")
a = metrics.summarize(AUTO(variable(seed=4, mean_rate=15.0), 600.0))
b = metrics.summarize(AUTO(variable(seed=4, mean_rate=15.0), 600.0))
ok("same seed -> identical autoscaling results", all((a[k] == b[k]) or (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k])) for k in a))
raises("min_vms=0 rejected", lambda: AUTO([], 10.0, min_vms=0))
raises("max_vms < min_vms rejected", lambda: AUTO([], 10.0, min_vms=3, max_vms=2))
raises("negative provisioning delay rejected", lambda: AUTO([], 10.0, provision_delay=-1))
raises("util_down >= util_up rejected", lambda: AUTO([], 10.0, util_up=0.5, util_down=0.6))
raises("eval_interval <= 0 rejected", lambda: AUTO([], 10.0, eval_interval=0))
raises("queue_limit = 0 rejected", lambda: AUTO([], 10.0, queue_limit=0))

print("-- config validation")
old = C.AUTOSCALE_MIN_VMS
C.AUTOSCALE_MIN_VMS = 0
raises("config.validate_config() catches a bad AUTOSCALE_MIN_VMS", C.validate_config)
C.AUTOSCALE_MIN_VMS = old
C.validate_config()
ok("config valid again after restoring", True)

# =====================================================================================
print("\n=== A. Analytical cold-start model ===")
# (1) it reduces to the textbook single-stream formula when traffic is sparse and the run is long
lam = 1 / 2000.0
phases, pat = [(0.0, 5e6, lam)], "poisson"
est = AN.estimate_cold_starts(phases, pat, idle_timeout=600.0, cold_start=0.5)
check("sparse Poisson stream ~ exp(-rate*(E+T))", est["p_cold"], AN.simple_cold_probability(lam), tol=0.02)
# (2) monotonicity: longer timeout -> fewer cold starts; lower cold start time -> fewer
p = lambda **k: AN.estimate_for_workload("bursty", **k)["p_cold"]
ok("longer idle timeout never increases p_cold", p(idle_timeout=5) >= p(idle_timeout=60) >= p(idle_timeout=600))
ok("shorter cold-start time never increases p_cold", p(cold_start=2.0) >= p(cold_start=0.5) >= p(cold_start=0.1))
ok("idle gap shorter than the timeout stays warm; longer one does not",
   AN.estimate_for_workload("idle_to_burst", idle=300)["p_cold"] < AN.estimate_for_workload("idle_to_burst", idle=900)["p_cold"])
# (3) timeout = 0 and sparse traffic: essentially every request is cold
ok("timeout 0 with sparse traffic -> p_cold close to 1",
   AN.estimate_cold_starts([(0.0, 1e5, 0.01)], "poisson", idle_timeout=0.0)["p_cold"] > 0.95)
# (4) rejection estimate equals Erlang-B for Poisson traffic (no cold-start effect: s = 0)
rho, cap = 3.0 * 0.2, 2
erlang_b = (rho ** 2 / 2) / (1 + rho + rho ** 2 / 2)
est = AN.estimate_cold_starts([(0.0, 1000.0, 3.0)], "poisson", mean_exec=0.2, cold_start=0.0, max_concurrency=cap)
check("p_reject = Erlang-B(2, 0.6)", est["p_reject"], erlang_b, tol=1e-6)
# (5) cost prediction uses the same formula as cost.serverless_cost
est = AN.estimate_for_workload("steady")
exp_cost = est["expected_completed"] * (P_REQ + (C.MEAN_EXEC_TIME_S + est["p_cold"] * C.COLD_START_S) * MEM * P_GBS)
check("predicted cost = N x (c_req + billed_seconds x GB x price)", est["cost_usd"], exp_cost)
# (6) agreement with the simulator (mean of 5 seeds) on the four default workloads: within 1.5 percentage points
worst = 0.0
for wl, gen in W.GENERATORS.items():
    runs = [simulate_serverless(gen(seed=s), W.horizon_of(wl)) for s in range(5)]
    sim = np.mean([x.n_cold / max(1, x.n_completed) for x in runs])
    mod = AN.estimate_for_workload(wl)["p_cold"]
    worst = max(worst, abs(mod - sim))
    print(f"    {wl:<14} simulated {sim*100:6.3f}%   model {mod*100:6.3f}%")
ok("model within 1.5 percentage points of the simulation on all four workloads", worst < 0.015, f"worst={worst*100:.2f} pp")
# (7) validation
raises("bad arrival pattern rejected", lambda: AN.estimate_cold_starts([(0, 1, 1.0)], "weird"))
raises("mean_exec <= 0 rejected", lambda: AN.estimate_cold_starts([(0, 1, 1.0)], mean_exec=0))
raises("negative idle timeout rejected", lambda: AN.estimate_cold_starts([(0, 1, 1.0)], idle_timeout=-1))
raises("max_concurrency < 1 rejected", lambda: AN.estimate_cold_starts([(0, 1, 1.0)], max_concurrency=0))
raises("unknown workload profile rejected", lambda: W.profile_of("nope"))
# (8) profiles mirror the generators: expected request counts match the generators' averages
for wl, gen in W.GENERATORS.items():
    phases, _ = W.profile_of(wl)
    expected_n = sum(r * (b - a) for a, b, r in phases)
    mean_n = np.mean([len(gen(seed=s)) for s in range(10)])
    ok(f"profile_of('{wl}') offered load matches generator ({expected_n:.0f} vs {mean_n:.0f})",
       abs(mean_n / expected_n - 1) < 0.05)
# (9) break-even with a cold-start probability
be0, be1 = K.breakeven_rate(), K.breakeven_rate(p_cold=0.5)
check("break-even default unchanged: 2 x 0.0416/3600 / (0.2e-6 + 0.2 x 0.5 x 1.66667e-5)", be0,
      2 * VMP / 3600 / (P_REQ + 0.2 * 0.5 * P_GBS))
ok("break-even falls when cold starts are billed (p_cold=0.5)", be1 < be0)
check("break-even with p_cold explicit formula", be1,
      2 * VMP / 3600 / (P_REQ + (0.2 + 0.5 * C.COLD_START_S) * 0.5 * P_GBS))

# =====================================================================================
print("\n=== Experiments: schema, reproducibility, backward compatibility ===")
saved_seeds = C.SEEDS
C.SEEDS = [0, 1]                               # keep the in-process runs quick
try:
    RE.INCLUDE_AUTOSCALE = True
    d8 = pd.DataFrame(RE.e8_exec_time())
    ok("E8 covers every execution time and all three systems",
       set(d8.value) == set(RE.EXEC_TIMES_S) and set(d8.system) == {"serverless", "vm", "vm_autoscale"})
    ok("E8 durations 50/200/500/1000 ms present", {0.05, 0.2, 0.5, 1.0} <= set(d8.value))
    d9 = pd.DataFrame(RE.e9_memory())
    ok("E9 covers 128/512/1024/2048 MB (+4096)", {128, 512, 1024, 2048, 4096} <= set(d9.value))
    s9 = d9[(d9.system == "serverless") & (d9.workload == "steady") & (d9.seed == 0)].set_index("value").total_cost_usd
    ok("E9 serverless cost rises with memory", s9.is_monotonic_increasing)
    v9 = d9[(d9.system == "vm") & (d9.workload == "steady") & (d9.seed == 0)].set_index("value").total_cost_usd
    ok("E9 fixed-VM cost does not depend on memory", v9.nunique() == 1)
    d10 = pd.DataFrame(RE.e10_concurrency())
    ok("E10 covers limits 10/25/50/100", {10, 25, 50, 100} <= set(d10.value))
    sv = d10[(d10.system == "serverless") & (d10.workload == "bursty_heavy")].groupby("value").agg(
        done=("n_completed", "mean"), rej=("n_rejected", "mean"), thr=("throughput_rps", "mean"))
    ok("E10 heavy burst: completed and throughput rise with the limit", sv.done.is_monotonic_increasing and sv.thr.is_monotonic_increasing)
    ok("E10 heavy burst: rejections fall with the limit", sv.rej.is_monotonic_decreasing)
    ok("E10 heavy burst: limit 10 rejects, default 1000 rejects nothing", sv.rej.loc[10] > 0 and sv.rej.loc[1000] == 0)
    d11 = pd.DataFrame(RE.e11_autoscale_delay())
    ok("E11 only the autoscaler's behaviour changes with the delay (serverless rows constant)",
       d11[d11.system == "serverless"].groupby(["workload", "seed"]).total_cost_usd.nunique().max() == 1)
    for name, d in (("E8", d8), ("E9", d9), ("E10", d10), ("E11", d11)):
        ok(f"{name}: key numeric columns contain no NaN/inf except per-request stats on empty runs",
           np.isfinite(d[["total_cost_usd", "success_rate", "throughput_rps", "sla_violation_rate"]].values).all())
        ok(f"{name}: success_rate + rejection_rate = 1 in every row", np.allclose(d.success_rate + d.rejection_rate, 1.0))
    again = pd.DataFrame(RE.e8_exec_time())
    ok("reproducible: running E8 twice gives identical results", d8.equals(again))
    # --no-autoscale -> the original two systems only, with the ORIGINAL numbers
    RE.INCLUDE_AUTOSCALE = False
    d2 = pd.DataFrame(RE.e2_workloads())
    ok("--no-autoscale: only serverless and vm rows", set(d2.system) == {"serverless", "vm"})
    w = W.steady(seed=0)
    direct = metrics.summarize(simulate_vm(w, 600.0))
    row = d2[(d2.workload == "steady") & (d2.seed == 0) & (d2.system == "vm")].iloc[0]
    ok("old-experiment rows equal a direct simulate_vm() + summarize() call",
       all(math.isclose(row[k], direct[k], rel_tol=0, abs_tol=0) for k in legacy[1:] if isinstance(direct[k], float) and not math.isnan(direct[k])))
    RE.INCLUDE_AUTOSCALE = True
    # all original experiments still execute (default config, reduced seeds)
    n_rows = {fn.__name__: len(fn()) for fn in (RE.e1_rate_sweep, RE.e2_workloads, RE.e3_cold_start, RE.e4_idle_gap,
                                                RE.e5_timeout, RE.e6_burst_size, RE.e7_vm_count)}
    ok("original experiments E1-E7 still run and return rows", all(v > 0 for v in n_rows.values()), str(n_rows))
finally:
    C.SEEDS = saved_seeds

# =====================================================================================
if not FAST:
    print("\n=== End-to-end: full run_experiments.py + plots.py in a scratch copy ===")
    src = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.mkdtemp()
    dst = os.path.join(tmp, "proj")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("results", "__pycache__", "*.zip"))
    env = dict(os.environ, MPLBACKEND="Agg")
    r1 = subprocess.run([sys.executable, "run_experiments.py"], cwd=dst, capture_output=True, text=True, env=env)
    ok("python run_experiments.py (the original command) exits 0", r1.returncode == 0, r1.stderr[-300:])
    res = os.path.join(dst, "results")
    for f in ("all_raw_results.csv", "summary_mean_std.csv", "analytical_vs_sim.csv") + tuple(
            f"raw_{n}.csv" for n in ("e1_rate_sweep", "e8_exec_time", "e9_memory", "e10_concurrency", "e11_autoscale_delay")):
        ok(f"results/{f} written and non-empty", os.path.exists(os.path.join(res, f)) and os.path.getsize(os.path.join(res, f)) > 0)
    big = pd.read_csv(os.path.join(res, "all_raw_results.csv"))
    ok("combined CSV: the 25 original columns come first, in the original order",
       list(big.columns[:25]) == ["experiment", "workload", "param", "value", "seed"] + legacy)
    ok("combined CSV has the three original systems plus the equal-capacity autoscaler, which appears ONLY in E12",
       set(big.system) == {"serverless", "vm", "vm_autoscale", "vm_autoscale_min2"}
       and set(big[big.system == "vm_autoscale_min2"].experiment) == {"E12_equal_capacity"})
    ok("combined CSV has experiments E1..E11", {f"E{i}" for i in range(1, 12)} <= {e.split("_")[0] for e in big.experiment})
    r2 = subprocess.run([sys.executable, "run_experiments.py", "--only", "e2_workloads"], cwd=dst, capture_output=True, text=True, env=env)
    ok("--only re-runs one experiment without truncating the combined CSV",
       r2.returncode == 0 and len(pd.read_csv(os.path.join(res, "all_raw_results.csv"))) == len(big))
    r3 = subprocess.run([sys.executable, "plots.py"], cwd=dst, capture_output=True, text=True, env=env)
    ok("python plots.py exits 0", r3.returncode == 0, r3.stderr[-400:])
    figs = sorted(os.listdir(os.path.join(res, "figures")))
    old_figs = [f for f in figs if int(f[3:5]) <= 15]
    ok("all 15 original figures still produced", len(old_figs) == 15, str(len(old_figs)))
    ok("new figures produced (fig16 and up)", len([f for f in figs if int(f[3:5]) >= 16]) >= 5, str([f for f in figs if int(f[3:5]) >= 16]))
    ok("every figure is a non-trivial PNG (> 10 kB)", all(os.path.getsize(os.path.join(res, "figures", f)) > 10_000 for f in figs))
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
