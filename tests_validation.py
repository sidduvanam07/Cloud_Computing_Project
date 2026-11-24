"""
tests_validation.py - Baseline validation: simulator output vs hand calculation.

Run:  python tests_validation.py
Every case uses a tiny, hand-made workload whose correct answer can be
worked out on paper. Expected values are written as plain arithmetic from
the config constants, NOT taken from the simulator code.
"""
import math
import numpy as np
import config as C
from workload import Request, steady
from serverless import simulate_serverless
from vm import simulate_vm
import cost as K
import metrics

PASS = FAIL = 0


def check(name, got, expected, tol=1e-9):
    global PASS, FAIL
    ok = math.isclose(got, expected, rel_tol=tol, abs_tol=1e-12)
    PASS += ok; FAIL += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: simulated={got:.8g}  expected={expected:.8g}")


def reqs(arrivals, exec_time):
    return [Request(i, a, exec_time) for i, a in enumerate(arrivals)]


COLD, EXEC = C.COLD_START_S, 0.2
P_REQ, P_GBS, MEM = C.LAMBDA_PRICE_PER_REQUEST, C.LAMBDA_PRICE_PER_GB_SECOND, C.FUNCTION_MEMORY_GB

print("Case 1: ONE request needing a COLD start (serverless)")
r = simulate_serverless(reqs([0.0], EXEC), horizon=10)
check("response = cold + exec = 0.5 + 0.2", r.response[0], 0.7)
check("cold starts", r.n_cold, 1)
# cost = request fee + (exec + init) * 0.5 GB * price  (INIT billed since Aug 2025)
check("cost", K.serverless_cost(r), 2e-7 + (0.2 + 0.5) * 0.5 * 0.0000166667)

print("Case 2: ONE request on a WARM instance")
r = simulate_serverless(reqs([0.0, 10.0], EXEC), horizon=20)
check("2nd response = exec only = 0.2", r.response[1], 0.2)
check("cold starts (only the first)", r.n_cold, 1)
check("2nd startup", r.startup[1], 0.0)

print("Case 3: MULTIPLE requests sharing one warm instance (5 requests, 1 s apart)")
r = simulate_serverless(reqs([0, 1, 2, 3, 4], EXEC), horizon=5.0)
check("instances created", r.n_instances, 1)
check("cold starts", r.n_cold, 1)
check("mean response = (0.7 + 4*0.2)/5", r.response.mean(), (0.7 + 4 * 0.2) / 5)
check("instance utilization = 5*0.2 busy / 5 s alive", metrics.summarize(r)["utilization"], 1.0 / 5.0)

print("Case 4: IDLE VM with no requests (1 hour)")
v = simulate_vm([], horizon=3600)
s = simulate_serverless([], horizon=3600)
check("VM cost = 2 VMs * $0.0416 * 1 h", K.vm_cost(v), 2 * 0.0416)
check("VM utilization", metrics.summarize(v)["utilization"], 0.0)
check("serverless cost when idle", K.serverless_cost(s), 0.0)

print("Case 5: VM at FULL capacity (4 slots, 1 s jobs)")
v = simulate_vm(reqs([0.0] * 4, 1.0), horizon=1.0)
check("all 4 start at once: mean response", v.response.mean(), 1.0)
check("utilization = 100%", metrics.summarize(v)["utilization"], 1.0)
v = simulate_vm(reqs([0.0] * 8, 1.0), horizon=2.0)
check("8 jobs on 4 slots: mean response = (4*1 + 4*2)/8", v.response.mean(), 1.5)
check("utilization = 100%", metrics.summarize(v)["utilization"], 1.0)

print("Case 6: SUDDEN BURST exceeding capacity (200 jobs of 1 s at t=0)")
v = simulate_vm(reqs([0.0] * 200, 1.0), horizon=1.0)
check("VM accepted = 4 running + 100 queued", v.n_completed, 104)
check("VM rejected", v.n_rejected, 96)
check("VM mean response = mean(1..26)", v.response.mean(), 13.5)
s = simulate_serverless(reqs([0.0] * 200, 1.0), horizon=1.0)
check("serverless rejected", s.n_rejected, 0)
check("serverless cold starts = 200", s.n_cold, 200)
check("serverless mean response = cold+exec", s.response.mean(), 1.5)

print("Case 7: IDLE TIMEOUT (600 s)")
check("gap 500 s -> instance still warm -> 1 cold start",
      simulate_serverless(reqs([0, 500], EXEC), horizon=600).n_cold, 1)
check("gap 700 s -> instance destroyed -> 2 cold starts",
      simulate_serverless(reqs([0, 700], EXEC), horizon=800).n_cold, 2)

print("Case 8: analytical cost vs simulation on a generated steady workload")
w = steady(seed=1, rate=10.0)
s = simulate_serverless(w, horizon=600.0)
expected = (len(w) * P_REQ + (sum(x.exec_time for x in w) + s.n_cold * COLD) * MEM * P_GBS)
check("serverless cost formula", K.serverless_cost(s), expected)
v = simulate_vm(w, horizon=600.0)
check("VM cost = 2 * 0.0416 * 600/3600", K.vm_cost(v), 2 * 0.0416 * 600 / 3600)
be = K.breakeven_rate()
print(f"  analytical break-even for {C.NUM_VMS} VMs = {be:.2f} req/s")
lo = K.serverless_cost(simulate_serverless(steady(seed=1, rate=be * 0.8), 600.0))
hi = K.serverless_cost(simulate_serverless(steady(seed=1, rate=be * 1.2), 600.0))
vmc = K.vm_cost(simulate_vm([], 600.0))
print(f"  at 0.8x break-even: serverless ${lo:.5f} < VM ${vmc:.5f} -> {lo < vmc};"
      f"  at 1.2x: serverless ${hi:.5f} > VM -> {hi > vmc}")
PASS += (lo < vmc) + (hi > vmc); FAIL += (not lo < vmc) + (not hi > vmc)

print("Case 9: determinism (same seed, same workload -> identical results)")
a = metrics.summarize(simulate_serverless(steady(seed=5), 600.0))
b = metrics.summarize(simulate_serverless(steady(seed=5), 600.0))
check("identical total cost", a["total_cost_usd"], b["total_cost_usd"])
check("identical P95", a["p95_response_s"], b["p95_response_s"])

print(f"\nRESULT: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
