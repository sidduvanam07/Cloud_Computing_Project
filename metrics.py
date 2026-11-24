"""
metrics.py - Turns a SimResult (+ its cost) into one row of numbers.

COLUMN GROUPS (the first group is the original schema and is unchanged):
  original      n_total ... window_s
  reliability   success_rate, sla_violations, sla_violation_rate(_completed),
                cost_per_successful_request_usd
  utilization   utilization (legacy, system-specific), utilization_basis,
                avg_concurrency, avg_provisioned_slots, utilization_ref_capacity
  scaling       peak_instances, scale_out_events, scale_in_events, vm_billed_seconds,
                scale_cycles (flapping indicator), utilization_paid_capacity

HOW TO READ LATENCY vs RELIABILITY
  avg/median/p95/p99_response_s and avg_queue_s / avg_startup_s describe COMPLETED
  requests only. A system that rejects heavily can look fast because the slow
  requests never complete. Always read them together with success_rate /
  rejection_rate and sla_violation_rate, which are measured over ALL offered requests.

DEFINITIONS (n_total = offered, n = completed, SLA = SLA_RESPONSE_S)
  success_rate                  = n / n_total
  rejection_rate                = n_rejected / n_total          (so the two sum to 1)
  sla_violation_rate            = (n_rejected + #completed with response > SLA) / n_total
                                  -> rejected requests COUNT as violations (they got no answer)
  sla_violation_rate_completed  = #completed with response > SLA / n   (latency only)
  cost_per_request_usd          = total cost / n   (kept; it already divided by COMPLETED requests)
  cost_per_successful_request_usd = same value, explicit name added for clarity

UTILIZATION - three different quantities, never mix them up:
  utilization (legacy)  = busy_time / capacity_time. The DENOMINATOR DIFFERS by system:
      serverless    : executing time / alive-instance time (instances exist only while kept warm,
                      so this measures keep-alive waste, NOT how loaded the platform is)
      vm            : executing time / (slots x window)  (fixed capacity, always paid)
      vm_autoscale  : executing time / ready-slot time  (capacity that exists changes over time)
      Not comparable across systems; utilization_basis names the denominator.
  avg_concurrency       = busy_time / window = average number of requests executing at once.
                          No capacity in the denominator -> directly comparable across systems.
  utilization_ref_capacity = avg_concurrency / ref_slots. "How much of the REFERENCE capacity does
                          the work that was actually completed consume?" The reference is the
                          fixed-VM pool of the experiment: ref_slots defaults to
                          REF_CAPACITY_SLOTS (4 = the original baseline) and run_experiments passes
                          the pool's real slot count when it differs (E7: VM count; E9: slots limited
                          by function memory). Same denominator for every system in a row, so it IS
                          comparable (can exceed 100 % when the work needs more than the reference).
                          It counts only COMPLETED work: a system that rejects requests consumes less.
  avg_provisioned_slots = capacity_time / window = capacity that existed on average (serverless:
                          average number of alive instances, each counted as one slot).
  READY vs PAID capacity (VM systems):
    utilization (legacy)       = busy_time / READY slot-time. For the autoscaler the ready slot-time
                                 excludes VMs that are still booting.
    utilization_paid_capacity  = busy_time / PAID slot-time (billed VM-seconds x slots per VM). This
                                 includes boot time and the per-VM minimum billing, so for the
                                 autoscaler it is <= the legacy value; for the fixed VM the two are
                                 equal. NaN for serverless (billed per execution, not for capacity).

BILLING WINDOW. Each system is billed over its own effective window `window_s` =
max(nominal horizon, time the last accepted request finishes). If a system's queue drains after the
nominal horizon, its window (and VM cost) is slightly longer than another system's on the same
workload. The original code already worked this way; it is reported as `window_s`, not corrected.

SCALE CYCLES (flapping). scale_cycles = min(scale_out_events, scale_in_events): the number of
complete "add a VM, later remove one" cycles. On a calm workload that never needs extra capacity,
scale_cycles > 0 means the thresholds made the autoscaler add and then drop a VM needlessly.
"""
import numpy as np
import config as C
import cost as K

_BASIS = {"serverless": "executing_time / alive_instance_time",
          "vm": "executing_time / (fixed_slots x window)",
          "vm_autoscale": "executing_time / ready_slot_time"}


def summarize(res, cold_start=None, memory_gb=None, sla_s=None, ref_slots=None):
    kw = {}
    if cold_start is not None:
        kw["cold_start"] = cold_start
    if memory_gb is not None:
        kw["memory_gb"] = memory_gb
    sla = C.SLA_RESPONSE_S if sla_s is None else sla_s
    ref = C.REF_CAPACITY_SLOTS if ref_slots is None else ref_slots
    total_cost = K.serverless_cost(res, **kw) if res.system == "serverless" else K.vm_cost(res)
    n = res.n_completed
    r = res.response
    has = n > 0
    n_slow = int(np.sum(r > sla)) if has else 0
    avg_conc = res.busy_time / res.horizon if res.horizon else 0.0
    return {
        "system": res.system,
        "n_total": res.n_total,
        "n_completed": n,
        "n_rejected": res.n_rejected,
        "rejection_rate": res.n_rejected / res.n_total if res.n_total else 0.0,
        "avg_response_s": float(np.mean(r)) if has else np.nan,
        "median_response_s": float(np.median(r)) if has else np.nan,
        "p95_response_s": float(np.percentile(r, 95)) if has else np.nan,
        "p99_response_s": float(np.percentile(r, 99)) if has else np.nan,
        "avg_queue_s": float(np.mean(res.queue)) if has else np.nan,
        "avg_startup_s": float(np.mean(res.startup)) if has else np.nan,
        "cold_starts": res.n_cold,
        "cold_start_rate": res.n_cold / n if (n and res.system == "serverless") else 0.0,
        "total_cost_usd": total_cost,
        "cost_per_request_usd": total_cost / n if has else np.nan,
        "cost_per_hour_usd": total_cost / res.horizon * 3600 if res.horizon else np.nan,
        "throughput_rps": n / res.horizon if res.horizon else 0.0,
        "utilization": res.busy_time / res.capacity_time if res.capacity_time else 0.0,
        "instances_created": res.n_instances,
        "window_s": res.horizon,
        # ---- added: reliability / fairness ----
        "success_rate": n / res.n_total if res.n_total else 1.0,
        "cost_per_successful_request_usd": total_cost / n if has else np.nan,
        "sla_violations": n_slow + res.n_rejected,
        "sla_violation_rate": (n_slow + res.n_rejected) / res.n_total if res.n_total else 0.0,
        "sla_violation_rate_completed": n_slow / n if has else np.nan,
        # ---- added: utilization (see module docstring) ----
        "utilization_basis": _BASIS.get(res.system, ""),
        "avg_concurrency": avg_conc,
        "avg_provisioned_slots": res.capacity_time / res.horizon if res.horizon else 0.0,
        "utilization_ref_capacity": avg_conc / ref if ref else np.nan,
        # ---- added: scaling behaviour ----
        "peak_instances": res.peak_instances,
        "scale_out_events": res.n_scale_out,
        "scale_in_events": res.n_scale_in,
        "vm_billed_seconds": res.vm_seconds if res.vm_seconds is not None else (
            res.n_instances * res.horizon if res.system == "vm" else np.nan),
        "scale_cycles": min(res.n_scale_out, res.n_scale_in),
        "utilization_paid_capacity": _paid_utilization(res),
    }


def _paid_utilization(res):
    """busy_time / paid slot-seconds (VM systems only; NaN for serverless). See module docstring."""
    if res.system == "vm":
        paid = res.capacity_time                           # fixed pool: ready == paid
    elif res.system == "vm_autoscale" and res.vm_seconds is not None and res.slots_per_vm:
        paid = res.vm_seconds * res.slots_per_vm
    else:
        return np.nan
    return res.busy_time / paid if paid else np.nan
