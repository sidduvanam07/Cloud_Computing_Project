"""
vm.py - Event-driven simulation of a fixed-capacity, always-on VM pool.

Rules modelled:
  * num_vms VMs run for the whole window; each has `slots` CPUs, i.e. it
    can execute `slots` requests at the same time. The pool shares ONE
    first-come-first-served queue (a simple load balancer).
  * A request starts immediately if a slot is free, otherwise it waits.
  * If `queue_limit` requests are already waiting, the new one is REJECTED.
  * Already-running VMs have no startup delay (startup = 0).
  * VMs are paid for the entire window, busy or idle.

Because service is first-come-first-served, the start time of a request is
  start = max(arrival, earliest time any slot becomes free).
"""
import heapq
from collections import deque
import numpy as np
import config as C
from common import SimResult


def effective_slots(function_memory_gb=C.FUNCTION_MEMORY_GB, vm_memory_gb=C.VM_MEMORY_GB,
                    cpu_slots=C.SLOTS_PER_VM):
    """Requests one VM can run at once = min(CPU slots, how many functions fit in its RAM).
    With the defaults (0.5 GB function, 4 GB VM) memory never binds, so this equals
    SLOTS_PER_VM and the original model is unchanged."""
    fit = int(vm_memory_gb / function_memory_gb + 1e-9)
    if fit < 1:
        raise ValueError(f"a {function_memory_gb} GB function does not fit in a {vm_memory_gb} GB VM")
    return min(cpu_slots, fit)


def simulate_vm(requests, horizon, num_vms=C.NUM_VMS, slots=C.SLOTS_PER_VM,
                queue_limit=C.QUEUE_LIMIT):
    total_slots = num_vms * slots
    free_at = [0.0] * total_slots          # heap: when each slot becomes free
    heapq.heapify(free_at)
    waiting_starts = deque()               # start times of accepted requests
    n_rejected = 0
    queue, execution = [], []
    last_finish = 0.0

    for r in requests:
        t = r.arrival_time
        # requests whose start time has passed are no longer waiting
        while waiting_starts and waiting_starts[0] <= t:
            waiting_starts.popleft()
        if len(waiting_starts) >= queue_limit:
            n_rejected += 1
            continue
        slot_free = heapq.heappop(free_at)
        start = max(t, slot_free)
        finish = start + r.exec_time
        heapq.heappush(free_at, finish)
        waiting_starts.append(start)
        queue.append(start - t); execution.append(r.exec_time)
        last_finish = max(last_finish, finish)

    horizon_eff = max(horizon, last_finish)
    exec_sum = float(np.sum(execution)) if execution else 0.0
    return SimResult(system="vm", n_total=len(requests), n_rejected=n_rejected,
                     n_cold=0, horizon=horizon_eff, busy_time=exec_sum,
                     capacity_time=total_slots * horizon_eff, n_instances=num_vms,
                     peak_instances=num_vms,
                     queue=np.array(queue), startup=np.zeros(len(execution)),
                     execution=np.array(execution))
