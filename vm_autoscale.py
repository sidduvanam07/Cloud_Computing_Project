"""
vm_autoscale.py - Event-driven simulation of an AUTOSCALING VM pool.

This is an OPTIONAL third system. The original fixed-capacity model in vm.py is
untouched and remains the baseline. Same request list in, same SimResult out, so
cost.py / metrics.py / the experiment runner treat it like the other two systems.

Why a separate module?  vm.py can be a one-pass loop because capacity never
changes. Autoscaling has time-driven events (monitoring ticks, VMs becoming
ready), so this module runs a small event loop instead.

Rules modelled (all defaults in config.py):
  * Start with `min_vms` ready VMs, each with `slots` parallel request slots.
  * One shared FCFS queue; a request that finds no free slot waits; if
    `queue_limit` requests are already waiting it is REJECTED (same as vm.py).
  * Dispatch fills the lowest-numbered VM that has a free slot, so load packs onto
    few VMs and the highest-numbered VM is the first to become idle.
  * Every `eval_interval` seconds the autoscaler looks at the slot utilization of
    the interval that just ended (busy slot-seconds / ready slot-seconds):
      SCALE OUT  if utilization >= util_up OR queue length >= queue_up,
                 and running+booting VMs < max_vms, and the cool-down since the
                 last scale-out has passed. Adds `step` VMs (capped by max_vms).
                 A new VM becomes usable only after `provision_delay` seconds.
      SCALE IN   if utilization < util_down AND the queue is empty AND no VM is
                 booting, continuously for `scale_in_after` seconds, and ready
                 VMs > min_vms. Removes ONE completely idle VM (the highest
                 numbered); if none is idle it waits for the next evaluation.
  * Cost: each VM is billed from launch (or from ready, if bill_provisioning is
    False) until terminated or the end of the window, at least `min_billed_s`.
  * Requests never see a per-request start-up delay (startup = 0): provisioning
    delay shows up as QUEUEING while capacity is missing.

Event order at equal timestamps: FINISH < READY < ARRIVAL < TICK. This matches
vm.py (a slot that frees at time t is usable by a request arriving at t), so with
min_vms == max_vms the results are identical to simulate_vm().
"""
import heapq
from collections import deque
import numpy as np
import config as C
from common import SimResult

_FINISH, _READY, _ARRIVAL, _TICK = 0, 1, 2, 3


class _VM:
    __slots__ = ("vid", "launched", "ready_at", "terminated", "running")

    def __init__(self, vid, launched, ready_at):
        self.vid = vid
        self.launched = launched
        self.ready_at = ready_at
        self.terminated = None
        self.running = 0


def _validate(min_vms, max_vms, slots, queue_limit, provision_delay, eval_interval,
              util_up, queue_up, util_down, scale_in_after, step, out_cooldown, min_billed_s):
    def need(cond, msg):
        if not cond:
            raise ValueError(f"autoscaling VM: {msg}")
    need(int(min_vms) == min_vms and min_vms >= 1, "min_vms must be an integer >= 1")
    need(int(max_vms) == max_vms and max_vms >= min_vms, "max_vms must be an integer >= min_vms")
    need(int(slots) == slots and slots >= 1, "slots must be an integer >= 1")
    need(queue_limit >= 1, "queue_limit must be >= 1")
    need(provision_delay >= 0, "provision_delay must be >= 0")
    need(eval_interval > 0, "eval_interval must be > 0")
    need(0 < util_down < util_up <= 1, "need 0 < util_down < util_up <= 1")
    need(queue_up >= 1, "queue_up must be >= 1")
    need(scale_in_after >= 0 and out_cooldown >= 0, "scale_in_after and out_cooldown must be >= 0")
    need(int(step) == step and step >= 1, "step must be an integer >= 1")
    need(min_billed_s >= 0, "min_billed_s must be >= 0")


def simulate_vm_autoscaling(requests, horizon,
                            min_vms=C.AUTOSCALE_MIN_VMS, max_vms=C.AUTOSCALE_MAX_VMS,
                            slots=C.SLOTS_PER_VM, queue_limit=C.QUEUE_LIMIT,
                            provision_delay=C.AUTOSCALE_PROVISION_DELAY_S,
                            eval_interval=C.AUTOSCALE_EVAL_INTERVAL_S,
                            util_up=C.AUTOSCALE_UTIL_UP, queue_up=C.AUTOSCALE_QUEUE_UP,
                            util_down=C.AUTOSCALE_UTIL_DOWN,
                            scale_in_after=C.AUTOSCALE_SCALE_IN_AFTER_S,
                            step=C.AUTOSCALE_STEP, out_cooldown=C.AUTOSCALE_OUT_COOLDOWN_S,
                            min_billed_s=C.AUTOSCALE_MIN_BILLED_S,
                            bill_provisioning=C.AUTOSCALE_BILL_PROVISIONING):
    _validate(min_vms, max_vms, slots, queue_limit, provision_delay, eval_interval,
              util_up, queue_up, util_down, scale_in_after, step, out_cooldown, min_billed_s)

    events = []                      # heap of (time, kind, seq, payload)
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, kind, seq, payload))

    vms = [_VM(i, 0.0, 0.0) for i in range(min_vms)]     # every VM ever created
    ready = list(vms)                # VMs that can serve, kept in vid order
    booting = 0                      # launched but not ready yet
    waiting = deque()                # FCFS queue of (arrival_time, exec_time)
    queue_w, execution = [], []
    n_rejected = n_out = n_in = 0
    last_finish = 0.0
    peak = min_vms
    timeline = [(0.0, min_vms, 0)]

    # time-integrals for utilization (whole run and current monitoring interval)
    now = 0.0
    running = 0                      # requests executing right now (all VMs)
    ready_slot_s = 0.0               # sum over time of ready slots (= capacity_time)
    win_busy = win_cap = 0.0         # same, but only since the last evaluation
    last_out = -1e18                 # time of the last scale-out decision
    low_since = None                 # when "low utilization" started (None = not low now)

    def advance(t):
        nonlocal now, ready_slot_s, win_busy, win_cap
        dt = t - now
        if dt > 0:
            cap = len(ready) * slots * dt
            ready_slot_s += cap
            win_cap += cap
            win_busy += running * dt
        now = t

    def start_waiting():
        """Start queued requests on free slots (lowest-numbered VM first)."""
        nonlocal running
        while waiting:
            vm = next((v for v in ready if v.running < slots), None)
            if vm is None:
                return
            t_arr, ex = waiting.popleft()
            vm.running += 1
            running += 1
            queue_w.append(now - t_arr)
            execution.append(ex)
            push(now + ex, _FINISH, vm)

    def dispatch(t_arr, ex):
        """A request arrives: run it, queue it, or reject it."""
        nonlocal n_rejected, running
        vm = next((v for v in ready if v.running < slots), None)
        if vm is not None:                         # a free slot implies an empty queue
            vm.running += 1
            running += 1
            queue_w.append(0.0)
            execution.append(ex)
            push(now + ex, _FINISH, vm)
        elif len(waiting) >= queue_limit:
            n_rejected += 1
        else:
            waiting.append((t_arr, ex))

    reqs = list(requests)
    i = 0                            # index of the next arrival
    push(eval_interval, _TICK, None)

    while True:
        next_arr = reqs[i].arrival_time if i < len(reqs) else None
        if events and (next_arr is None or (events[0][0], events[0][1]) < (next_arr, _ARRIVAL)):
            t, kind, _, payload = events[0]
        elif next_arr is not None:
            t, kind, payload = next_arr, _ARRIVAL, None
        else:
            break
        # nothing left to do and the window is over -> stop (ignore later ticks / boots)
        if next_arr is None and not waiting and running == 0 and t >= horizon:
            break
        if kind == _ARRIVAL:
            advance(t)
            dispatch(t, reqs[i].exec_time)
            i += 1
            continue
        heapq.heappop(events)
        advance(t)
        if kind == _FINISH:
            payload.running -= 1
            running -= 1
            last_finish = max(last_finish, t)
            start_waiting()
        elif kind == _READY:
            booting -= 1
            ready.append(payload)
            ready.sort(key=lambda v: v.vid)
            timeline.append((t, len(ready), booting))
            start_waiting()
        else:                                          # _TICK: the autoscaler decides
            util = win_busy / win_cap if win_cap > 0 else 0.0
            win_busy = win_cap = 0.0
            total = len(ready) + booting
            if (util >= util_up or len(waiting) >= queue_up) and total < max_vms \
                    and t - last_out >= out_cooldown:
                add = min(step, max_vms - total)
                for _ in range(add):
                    vm = _VM(len(vms), t, t + provision_delay)
                    vms.append(vm)
                    booting += 1
                    n_out += 1
                    push(vm.ready_at, _READY, vm)
                last_out = t
                low_since = None
                peak = max(peak, len(ready) + booting)
                timeline.append((t, len(ready), booting))
            elif util < util_down and not waiting and booting == 0 and len(ready) > min_vms:
                if low_since is None:
                    low_since = t
                if t - low_since >= scale_in_after:
                    idle_vm = next((v for v in reversed(ready) if v.running == 0), None)
                    if idle_vm is not None:
                        ready.remove(idle_vm)
                        idle_vm.terminated = t
                        n_in += 1
                        low_since = t            # next removal needs another full quiet period
                        timeline.append((t, len(ready), booting))
            else:
                low_since = None
            # keep monitoring while the window is open or work remains
            nxt = t + eval_interval
            if nxt <= horizon or waiting or running > 0 or i < len(reqs):
                push(nxt, _TICK, None)

    horizon_eff = max(horizon, last_finish)
    advance(horizon_eff)

    # ---- billing: each VM from launch (or ready) to termination / end of window ----
    vm_seconds = 0.0
    for vm in vms:
        start = vm.launched if bill_provisioning else vm.ready_at
        if start >= horizon_eff:                       # never became billable inside the window
            continue
        end = vm.terminated if vm.terminated is not None else horizon_eff
        vm_seconds += max(min(end, horizon_eff) - start, min_billed_s)

    exec_sum = float(np.sum(execution)) if execution else 0.0
    return SimResult(system="vm_autoscale", n_total=len(reqs), n_rejected=n_rejected,
                     n_cold=0, horizon=horizon_eff, busy_time=exec_sum,
                     capacity_time=ready_slot_s, n_instances=len(vms),
                     peak_instances=peak, vm_seconds=vm_seconds,
                     n_scale_out=n_out, n_scale_in=n_in, slots_per_vm=slots, timeline=timeline,
                     queue=np.array(queue_w), startup=np.zeros(len(execution)),
                     execution=np.array(execution))
