"""
serverless.py - Event-driven simulation of a serverless (FaaS) platform.

Rules modelled (all parameters come from config.py):
  1. A request needs one instance (one request at a time per instance).
  2. If an idle ("warm") instance exists -> reuse it, startup = 0.
     We pick the most recently used one (how real platforms behave).
  3. Otherwise create a NEW instance -> COLD START: startup = cold_start.
  4. An instance idle for longer than idle_timeout is destroyed.
  5. There is no queue: the platform scales out, up to max_concurrency
     instances. Beyond that the request is rejected (throttled).

How the simulation proceeds: requests are handled in arrival order. A
min-heap holds the finish times of busy instances; when a new request
arrives, every instance that has finished by then is moved to the idle
pool. Time jumps from event to event (discrete-event simulation).
"""
import heapq
from collections import deque
import numpy as np
import config as C
from common import SimResult


class _Instance:
    __slots__ = ("created", "free_at", "exec_total")

    def __init__(self, created):
        self.created = created      # when this instance was created
        self.free_at = created      # when it last became free
        self.exec_total = 0.0       # total seconds spent executing requests


def simulate_serverless(requests, horizon, cold_start=C.COLD_START_S,
                        idle_timeout=C.IDLE_TIMEOUT_S,
                        max_concurrency=C.MAX_CONCURRENCY):
    if cold_start < 0 or idle_timeout < 0:
        raise ValueError("cold_start and idle_timeout must be >= 0")
    if int(max_concurrency) != max_concurrency or max_concurrency < 1:
        raise ValueError("max_concurrency must be an integer >= 1")
    peak = 0                  # most instances alive at once (busy + idle)
    busy = []                 # heap of (finish_time, tie_breaker, instance)
    idle = deque()            # idle instances, oldest-free first, newest last
    destroyed_alive = 0.0     # alive-time of instances already destroyed
    n_created = n_cold = n_rejected = 0
    queue, startup, execution = [], [], []
    last_finish = 0.0
    tie = 0

    for r in requests:
        t = r.arrival_time
        # 1) instances that finished before t become idle (in finish-time order)
        while busy and busy[0][0] <= t:
            fin, _, inst = heapq.heappop(busy)
            inst.free_at = fin
            idle.append(inst)
        # 2) destroy instances idle for longer than the timeout (oldest first)
        while idle and t - idle[0].free_at > idle_timeout:
            inst = idle.popleft()
            destroyed_alive += (inst.free_at + idle_timeout) - inst.created
        # 3) warm start, cold start, or reject
        if idle:
            inst = idle.pop()                    # most recently used
            s = 0.0
        elif len(busy) + len(idle) < max_concurrency:
            inst = _Instance(created=t)
            n_created += 1
            n_cold += 1
            s = cold_start
        else:
            n_rejected += 1
            continue
        peak = max(peak, len(busy) + len(idle) + 1)
        finish = t + s + r.exec_time
        inst.free_at = finish                    # when it will next be free
        inst.exec_total += r.exec_time
        tie += 1
        heapq.heappush(busy, (finish, tie, inst))
        queue.append(0.0); startup.append(s); execution.append(r.exec_time)
        last_finish = max(last_finish, finish)

    horizon_eff = max(horizon, last_finish)
    # instances still alive at the end live until their timeout expires or the window ends
    alive = destroyed_alive
    for inst in [x[2] for x in busy] + list(idle):
        alive += min(inst.free_at + idle_timeout, horizon_eff) - inst.created
    exec_sum = float(np.sum(execution)) if execution else 0.0

    return SimResult(system="serverless", n_total=len(requests),
                     n_rejected=n_rejected, n_cold=n_cold, horizon=horizon_eff,
                     busy_time=exec_sum, capacity_time=alive,
                     n_instances=n_created, peak_instances=peak,
                     queue=np.array(queue), startup=np.array(startup),
                     execution=np.array(execution))
