"""
common.py - the result object shared by both simulators.

Both simulators return the same kind of object, so cost.py, metrics.py
and the experiment runner treat serverless and VM identically.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SimResult:
    system: str                 # "serverless", "vm" (fixed) or "vm_autoscale"
    n_total: int = 0            # requests offered
    n_rejected: int = 0         # requests refused (VM queue full / serverless concurrency cap)
    n_cold: int = 0             # requests that triggered a cold start (serverless only)
    horizon: float = 0.0        # effective time window = max(nominal length, last completion)
    busy_time: float = 0.0      # seconds spent actually executing requests
    capacity_time: float = 0.0  # seconds of capacity that existed (VM slots / alive instances)
    n_instances: int = 0        # serverless: instances created; VM: number of VMs
    queue: np.ndarray = field(default_factory=lambda: np.array([]))     # per completed request
    startup: np.ndarray = field(default_factory=lambda: np.array([]))
    execution: np.ndarray = field(default_factory=lambda: np.array([]))
    # ---- added later; all optional so the original simulators are unaffected ----
    peak_instances: int = 0     # serverless: most instances alive at once; VM: most VMs (running + booting)
    vm_seconds: Optional[float] = None   # autoscaling VM: billed VM-seconds (None -> fixed VM formula)
    n_scale_out: int = 0        # autoscaling VM: VMs launched by the autoscaler
    n_scale_in: int = 0         # autoscaling VM: VMs terminated by the autoscaler
    slots_per_vm: int = 0       # autoscaling VM: parallel slots per VM (used for paid-capacity utilization)
    timeline: list = field(default_factory=list)   # autoscaling VM: (time, ready_vms, booting_vms) at every change

    @property
    def response(self):
        """T_response = T_queue + T_startup + T_execution (per completed request)."""
        return self.queue + self.startup + self.execution

    @property
    def n_completed(self):
        return len(self.execution)
