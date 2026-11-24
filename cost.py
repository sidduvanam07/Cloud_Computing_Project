"""
cost.py - Cost models (US dollars). Prices come from config.py.

Serverless  (pay per use):
    cost = N_completed * price_per_request
         + (sum of execution seconds [+ cold-start seconds if INIT is billed])
           * memory_GB * price_per_GB_second
    Rejected (throttled) requests are not billed.
    Billing is in exact seconds (real AWS rounds up to 1 ms - ignored).

VM  (pay per time):
    cost = num_vms * price_per_hour * (window_seconds / 3600)
    Charged whether the VMs are busy or idle.

Autoscaling VM (pay per VM-second that a VM exists):
    cost = billed_vm_seconds * price_per_hour / 3600
    billed_vm_seconds is computed by vm_autoscale.py (launch -> termination, with
    the per-VM minimum billing time); with min_vms == max_vms it equals the fixed formula.
"""
import config as C


def serverless_cost(res, cold_start=C.COLD_START_S, memory_gb=C.FUNCTION_MEMORY_GB,
                    bill_init=C.BILL_INIT_PHASE):
    gb_seconds = float(res.execution.sum())
    if bill_init:
        gb_seconds += res.n_cold * cold_start
    gb_seconds *= memory_gb
    request_fee = res.n_completed * C.LAMBDA_PRICE_PER_REQUEST
    duration_fee = gb_seconds * C.LAMBDA_PRICE_PER_GB_SECOND
    return request_fee + duration_fee


def vm_cost(res):
    if res.vm_seconds is not None:               # autoscaling VM: pay for the VM-seconds that existed
        return res.vm_seconds * C.VM_PRICE_PER_HOUR / 3600.0
    return res.n_instances * C.VM_PRICE_PER_HOUR * (res.horizon / 3600.0)


def breakeven_rate(num_vms=C.NUM_VMS, mean_exec=C.MEAN_EXEC_TIME_S,
                   memory_gb=C.FUNCTION_MEMORY_GB, p_cold=0.0, cold_start=C.COLD_START_S):
    """Analytical break-even arrival rate (req/s):
    serverless cost/s = rate * c_req   ;   VM cost/s = num_vms * price/3600
    With the default p_cold = 0 cold starts are ignored (original behaviour). Pass a
    cold-start probability (e.g. from analytical.estimate_cold_starts) to include the
    billed INIT time: each request then occupies mean_exec + p_cold * cold_start.
    """
    billed = mean_exec + (p_cold * cold_start if C.BILL_INIT_PHASE else 0.0)
    c_req = C.LAMBDA_PRICE_PER_REQUEST + billed * memory_gb * C.LAMBDA_PRICE_PER_GB_SECOND
    return num_vms * C.VM_PRICE_PER_HOUR / 3600.0 / c_req
