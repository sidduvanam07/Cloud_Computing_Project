"""
config.py - ALL experiment parameters live here.

Rule: no other file hard-codes a number. If you want to change an
assumption (cold start time, prices, ...), change it here only.

Legend:
  [PRICE]      real published price (AWS us-east-1, Linux/x86), ignoring free tier
  [ASSUMPTION] our modelling choice, justified/swept in the report
"""

# ---------------- Reproducibility ----------------
SEEDS = list(range(10))          # [ASSUMPTION] 10 repeated runs per experiment
DEFAULT_SEED = 0

# ---------------- Request (function) properties ----------------
MEAN_EXEC_TIME_S = 0.200         # [ASSUMPTION] mean execution time = 200 ms
FUNCTION_MEMORY_GB = 0.5         # [ASSUMPTION] 512 MB function

# ---------------- Serverless model ----------------
COLD_START_S = 0.500             # [ASSUMPTION] extra init time for a new instance (sweep 0.2-1.5)
IDLE_TIMEOUT_S = 600.0           # [ASSUMPTION] warm instance destroyed after 10 min idle (sweep 60-1200)
MAX_CONCURRENCY = 1000           # [ASSUMPTION] max simultaneous instances

# ---------------- VM model ----------------
NUM_VMS = 2                      # [ASSUMPTION] fixed-capacity baseline
SLOTS_PER_VM = 2                 # t3.medium has 2 vCPUs -> 2 requests in parallel
QUEUE_LIMIT = 100                # [ASSUMPTION] requests beyond this are rejected

# ---------------- Pricing ----------------
LAMBDA_PRICE_PER_REQUEST = 0.20 / 1_000_000   # [PRICE] $0.20 per million requests
LAMBDA_PRICE_PER_GB_SECOND = 0.0000166667     # [PRICE] x86 duration price
VM_PRICE_PER_HOUR = 0.0416                    # [PRICE] t3.medium on-demand
BILL_INIT_PHASE = True           # [PRICE RULE] AWS bills the cold-start (INIT) phase since 1 Aug 2025

# ---------------- Workload defaults ----------------
SIM_DURATION_S = 600.0           # [ASSUMPTION] 10 simulated minutes per run

# Steady
STEADY_RATE = 5.0                # requests/second
STEADY_JITTER = 0.10             # +-10% randomness on each interval

# Bursty: low background traffic + short intense bursts
BURSTY_BASE_RATE = 1.0           # requests/second between bursts
BURSTY_NUM_BURSTS = 3
BURSTY_BURST_SIZE = 300          # requests per burst
BURSTY_BURST_WINDOW_S = 5.0      # each burst arrives within 5 seconds

# Variable: rate rises and falls smoothly (sine wave)
VARIABLE_MEAN_RATE = 5.0         # requests/second (average)
VARIABLE_AMPLITUDE = 0.8         # rate swings between 20% and 180% of mean
VARIABLE_PERIOD_S = 200.0        # one full up-down cycle

# Idle-to-burst: short warm-up, long silence, then sudden heavy traffic
WARMUP_DURATION_S = 5.0          # traffic at the start so that some warm instances exist
IDLE_DURATION_S = 900.0          # burst starts at t=900 s (15 min > 10 min idle timeout)
IDLE_BURST_RATE = 30.0           # requests/second (also used during warm-up)
IDLE_BURST_DURATION_S = 20.0


# =====================================================================
# ADDED IN THE IMPROVEMENT PASS (everything above is unchanged)
# =====================================================================

# ---------------- Evaluation metrics ----------------
SLA_RESPONSE_S = 1.0             # [ASSUMPTION] a request meets its SLA if it COMPLETES within 1 s.
                                 # Rejected requests always count as SLA violations.
REF_CAPACITY_SLOTS = NUM_VMS * SLOTS_PER_VM   # reference capacity (4 slots) for the common
                                 # utilization metric = the original fixed-VM baseline

# ---------------- VM hardware (used for memory-aware slot packing) ----------------
VM_MEMORY_GB = 4.0               # [PRICE] t3.medium has 4 GiB. A VM can host at most
                                 # floor(VM_MEMORY_GB / function memory) requests at once.

# ---------------- Autoscaling VM model (vm_autoscale.py) ----------------
AUTOSCALE_MIN_VMS = 1            # [ASSUMPTION] never fewer than this many VMs (>= 1)
AUTOSCALE_MAX_VMS = 8            # [ASSUMPTION] never more than this many (running + booting)
AUTOSCALE_PROVISION_DELAY_S = 60.0   # [ASSUMPTION] launch -> ready to serve (boot + app start)
AUTOSCALE_EVAL_INTERVAL_S = 10.0     # [ASSUMPTION] how often the autoscaler looks at the metrics
AUTOSCALE_UTIL_UP = 0.70         # [ASSUMPTION] scale OUT if slot utilization over the last interval >= 70 %
AUTOSCALE_QUEUE_UP = 10          # [ASSUMPTION] ... or if >= 10 requests are waiting
AUTOSCALE_UTIL_DOWN = 0.30       # [ASSUMPTION] scale IN only while utilization stays < 30 % ...
AUTOSCALE_SCALE_IN_AFTER_S = 120.0   # [ASSUMPTION] ... continuously for this long (and the queue is empty)
AUTOSCALE_STEP = 1               # [ASSUMPTION] VMs added per scale-out decision
AUTOSCALE_OUT_COOLDOWN_S = 30.0  # [ASSUMPTION] minimum time between two scale-out decisions
AUTOSCALE_MIN_BILLED_S = 60.0    # [PRICE RULE] EC2 Linux per-second billing, 60 s minimum per VM
AUTOSCALE_BILL_PROVISIONING = True   # [ASSUMPTION] a VM is paid for from launch (boot time is billed)


def validate_config():
    """Fail fast on impossible parameter combinations. Called when config is imported."""
    def need(cond, msg):
        if not cond:
            raise ValueError(f"config.py: {msg}")
    need(MEAN_EXEC_TIME_S > 0, "MEAN_EXEC_TIME_S must be > 0")
    need(FUNCTION_MEMORY_GB > 0, "FUNCTION_MEMORY_GB must be > 0")
    need(COLD_START_S >= 0 and IDLE_TIMEOUT_S >= 0, "COLD_START_S and IDLE_TIMEOUT_S must be >= 0")
    need(MAX_CONCURRENCY >= 1, "MAX_CONCURRENCY must be >= 1")
    need(NUM_VMS >= 1 and SLOTS_PER_VM >= 1 and QUEUE_LIMIT >= 0, "NUM_VMS/SLOTS_PER_VM >= 1 and QUEUE_LIMIT >= 0")
    need(SLA_RESPONSE_S > 0, "SLA_RESPONSE_S must be > 0")
    need(VM_MEMORY_GB >= FUNCTION_MEMORY_GB, "a function must fit in one VM (VM_MEMORY_GB >= FUNCTION_MEMORY_GB)")
    need(1 <= AUTOSCALE_MIN_VMS <= AUTOSCALE_MAX_VMS, "need 1 <= AUTOSCALE_MIN_VMS <= AUTOSCALE_MAX_VMS")
    need(AUTOSCALE_PROVISION_DELAY_S >= 0, "AUTOSCALE_PROVISION_DELAY_S must be >= 0")
    need(AUTOSCALE_EVAL_INTERVAL_S > 0, "AUTOSCALE_EVAL_INTERVAL_S must be > 0")
    need(0 < AUTOSCALE_UTIL_DOWN < AUTOSCALE_UTIL_UP <= 1, "need 0 < AUTOSCALE_UTIL_DOWN < AUTOSCALE_UTIL_UP <= 1")
    need(AUTOSCALE_QUEUE_UP >= 1, "AUTOSCALE_QUEUE_UP must be >= 1")
    need(AUTOSCALE_SCALE_IN_AFTER_S >= 0 and AUTOSCALE_OUT_COOLDOWN_S >= 0, "scale-in period and cooldown must be >= 0")
    need(AUTOSCALE_STEP >= 1, "AUTOSCALE_STEP must be >= 1")
    need(AUTOSCALE_MIN_BILLED_S >= 0, "AUTOSCALE_MIN_BILLED_S must be >= 0")


validate_config()
