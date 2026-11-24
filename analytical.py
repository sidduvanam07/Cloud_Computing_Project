"""
analytical.py - Closed-form-style ESTIMATE of the cold-start probability (and the
cost / latency that follow from it), to be compared with the simulation.

The simulator is the reference; this module is the "analytical + simulation" pair
the project title asks for. It is deliberately a small approximation.

MODEL (piecewise-stationary M/G/infinity with keep-alive)
---------------------------------------------------------
Inputs: arrival-rate profile (rate per phase), mean execution time E, cold-start
time s, idle timeout T, arrival pattern ("poisson" or "regular"), optional
concurrency cap.

1. Occupancy.  A request occupies an instance for E_eff = E + p_cold * s seconds
   (a cold request also pays the start-up time). Offered concurrency is
   rho = lambda * E_eff   (Little's law).

2. Concurrency levels.  Because the simulator re-uses the MOST recently used warm
   instance, instance "j" is only needed when j-1 other instances are busy.
   An arrival sees k busy instances with probability pi_k:
       poisson arrivals : pi_k = Poisson(rho)
       regular arrivals : pi_k = Poisson-binomial with p_i = exp(-i/(lambda*E_eff))
       concurrency cap C: pi is truncated at k <= C and renormalised (Erlang-loss);
                          pi_C is the probability an arrival is rejected.
   Level j therefore receives requests at rate  lambda_j = lambda * pi_{j-1}.

3. Cold starts per level.  A request at level j is COLD when
     (a) it is the first use of level j (instance never existed), or
     (b) the idle gap since level j was last used exceeds T.
   POISSON arrivals: gaps are modelled as exponential with rate lambda_j (thinned
   Poisson), so over a phase of length L:
        E[#gaps longer than g] = lambda_j * max(L-g, 0) * exp(-lambda_j * g),
   with g = T + E_eff (arrival-to-arrival gap needed to leave the instance idle > T).
   Across phase boundaries (e.g. silence between bursts) the idle gap is computed
   from the expected last/first use time in the neighbouring phases.

3a. REGULAR arrivals (evenly spaced, spacing D = 1/lambda with +-`jitter` randomness,
   as produced by workload.steady).  The gap between two uses of level j is NOT
   exponential: it is G * D, where G is the number of arrivals between the two uses
   (G ~ Geometric(pi_{j-1}), because each arrival uses level j with probability
   pi_{j-1}).  The idle time of the instance is  gap - (s if its last use was cold) - X,
   with X the execution time of that use (exponential, mean E, as in the simulator).
   The next use is cold iff idle > T, i.e. iff  X < gap - T - s_prev.  So for a gap
   made of G jittered spacings (sum of G uniforms; exact for G = 1, normal
   approximation for G >= 2):
        c(theta) = E_gap[ 1 - exp(-(gap - theta)^+ / E) ],  theta = T (previous use warm)
                                                            or T + s (previous use cold)
   This is step-like: if every possible gap is <= T the cold probability is 0, if the
   gap clearly exceeds T + (a few E) it is 1, with a smooth transition only inside
   the +-jitter band and the execution-time tail.  A two-state (warm/cold) chain per
   level with transition probabilities c_w = c(T) and c_c = c(T+s) then gives the
   expected number of cold uses among M uses:
        sum_{m<=M} p_m = M*pi + (p1 - pi)(1 - kappa^M)/(1 - kappa),
        kappa = c_c - c_w,  pi = c_w/(1 - kappa),  p1 = 1 if the first use is cold else 0.
   The probability that level j is used at all is 1 - (1 - pi_{j-1})^n for n = lambda*L
   evenly spaced arrivals (not the exponential 1 - exp(-lambda_j L)).
   Boundary conditions: gap == T is WARM (the simulator destroys an instance only if
   idle > T, strictly); a gap shorter than the execution time is not a reuse of that
   level (the instance is still busy) and is ignored by this approximation.

3b. Ramp-up transient.  When a phase starts (e.g. a burst), the first `s` seconds
   see every arrival find all instances busy (the ones just created are still
   cold-starting), so levels 1..lambda*min(s, L) are certain to be used in that
   phase.  Levels that already exist and are still within the timeout stay warm,
   so only genuinely new levels count as cold starts.
4. p_cold = (sum over levels and phases of expected cold starts) / (completed requests).
   E_eff depends on p_cold, so steps 1-4 are iterated a few times (it converges fast).

ASSUMPTIONS / LIMITATIONS (also listed in README):
  * Poisson (or regular) arrivals within a phase; no correlation beyond the phase profile.
  * Regular pattern: the spacing jitter is the workload's configured `jitter` (default
    config.STEADY_JITTER), a workload parameter, never a simulator output. Gaps of G >= 2
    spacings use a normal approximation; consecutive uses of one level are treated as a
    two-state Markov chain; use of a level by successive arrivals is treated as independent.
  * Regular pattern with a concurrency cap: the truncated Poisson-binomial distribution
    is NOT an exact loss system, and is known to OVER-estimate rejections for evenly
    spaced traffic (see README / results/analytical_validation.csv).
  * Level j is treated as an independent Poisson stream (thinning) - an approximation,
    worst for the highest levels and for strongly clustered arrivals.
  * Stationarity inside each phase, except the ramp-up rule (3b). The ramp rule
    assumes arrivals in the first `s` seconds find no idle instance, so it
    slightly OVER-estimates cold starts when many idle instances already exist.
  * Execution times are treated through their mean only (E), except that the
    'regular' pattern assumes exponential durations like the simulator does.
  * Cost / latency predictions ignore queueing (serverless has none) and rounding.
"""
import math
import numpy as np
import config as C
import workload as W


# ---------------------------------------------------------------- helpers
def _poisson_pmf(rho, kmax):
    k = np.arange(kmax + 1)
    if rho <= 0:
        out = np.zeros(kmax + 1); out[0] = 1.0
        return out
    logp = -rho + k * math.log(rho) - np.array([math.lgamma(i + 1) for i in k])
    return np.exp(logp)


def _regular_pmf(lam, e_eff, kmax):
    """Number of busy instances seen by an arrival when arrivals are evenly spaced
    (gap 1/lam) and durations are exponential(mean e_eff): Poisson-binomial."""
    pmf = np.zeros(kmax + 1); pmf[0] = 1.0
    for i in range(1, 100000):
        p = math.exp(-i / (lam * e_eff))
        if p < 1e-12:
            break
        new = pmf * (1 - p)
        new[1:] += pmf[:-1] * p
        pmf = new
    return pmf


def _first_offset(lam, length):
    """E[X | X < length] for X ~ Exp(lam): expected time to the first (or, by
    symmetry, from the last) level-j request inside a phase, given there is one."""
    x = lam * length
    if x < 1e-9:
        return length / 2
    return 1.0 / lam - length * math.exp(-x) / (1.0 - math.exp(-x))


def concurrency_pmf(lam, e_eff, pattern, cap=None):
    """P(an arriving request finds k instances busy), k = 0..kmax (see step 2)."""
    rho = lam * e_eff
    kmax = int(rho + 10 * math.sqrt(rho) + 20)
    if cap is not None:
        kmax = min(kmax, cap)
    pmf = _regular_pmf(lam, e_eff, kmax) if pattern == "regular" else _poisson_pmf(rho, kmax)
    return pmf / pmf.sum()          # renormalise (matters when truncated by the cap)



# ---------------------------------------------------------------- regular-arrival gap model (3a)
def _phi(z):
    """Standard normal CDF for a numpy array."""
    return 0.5 * np.vectorize(math.erfc, otypes=[float])(-np.asarray(z, dtype=float) / math.sqrt(2.0))


def _c_uniform(a, b, theta, e):
    """c(theta) for ONE gap ~ Uniform(a, b): E[1 - exp(-(gap-theta)^+/e)] (closed form)."""
    if b <= theta:
        return 0.0
    if b - a < 1e-15:
        return 1.0 - math.exp(-(a - theta) / e) if a > theta else 0.0
    lo = max(a, theta)
    val = (b - lo) - e * (math.exp(-(lo - theta) / e) - math.exp(-(b - theta) / e))
    return val / (b - a)


def _c_normal(mu, sigma, theta, e):
    """c(theta) for gaps ~ Normal(mu, sigma) (arrays), closed form:
       Phi(z) - exp(-(mu-theta)/e + sigma^2/(2e^2)) * Phi(z - sigma/e),  z = (mu-theta)/sigma.
       When sigma >> e the exponential is a step and the first-order form
       Phi(z) - (e/sigma) * pdf(z) is used (avoids overflow)."""
    d = mu - theta
    z = d / sigma
    x = sigma / e
    out = np.zeros_like(d)
    ok = z > -8.0                                    # below this the probability is < 1e-15
    big = ok & (x > 20.0)
    small = ok & ~big
    out[big] = _phi(z[big]) - (e / sigma[big]) * np.exp(-0.5 * z[big] ** 2) / math.sqrt(2 * math.pi)
    out[small] = _phi(z[small]) - np.exp(-d[small] / e + 0.5 * x[small] ** 2) * _phi(z[small] - x[small])
    return np.clip(out, 0.0, 1.0)


def _first_g(f, hi):
    """Smallest integer g in [1, hi] with f(g) True (f monotone: False ... True)."""
    lo = 1
    if f(lo):
        return 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _level_cold_prob(p, delta, jitter, theta, e):
    """Sum over G >= 1 of P(G) * c_G(theta) for G ~ Geometric(p) arrivals between two uses
    of the same level, gap = sum of G spacings of mean `delta` and relative jitter `jitter`."""
    if p >= 1.0 - 1e-12:
        weights_g1 = 1.0
        if jitter == 0:
            return _c_uniform(delta, delta, theta, e)
        return _c_uniform(delta * (1 - jitter), delta * (1 + jitter), theta, e) * weights_g1
    lq = -math.log1p(-p)                              # -log(1-p) > 0
    root = lambda g: delta * jitter * math.sqrt(g / 3.0)
    # g0: smallest G whose gap can exceed theta; g1: smallest G whose gap is surely >> theta + tail
    g0 = _first_g(lambda g: g * delta + 8 * root(g) > theta, int(2 * theta / delta) + 50)
    if (g0 - 1) * lq > 40.0:                          # such long gaps (in arrivals) never occur
        return 0.0
    g1 = _first_g(lambda g: g * delta - 8 * root(g) - 28 * e >= theta, int(2 * (theta + 28 * e) / delta) + 50)
    g1 = min(g1, g0 + 400000)
    total = math.exp(-(g1 - 1) * lq)                  # all G >= g1: cold with probability ~1
    g = np.arange(g0, g1)
    if len(g):
        w = np.exp(-(g - 1) * lq) * p
        if jitter == 0:
            c = 1.0 - np.exp(-np.maximum(g * delta - theta, 0.0) / e)
        else:
            c = _c_normal(g * delta, np.maximum(delta * jitter * np.sqrt(g / 3.0), 1e-12), theta, e)
            if g0 == 1:                               # a single spacing is exactly uniform
                c[0] = _c_uniform(delta * (1 - jitter), delta * (1 + jitter), theta, e)
        total += float(np.sum(w * c))
    return total


def regular_expected_colds(p_use, n_arrivals, delta, jitter, idle_timeout, cold_start, mean_exec, first_cold,
                           force_use=False):
    """Expected cold starts at ONE concurrency level over a phase of evenly spaced arrivals.
       p_use       probability that a given arrival uses this level (pi_{j-1})
       n_arrivals  arrivals in the phase (lambda * L)
       first_cold  True if the first use of the level in this phase finds no warm instance
       force_use   True if the ramp-up rule (3b) makes use of this level certain (q = 1)."""
    n_j = n_arrivals * p_use                           # expected uses of this level
    q = 1.0 if force_use else 1.0 - (1.0 - min(p_use, 1.0)) ** n_arrivals    # P(level used at least once)
    if q <= 0.0:
        return 0.0, 0.0, 0.0
    p1 = 1.0 if first_cold else 0.0
    m = max(int(round(n_j / q)), 1)                    # expected uses given the level is used at all
    if m <= 1:                                         # a single use: cold iff it is the first use of the level
        return q * p1, q, 0.0
    c_w = _level_cold_prob(p_use, delta, jitter, idle_timeout, mean_exec)
    c_c = _level_cold_prob(p_use, delta, jitter, idle_timeout + cold_start, mean_exec)
    kappa = c_c - c_w                                  # in (-1, 0]
    pi = c_w / (1.0 - kappa)
    colds = m * pi + (p1 - pi) * (1.0 - kappa ** m) / (1.0 - kappa)
    return q * colds, q, c_w


# ---------------------------------------------------------------- main model
def estimate_cold_starts(phases, pattern="poisson", mean_exec=C.MEAN_EXEC_TIME_S,
                         cold_start=C.COLD_START_S, idle_timeout=C.IDLE_TIMEOUT_S,
                         max_concurrency=None, memory_gb=C.FUNCTION_MEMORY_GB, iterations=8,
                         jitter=C.STEADY_JITTER):
    """Return dict(p_cold, expected_cold_starts, expected_requests, p_reject,
    mean_response_s, cost_usd) for a piecewise-constant arrival-rate profile
    `phases = [(t0, t1, rate), ...]`.
      mean_response_s = E + p_cold * s       (serverless has no queue)
      cost_usd        = N_completed * (c_req + (E + p_cold*s [if INIT billed]) * memory_gb * c_gbs)
    which is the same formula cost.serverless_cost() applies to simulated counts."""
    if pattern not in ("poisson", "regular"):
        raise ValueError("pattern must be 'poisson' or 'regular'")
    if mean_exec <= 0 or cold_start < 0 or idle_timeout < 0:
        raise ValueError("mean_exec must be > 0; cold_start and idle_timeout must be >= 0")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    if memory_gb <= 0:
        raise ValueError("memory_gb must be > 0")
    if not (0.0 <= jitter < 1.0):
        raise ValueError("jitter must be in [0, 1)")
    active = [(a, b, r) for a, b, r in phases if r > 0 and b > a]
    p_phase = [0.0] * len(active)          # cold probability per phase (fixed-point variable)
    out = None
    for _ in range(iterations):
        e_eff = [mean_exec + pc * cold_start for pc in p_phase]
        pmfs = [concurrency_pmf(r, e, pattern, max_concurrency) for (a, b, r), e in zip(active, e_eff)]
        colds = [0.0] * len(active)
        # ramp-up (3b): levels 1..J are certain to be used in the first `s` seconds of a phase
        ramp = [int(r * min(cold_start, b - a)) for (a, b, r) in active]
        kmax = max([len(p) for p in pmfs] + ramp) if pmfs else 0
        # level j = k+1 may only be used if k < cap  (pmf[cap] is the rejection state)
        for k in range(kmax):
            if max_concurrency is not None and k >= max_concurrency:
                break
            last_end = None
            for i, ((a, b, r), e, pmf) in enumerate(zip(active, e_eff, pmfs)):
                lam_j = r * pmf[k] if k < len(pmf) else 0.0
                L = b - a
                m = lam_j * L
                forced = (k + 1) <= ramp[i]
                if m < 1e-12 and not forced:
                    continue
                if pattern == "regular":
                    # --- evenly spaced arrivals: deterministic spacing, step-like idle-timeout behaviour (3a)
                    p_use = pmf[k] if k < len(pmf) else 0.0
                    q = 1.0 if forced else 1.0 - (1.0 - min(p_use, 1.0)) ** (r * L)
                    off = (k + 1) / r if forced else min(1.0 / lam_j, L)
                    first_cold = last_end is None or (a + off) - last_end > idle_timeout
                    cs, _, _ = regular_expected_colds(p_use, r * L, 1.0 / r, jitter, idle_timeout, cold_start,
                                                      mean_exec, first_cold, force_use=forced)
                    colds[i] += cs
                else:
                    q = 1.0 if forced else 1.0 - math.exp(-m)    # P(level j used at least once in phase)
                    off = (k + 1) / r if forced else _first_offset(lam_j, L)
                    if last_end is None or (a + off) - last_end > idle_timeout:
                        colds[i] += q                            # first use, or instance expired between phases
                    g = idle_timeout + e                         # arrival-to-arrival gap that leaves it idle > T
                    colds[i] += lam_j * max(L - g, 0.0) * math.exp(-lam_j * g)
                if q >= 0.5:
                    last_end = (b - off) + e
        p_rej = [float(pmf[max_concurrency]) if (max_concurrency is not None and len(pmf) > max_concurrency) else 0.0
                 for pmf in pmfs]
        offered = [r * (b - a) for a, b, r in active]
        done = [o * (1 - pr) for o, pr in zip(offered, p_rej)]
        p_phase = [c / d if d > 0 else 0.0 for c, d in zip(colds, done)]
        out = (sum(colds), sum(offered), sum(done))
    n_cold, n_off, n_done = out if out else (0.0, 0.0, 0.0)
    p_cold = n_cold / n_done if n_done else 0.0
    e_overall = mean_exec + p_cold * cold_start
    billed = mean_exec + (p_cold * cold_start if C.BILL_INIT_PHASE else 0.0)
    cost = n_done * (C.LAMBDA_PRICE_PER_REQUEST + billed * memory_gb * C.LAMBDA_PRICE_PER_GB_SECOND)
    return {"p_cold": p_cold,
            "expected_cold_starts": n_cold,
            "expected_requests": n_off,
            "expected_completed": n_done,
            "p_reject": 1.0 - n_done / n_off if n_off else 0.0,
            "mean_response_s": e_overall,
            "cost_usd": cost}


def estimate_for_workload(name, mean_exec=C.MEAN_EXEC_TIME_S, cold_start=C.COLD_START_S,
                          idle_timeout=C.IDLE_TIMEOUT_S, max_concurrency=None,
                          memory_gb=C.FUNCTION_MEMORY_GB, **workload_kw):
    """Convenience: workload keyword arguments (rate, burst_size, idle, ...) are the same
    as for the generators in workload.py; the model parameters (mean_exec, cold_start,
    idle_timeout, max_concurrency, memory_gb) are the same as for the simulator/cost code."""
    jitter = workload_kw.pop("jitter", C.STEADY_JITTER)      # spacing randomness of the 'regular' pattern
    phases, pattern = W.profile_of(name, **workload_kw)
    return estimate_cold_starts(phases, pattern, mean_exec, cold_start, idle_timeout,
                                max_concurrency, memory_gb, jitter=jitter)


def simple_cold_probability(rate, mean_exec=C.MEAN_EXEC_TIME_S, idle_timeout=C.IDLE_TIMEOUT_S):
    """Single-instance special case, for intuition only (a standard one-line derivation, not
    taken from a cited source): one Poisson stream
    that never overlaps itself. A request is cold iff the gap since the previous
    arrival exceeds mean_exec + idle_timeout:   P(cold) = exp(-rate * (E + T)).
    (The full model above generalises this to many concurrent instances.)"""
    return math.exp(-rate * (mean_exec + idle_timeout))


if __name__ == "__main__":
    # Quick demo:  python analytical.py
    print(f"{'workload':<15}{'p_cold (model)':>16}{'cold starts':>14}{'p_reject':>10}{'cost (USD)':>12}")
    for n in W.GENERATORS:
        e = estimate_for_workload(n)
        print(f"{n:<15}{e['p_cold']*100:>15.3f}%{e['expected_cold_starts']:>14.1f}"
              f"{e['p_reject']*100:>9.2f}%{e['cost_usd']:>12.5f}")
