"""
Time-dependent (transient) availability via a continuous-time Markov chain.

The paper's `calc_subsystem_availability` gives only the STEADY-STATE
availability (the t -> infinity limit). This module reconstructs the
underlying CTMC so we can compute availability as a function of mission
time, A(t), and the interval (mission-average) availability over [0, T].

State = number of working units in the subsystem, j in {0, 1, ..., n};
the subsystem is DOWN only in state 0 (all units failed).

Transitions (chosen so the chain's stationary availability reproduces the
paper's closed form exactly -- see self-check in __main__):
  * failure  j -> j-1 : rate  lam * j**(-e)         (one operating unit fails)
  * repair   j -> j+1 : rate  min(n-j, r) * mu       (failed units under repair)
where e encodes the failure-dependency type: e=0 (independent),
e=-0.5 (type 1), e=+0.5 (type 2) -- matching the factorial(j)**e weighting
in calc_subsystem_availability.
"""
import numpy as np
from scipy.linalg import expm

from backend.core_math import (calc_subsystem_availability, calc_system_availability,
                               decode_solution)
from backend.data import SYSTEM_DATA


def _dep_exponent(dep_type):
    return {0: 0.0, 1: -0.5, 2: 0.5}[int(dep_type)]


def build_generator(n, r, lam, mu, dep_type):
    """Return the (n+1)x(n+1) CTMC generator Q (rows = working count 0..n)."""
    n = int(n)
    r = max(1, min(int(r), n))
    e = _dep_exponent(dep_type)
    size = n + 1
    Q = np.zeros((size, size))
    for j in range(size):
        if j >= 1:                                   # failure: j -> j-1
            Q[j, j - 1] += lam * (j ** (-e))
        if j <= n - 1:                               # repair: j -> j+1
            Q[j, j + 1] += min(n - j, r) * mu
    for j in range(size):
        Q[j, j] = -np.sum(Q[j, :])                   # rows sum to zero
    return Q


def steady_state_availability(n, r, lam, mu, dep_type):
    """Stationary availability solved directly from Q (used to validate
    against the paper's closed-form formula)."""
    Q = build_generator(n, r, lam, mu, dep_type)
    size = Q.shape[0]
    # Solve pi @ Q = 0 with sum(pi) = 1.
    A = np.vstack([Q.T, np.ones(size)])
    b = np.zeros(size + 1); b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return 1.0 - pi[0]


def instantaneous_availability(n, r, lam, mu, dep_type, t):
    """A(t): probability the subsystem is up at time t, starting fully working."""
    Q = build_generator(n, r, lam, mu, dep_type)
    size = Q.shape[0]
    p0 = np.zeros(size); p0[n] = 1.0                 # start: all units working
    pt = p0 @ expm(Q * t)
    return 1.0 - pt[0]


def interval_availability(n, r, lam, mu, dep_type, T):
    """Mission-average availability over [0, T] = (1/T) * integral_0^T A(t) dt.

    Uses the block-matrix identity:
        expm([[Q, I],[0, 0]] * T) = [[expm(QT), M(T)], [0, I]],
    where M(T) = integral_0^T expm(Q t) dt -- one expm call, exact.
    """
    Q = build_generator(n, r, lam, mu, dep_type)
    size = Q.shape[0]
    block = np.zeros((2 * size, 2 * size))
    block[:size, :size] = Q
    block[:size, size:] = np.eye(size)
    M = expm(block * T)[:size, size:]                # integral of expm(Qt)
    p0 = np.zeros(size); p0[n] = 1.0
    down_time = (p0 @ M)[0]                           # expected time spent down
    return 1.0 - down_time / T


def system_availability_at(n_vars, r_vars, m, sys_type, t):
    """System availability at time t: each subsystem's A_i(t) combined through
    the structure function. Exact at each instant because the subsystems are
    independent CTMCs (same combination the paper uses for steady state)."""
    A_subs = []
    for i in range(m):
        dep_type, lam, mu, _, _ = SYSTEM_DATA[i + 1]
        A_subs.append(instantaneous_availability(n_vars[i], r_vars[i], lam, mu, dep_type, t))
    return calc_system_availability(A_subs, sys_type)


def mission_penalised_objective(x, m, sys_type, As_min, T):
    """Like core_math.penalised_objective, but the availability constraint is
    evaluated at mission time T (point availability) instead of at steady state.
    Minimise cost subject to system A(T) >= As_min."""
    n_vars, r_vars = decode_solution(x, m)
    cost = 0.0
    penalty = 0.0
    for i in range(m):
        _, _, _, c_comp, c_rep = SYSTEM_DATA[i + 1]
        cost += c_comp * n_vars[i] + c_rep * r_vars[i]
        raw_n = int(round(x[2 * i])); raw_r = int(round(x[2 * i + 1]))
        if raw_r > raw_n:
            penalty += 1e5 * (raw_r - raw_n)
    As_T = system_availability_at(n_vars, r_vars, m, sys_type, T)
    if As_T < As_min:
        penalty += 1e8 * (As_min - As_T)
    return cost + penalty


if __name__ == "__main__":
    # Validation: the CTMC's steady state must reproduce the paper's formula,
    # and A(t) must converge to it as t grows.
    rng = np.random.default_rng(0)
    print("Validating transient CTMC against the paper's steady-state formula")
    print(f"{'n':>3}{'r':>3}{'dep':>4}{'lam':>7}{'mu':>7}{'paper A_ss':>13}{'CTMC A_ss':>12}{'A(t=1e4)':>12}{'err':>11}")
    max_err = 0.0
    for _ in range(12):
        n = int(rng.integers(2, 9))
        r = int(rng.integers(1, n + 1))
        dep = int(rng.integers(0, 3))
        lam = float(rng.uniform(0.01, 0.1))
        mu = float(rng.uniform(0.1, 0.3))
        paper = calc_subsystem_availability(n, r, lam, mu, dep)
        ctmc = steady_state_availability(n, r, lam, mu, dep)
        a_big = instantaneous_availability(n, r, lam, mu, dep, 1e4)
        err = abs(paper - ctmc)
        max_err = max(max_err, err)
        print(f"{n:>3}{r:>3}{dep:>4}{lam:>7.3f}{mu:>7.3f}{paper:>13.6f}{ctmc:>12.6f}{a_big:>12.6f}{err:>11.2e}")
    print(f"\nmax |paper - CTMC| = {max_err:.2e}  ->  {'MATCH' if max_err < 1e-6 else 'MISMATCH'}")
