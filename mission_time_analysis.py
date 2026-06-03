"""
Mission-time analysis -- extending the paper from an infinite horizon to a
finite mission.

The paper optimises STEADY-STATE availability (mission length = infinity).
Using the validated transient Markov model (backend.transient), this script
finds the cheapest design whose system availability at a finite mission time T
meets the target, for a range of T.

Finding: short missions tolerate less redundancy (failures are unlikely early
on), so they are cheaper; as T grows the optimum rises and converges to the
paper's steady-state design -- i.e. the paper's result is the worst case
(longest mission).

    python mission_time_analysis.py
"""
import numpy as np

from backend.algorithms import optimize_shade
from backend.core_math import evaluate_solution
from backend.transient import system_availability_at, mission_penalised_objective


def optimise_for_mission(m, sys_type, As_min, T, pop_size=60, max_gen=120, seed=0):
    np.random.seed(seed)
    if np.isfinite(T):
        obj = lambda x: mission_penalised_objective(x, m, sys_type, As_min, T)
    else:
        obj = None  # default steady-state objective (the paper's setting)
    n_v, r_v, _ = optimize_shade(m, sys_type, As_min,
                                 pop_size=pop_size, max_gen=max_gen, objective=obj)
    x = np.zeros(m * 2); x[0::2] = n_v; x[1::2] = r_v
    cost, A_ss = evaluate_solution(x, m, sys_type)
    A_T = system_availability_at(n_v, r_v, m, sys_type, T) if np.isfinite(T) else A_ss
    return n_v, r_v, cost, A_ss, A_T


def main():
    m, sys_type, As_min = 5, 'bridge', 0.99
    print(f"Mission-time optimisation -- bridge system (m={m}), target system A >= {As_min}\n")
    print(f"{'mission T':>12}{'opt cost':>10}{'A(T)':>9}{'A_ss':>9}   design")
    print("-" * 78)
    for T in [5, 20, 50, 100, 300, np.inf]:
        n_v, r_v, cost, A_ss, A_T = optimise_for_mission(m, sys_type, As_min, T)
        label = "inf (paper)" if not np.isfinite(T) else f"{T:.0f}"
        print(f"{label:>12}{cost:>10.0f}{A_T:>9.4f}{A_ss:>9.4f}   "
              f"n={list(map(int, n_v))} r={list(map(int, r_v))}")
    print("\nShorter missions -> cheaper designs; T -> inf recovers the paper's design.")


if __name__ == "__main__":
    main()
