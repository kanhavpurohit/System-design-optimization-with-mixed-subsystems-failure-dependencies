"""
DE vs SHADE comparison with a Wilcoxon rank-sum significance test.

Runs both optimizers N independent times on each system/target, records the
best *feasible* cost per run, and reports mean/median/std plus the Wilcoxon
rank-sum p-value -- the standard test in the metaheuristics literature for
comparing two algorithms across independent runs.

    python shade_benchmark.py
"""
import numpy as np
from scipy import stats

from backend.algorithms import optimize_de, optimize_shade
from backend.core_math import evaluate_solution


def best_feasible_cost(opt_fn, m, sys_type, As_min):
    """Run one optimization; return the solution cost if it meets the
    availability target, else +inf (infeasible)."""
    n_v, r_v, _ = opt_fn(m, sys_type, As_min)
    x = np.zeros(m * 2)
    x[0::2] = n_v
    x[1::2] = r_v
    cost, avail = evaluate_solution(x, m, sys_type)
    return cost if avail >= As_min else np.inf


def run_comparison(num_runs=25, pop_size=100, max_gen=200, As_min=0.99, seed=42):
    np.random.seed(seed)
    systems = [
        {'name': 'Complex bridge network', 'm': 5,  'type': 'bridge'},
        {'name': 'Parallel-series system', 'm': 10, 'type': 'series_parallel'},
    ]
    algos = {
        'DE':    lambda m, t, a: optimize_de(m, t, a, pop_size=pop_size, max_gen=max_gen),
        'SHADE': lambda m, t, a: optimize_shade(m, t, a, pop_size=pop_size, max_gen=max_gen),
    }

    print(f"\nDE vs SHADE  -  {num_runs} runs, pop={pop_size}, gen={max_gen}, A_s,min={As_min}")
    print("=" * 80)

    for cfg in systems:
        results = {}
        for name, fn in algos.items():
            results[name] = np.array(
                [best_feasible_cost(fn, cfg['m'], cfg['type'], As_min) for _ in range(num_runs)],
                dtype=float,
            )

        de_f = results['DE'][np.isfinite(results['DE'])]
        sh_f = results['SHADE'][np.isfinite(results['SHADE'])]
        _, p = stats.ranksums(de_f, sh_f)

        print(f"\n>>> {cfg['name']} (m={cfg['m']})")
        print(f"  {'Algo':<6}{'best':>9}{'mean':>11}{'median':>9}{'std':>9}{'feasible':>11}")
        for name in ('DE', 'SHADE'):
            a = results[name][np.isfinite(results[name])]
            print(f"  {name:<6}{a.min():>9.0f}{a.mean():>11.1f}{np.median(a):>9.0f}"
                  f"{a.std():>9.2f}{len(a):>8}/{num_runs}")
        winner = 'SHADE' if np.median(sh_f) < np.median(de_f) else 'DE'
        sig = 'SIGNIFICANT (p<0.05)' if p < 0.05 else 'not significant'
        print(f"  Wilcoxon rank-sum: p = {p:.4g}  ->  {sig};  lower median = {winner}")


if __name__ == '__main__':
    run_comparison()
