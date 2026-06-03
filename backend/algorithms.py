import numpy as np
from backend.core_math import decode_solution, penalised_objective

def optimize_de(m, sys_type, As_min, pop_size=100, max_gen=200, track_history=False):
    dim    = m * 2
    bounds = np.array([[1, 8]] * dim, dtype=float)
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, dim))
    fitness = np.array([penalised_objective(ind, m, sys_type, As_min) for ind in pop])

    best_idx = np.argmin(fitness)
    best_x   = pop[best_idx].copy()
    best_fit = fitness[best_idx]
    history = [best_fit] if track_history else None
    CR = 0.9

    for gen in range(max_gen):
        for i in range(pop_size):
            F = np.random.uniform(0.5, 1.0)
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
            mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

            cross_mask = np.random.rand(dim) < CR
            cross_mask[np.random.randint(0, dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])

            trial_fit = penalised_objective(trial, m, sys_type, As_min)

            if trial_fit <= fitness[i]:
                pop[i]     = trial
                fitness[i] = trial_fit
                if trial_fit < best_fit:
                    best_x   = trial.copy()
                    best_fit = trial_fit

        if track_history:
            history.append(best_fit)

    n_vars, r_vars = decode_solution(best_x, m)
    return n_vars, r_vars, history

def optimize_shade(m, sys_type, As_min, pop_size=100, max_gen=200, track_history=False):
    """
    SHADE — Success-History based Adaptive Differential Evolution
    (Tanabe & Fukunaga, IEEE CEC 2013).

    A self-adaptive upgrade over optimize_de's fixed F=U(0.5,1)/CR=0.9:
      * F and CR are sampled per-trial from a success-history memory that is
        continually updated with the (F, CR) values that produced improving
        offspring, weighted by how much they improved the objective.
      * Mutation is current-to-pbest/1 with an external archive of replaced
        parents, which preserves diversity and avoids premature convergence.
    """
    dim    = m * 2
    bounds = np.array([[1, 8]] * dim, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]

    pop     = np.random.uniform(lo, hi, (pop_size, dim))
    fitness = np.array([penalised_objective(ind, m, sys_type, As_min) for ind in pop])

    best_idx = np.argmin(fitness)
    best_x   = pop[best_idx].copy()
    best_fit = fitness[best_idx]
    history  = [best_fit] if track_history else None

    H    = pop_size                 # success-history memory size
    M_CR = np.full(H, 0.5)
    M_F  = np.full(H, 0.5)
    k    = 0                        # memory update pointer
    archive = []                    # external archive of replaced parents
    p_min = 2.0 / pop_size

    for gen in range(max_gen):
        S_CR, S_F, weights = [], [], []          # successful params this generation
        sorted_idx = np.argsort(fitness)          # for pbest selection

        for i in range(pop_size):
            ri = np.random.randint(H)
            # CR ~ Normal(M_CR, 0.1), clipped to [0, 1]
            cr = float(np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0))
            # F ~ Cauchy(M_F, 0.1), resampled until > 0, capped at 1
            f = -1.0
            while f <= 0.0:
                f = M_F[ri] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
            f = min(f, 1.0)

            # current-to-pbest/1: pbest drawn from the top p% individuals
            p           = np.random.uniform(p_min, 0.2)
            pbest_count = max(1, int(round(p * pop_size)))
            x_pbest     = pop[sorted_idx[np.random.randint(pbest_count)]]

            # r1 from population; r2 from population ∪ archive (distinct from i, r1)
            idxs = list(range(pop_size)); idxs.remove(i)
            r1   = int(np.random.choice(idxs))
            n_union = pop_size + len(archive)
            while True:
                r2 = np.random.randint(n_union)
                if r2 != i and r2 != r1:
                    break
            x_r1 = pop[r1]
            x_r2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]

            mutant = pop[i] + f * (x_pbest - pop[i]) + f * (x_r1 - x_r2)
            mutant = np.clip(mutant, lo, hi)

            cross = np.random.rand(dim) < cr
            cross[np.random.randint(dim)] = True   # ensure ≥1 gene from the mutant
            trial = np.where(cross, mutant, pop[i])

            trial_fit = penalised_objective(trial, m, sys_type, As_min)

            if trial_fit < fitness[i]:
                archive.append(pop[i].copy())      # replaced parent -> archive
                S_CR.append(cr); S_F.append(f)
                weights.append(fitness[i] - trial_fit)
                pop[i]     = trial
                fitness[i] = trial_fit
                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_x   = trial.copy()
            elif trial_fit == fitness[i]:
                pop[i] = trial                     # accept equal moves (drift)

        # keep the archive bounded
        while len(archive) > pop_size:
            archive.pop(np.random.randint(len(archive)))

        # update success-history memory from this generation's wins
        if S_CR:
            w  = np.array(weights, dtype=float)
            w  = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
            scr = np.array(S_CR); sf = np.array(S_F)
            M_CR[k] = float(np.sum(w * scr))                       # weighted arithmetic mean
            M_F[k]  = float(np.sum(w * sf * sf) / np.sum(w * sf))  # weighted Lehmer mean
            k = (k + 1) % H

        if track_history:
            history.append(best_fit)

    n_vars, r_vars = decode_solution(best_x, m)
    return n_vars, r_vars, history

def optimize_mrfo(m, sys_type, As_min, pop_size=100, max_gen=200, track_history=False):
    dim    = m * 2
    bounds = np.array([[1, 8]] * dim, dtype=float)
    X       = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, dim))
    fitness = np.array([penalised_objective(ind, m, sys_type, As_min) for ind in X])

    best_idx = np.argmin(fitness)
    X_best   = X[best_idx].copy()
    fit_best = fitness[best_idx]
    history = [fit_best] if track_history else None
    S = 2.0

    for t in range(max_gen):
        rand_val = np.random.rand()
        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            if rand_val < 0.5:
                if i == 0:
                    X_new = X[i] + r1 * (X_best - X[i])
                else:
                    X_new = X[i] + r1 * (X[i-1] - X[i]) + r2 * (X_best - X[i])
            else:
                if t / max_gen < 0.5:
                    X_rand = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
                    X_ref  = X_rand
                else:
                    X_ref  = X_best

                beta  = 2 * np.exp(r1 * (max_gen - t) / max_gen) * np.sin(2 * np.pi * r1)
                X_new = X_ref + r1 * (X_ref - X[i]) + beta * (X_ref - X[i])

            X_new = np.clip(X_new, bounds[:, 0], bounds[:, 1])
            fit_new = penalised_objective(X_new, m, sys_type, As_min)

            if fit_new < fitness[i]:
                X[i]       = X_new
                fitness[i] = fit_new
                if fit_new < fit_best:
                    X_best   = X_new.copy()
                    fit_best = fit_new

        for i in range(pop_size):
            r3    = np.random.rand(dim)
            r4    = np.random.rand(dim)
            X_new = X[i] + S * (r3 * X_best - r4 * X[i])
            X_new = np.clip(X_new, bounds[:, 0], bounds[:, 1])
            fit_new = penalised_objective(X_new, m, sys_type, As_min)

            if fit_new < fitness[i]:
                X[i]       = X_new
                fitness[i] = fit_new
                if fit_new < fit_best:
                    X_best   = X_new.copy()
                    fit_best = fit_new

        if track_history:
            history.append(fit_best)

    n_vars, r_vars = decode_solution(X_best, m)
    return n_vars, r_vars, history

def optimize_sfla(m_subsys, sys_type, As_min, pop_size=100, num_memeplexes=5, local_iters=10, max_gen=200, track_history=False):
    dim    = m_subsys * 2
    bounds = np.array([[1, 8]] * dim, dtype=float)
    q      = pop_size // num_memeplexes

    frogs   = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, dim))
    fitness = np.array([penalised_objective(f, m_subsys, sys_type, As_min) for f in frogs])

    sort_idx = np.argsort(fitness)
    frogs    = frogs[sort_idx]
    fitness  = fitness[sort_idx]

    X_global_best  = frogs[0].copy()
    fit_global_best = fitness[0]
    history = [fit_global_best] if track_history else None

    for shuffle in range(max_gen):
        for p in range(num_memeplexes):
            mem_idx = list(range(p, pop_size, num_memeplexes))
            if len(mem_idx) < 2:
                continue

            for _ in range(local_iters):
                sub_fitness = fitness[mem_idx]
                local_best_pos  = mem_idx[np.argmin(sub_fitness)]
                local_worst_pos = mem_idx[np.argmax(sub_fitness)]

                X_lb = frogs[local_best_pos]
                X_lw = frogs[local_worst_pos]

                step  = np.random.rand(dim) * (X_lb - X_lw)
                X_new = np.clip(X_lw + step, bounds[:, 0], bounds[:, 1])
                fit_new = penalised_objective(X_new, m_subsys, sys_type, As_min)

                if fit_new < fitness[local_worst_pos]:
                    frogs[local_worst_pos]   = X_new
                    fitness[local_worst_pos] = fit_new
                else:
                    step  = np.random.rand(dim) * (X_global_best - X_lw)
                    X_new = np.clip(X_lw + step, bounds[:, 0], bounds[:, 1])
                    fit_new = penalised_objective(X_new, m_subsys, sys_type, As_min)

                    if fit_new < fitness[local_worst_pos]:
                        frogs[local_worst_pos]   = X_new
                        fitness[local_worst_pos] = fit_new
                    else:
                        X_rand = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
                        frogs[local_worst_pos]   = X_rand
                        fitness[local_worst_pos] = penalised_objective(X_rand, m_subsys, sys_type, As_min)

        sort_idx = np.argsort(fitness)
        frogs    = frogs[sort_idx]
        fitness  = fitness[sort_idx]

        if fitness[0] < fit_global_best:
            fit_global_best = fitness[0]
            X_global_best   = frogs[0].copy()

        if track_history:
            history.append(fit_global_best)

    n_vars, r_vars = decode_solution(X_global_best, m_subsys)
    return n_vars, r_vars, history