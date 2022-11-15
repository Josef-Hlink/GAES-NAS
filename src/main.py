import os
from time import perf_counter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ioh import get_problem
import ioh

from evolution_strategies import EvolutionStrategies
from utils import get_directories


def main():
    global dirs, seed
    dirs = get_directories(__file__)
    seed = 42
    np.random.seed(seed)
    tic = perf_counter()
    for experiment_id in range(1, 25):
        run_experiment(experiment_id)
    toc = perf_counter()
    print(f'\nTotal time elapsed: {toc - tic:.3f} seconds')
    

def run_experiment(problem_id: int) -> None:
    problem = get_problem(problem_id, dimension=5)
    problem_name = problem.meta_data.name
    print(f'Running experiment for {problem_name}...')
    taus = [np.sqrt(2 / problem.meta_data.n_variables) * i for i in [0.1, 0.5, 1]]
    recombinations = ['d', 'dg', 'i', 'ig']
    sigmas = [0.001, 0.01, 0.1]
    combinations = product(recombinations, sigmas, taus)
    
    df_path = dirs['pkl']+f'ES-{problem_name}-{seed}.pkl'
    if not os.path.exists(df_path):
        print('No data found, creating dataframe...')
        tic = perf_counter()
        res_df = create_dataframe(problem, combinations)
        toc = perf_counter()
        print(f'Runtime: {toc - tic:.3f} seconds')
        res_df.to_pickle(df_path)
    else:
        res_df = pd.read_pickle(df_path)
    
    fig = create_plot(res_df, recombinations, sigmas, taus, problem_name)
    fig.savefig(dirs['plots']+f'ES-{problem_name}-{seed}.png', dpi=300)
    plt.close(fig)
    return


def create_dataframe(problem: ioh.ProblemType, combinations: product) -> pd.DataFrame:
    df = pd.DataFrame(columns=['recombination', 'sigma', 'tau', 'best_fitness', 'history'])
    df.index.name = 'run_id'

    for recombination, sigma_, tau_ in combinations:
        run_id = f'{recombination}-{sigma_}-{tau_:.3f}'
        es = EvolutionStrategies(
            problem = problem,
            pop_size = 100,
            mu_ = 40,
            lambda_ = 60,
            tau_ = tau_,
            sigma_ = sigma_,
            budget = 5_000,
            recombination = recombination,
            individual_sigmas = True,
            run_id = run_id,
            verbose = False
        )
        x_opt, f_opt, history = es.optimize(return_history=True)
        df.loc[run_id] = (recombination, sigma_, tau_, f_opt, history)
    return df


def create_plot(df: pd.DataFrame, recombinations: list, sigmas: list, taus: list, problem_name: str) -> plt.Figure:
    rec_names = {
        'd': 'Discrete',
        'i': 'Intermediate',
        'dg': 'Discrete Global',
        'ig': 'Intermediate Global'
    }

    fig, axes = plt.subplots(len(recombinations), len(sigmas), figsize=(10, 10))
    # plot histories, rows are recombinations, columns are sigmas, line colors are taus
    for i, recombination in enumerate(recombinations):
        for j, sigma_ in enumerate(sigmas):
            ax: plt.Axes = axes[i, j]
            for tau_ in taus:
                run_id = f'{recombination}-{sigma_}-{tau_:.3f}'
                history = df.loc[run_id]['history']
                ax.plot(history, label=f'tau: {tau_:.3f}')
            if i == 0:
                ax.set_title(r'$\sigma = $' + f'{sigma_}')
            if j == 0:
                ax.set_ylabel(rec_names[recombination])
            ax.tick_params(
                axis='both',
                which = 'both',
                bottom = False,
                left = False,
                labelbottom = True,
                labelleft = True,
                labelsize = 6
            )
    axes[0, 0].legend()
    fig.suptitle(f'Evolution Strategies ({problem_name})', fontsize=16, weight='bold')
    fig.supxlabel('generation')
    fig.supylabel('fitness')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    main()
