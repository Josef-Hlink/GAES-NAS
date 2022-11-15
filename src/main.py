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
    dirs = get_directories(__file__)
    sphere = get_problem('Sphere', dimension=5)
    taus = [np.sqrt(2 / sphere.meta_data.n_variables) * i for i in [0.1, 0.5, 1]]
    recombinations = ['d', 'i', 'dg', 'ig']
    sigmas = [0.001, 0.01, 0.1]
    combinations = product(recombinations, sigmas, taus)
    
    df_path = dirs['results']+'ES.pkl'
    if not os.path.exists(df_path):
        tic = perf_counter()
        res_df = create_dataframe(sphere, combinations)
        toc = perf_counter()
        print(f'time: {toc - tic:.3f} seconds')
        res_df.to_pickle(df_path)
    else:
        res_df = pd.read_pickle(df_path)
    
    fig = create_plot(res_df, recombinations, sigmas, taus)
    fig.savefig(dirs['plots']+'ES.png', dpi=300)
    

def create_plot(df: pd.DataFrame, recombinations: list, sigmas: list, taus: list) -> plt.Figure:
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
            ax.legend()
            ax.tick_params(
                axis='both',
                which = 'both',
                bottom = False,
                left = False,
                labelbottom = True,
                labelleft = True,
                labelsize = 6
            )
    fig.suptitle('Evolution Strategies', fontsize=16, weight='bold')
    fig.supxlabel('generation')
    fig.supylabel('fitness')
    fig.tight_layout()
    return fig

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
            run_id = run_id
        )
        x_opt, f_opt, history = es.optimize(return_history=True)
        print(f'f_opt: {f_opt:.5f}')
        print(f'x_opt: {x_opt}')
        df.loc[run_id] = (recombination, sigma_, tau_, f_opt, history)
    return df


if __name__ == '__main__':
    main()
