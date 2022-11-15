import os
import argparse
from time import perf_counter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ioh import get_problem
import ioh

from genetic_algorithm import GeneticAlgorithm
from utils import get_directories, ParseWrapper


def main():
    global dirs, seed, verbose
    dirs = get_directories(__file__)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(parser)()
    seed = args.get('seed')
    verbose = args.get('verbose')
    plot_target = 'GA' + os.sep + f'{seed}'
    if not os.path.exists(dirs['plots']+plot_target):
        os.mkdir(dirs['plots']+plot_target)
    dirs['plots'] += plot_target + os.sep
    np.random.seed(seed)
    tic = perf_counter()
    for experiment_id in range(1, 26):
        run_experiment(experiment_id)
    toc = perf_counter()
    print(f'\nTotal time elapsed: {toc - tic:.3f} seconds')
    

def run_experiment(problem_id: int) -> None:
    problem = get_problem(problem_id, dimension=64, problem_type='PBO')
    problem_name = problem.meta_data.name
    print(f'Running experiment for {problem_name}...')
    selections = ['rw', 'ts', 'rk', 'su']
    recombinations = ['kp', 'u']
    mutations = ['u', 'b']
    combinations = product(selections, recombinations, mutations)
    
    df_path = dirs['pkl']+f'GA-{problem_name}-{seed}.pkl'
    if not os.path.exists(df_path):
        print('No data found, creating dataframe...')
        tic = perf_counter()
        res_df = create_dataframe(problem, combinations)
        toc = perf_counter()
        print(f'Runtime: {toc - tic:.3f} seconds')
        res_df.to_pickle(df_path)
    else:
        res_df = pd.read_pickle(df_path)
    
    fig = create_plot(res_df, selections, recombinations, mutations, problem_name)
    fig.savefig(dirs['plots']+f'GA-{problem_name}-{seed}.png', dpi=300)
    plt.close(fig)
    return


def create_dataframe(problem: ioh.ProblemType, combinations: product) -> pd.DataFrame:
    df = pd.DataFrame(columns=['mutation', 'recombination', 'mutation', 'best_fitness', 'history'])
    df.index.name = 'run_id'

    for selection, recombination, mutation in combinations:
        run_id = f'{selection}-{recombination}-{mutation}'
        ga = GeneticAlgorithm(
            problem,
            pop_size = 100,
            mu_ = 40,
            lambda_ = 60,
            budget = 5_000,
            selection = selection,
            recombination = recombination,
            mutation = mutation,
            run_id = run_id,
            verbose = verbose
        )
        x_opt, f_opt, history = ga.optimize(return_history=True)
        df.loc[run_id] = (selection, recombination, mutation, f_opt, history)
    return df


def create_plot(df: pd.DataFrame, selections: list, recombinations: list, mutations: list, problem_name: str) -> plt.Figure:
    sel_names = {
        'rw': 'Roulette Wheel Selection',
        'ts': 'Tournament Selection',
        'rk': 'Rank Selection',
        'su': 'Stochastic Universal Sampling'
    }
    
    rec_names = {
        'kp': 'K-Point Crossover',
        'u': 'Uniform Crossover'
    }

    mut_names = {
        'u': 'Uniform Mutation',
        'b': 'Bitflip Mutation'
    }

    fig, axes = plt.subplots(len(recombinations), len(mutations), figsize=(10, 10))
    # plot histories, rows are recombinations, columns are sigmas, line colors are taus
    for i, recombination in enumerate(recombinations):
        for j, mutation in enumerate(mutations):
            ax: plt.Axes = axes[i, j]
            for selection in selections:
                run_id = f'{selection}-{recombination}-{mutation}'
                history = df.loc[run_id]['history']
                ax.plot(history, label=sel_names[selection])
            if i == 0:
                ax.set_title(mut_names[mutation])
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
    fig.suptitle(f'Genetic Algorithm ({problem_name})', fontsize=16, weight='bold')
    fig.supxlabel('generation')
    fig.supylabel('fitness')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    main()
