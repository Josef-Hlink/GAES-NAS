import os
import argparse
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ioh import problem, get_problem, logger, OptimizationType, ProblemType
from nasbench.api import NASBench, ModelSpec

from genetic_algorithm import GeneticAlgorithm
from evolution_strategies import EvolutionStrategies
from utils import get_directories, ParseWrapper, ProgressBar


def main():
    
    global DIRS, ARGS, NB, PROB
    DIRS = get_directories(__file__)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = ParseWrapper(parser)()

    # loop over all args and print them
    arg_names = list(ARGS.keys())
    print('\033[1m' + 'ARGS:' + '\033[0m')
    # print 4 args per line
    for i in range(0, len(arg_names), 4):
        print('|'.join([f'{name:<20}' for name in arg_names[i:i+4]]))

    dir_names = list(DIRS.keys())
    print('\033[1m' + 'DIRS:' + '\033[0m')
    print(', '.join([f'{name}' for name in dir_names]))

    np.random.seed(ARGS['seed'])
    NB = NASBench(ARGS['problem_path'], seed=ARGS['seed'])
    PROB = create_problem()
    
    if ARGS['verbose'] == 1 and ARGS['repetitions'] > 1:
        progress_1 = ProgressBar(ARGS['repetitions'], ARGS['run_id'])
    
    df = pd.DataFrame(columns=list(range(ARGS['repetitions'])))
    df.index.name = 'generation'

    tic = perf_counter()
    for i in range(ARGS['repetitions']):
        
        res = run_experiment(i, ARGS['repetitions'])  # no real args, because everything is already in global ARGS
        df[i] = res

        if ARGS['verbose'] == 1 and ARGS['repetitions'] > 1:
            progress_1(i)
    
    toc = perf_counter()
    print(f'\nTotal time elapsed: {toc - tic:.3f} seconds')

    # --- TODO rename this to something like save --- #
    if ARGS['overwrite']:
        df.to_csv(DIRS['csv'] + f'{ARGS["run_id"]}_{ARGS["pid"]}.csv', index=True)
    # ----------------------------------------------- #

def run_experiment(i: int, n_reps: int) -> pd.Series:
    
    history = pd.Series(dtype=np.float64)
    history.index.name = 'generation'

    if ARGS['optimizer'] == 'GA':
        optimizer = GeneticAlgorithm(
            problem = PROB,
            pop_size = ARGS['population_size'],
            mu_ = ARGS['mu_'],
            lambda_ = ARGS['lambda_'],
            budget = ARGS['budget'],
            selection = ARGS['selection'],
            recombination = ARGS['recombination'],
            mutation = ARGS['mutation'],
            xp = ARGS['xp'],
            mut_rate = ARGS['mut_r'],
            mut_nb = ARGS['mut_b'],
            run_id = f'Repetition {i+1}/{n_reps}...',
            verbose = True if ARGS['verbose'] == 2 else False
        )
    else:  # ES
        raise NotImplementedError('ES not implemented yet')
    
    _, _, history = optimizer.optimize(return_history=True)
    return history


def create_problem() -> ProblemType:
    global _options; _options = ('maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu')
    problem.wrap_integer_problem(
        f = nas_ioh,
        name = 'nas101',
        optimization_type = OptimizationType.MAX
    )
    prob = get_problem(
        fid = 'nas101',
        instance = 0,
        dimension = 26,
        problem_type = 'Integer'
    )
    my_logger = logger.Analyzer(
        root = os.path.join(DIRS['logs']),
        folder_name = 's2233827_s2714892',
        algorithm_name = ARGS['run_id'],
        store_positions = True
    )
    prob.attach_logger(my_logger)
    return prob


def nas_ioh(x: np.ndarray) -> float:
    """ Gets wrapped by an ioh Problem """
    
    print('-' * 80)
    
    print(x)

    # create adjacency matrix of first 21 elements
    matrix = np.empty((7, 7), dtype=int)
    matrix = np.triu(matrix, 1)
    index = 0
    for i in range(7):
        for j in range(i + 1, 7):
            matrix[i][j] = x[index]
            index += 1
    
    print(matrix)

    # create operations list of last 5 elements
    ops = ['input'] + [_options[i] for i in x[21:]] + ['output']

    print('ops:', ops)

    # create model spec
    model_spec = ModelSpec(matrix=matrix, ops=ops)
    
    print(model_spec)

    # get validation accuracy
    tmp = NB.get_metrics_from_spec(model_spec)
    epoch = tmp[1][108]
    print('epoch:', epoch)
    res = 0
    for e in epoch:
        print(e)
        res += e['final_validation_accuracy']
    res = res / 3.0
    
    print('res:', res)

    print('\n' * 10)

    return res


def create_plot(df: pd.DataFrame) -> plt.Figure:
    
    fig, ax = plt.subplots()
    ax.set_title(f'{ARGS["optimizer"]}: {PROB.meta_data.name}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_yscale('log')
    ax.grid(True)
    ax.plot(df.mean(axis=1), label='Mean')
    ax.plot(df.median(axis=1), label='Median')
    ax.plot(df.min(axis=1), label='Min')
    ax.plot(df.max(axis=1), label='Max')
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    main()
