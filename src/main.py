#!/usr/bin/env python

import argparse
from time import perf_counter

import numpy as np
import pandas as pd

from ioh import problem, get_problem, logger, OptimizationType
from nasbench.api import NASBench, ModelSpec

from genetic_algorithm import GeneticAlgorithm
from evolution_strategies import EvolutionStrategies
from utils import get_directories, ParseWrapper, ProgressBar


def init_cli(save_to: str = None):
    global DIRS, ARGS, NB, OPTS, PROB
    DIRS = get_directories(__file__)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = ParseWrapper(parser)()
    np.random.seed(ARGS['seed'])

    # create global NASBench object and specify options for usage in nas_ioh
    NB = NASBench(ARGS['problem_path'], seed=ARGS['seed'])
    OPTS = ('maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu')

    problem.wrap_integer_problem(
        f = nas_ioh,
        name = 'nas101',
        optimization_type = OptimizationType.MAX
    )
    PROB = get_problem(
        fid = 'nas101',
        instance = 0,
        dimension = 26,
        problem_type = 'Integer'
    )
    if ARGS['log']:
        my_logger = logger.Analyzer(
            root = DIRS['logs'],
            folder_name = 's2233827_s2714892',
            algorithm_name = ARGS['run_id'],
            store_positions = True
        )
        PROB.attach_logger(my_logger)
        # note that it is possible to detach the logger if we want to run multiple experiments with different configurations
        # with PROB.detach_logger()
    
    if ARGS['verbose'] == 1:
        print('-' * 80)

    if ARGS['verbose'] == 1 and ARGS['repetitions'] > 1:
        progress_1 = ProgressBar(ARGS['repetitions'], ARGS['run_id'])
    
    df = pd.DataFrame(columns=list(range(ARGS['repetitions'])))
    df.index.name = 'generation'

    tic = perf_counter()
    for i in range(ARGS['repetitions']):
        
        res = run_experiment(i, ARGS['repetitions'])  # no real args passed here, because everything is already in global ARGS
        df[i] = res

        if ARGS['verbose'] == 1 and ARGS['repetitions'] > 1:
            progress_1(i)
        
        PROB.reset()

    toc = perf_counter()
    if ARGS['verbose'] == 1:
        print('\n' + f'Total time elapsed: {toc - tic:.3f} seconds')
        print('Saving results...')
    
    if save_to is None:
        df.to_csv(DIRS['csv'] + f'{ARGS["run_id"]}.csv', index=True)
    else:
        df.to_csv(save_to + f'{ARGS["run_id"]}.csv', index=True)

    return

def init_ipynb(nasbench_object: NASBench, args: dict[str, any], save_to: str = None):

    global DIRS, ARGS, NB, OPTS, PROB
    DIRS = get_directories(__file__)
    ARGS = args
    np.random.seed(ARGS['seed'])

    # create global NASBench object and specify options for usage in nas_ioh
    NB = nasbench_object
    OPTS = ('maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu')

    problem.wrap_integer_problem(
        f = nas_ioh,
        name = 'nas101',
        optimization_type = OptimizationType.MAX
    )
    PROB = get_problem(
        fid = 'nas101',
        instance = 0,
        dimension = 26,
        problem_type = 'Integer'
    )
    if ARGS['log']:
        my_logger = logger.Analyzer(
            root = DIRS['logs'],
            folder_name = 's2233827_s2714892',
            algorithm_name = ARGS['run_id'],
            store_positions = True
        )
        PROB.attach_logger(my_logger)
    
    if ARGS['verbose'] == 1:
        print('-' * 80)

    if ARGS['verbose'] == 1 and ARGS['repetitions'] > 1:
        progress_1 = ProgressBar(ARGS['repetitions'], ARGS['run_id'])
    
    df = pd.DataFrame(columns=list(range(ARGS['repetitions'])))
    df.index.name = 'generation'

    tic = perf_counter()
    for i in range(ARGS['repetitions']):
        
        res = run_experiment(i, ARGS['repetitions'])  # no real args passed here, because everything is already in global ARGS
        df[i] = res

        if ARGS['verbose'] == 1 and ARGS['repetitions'] > 1:
            progress_1(i)
        
        PROB.reset()

    toc = perf_counter()
    if ARGS['verbose'] == 1:
        print('\n' + f'Total time elapsed: {toc - tic:.3f} seconds')
        print('Saving results...')
    
    if save_to is None:
        df.to_csv(DIRS['csv'] + f'{ARGS["run_id"]}.csv', index=True)
    else:
        df.to_csv(save_to + f'{ARGS["run_id"]}.csv', index=True)

    return


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
            run_id = f'Repetition {i+1}/{n_reps}',
            verbose = True if ARGS['verbose'] == 2 else False
        )
    else:  # ES
        optimizer = EvolutionStrategies(
            problem = PROB,
            pop_size = ARGS['population_size'],
            mu_ = ARGS['mu_'],
            lambda_ = ARGS['lambda_'],
            tau_ = ARGS['tau_'],
            sigma_ = ARGS['sigma_'],
            chunk_size = ARGS['chunk_size'],
            budget = ARGS['budget'],
            recombination = ARGS['recombination'],
            individual_sigmas = ARGS['individual_sigmas'],
            run_id = f'Repetition {i+1}/{n_reps}',
            verbose = True if ARGS['verbose'] == 2 else False
        )
    
    _, _, history = optimizer.optimize(return_history=True)
    return history


def nas_ioh(x: np.ndarray) -> float:
    """ Gets wrapped by an ioh Problem """

    # create adjacency matrix of first 21 elements
    matrix = np.triu(np.ones((7, 7), dtype=int), 1)
    matrix[matrix == 1] = x[:21]

    # create operations list of last 5 elements
    ops = ['input'] + [OPTS[i] for i in x[21:]] + ['output']

    # create model spec
    model_spec = ModelSpec(matrix=matrix, ops=ops)
    
    # assert NB.is_valid(model_spec), "Invalid model spec:\n" + \
    #     f"matrix:\n{model_spec.matrix}\n" + \
    #     f"ops:\n{model_spec.ops}"
    
    # check validity
    if not NB.is_valid(model_spec):
        return 0

    # get validation accuracy
    epoch = NB.get_metrics_from_spec(model_spec)[1][108]
    res = sum([e['final_validation_accuracy'] for e in epoch]) / 3.0

    return res


if __name__ == '__main__':
    init_cli()
