#!/usr/bin/env python

import os
import datetime
import argparse
import numpy as np
from ioh import problem, get_problem, logger, OptimizationType
from nasbench.api import NASBench, ModelSpec
from genetic_algorithm import GeneticAlgorithm
from utils import ParseWrapperGA, ProgressBar


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapperGA(parser)()
    pop_size, mu_, lambda_, budget, reps, seed, selection, recombination, mutation, xp, mut_rate, mut_nb = \
        args['population_size'], args['mu_'], args['lambda_'], args['budget'], args['repetitions'], args['seed'], \
        args['selection'], args['recombination'], args['mutation'], args['xp'], args['mut_r'], args['mut_b']
    
    # define run id
    mut_id = f'u-{mut_rate:.3f}' if mutation == 'u' else f'n-{mut_nb}'
    rec_id = f'kp-{xp}' if recombination == 'xp' else 'u'
    run_id = f'GA_P{pop_size}_M{mu_}_L{lambda_}_S{selection}_R{rec_id}_M{mut_id}'

    # create global NASBench object and specify options for usage in nas_ioh function
    global NB, OPTS
    NB = NASBench(args['problem_path'], seed=args['seed'])
    OPTS = ('maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu')

    # create IOH problem object
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

    # conditionally attach logger to problem
    if not args['no_log']:
        log_dir = __file__.replace('s2714892_s2233827_GA.py', 'logs')
        if not os.path.exists(log_dir): os.mkdir(log_dir)
        my_logger = logger.Analyzer(
            root = log_dir,
            folder_name = 'GA_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            algorithm_name = run_id,
            store_positions = True
        )
        prob.attach_logger(my_logger)
    
    # make stdout prettier
    print('-' * 80)
    progress = ProgressBar(reps, run_id)

    # run actual experiment
    np.random.seed(seed)
    for i in range(reps):
        ga = GeneticAlgorithm(
            problem = prob,
            pop_size = pop_size,
            mu_ = mu_,
            lambda_ = lambda_,
            budget = budget,
            selection = selection,
            recombination = recombination,
            mutation = mutation,
            xp = xp,
            mut_rate = mut_rate,
            mut_nb = mut_nb,
        )
        ga.optimize()
        prob.reset()
        progress(i)

    return


def nas_ioh(x: np.ndarray) -> float:
    """ Gets wrapped by an ioh Problem """

    # create adjacency matrix of first 21 elements
    matrix = np.triu(np.ones((7, 7), dtype=int), 1)
    matrix[matrix == 1] = x[:21]

    # create operations list of last 5 elements
    ops = ['input'] + [OPTS[i] for i in x[21:]] + ['output']

    # create model spec
    model_spec = ModelSpec(matrix=matrix, ops=ops)
    
    # check validity
    if not NB.is_valid(model_spec): return 0

    # get validation accuracy
    epoch = NB.get_metrics_from_spec(model_spec)[1][108]
    res = sum([e['final_validation_accuracy'] for e in epoch]) / 3.0

    return res


if __name__ == '__main__':
    main()
