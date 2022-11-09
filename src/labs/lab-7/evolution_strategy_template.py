from ioh import get_problem
from ioh import logger
import ioh
import sys
import numpy as np
import time
from typing import Callable

dimension = 5
def sphere(x: np.ndarray) -> float:
    return np.sum(np.power(x, 2))

ioh.problem.wrap_real_problem(
    sphere,                                     # Handle to the function
    name="Sphere",                               # Name to be used when instantiating
    optimization_type=ioh.OptimizationType.MIN, # Specify that we want to minimize
    lb=-5,                                               # The lower bound
    ub=5,                                                # The upper bound
)
sphere = get_problem("Sphere", dimension=dimension)
# The optimum of Sphere is 0
optimum = 0

# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
l = logger.Analyzer(root="data", 
    folder_name="run", 
    algorithm_name="evolution strategy", 
    algorithm_info="The lab session of the evolutionary algorithm course in LIACS")

sphere.attach_logger(l)


# Initialization
def initialization(
    mu: int,
    dimension: int,
    lowerbound = -5.0,
    upperbound = 5.0
    ) -> tuple[np.ndarray, np.ndarray]:
    """ Initializes mu individuals with dimension dimensions, along with their sigma vectors """
    parents = np.array([
        np.random.uniform(lowerbound, upperbound, dimension)
        for _ in range(mu)
    ])
    parents_sigma = np.array([
        np.random.uniform(lowerbound/1000, upperbound/1000, dimension)
        for _ in range(mu)
    ])
    return parents, parents_sigma


# Mutation
def mutation(
    parents: np.ndarray,
    parents_sigma: np.ndarray,
    tau: float
    ) -> tuple[np.ndarray, np.ndarray]:
    """ Mutation of the parents? (children makes more sense) """
    for i in range(len(parents)):
        parents[i] = parents[i] + parents_sigma[i]
        parents_sigma[i] = parents_sigma[i] * np.exp(tau * np.random.normal(0, 1))
    return parents, parents_sigma


# Recombination
def recombination(
    parents: np.ndarray,
    parent_sigmas: np.ndarray
    ) -> np.ndarray:
    """ Recombination of the parents to create one child """
    offspring = parents[np.random.choice(parents.shape[0])]
    offspring_sigma = parent_sigmas[np.random.choice(parent_sigmas.shape[0])]
    return offspring, offspring_sigma


def evolution_strategy(
    func : Callable[[np.ndarray], float],
    budget: int = None
    ) -> tuple[float, np.ndarray]:
    
    # Budget of each run: 50,000
    if budget is None:
        budget = 50_000
    
    f_opt = sys.float_info.max
    x_opt = None

    # Parameters setting
    mu_ = 10
    lambda_ = 100
    tau =  1 / np.sqrt(2 * dimension)

    # Initialization and Evaluation
    parents, parents_sigma = initialization(lambda_, func.meta_data.n_variables)
    parents_f = np.array([func(parent) for parent in parents])
    budget = budget - lambda_

    # Optimization Loop
    while (f_opt > optimum and budget > 0):        
        offspring = np.zeros(shape=(lambda_, dimension))
        offspring_sigma = np.zeros(shape=(lambda_, dimension))
        offspring_f = np.zeros(shape=(lambda_))

        # Recombination
        for i in range(lambda_):
            o, s = recombination(parents, parents_sigma)
            offspring[i] = o
            offspring_sigma[i] = s
            offspring_f[i] = func(o)
            budget -= 1

        # Mutation
        offspring, offspring_sigma = mutation(offspring, offspring_sigma, tau)

        # Selection
        ranking = np.argsort(offspring_f)
        offspring = offspring[ranking]
        offspring_sigma = offspring_sigma[ranking]
        offspring_f = offspring_f[ranking]

        parents = offspring[:mu_]
        parents_sigma = offspring_sigma[:mu_]
        parents_f = offspring_f[:mu_]
        
        # Update the best solution
        best_in_pop = np.argmin(parents_f)
        if parents_f[best_in_pop] < f_opt:
            f_opt = parents_f[best_in_pop]
            x_opt = parents[best_in_pop].copy()

    # ioh function, to reset the recording status of the function.
    func.reset()
    print(f_opt, x_opt)
    return f_opt, x_opt


def main():
    # We run the algorithm 20 independent times.
    for _ in range(20):
        evolution_strategy(sphere)

if __name__ == '__main__':
  start = time.time()
  main()
  end = time.time()
  print("The program takes %s seconds" % (end-start))
