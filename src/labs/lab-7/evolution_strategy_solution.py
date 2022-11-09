from ioh import get_problem
from ioh import logger
import ioh
import sys
import numpy as np
import time

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
def initialization(mu,dimension,lowerbound = -5.0, upperbound = 5.0):
    parent = []
    parent_sigma = []
    for i in range(mu):
        parent.append(np.random.uniform(low = lowerbound,high = upperbound, size = dimension))
        parent_sigma.append(0.05 * (upperbound - lowerbound))
    return parent,parent_sigma

# One-sigma mutation
def mutation(parent, parent_sigma,tau):
    for i in range(len(parent)):
        parent_sigma[i] = parent_sigma[i] * np.exp(np.random.normal(0,tau))
        for j in range(len(parent[i])):
            parent[i][j] = parent[i][j] + np.random.normal(0,parent_sigma[i])
            parent[i][j] = parent[i][j] if parent[i][j] < 5.0 else 5.0
            parent[i][j] = parent[i][j] if parent[i][j] > -5.0 else -5.0            

# Intermediate recombination
def recombination(parent,parent_sigma):
    [p1,p2] = np.random.choice(len(parent),2,replace = False)
    offspring = (parent[p1] + parent[p2])/2
    sigma = (parent_sigma[p1] + parent_sigma[p2])/2 

    return offspring,sigma

def evolution_strategy(func, budget = None):
    
    # Budget of each run: 50,000
    if budget is None:
        budget = 50000
    
    f_opt = sys.float_info.max
    x_opt = None

    # Parameters setting
    mu_ = 5
    lambda_ = 10
    tau =  1.0 / np.sqrt(func.meta_data.n_variables)

    
    # Initialization and Evaluation
    parent,parent_sigma = initialization(mu_,func.meta_data.n_variables)
    parent_f = []
    for i in range(mu_):
        parent_f.append(func(parent[i]))
        budget = budget - 1
        if parent_f[i] < f_opt:
            f_opt = parent_f[i]
            x_opt = parent[i].copy()

    # Optimization Loop
    while (f_opt > optimum and budget > 0):        
        offspring = []
        offspring_sigma = []
        offspring_f = []

        # Recombination
        for i in range(lambda_):
            o, s = recombination(parent,parent_sigma)
            offspring.append(o)
            offspring_sigma.append(s)

        # Mutation
        mutation(offspring,offspring_sigma,tau)

        # Evaluation
        for i in range(lambda_) : 
            offspring_f.append(func(offspring[i]))
            budget = budget - 1
            if offspring_f[i] < f_opt:
                    f_opt = offspring_f[i]
                    x_opt = offspring[i].copy()
            if (f_opt <= optimum) | (budget <= 0):
                break

        # Selection
        rank = np.argsort(offspring_f)
        parent = []
        parent_sigma = []
        parent_f = []
        i = 0
        while ((i < lambda_) & (len(parent) < mu_)):
            if (rank[i] < mu_):
                parent.append(offspring[i])
                parent_f.append(offspring_f[i])
                parent_sigma.append(offspring_sigma[i])
            i = i + 1
        
    # ioh function, to reset the recording status of the function.
    func.reset()
    print(f_opt,x_opt)
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
