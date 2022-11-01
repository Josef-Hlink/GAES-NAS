from ioh import get_problem
from ioh import logger
import sys
import numpy as np
import time


# Declaration of problems to be tested.
# We obtain an interface of the OneMax problem here.
# om(x) return the fitness value of 'x'
dimension = 50
om = get_problem(1, dimension=dimension, instance=1, problem_type = 'PBO')
# We know the optimum of onemax
optimum = dimension

# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
l = logger.Analyzer(root="data", 
    folder_name="run", 
    algorithm_name="genetic_algorithm", 
    algorithm_info="The lab session of the evolutionary algorithm course in LIACS")

om.attach_logger(l)


# Parameters setting
pop_size = 10
tournament_k = 10
mutation_rate = 0.02
crossover_probability = 0.5


# Uniform Crossover
def crossover(p1, p2):
   if(np.random.uniform(0,1) < crossover_probability):
        for i in range(len(p1)) :
            if np.random.uniform(0,1) < 0.5:
                t = p1[i]
                p1[i] = p2[i]
                p2[i] = t

# Standard bit mutation using mutation rate p
def mutation(p):
    for i in range(len(p)) :
        if np.random.uniform(0,1) < mutation_rate:
            p[i] = 1 - p[i]


def mating_seletion(parent, parent_f):
    # Using the tournament selection
    # select_parent = []
    # for i in range(len(parent)) :
    #     pre_select = np.random.choice(len(parent_f),tournament_k,replace = False)
    #     max_f = sys.float_info.min
    #     for p in pre_select:
    #         if parent_f[p] > max_f:
    #             index = p
    #             max_f = parent_f[p]
    #     select_parent.append(parent[index].copy())
    # return select_parent

    # Using the proportional selection

    # Plusing 0.001 to avoid dividing 0
    f_min = min(parent_f)
    f_sum = sum(parent_f) - (f_min - 0.001) * len(parent_f)
    
    rw = [(parent_f[0] - f_min + 0.001)/f_sum]
    for i in range(1,len(parent_f)):
        rw.append(rw[i-1] + (parent_f[i] - f_min + 0.001) / f_sum)
    
    select_parent = []
    for i in range(len(parent)) :
        r = np.random.uniform(0,1)
        index = 0
        # print(rw,r)
        while(r > rw[index]) :
            index = index + 1
        
        select_parent.append(parent[index].copy())
    return select_parent

def genetic_algorithm(func, budget = None):
    
    # budget of each run: 10000
    if budget is None:
        budget = 10000
    
    f_opt = sys.float_info.min
    x_opt = None
    
    parent = []
    parent_f = []
    for i in range(pop_size):

        # Initialization
        parent.append(np.random.randint(2, size = func.meta_data.n_variables))
        parent_f.append(func(parent[i]))
        budget = budget - 1

    while (f_opt < optimum and budget > 0):
            
        offspring = mating_seletion(parent,parent_f)

        for i in range(0,pop_size - (pop_size%2),2) :
            crossover(offspring[i], offspring[i+1])


        for i in range(pop_size):
            mutation(offspring[i])

        parent = offspring.copy()
        for i in range(pop_size) : 
            parent_f[i] = func(parent[i])
            budget = budget - 1
            if parent_f[i] > f_opt:
                    f_opt = parent_f[i]
                    x_opt = parent[i].copy()
            if f_opt >= optimum:
                break
        
    # ioh function, to reset the recording status of the function.
    func.reset()
    print(f_opt,x_opt)
    return f_opt, x_opt

def main():
    # We run the algorithm 20 independent times.
    for _ in range(20):
        genetic_algorithm(om)

if __name__ == '__main__':
  start = time.time()
  main()
  end = time.time()
  print("The program takes %s seconds" % (end-start))
