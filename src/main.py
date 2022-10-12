from ioh import problem, OptimizationType
from ioh import get_problem
import numpy as np

f = get_problem(7, dimension=5, instance=1, problem_type = 'Real')

print(f.meta_data)
