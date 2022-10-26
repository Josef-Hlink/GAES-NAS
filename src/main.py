from ioh import get_problem
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    p_size = 100
    mu = 0.1
    survivors = int(p_size*mu)
    n_gen = 1000
    n_dim = 10
    p = get_problem(23, 1, n_dim, 'Real')
    population = np.random.randint(0, 2, (p_size, n_dim))
    log = np.zeros(n_gen)
    
    for gen in range(n_gen):
        
        fitness = np.array([p(individual) for individual in population])
        log[gen] = np.min(fitness)

        # selection
        parents_ids = np.array([roulette_wheel_selection(fitness) for _ in range(survivors)])
        new_population = []

        # new generation
        while len(new_population) < p_size:
            pid_1, pid_2 = np.random.choice(parents_ids, 2, replace=False)
            parent_1, parent_2 = population[pid_1], population[pid_2]

            # crossover and mutation
            child = crossover_1p(parent_1, parent_2)
            child = mutate_uniform(child)
            new_population.append(child)

        population = np.array(new_population)

    plt.plot(log)
    plt.show()


def roulette_wheel_selection(fitness: np.ndarray) -> int:
    """Roulette wheel selection."""
    fitness = fitness - np.min(fitness)
    fitness = fitness / np.sum(fitness)
    return np.random.choice(fitness.shape[0], p=fitness)

def mutate_uniform(individual: np.ndarray) -> np.ndarray:
    """Uniform mutation."""
    for i in range(individual.shape[0]):
        if np.random.rand() < 1 / individual.shape[0]:
            individual[i] = not individual[i]
    return individual

def crossover_1p(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """One point crossover."""
    cutoff = np.random.choice(parent1.shape[0])
    child = np.concatenate((parent1[:cutoff], parent2[cutoff:]))
    return child


if __name__ == "__main__":
    np.random.seed(123)
    main()
