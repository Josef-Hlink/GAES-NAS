from ioh import get_problem
import numpy as np
import matplotlib.pyplot as plt


def main():

    iohp_kwargs = dict(
        fid = 1,
        instance = 1,
        dimension = 100,
        problem_type = 'PBO'
    )

    x, f, log = genetic_algorithm(100, 0.1, 100_000, iohp_kwargs, return_log=True)
    print(f'Best solution: {x}')
    print(f'Best fitness: {f}')
    print(f'Number of generations: {len(log)}')
    fig = plot_results(log)
    fig.savefig('../plots/results.png')


def genetic_algorithm(
    population_size: int = 100,
    mu: float = 0.1,
    budget: int = 10_000,
    iohp_kwargs: dict = None,
    return_log: bool = False
    ) ->  tuple[np.ndarray, float] | tuple[np.ndarray, float, list[dict]]:
    
    """
    Genetic algorithm
    ===
    Parameters
    ---
    population_size: int
        Size of the population.
    mu: float
        Survival rate, [0, 1].
    budget: int
        Number of function evaluations.
    iohp_args: dict
        Arguments for the IOH problem, will be passed directly (defaults to onemax problem).
    return_log: bool
        Whether to return the log.
    
    Returns
    ---
    best_x: np.ndarray
        Best found solution.
    best_f: float
        Fitness value associated with the best solution.
    log: list[dict] (optional)
        Log of the algorithm. Each entry is a dictionary with the following keys:
        
        * best_so_far (float): Best fitness value found so far.
        * best_in_pop (float): Best fitness value in the population.
        * worst_in_pop (float): Worst fitness value in the population.
        * mean_in_pop (float): Mean fitness value in the population.
    """
    
    if iohp_kwargs is None:
        iohp_kwargs = dict(
            fid = 1,
            instance = 1,
            dimension = 10,
            problem_type = 'PBO'
        )
    
    n_survivors = int(population_size*mu)
    n_dimensions = iohp_kwargs['dimension']
    p = get_problem(**iohp_kwargs)
    population = np.random.randint(0, 2, (population_size, n_dimensions))
    best_x, best_f = np.empty(n_dimensions), -np.inf
    if return_log:
        log = []
    
    while budget >= population_size:
        
        # evaluate population
        fitness = np.array([p(individual) for individual in population])
        budget -= population_size
        
        # necessary logging
        best_x_in_pop, best_f_in_pop = population[np.argmax(fitness)], np.max(fitness)
        if best_f_in_pop > best_f:
            best_x, best_f = best_x_in_pop, best_f_in_pop

        # optional logging
        if return_log:
            log.append(
                dict(
                    best_so_far = best_f,
                    best_in_pop = best_f_in_pop,
                    worst_in_pop = np.min(fitness),
                    mean_in_pop = np.mean(fitness)
                )
            )

        # selection
        parents_ids = np.array([roulette_wheel_selection(fitness) for _ in range(n_survivors)])
        new_population = []

        # new generation
        while len(new_population) < population_size:
            pid_1, pid_2 = np.random.choice(parents_ids, 2, replace=False)
            parent_1, parent_2 = population[pid_1], population[pid_2]

            # crossover and mutation
            child = crossover_1p(parent_1, parent_2)
            child = mutate_uniform(child)
            new_population.append(child)

        population = np.array(new_population)

    if return_log:
        return best_x, best_f, log
    else:
        return best_x, best_f


def plot_results(log: list[dict]) -> plt.Figure:
    """Plot the results of the genetic algorithm."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot([entry['best_so_far'] for entry in log], label='Best so far')
    ax.plot([entry['best_in_pop'] for entry in log], label='Best in population')
    ax.plot([entry['worst_in_pop'] for entry in log], label='Worst in population')
    ax.plot([entry['mean_in_pop'] for entry in log], label='Mean in population')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()
    return fig


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
