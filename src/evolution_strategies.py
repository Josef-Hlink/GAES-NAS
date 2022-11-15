import time

import numpy as np
import matplotlib.pyplot as plt
from ioh import get_problem
import ioh

from utils import ProgressBar, get_directories


def main():
    dirs = get_directories(__file__)
    sphere = get_problem('Sphere', dimension=5)
    es = EvolutionStrategies(
        problem = sphere,
        pop_size = 100,
        mu_ = 40,
        lambda_ = 60,
        tau_ = 1 / np.sqrt(2 * sphere.meta_data.n_variables),
        sigma_ = 0.1,
        budget = 500_000,
        recombination = 'd',
        individual_sigmas = True,
        run_id = 'discrete recombination'
    )
    tic = time.perf_counter()
    x_opt, f_opt, history = es.optimize(return_history=True)
    toc = time.perf_counter()
    print(f'time: {toc - tic:.3f} seconds')
    print(f'f_opt: {f_opt:.5f}')
    print(f'x_opt: {x_opt}')
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_title('Sphere')
    ax.set_xlabel('generation')
    ax.set_ylabel(r'$f_\mathrm{opt}$')
    fig.tight_layout()
    fig.savefig(dirs['plots']+'sphere.png', dpi=100)
    return


class EvolutionStrategies:
    
    def __init__(
        self,
        problem: ioh.ProblemType,
        pop_size: int,
        mu_: int,
        lambda_: int,
        tau_: float,
        sigma_: float,
        budget: int = 50_000,
        recombination: str = 'd',
        individual_sigmas: bool = False,
        run_id: str | None = None
        ) -> None:
        
        """ Sets all parameters """

        kwargs = locals(); kwargs.pop('self'); kwargs.pop('run_id')
        self.validate_parameters(**kwargs)

        self.problem = problem
        self.pop_size = pop_size
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.tau_ = tau_
        self.sigma_prop = sigma_  # what gets passed as sigma_ should be interpreted as the proportion wrt the bounds
        self.budget = budget
        self.isig = individual_sigmas

        self.n_dimensions = problem.meta_data.n_variables
        self.lb = problem.bounds.lb[0]
        self.ub = problem.bounds.ub[0]

        if self.pop_size == self.lambda_:
            self.selection = ','
        else:  # pop_size is mu + lambda
            self.selection = '+'

        self.recombination = dict(
            d = self.recombination_discrete,
            i = self.recombination_intermediate,
            dg = self.recombination_discrete_global,
            ig = self.recombination_intermediate_global
        )[recombination]

        self.n_generations = self.budget // self.pop_size
        self.history = np.zeros(self.n_generations)
        self.progress = ProgressBar(self.n_generations, run_id=run_id)

        self.f_opt = np.inf
        self.x_opt = None

        return


    def optimize(self, return_history: bool = False) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
        """
        Runs the optimization algorithm and returns the best candidate solution found and its fitness.
        If return_history is set to True, it will also return the history
        of the best candidate solutions found in each population.
        """

        self.initialize_population()

        for gen in range(self.n_generations):
            self.population, self.pop_sigmas, f_opt_in_pop = self.evaluate_population()
            self.history[gen] = f_opt_in_pop

            if f_opt_in_pop < self.f_opt:
                self.f_opt = f_opt_in_pop
                self.x_opt = self.population[0]
            
            parents = self.population[:self.mu_]
            parents_sigmas = self.pop_sigmas[:self.mu_]

            children, children_sigmas = [], []
            for _ in range(self.lambda_):
                c, cs = self.recombination(parents, parents_sigmas)
                children.append(c)
                children_sigmas.append(cs)
            
            children, children_sigmas = np.array(children), np.array(children_sigmas)
            children, children_sigmas = self.mutate(children, children_sigmas)

            if self.selection == '+':
                self.population = np.concatenate((parents, children), axis=0)
                self.pop_sigmas = np.concatenate((parents_sigmas, children_sigmas), axis=0)
            else:  # selection is ,
                self.population = children
                self.pop_sigmas = children_sigmas

            self.progress(gen)

        if return_history:
            return self.x_opt, self.f_opt, self.history
        else:
            return self.x_opt, self.f_opt


    def initialize_population(self) -> None:
        """ Initializes population and sigmas with random values between lower and upper bounds """

        self.population = np.random.uniform(
            self.lb,
            self.ub,
            (self.pop_size, self.n_dimensions)
        )

        sigma = self.sigma_prop * (self.ub - self.lb)

        if self.isig:  # every parameter has its own sigma associated with it
            self.pop_sigmas = np.random.uniform(
                self.lb * sigma,
                self.ub * sigma,
                (self.pop_size, self.n_dimensions)
            )
        else:  # every parameter has the same sigma associated with it, but it still differs from candidate to candidate
            self.pop_sigmas = np.repeat(
                np.random.uniform(
                    self.lb * sigma,
                    self.ub * sigma,
                    self.pop_size
                ),
                self.n_dimensions
            ).reshape(self.pop_size, self.n_dimensions)

        return


    def mutate(self, individuals: np.ndarray, sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Mutates the individuals in the given 2D array (will also work for 1D array).
        Returns both the updated individuals as well as the updated sigmas.
        """
        
        mutated = individuals + sigmas
        mutated = np.clip(mutated, self.lb, self.ub)  # clip to bounds
        
        updated_sigmas = sigmas * np.exp(self.tau_ * np.random.normal(0, 1, sigmas.shape))

        return mutated, updated_sigmas


    def recombination_discrete(self, parents: np.ndarray, parents_sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Chooses two parents and returns one child, randomly choosing a parameter from each parent """
        
        pi1, pi2 = np.random.choice(parents.shape[0], 2, replace=False)
        c, cs = np.zeros(self.n_dimensions), np.zeros(self.n_dimensions)
        for i in range(self.n_dimensions):
            if np.random.uniform() < 0.5:
                c[i], cs[i] = parents[pi1, i], parents_sigmas[pi1, i]
            else:
                c[i], cs[i] = parents[pi2, i], parents_sigmas[pi2, i]

        return c, cs
    
    def recombination_intermediate(self, parents: np.ndarray, parents_sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Chooses two parents and returns one child, averaging the parameters of the two parents """

        pi1, pi2 = np.random.choice(parents.shape[0], 2, replace=False)
        c, cs = np.zeros(self.n_dimensions), np.zeros(self.n_dimensions)
        for i in range(self.n_dimensions):
            c[i], cs[i] = (parents[pi1, i] + parents[pi2, i]) / 2, (parents_sigmas[pi1, i] + parents_sigmas[pi2, i]) / 2

        return c, cs
    
    def recombination_discrete_global(self, parents: np.ndarray, parents_sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Randomly chooses each parameter from all parents """

        c, cs = np.zeros(self.n_dimensions), np.zeros(self.n_dimensions)
        for i in range(self.n_dimensions):
            pi = np.random.choice(parents.shape[0])
            c[i], cs[i] = parents[pi, i], parents_sigmas[pi, i]

        return c, cs
    
    def recombination_intermediate_global(self, parents: np.ndarray, parents_sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Averages all parameters of all parents """

        c, cs = np.mean(parents, axis=0), np.mean(parents_sigmas, axis=0)

        return c, cs


    def evaluate_population(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluates the fitness of all candidate solutions in the population.
        Returns the candidate solutions ranked by fitness values, along with their sigmas and the highest fitness value.
        """
        pop_fitness = np.array([self.problem(x) for x in self.population])
        ranking = np.argsort(pop_fitness)

        return self.population[ranking], self.pop_sigmas[ranking], np.max(pop_fitness)


    def validate_parameters(
        self,
        problem: ioh.ProblemType,
        pop_size: int,
        mu_: int,
        lambda_: int,
        tau_: float,
        sigma_: float,
        budget: int,
        recombination: str,
        individual_sigmas: bool
        ) -> None:

        """ Validates all parameters of __init__ """
        
        assert isinstance(problem, ioh.ProblemType), "problem must be an instance of <ioh.ProblemType>"

        assert isinstance(pop_size, int), "population size must be an integer"
        assert pop_size in range(0, 500), "population size must be between 0 and 500"

        assert isinstance(mu_, int), "mu must be an integer"
        assert mu_ > 0, "mu must be greater than 0"
        assert mu_ < pop_size, "mu must be less than population size"

        assert isinstance(lambda_, int), "lambda must be an integer"
        assert lambda_ > 0, "lambda must be greater than 0"
        assert lambda_ < pop_size, "lambda must be less than population size"

        assert pop_size == lambda_ + mu_ or pop_size == lambda_, "population size must be" + \
        " either number of parents + number of offspring or just number of offspring"

        assert isinstance(tau_, float), "tau must be a float"
        assert tau_ > 0, "tau must be greater than 0"
        assert tau_ < 1, "tau must be less than 1"

        assert isinstance(sigma_, float), "sigma must be a float"
        assert sigma_ > 0, "sigma must be greater than 0"
        assert sigma_ < 1, "sigma must be less than 1"

        assert recombination in ['d', 'i', 'dg', 'ig'], "recombination must be one of the following: 'd', 'i', 'dg', 'ig'"

        assert isinstance(budget, int), "budget must be an integer"
        assert budget > 0, "budget must be greater than 0"
        assert budget < 100_000_000, "budget must be less than 100 million"

        assert isinstance(individual_sigmas, bool), "individual_sigmas must be a boolean"

        return



if __name__ == '__main__':
    main()
