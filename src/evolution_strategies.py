from datetime import datetime

import numpy as np
import ioh

from utils import ProgressBar

class EvolutionStrategies:
    
    def __init__(
        self,
        problem: ioh.problem.Integer,
        pop_size: int,
        mu_: int,
        lambda_: int,
        tau_: float,
        sigma_: float,
        chunk_size: int = 3,
        budget: int = 5_000,
        recombination: str = 'd',
        individual_sigmas: bool = False,
        run_id: any = None,
        verbose: bool = False
        ) -> None:
        
        """ Sets all parameters """

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        kwargs = locals(); kwargs.pop('self')
        self.validate_parameters(**kwargs)

        self.problem = problem
        self.pop_size = pop_size
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.tau_ = tau_
        self.sigma_prop = sigma_  # what gets passed as sigma_ should be interpreted as the proportion wrt the bounds
        self.chunk_size = chunk_size
        self.budget = budget
        self.isig = individual_sigmas
        self.run_id = str(run_id)
        self.verbose = verbose

        self.n_dims_mat = 21 // chunk_size
        self.lb_mat = 0
        self.ub_mat = 2 ** chunk_size - 1
        
        self.n_dims_ops = 1
        self.lb_ops = 0
        self.ub_ops = 2 ** 5 - 1

        if self.pop_size == self.lambda_:
            self.selection_kind = ','
        else:  # pop_size is mu + lambda
            self.selection_kind = '+'

        self.recombination = dict(
            d = self.recombination_discrete,
            i = self.recombination_intermediate,
            dg = self.recombination_discrete_global,
            ig = self.recombination_intermediate_global
        )[recombination]

        self.n_generations = self.budget // self.pop_size
        self.history = np.zeros(self.n_generations)
        if self.verbose:
            self.progress = ProgressBar(self.n_generations, p_id=self.run_id)

        self.f_opt = -np.inf
        self.x_opt = None

        return


    def optimize(self, return_history: bool = False) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
        """
        Runs the optimization algorithm and returns the best candidate solution found and its fitness.
        If return_history is set to True, it will also return the history
        of the best fitness value found in each population.
        """

        self.initialize_population()
        improvement = lambda x, y: x > y

        for gen in range(self.n_generations):
            self.population, self.pop_sigmas, self.pop_fitness = self.evaluate_population()
            f_opt_in_pop = self.pop_fitness[0]
            self.history[gen] = f_opt_in_pop

            if improvement(f_opt_in_pop, self.f_opt):
                self.f_opt = f_opt_in_pop
                self.x_opt = self.population[0]

            # deterministic selection
            parents = self.population[:self.mu_]
            parents_sigmas = self.pop_sigmas[:self.mu_]

            offspring = np.zeros((self.lambda_, self.n_dimensions))
            offspring_sigmas = np.zeros((self.lambda_, self.n_dimensions))
            for i in range(self.lambda_):
                offspring[i], offspring_sigmas[i] = self.recombination(parents, parents_sigmas)
            
            offspring, offspring_sigmas = self.mutate(offspring, offspring_sigmas)

            if self.selection_kind == '+':
                self.population = np.concatenate((parents, offspring), axis=0)
                self.pop_sigmas = np.concatenate((parents_sigmas, offspring_sigmas), axis=0)
            else:  # selection kind is ,
                self.population = offspring
                self.pop_sigmas = offspring_sigmas

            if self.verbose:
                self.progress(gen)

        if self.verbose:
            print(f'f_opt: {self.f_opt:.6f}')
            print(f'x_opt: {np.round(self.x_opt, 2)}')
            print(f'x_opt_clp: {np.round(self.x_opt).astype(int)}')
            x_opt_encoded = self.encode(self.x_opt[None, :])[0]
            print(f'x_opt_enc: {x_opt_encoded}')
            self.print_as_matrix(x_opt_encoded)
            print('-' * 80)

        if return_history:
            return self.x_opt, self.f_opt, self.history
        else:
            return self.x_opt, self.f_opt


    def initialize_population(self) -> None:
        """ Initializes population and sigmas with random values between lower and upper bounds """

        # create the parts of the matrix and the operators
        self.n_dimensions = self.n_dims_mat + self.n_dims_ops
        self.population = np.zeros((self.pop_size, self.n_dimensions), dtype=np.float32)
        self.pop_sigmas = np.zeros((self.pop_size, self.n_dimensions), dtype=np.float32)

        # initialize the matrix part
        self.population[:, :self.n_dims_mat] = np.random.uniform(
            low = self.lb_mat,
            high = self.ub_mat,
            size = (self.pop_size, self.n_dims_mat)
        )

        # initialize the operators part
        self.population[:, self.n_dims_mat:] = np.random.uniform(
            low = self.lb_ops,
            high = self.ub_ops,
            size = (self.pop_size, self.n_dims_ops)
        )

        # initialize the sigmas
        sigma_mat = self.sigma_prop * (self.ub_mat - self.lb_mat)
        sigma_ops = self.sigma_prop * (self.ub_ops - self.lb_ops)

        if self.isig:
            self.pop_sigmas[:, :self.n_dims_mat] = np.random.uniform(
                low = -sigma_mat,
                high = sigma_mat,
                size = (self.pop_size, self.n_dims_mat)
            )
            self.pop_sigmas[:, self.n_dims_mat:] = np.random.uniform(
                low = -sigma_ops,
                high = sigma_ops,
                size = (self.pop_size, self.n_dims_ops)
            )
        else:
            sigma_mat = np.random.uniform(low = -sigma_mat, high = sigma_mat)
            self.pop_sigmas[:, :self.n_dims_mat] = sigma_mat
            sigma_ops = np.random.uniform(low = -sigma_ops, high = sigma_ops)
            self.pop_sigmas[:, self.n_dims_mat:] = sigma_ops

        return


    def evaluate_population(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates the fitness of all candidate solutions in the population.
        Returns the candidate solutions ranked by fitness values, along with their sigmas and these fitness values.
        """

        encoded_pop = self.encode(self.population)

        pop_fitness = np.array([self.problem(x) for x in encoded_pop])
        ranking = np.argsort(pop_fitness)[::-1]

        return self.population[ranking], self.pop_sigmas[ranking], pop_fitness[ranking]


    def mutate(self, individuals: np.ndarray, sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Mutates the individuals in the given 2D array (will also work for 1D array).
        Returns both the updated individuals as well as the updated sigmas.
        """
        
        mutated = individuals + sigmas
        
        updated_sigmas = sigmas * np.exp(self.tau_ * np.random.normal(0, 1, sigmas.shape))

        mutated[:, :self.n_dims_mat] = np.clip(mutated[:, :self.n_dims_mat], self.lb_mat, a_max = self.ub_mat)
        mutated[:, self.n_dims_mat:] = np.clip(mutated[:, self.n_dims_mat:], self.lb_ops, a_max = self.ub_ops)

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


    def encode(self, individuals: np.ndarray) -> np.ndarray:
        """
        Encodes the individuals into a 2D array of integers.
        We get individuals with 21//chunk_size real params for the matrix and one real param for operations.
        We clip these real values to the respective bounds and encode them to:
            - binary [21 bits] for the matrix, by concatenating the bits of each integer
            - ternary [5 bits] for the operations, simply converting the single int to ternary
        Binary and ternary are concatenated to form the final encoded individual.
        Resulting array has shape (26 x individuals.shape[0]).
        """
        
        encoded = np.zeros((individuals.shape[0], 26), dtype=int)
        
        # encode the matrix part
        for i in range(self.n_dims_mat):
            encoded[:, i * self.chunk_size: (i + 1) * self.chunk_size] = self.encode_param_bin(individuals[:, i], self.chunk_size)
        
        # encode the operations integer
        encoded[:, -5:] = self.encode_param_ter(individuals[:, -1])

        return encoded
    

    def encode_param_bin(self, real_values: np.ndarray, chunk_size: int) -> np.ndarray:
        """ Encodes a slice of real values to binary: (pop_size x 1) -> (pop_size x chunk_size). """
        
        # encode to binary
        encoded = np.zeros((real_values.shape[0], chunk_size), dtype=int)
        for i in range(chunk_size):
            encoded[:, i] = np.floor(np.round(real_values) / 2**i) % 2
        
        return encoded
    
    def encode_param_ter(self, real_values: np.array) -> np.ndarray:
        """ Encodes a slice of real values to ternary: (ps x 1) -> (ps x 5). """

        # encode to ternary
        encoded = np.zeros((real_values.shape[0], 5), dtype=int)
        for i in range(5):
            encoded[:, i] = np.floor(np.round(real_values) / 3**i) % 3
        
        return encoded


    def print_as_matrix(self, individual: np.ndarray) -> None:
        """ Prints matrix part of the individual as a 7x7 triu matrix. """

        mat = np.ones((7, 7), dtype=int)
        mat = np.triu(mat, 1)
        mat[mat == 1] = individual[:21]
        print(f'x_opt_mat:\n{mat}')


    def validate_parameters(
        self,
        problem: ioh.problem.Integer,
        pop_size: int,
        mu_: int,
        lambda_: int,
        tau_: float,
        sigma_: float,
        chunk_size: int,
        budget: int,
        recombination: str,
        individual_sigmas: bool,
        run_id: any,
        verbose: bool
        ) -> None:

        """ Validates all parameters passed to the constructor """
        
        assert isinstance(problem, ioh.ProblemType), "problem must be an instance of <ioh.ProblemType>"

        assert isinstance(pop_size, int), "population size must be an integer"
        assert pop_size in range(0, 500), "population size must be between 0 and 500"

        assert isinstance(mu_, int), "mu_ must be an integer"
        assert mu_ > 0, "mu_ must be greater than 0"
        assert mu_ < pop_size, "mu_ must be less than population size"

        assert isinstance(lambda_, int), "lambda_ must be an integer"
        assert lambda_ > 0, "lambda_ must be greater than 0"
        assert lambda_ <= pop_size, "lambda_ must be less than or equal to population size"

        assert pop_size == lambda_ + mu_ or pop_size == lambda_, "population size must be" + \
        " either number of parents + number of offspring or just number of offspring"

        assert isinstance(tau_, float), "tau_ must be a float"
        assert tau_ > 0, "tau_ must be greater than 0"
        assert tau_ < 1, "tau_ must be less than 1"

        assert isinstance(sigma_, float), "sigma_ must be a float"
        assert sigma_ > 0, "sigma_ must be greater than 0"
        assert sigma_ < 1, "sigma_ must be less than 1"

        assert isinstance(chunk_size, int), "chunk size must be an integer"
        assert chunk_size > 0, "chunk size must be greater than 0"
        assert 21 % chunk_size == 0, "chunk size a divisor of 21"

        assert isinstance(budget, int), "budget must be an integer"
        assert budget > 0, "budget must be greater than 0"
        assert budget < 10_000_000, "budget must be less than 100 million"

        assert recombination in ['d', 'i', 'dg', 'ig'], "recombination must be one of the following: 'd', 'i', 'dg', 'ig'"

        assert isinstance(individual_sigmas, bool), "individual_sigmas must be a boolean"
        
        assert len(str(run_id)) > 0, "run_id must be representable as a string"

        assert isinstance(verbose, bool), "verbose must be a boolean"

        return
