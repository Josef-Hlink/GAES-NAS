from argparse import ArgumentParser
from time import perf_counter



class ParseWrapper:
    """ Base class, implemented by ParseWrapperES and ParseWrapperGA """

    def __init__(self, parser: ArgumentParser) -> None:

        parser.add_argument('-b', dest='budget', type=int, default=5_000,
                            help="Number of function evaluations.")
        parser.add_argument('-p', dest='population_size', type=int, default=100,
                            help="Population size: [10, 1000].")
        parser.add_argument('-m', dest='mu_', type=int, default=40,
                            help="Number of parents: [2, pop_size-1].")
        parser.add_argument('-l', dest='lambda_', type=int, default=100,
                            help="Number of offspring: [2, pop_size].")

        parser.add_argument('-P', '--problem_path', type=str, default='nasbench_only108.tfrecord',
                            help=f'path to the problem file.')
        parser.add_argument('-R', '--repetitions', type=int, default=20,
                            help="Number of repetitions for each experiment: [1, 100].")
        parser.add_argument('-S', '--seed', type=int, default=42,
                            help="Seed for the random number generator: [0, 999999].")
        parser.add_argument('--no-log', action='store_true',
                            help="Do not attach an IOH logger to the problem.")

    def __call__(self) -> dict[str, any]:
        
        print('-' * 80)
        print('Experiment will be ran with the following parameters:')
        for arg, value in self.args.items():
            if self.defaults[arg] != value:
                print(f'\033[1m{arg:>19}\033[0m | {value}')
            else:
                if arg == 'mut_r': value = f'{value:.3f}'
                print(f'{arg:>19} | {value}')
        print('-' * 80)
        
        return self.args

    def validate_args(self) -> None:

        assert self.args['budget'] in range(1, 1_000_001), "Budget must be in [1, 1 million]."

        assert self.args['population_size'] in range(10, 1001), "Population size must be in [10, 1000]."

        assert self.args['mu_'] in range(2, self.args['population_size']), "Number of parents must be in [2, pop_size-1]."

        assert self.args['lambda_'] in range(2, self.args['population_size']+1), "Number of offspring must be in [2, pop_size]."

        assert self.args['mu_'] + self.args['lambda_'] == self.args['population_size'] \
            or self.args['lambda_'] == self.args['population_size'], \
            "mu + lambda must be popsize or lambda must be popsize."

        assert self.args['repetitions'] in range(1, 101), "Number of repetitions must be in [1, 100]."

        assert self.args['seed'] in range(0, 1_000_000), "Seed must be an integer in [0, 999_999]."

        return


class ParseWrapperES(ParseWrapper):

    def __init__(self, parser: ArgumentParser) -> None:

        super().__init__(parser)
        BOLD = lambda x: f'\033[1m{x}\033[0m'

        parser.add_argument('-s', dest='sigma_', type=float, default=0.01,
                            help='Initial mutation strength (sigma): [0.001, 1].')
        parser.add_argument('-t', dest='tau_', type=float, default=0.1,
                            help='Perturbation rate for sigma: [0.01, 1].')
        parser.add_argument('--rec', dest='recombination', type=str, default='d',
                            choices=['d', 'i', 'dg', 'ig'],
                            help=f'''
                                Recombination method.
                                {BOLD("d")}iscrete,
                                {BOLD("i")}ntermediate,
                                {BOLD("d")}iscrete {BOLD("g")}lobal,
                                {BOLD("i")}ntermediate {BOLD("g")}lobal.
                            ''')
        parser.add_argument('--is', dest='individual_sigmas', action='store_true',
                            help='Use individual mutation strengths (sigmas).')
        parser.add_argument('--cs', dest='chunk_size', type=int, default=7,
                            help='Number of bits represented by one real value.')

        self.defaults = vars(parser.parse_args([]))
        self.args = vars(parser.parse_args())
        self.validate_args()

    def validate_args(self) -> None:

        super().validate_args()    
        
        assert self.args['sigma_'] >= 0.001 and self.args['sigma_'] <= 1, "Initial mutation strength (sigma) must be in [0.001, 1]."

        assert self.args['tau_'] >= 0.01 and self.args['tau_'] <= 1, "Perturbation rate for sigma must be in [0.01, 1]."

        assert self.args['chunk_size'] in range(1, 22), "Chunk size must be an int in [1, 21]."
        assert 21 % self.args['chunk_size'] == 0, "Chunk size must be a divisor of 21."

        return


class ParseWrapperGA(ParseWrapper):

    def __init__(self, parser: ArgumentParser) -> None:
        
        super().__init__(parser)
        BOLD = lambda x: f'\033[1m{x}\033[0m'

        parser.add_argument('--sel', dest='selection', type=str, default='ts',
                            choices=['ts', 'rw', 'rk', 'su'],
                            help=f'''
                                Selection method.
                                {BOLD("t")}ournament {BOLD("s")}election,
                                {BOLD("r")}oulette {BOLD("w")}heel selection,
                                {BOLD("r")}an{BOLD("k")} selection,
                                {BOLD("s")}tochastic {BOLD("u")}niversal sampling.
                            ''')
        parser.add_argument('--mut', dest='mutation', type=str, default='u',
                            choices=['u', 'b'],
                            help=f'''
                                Mutation method.
                                {BOLD("u")}niform,
                                {BOLD("b")}itflip.
                            ''')
        parser.add_argument('--rec', dest='recombination', type=str, default='kp',
                            choices=['kp', 'u'],
                            help=f'''
                                Recombination method.
                                {BOLD("k")}-point,
                                {BOLD("u")}niform.
                            ''')
        parser.add_argument('--xp', dest='xp', type=int, default=1,
                            help=f'({BOLD("kp only")}) Number of crossover points in recombination.')
        parser.add_argument('--mb', dest='mut_b', type=int, default=1,
                            help=f'({BOLD("b only")}) Number of bits to flip in mutation.')
        parser.add_argument('--mr', dest='mut_r', type=float, default=1/26,
                            help=f'({BOLD("u only")}) Mutation rate: [0, 1].')

        self.defaults = vars(parser.parse_args([]))
        self.args = vars(parser.parse_args())
        
        self.validate_args()
    
    def validate_args(self) -> None:
            
        super().validate_args()

        assert self.args['xp'] in range(1, 26), "Number of crossover points must be in [1, 25]."

        assert self.args['mut_b'] in range(1, 26), "Number of bits to flip must be in [1, 25]."

        assert self.args['mut_r'] >= 0 and self.args['mut_r'] <= 1, "Mutation rate must be in [0, 1]."

        return



class ProgressBar:
    done_char = '\033[32m' + '\033[1m' + '\u2501' + '\033[0m'   # green bold ━, reset after
    todo_char = '\033[31m' + '\033[2m' + '\u2500' + '\033[0m'   # red faint ─, reset after

    def __init__(self, n_iters: int, p_id: str) -> None:
        self.n_iters = n_iters
        self.len_n_iters = len(str(n_iters))
        start_suffix = ' ' + '0'.zfill(self.len_n_iters) + '/' + str(n_iters)
        print(p_id)
        print('\r' + 50 * self.todo_char + start_suffix, end='')
        self.start_ts = perf_counter()

    def __call__(self, iteration: int) -> None:
        """Updates and displays a progress bar on the command line"""
        steps = 50 * (iteration+1) // self.n_iters                  # chars representing progress
        bar = (steps)*self.done_char + (50-steps)*self.todo_char    # the actual bar
        
        runtime = perf_counter() - self.start_ts
        if iteration+1 == self.n_iters:
            suffix = ' completed in ' + f'{runtime:.2f} sec'
            # add 30 - len(suffix) spaces to clear the line and move carriage to newline
            suffix += ' ' * (30 - len(suffix)) + '\n'
        else:                                       # print iteration number
            percentage_float = (100 * (iteration+1) / self.n_iters)
            eta = (100-percentage_float) / percentage_float * runtime
            suffix = f' {str(iteration+1).zfill(self.len_n_iters)}/{self.n_iters} (ETA {eta:.1f} sec) '
        
        print('\r' + bar + suffix, end='')
        return
