import os
from warnings import warn
from time import perf_counter
from argparse import ArgumentParser

def get_directories(parent_file: str) -> dict[str, str]:
    """Returns a tuple of directories to be used in the program."""
    src = os.path.dirname(os.path.abspath(parent_file))
    root = os.sep.join(src.split(os.sep)[:-1])
    data = os.path.join(root, 'data')
    results = os.path.join(root, 'results')
    pkl = os.path.join(results, 'pkl')
    plots = os.path.join(results, 'plots')
    
    dirs: dict[str, str] = {}
    for directory in (data, results, pkl, plots):
        basename = os.path.basename(directory)
        if not os.path.exists(directory):
            os.mkdir(directory)
            print()
            warn(f'Created empty {basename} directory at "{directory}".')
            print()
        dirs[basename] = directory + '/'
    return dirs


class ParseWrapper:

    valid_long: dict[str, tuple[int | float]] = {
        'seed': (0, 999999)
    }
    short_args = ['s']
    valid: dict[str, tuple[int | float]] = dict(zip(short_args, list(valid_long.values())))

    def __init__(self, parser: ArgumentParser) -> None:

        parser.add_argument('-s', '--seed', type=int, default=42,
                            help=("dimensions of the environment " +
                            f"[{self.valid['s'][0]}-{self.valid['s'][1]}] "))
        parser.add_argument('-o', '--overwrite', action='store_true',
                            help=("overwrite existing data if already present"))
        parser.add_argument('-v', '--verbose', action='store_true',
                            help=("print progress (and more) to stdout"))

        self.args = parser.parse_args()
        self.argdict = vars(self.args)
        self.check_validity()

    def __call__(self) -> dict[str, int | bool]:
        print('\nExperiment will be ran with the following parameters:')
        for arg, value in self.argdict.items():
            print(f'{arg:>10}|{value}')
        return self.argdict

    def check_validity(self) -> None:
        for arg, value in self.argdict.items():
            if value is None: continue
            if type(value) in (str, bool): continue
            if value < self.valid_long[arg][0] or value > self.valid_long[arg][1]:
                raise ValueError(f'Invalid value for argument "{arg}": {value}\n' +
                                 f"Please choose between {self.valid_long[arg][0]} and {self.valid_long[arg][1]}")


class ProgressBar:
    frames = [f'\033[32m\033[1m{s}\033[0m' for s in ['╀', '╄', '┾', '╆', '╁', '╅', '┽', '╃']]   # spinner frames
    done_char = '\033[32m\033[1m━\033[0m'   # green bold ━, reset after
    todo_char = '\033[31m\033[2m─\033[0m'   # red faint ─, reset after
    spin_frame = 0

    def __init__(self, n_iters: int, run_id: str) -> None:
        self.n_iters = n_iters
        self.len_n_iters = len(str(n_iters))
        print(run_id)
        print('\r' + 50 * self.todo_char + ' ' + self.frames[0] + ' 0%', end='')
        self.start_ts = perf_counter()

    def __call__(self, iteration: int) -> None:
        """Updates and displays a progress bar on the command line"""
        percentage = 100 * (iteration+1) // self.n_iters            # floored percentage
        if percentage == 100 * iteration // self.n_iters: return    # prevent printing same line multiple times
        steps = 50 * (iteration+1) // self.n_iters                  # chars representing progress
        self.spin_frame += 1

        spin_char = self.frames[self.spin_frame%8]
        bar = (steps)*self.done_char + (50-steps)*self.todo_char        # the actual bar
        
        runtime = perf_counter() - self.start_ts
        if iteration+1 == self.n_iters:             # flush last suffix with spaces and place carriage at newline
            suffix = ' completed in ' + f'{runtime:.2f} sec'  + ' ' * 50 + '\n'
        else:                                       # print iteration number
            percentage_float = (100 * (iteration+1) / self.n_iters)
            eta = (100-percentage_float) / percentage_float * runtime
            suffix = f' {str(iteration+1).zfill(self.len_n_iters)}/{self.n_iters} (ETA {eta:.1f} sec) '
        
        print('\r' + bar + suffix, end='')
        return
