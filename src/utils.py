import os
from warnings import warn
from time import perf_counter

def get_directories(parent_file: str) -> dict[str, str]:
    """Returns a tuple of directories to be used in the program."""
    src = os.path.dirname(os.path.abspath(parent_file))
    root = os.sep.join(src.split(os.sep)[:-1])
    data = os.path.join(root, 'data')
    results = os.path.join(root, 'results')
    plots = os.path.join(results, 'plots')
    
    dirs: dict[str, str] = {}
    for directory in (data, results, plots):
        basename = os.path.basename(directory)
        if not os.path.exists(directory):
            os.mkdir(directory)
            print()
            warn(f'Created empty {basename} directory at "{directory}".')
            print()
        dirs[basename] = directory + '/'
    return dirs

class ProgressBar:
    frames = [f'\033[32m\033[1m{s}\033[0m' for s in ['╀', '╄', '┾', '╆', '╁', '╅', '┽', '╃']]   # spinner frames
    done_char = '\033[32m\033[1m━\033[0m'   # green bold ━, reset after
    todo_char = '\033[31m\033[2m─\033[0m'   # red faint ─, reset after
    spin_frame = 0

    def __init__(self, n_iters: int, run_id: str | None) -> None:
        if run_id is not None:
            print(run_id)
        self.n_iters = n_iters
        self.len_n_iters = len(str(n_iters))
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
