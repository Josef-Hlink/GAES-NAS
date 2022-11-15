class ProgressBar:
    frames = [f'\033[32m\033[1m{s}\033[0m' for s in ['╀', '╄', '┾', '╆', '╁', '╅', '┽', '╃']]   # spinner frames
    done_char = '\033[32m\033[1m━\033[0m'   # green bold ━, reset after
    todo_char = '\033[31m\033[2m─\033[0m'   # red faint ─, reset after
    spin_frame = 0

    def __init__(self, n_iters: int, run_id: str | None) -> None:
        self.n_iters = n_iters
        if run_id is not None:
            print(run_id)
        print('\r' + 50 * self.todo_char + ' ' + self.frames[0] + ' 0%', end='')

    def __call__(self, iteration: int) -> None:
        """Updates and displays a progress bar on the command line"""
        percentage = 100 * (iteration+1) // self.n_iters    # floored percentage
        if percentage == 100 * iteration // self.n_iters: return    # prevent printing same line multiple times
        steps = 50 * (iteration+1) // self.n_iters          # chars representing progress
        self.spin_frame += 1

        spin_char = self.frames[self.spin_frame%8]
        bar = (steps)*self.done_char + (50-steps)*self.todo_char        # the actual bar
        
        if iteration+1 == self.n_iters: suffix = ' complete\n'
        else: suffix = ' ' + spin_char + ' ' + str(percentage) + '% '	# spinner and percentage
        print('\r' + bar + suffix, end='')
        return