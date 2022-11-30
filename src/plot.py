import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import get_directories

DIRS = get_directories(__file__)


def plot_results(run_id: str, save: bool = False):
    df = pd.read_csv(DIRS['csv'] + f'{run_id}.csv', index_col=0).cummax() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in df.columns:
        ax.plot(df.index, df[i], color='tab:blue', alpha=0.2, linewidth=0.75)
    ax.plot(df.mean(axis=1), color='tab:blue', linewidth=2, label=f'max: {df.mean(axis=1).iloc[-1]:.4f}%')
    ax.fill_between(df.index, df.min(axis=1), df.max(axis=1), color='tab:blue', alpha=0.2)
    ax.set_xlabel('generation'); ax.set_ylabel('fitness (%)')
    ax.legend()
    ax.set_title(
        run_id.replace('_', ' ').replace('GA', 'Genetic Algorithm').replace('ES', 'Evolution Strategy'),
        weight = 'bold'
    )
    fig.tight_layout()
    if save: fig.savefig(DIRS['plots'] + f'{run_id}.png', dpi=300)
    else: plt.show()
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('run_id', type=str, help='run id')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    plot_results(args.run_id, args.save)
