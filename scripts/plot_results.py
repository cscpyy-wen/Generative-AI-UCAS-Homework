"""Plot recorded or newly trained histories and complete generated batches."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('reproduced/figures'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix',
                         'font.size': 10, 'axes.linewidth': .65, 'lines.linewidth': 1.2,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'pdf.fonttype': 42, 'savefig.dpi': 300})
    standard, fine = plt.subplots(1, 2, figsize=(10, 3.5))[0], plt.subplots(1, 2, figsize=(10, 3.5))[0]
    standard_axes, fine_axes = standard.axes, fine.axes
    for name, linestyle in zip(['s0', 's1', 's2', 's3', 'g', 'f0', 'f02'],
                               ['-.', '-', '--', ':', '-', '-', '--']):
        directory = args.input / name
        if not (directory / 'history.csv').exists():continue
        with (directory / 'history.csv').open() as f:rows = list(csv.DictReader(f))
        x = [int(r['epoch']) for r in rows]
        if name in ('f0', 'f02'):
            for key, style in [('raw_val_nll', '-'), ('ema_val_nll', '--')]:
                fine_axes[0].plot(x, [float(r[key]) for r in rows], style, label=name.upper() + ' ' + key.split('_')[0])
            online = [r for r in rows if int(r['epoch']) > 0]
            fine_axes[1].plot([int(r['epoch']) for r in online],
                              [float(r['train_nll']) for r in online], linestyle, label=name.upper())
        else:
            axes = standard_axes[1 if name == 'g' else 0]
            axes.plot(x, [float(r['val_nll']) for r in rows], linestyle, label=name.upper())
            if name == 'g':axes.plot(x, [float(r['train_nll']) for r in rows], '--', label='G training (online)')
        samples_path = directory / 'samples_final.npy'
        if not samples_path.exists():samples_path = directory / 'samples.npy'
        if samples_path.exists():
            samples = np.load(samples_path).reshape(-1, 28, 28)
            figure, axes = plt.subplots(8, 8, figsize=(6, 6))
            for ax, sample in zip(axes.flat, samples):
                ax.imshow(sample, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
                ax.set_axis_off()
            figure.subplots_adjust(wspace=.05, hspace=.05)
            for ext in ['png', 'pdf']:figure.savefig(args.output / (name + '-samples.' + ext), bbox_inches='tight')
            plt.close(figure)
    standard_axes[0].set_yscale('log')
    for figure in [standard, fine]:
        for ax in figure.axes:
            ax.set(xlabel='Additional epoch' if figure is fine else 'Epoch', ylabel='NLL (nats/image)')
            ax.legend(fontsize=8, frameon=True)
        figure.tight_layout()
    for figure, name in [(standard, 'standard-and-gated-history'), (fine, 'fine-tuning-history')]:
        for ext in ['png', 'pdf']:figure.savefig(args.output / (name + '.' + ext), bbox_inches='tight')
        plt.close(figure)
    print('Figures saved to', args.output)


if __name__ == '__main__':main()
