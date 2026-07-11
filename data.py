"""Colored MNIST dataset construction (Arjovsky et al., 2019)."""

import os
import random
import torch
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_envs(data_dir='./data', seed=0):
    set_seeds(seed)

    mnist = datasets.MNIST(data_dir, train=True, download=True)
    imgs = mnist.data[:, ::2, ::2].float()
    labels = mnist.targets

    splits = [
        (imgs[:25000], labels[:25000]),
        (imgs[25000:50000], labels[25000:50000]),
        (imgs[50000:], labels[50000:]),
    ]

    flip_probs = [0.1, 0.2, 0.9]

    envs = []
    for (x, y), flip_p in zip(splits, flip_probs):
        n = len(x)
        y_bin = (y >= 5).float() # binarize labels
        y_bin = (y_bin - (torch.rand(n) < 0.25).float()).abs() # flip labels
        colors = (y_bin - (torch.rand(n) < flip_p).float()).abs() # assign colors
        x_2ch = torch.stack([x, x], dim=1) # (N, 2, 14, 14)
        x_2ch[torch.arange(n), (1 - colors).long()] = 0

        envs.append({
            'images': (x_2ch / 255.).reshape(n, -1),
            'labels': y_bin[:, None],
            'colors': colors[:, None],
        })

    return envs


def save_sample_grid(envs, path='figures/sample_envs.png', n_cols=10):
    env_names = ['Env1 (90% corr)', 'Env2 (80% corr)', 'Test (10% corr)']
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 1.2, 4.2))

    for row, (env, name) in enumerate(zip(envs, env_names)):
        x = env['images'][:n_cols].reshape(-1, 2, 14, 14)
        y = env['labels'][:n_cols, 0]

        for col in range(n_cols):
            rgb = torch.zeros(14, 14, 3)
            rgb[:, :, 0] = x[col, 0]
            rgb[:, :, 1] = x[col, 1]

            axes[row, col].imshow(rgb.numpy(), interpolation='nearest')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f'y={int(y[col])}', fontsize=8)
        axes[row, 0].set_ylabel(name, fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    envs = make_envs()
    for i, name in enumerate(['Env1 (90%)', 'Env2 (80%)', 'Test (10%)']):
        n = len(envs[i]['labels'])
        agreement = envs[i]['colors'].eq(envs[i]['labels']).float().mean()
        print(f'{name}: {n:,} samples, color-label agreement: {agreement:.1%}')

    save_sample_grid(envs)
