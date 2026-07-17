"""Linear probing on frozen representations."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from data import set_seeds, make_envs
from model import MLP
from train import accuracy


plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linestyle': '-',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

ERM_COLOR = '#D55E00' 
IRM_COLOR = '#0072B2'  


def extract_features(model, envs, device):
    model.eval()
    feats, labels, colors = [], [], []
    with torch.no_grad():
        for env in envs:
            feats.append(model.features(env['images'].to(device)).cpu().numpy())
            labels.append(env['labels'].cpu().numpy())
            colors.append(env['colors'].cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels).ravel(), np.concatenate(colors).ravel()


def train_probe(X_tr, y_tr, X_te, y_te):
    return LogisticRegression(max_iter=1000, C=1.0).fit(X_tr, y_tr).score(X_te, y_te)

def plot_accuracy(erm_m, erm_s, irm_m, irm_s):
    labels = ['Env1\n(90% corr)', 'Env2\n(80% corr)', 'Test\n(10% corr)']
    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, erm_m, w, yerr=erm_s, label='ERM', color=ERM_COLOR,
           capsize=3, error_kw={'linewidth': 1.2})
    ax.bar(x + w/2, irm_m, w, yerr=irm_s, label='IRM', color=IRM_COLOR,
           capsize=3, error_kw={'linewidth': 1.2})

    ax.axhline(0.50, color='gray', ls='--', lw=0.8, alpha=0.6, label='Chance')
    ax.axhline(0.75, color='gray', ls=':', lw=0.8, alpha=0.6, label='Shape ceiling')
    ax.set_ylabel('Accuracy')
    ax.set_title('ERM vs IRM Generalization')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', framealpha=0.9)

    for bars, means in [(ax.patches[:3], erm_m), (ax.patches[3:], irm_m)]:
        for bar, val in zip(bars, means):
            y = bar.get_height()
            if y > 0.85:
                ax.text(bar.get_x() + bar.get_width()/2, y - 0.06,
                        f'{val:.1%}', ha='center', va='top', fontsize=9,
                        fontweight='bold', color='white')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, y + 0.02,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/erm_vs_irm_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/erm_vs_irm_accuracy.png')


def plot_probes(means, stds):
    """Color vs Digit probe bar chart."""
    labels = ['Color probe\n(spurious)', 'Digit probe\n(causal)']
    x = np.arange(len(labels))
    w = 0.32

    erm_vals = [means['erm_color'], means['erm_digit']]
    irm_vals = [means['irm_color'], means['irm_digit']]
    erm_err = [stds['erm_color'], stds['erm_digit']]
    irm_err = [stds['irm_color'], stds['irm_digit']]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - w/2, erm_vals, w, yerr=erm_err, label='ERM repr.', color=ERM_COLOR,
           capsize=3, error_kw={'linewidth': 1.2})
    ax.bar(x + w/2, irm_vals, w, yerr=irm_err, label='IRM repr.', color=IRM_COLOR,
           capsize=3, error_kw={'linewidth': 1.2})

    ax.axhline(0.50, color='gray', ls='--', lw=0.8, alpha=0.6, label='Chance')
    ax.set_ylabel('Probe accuracy')
    ax.set_title('Representation probing')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', framealpha=0.9)

    for bars, vals in [(ax.patches[:2], erm_vals), (ax.patches[2:], irm_vals)]:
        for bar, val in zip(bars, vals):
            y = bar.get_height()
            if y > 0.85:
                ax.text(bar.get_x() + bar.get_width()/2, y - 0.06,
                        f'{val:.1%}', ha='center', va='top', fontsize=9,
                        fontweight='bold', color='white')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, y + 0.02,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/probe_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/probe_results.png')


def plot_dynamics(erm_hist, irm_hist, anneal_step=190):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    erm_train = (erm_hist[:, 1] + erm_hist[:, 2]) / 2
    irm_train = (irm_hist[:, 1] + irm_hist[:, 2]) / 2

    ax.plot(erm_hist[:, 0], erm_train, color=ERM_COLOR, ls='--', alpha=0.5, lw=1.2, label='ERM train')
    ax.plot(erm_hist[:, 0], erm_hist[:, 3], color=ERM_COLOR, lw=2.2, label='ERM test')
    ax.plot(irm_hist[:, 0], irm_train, color=IRM_COLOR, ls='--', alpha=0.5, lw=1.2, label='IRM train')
    ax.plot(irm_hist[:, 0], irm_hist[:, 3], color=IRM_COLOR, lw=2.2, label='IRM test')

    ax.axvline(anneal_step, color='black', ls=':', lw=1, alpha=0.6)
    ax.text(anneal_step + 8, 0.08, 'penalty on', fontsize=9, alpha=0.7)

    ax.axhline(0.50, color='gray', ls='--', lw=0.7, alpha=0.4)
    ax.axhline(0.75, color='gray', ls=':', lw=0.7, alpha=0.4)
    ax.text(515, 0.505, '50%', fontsize=8, alpha=0.5, va='bottom')
    ax.text(515, 0.755, '75%', fontsize=8, alpha=0.5, va='bottom')

    ax.set_xlabel('Training step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training dynamics (ERM vs IRM)')
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), framealpha=0.9, borderaxespad=0)

    plt.tight_layout()
    plt.savefig('figures/training_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/training_dynamics.png')


def plot_pca(erm_model, irm_model, envs, device):
    test_imgs = envs[2]['images'].to(device)
    test_labels = envs[2]['labels'].cpu().numpy().ravel()
    test_colors = envs[2]['colors'].cpu().numpy().ravel()

    with torch.no_grad():
        erm_feats = erm_model.features(test_imgs).cpu().numpy()
        irm_feats = irm_model.features(test_imgs).cpu().numpy()

    erm_pca = PCA(n_components=2).fit_transform(erm_feats)
    irm_pca = PCA(n_components=2).fit_transform(irm_feats)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    n = min(2000, len(test_labels))
    idx = np.random.permutation(len(test_labels))[:n]

    for row, (pca, name) in enumerate([(erm_pca, 'ERM'), (irm_pca, 'IRM')]):
        for val, c, lab in [(0, IRM_COLOR, 'digit 0\u20134'), (1, ERM_COLOR, 'digit 5\u20139')]:
            m = test_labels[idx] == val
            axes[row, 0].scatter(pca[idx[m], 0], pca[idx[m], 1],
                                 c=c, s=4, alpha=0.35, label=lab, edgecolors='none')
        axes[row, 0].set_title(f'{name} \u2014 colored by digit label')
        axes[row, 0].legend(markerscale=3, fontsize=9, loc='best', framealpha=0.9)

        for val, c, lab in [(0, '#d62728', 'red'), (1, '#2ca02c', 'green')]:
            m = test_colors[idx] == val
            axes[row, 1].scatter(pca[idx[m], 0], pca[idx[m], 1],
                                 c=c, s=4, alpha=0.35, label=lab, edgecolors='none')
        axes[row, 1].set_title(f'{name} \u2014 colored by input color')
        axes[row, 1].legend(markerscale=3, fontsize=9, loc='best', framealpha=0.9)

    for ax in axes.flat:
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.tick_params(labelsize=0, length=0)

    fig.suptitle('Representation structure', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('figures/pca_representations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/pca_representations.png')


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEEDS = [0, 1, 2]

    all_erm_accs, all_irm_accs = [], []
    erm_color_p, erm_digit_p, irm_color_p, irm_digit_p = [], [], [], []


    for seed in SEEDS:
        set_seeds(seed)
        envs = make_envs(seed=seed)

        erm_model, irm_model = MLP().to(device), MLP().to(device)
        erm_model.load_state_dict(torch.load(f'erm_model_seed{seed}.pt', map_location=device))
        irm_model.load_state_dict(torch.load(f'irm_model_seed{seed}.pt', map_location=device))

        all_erm_accs.append([accuracy(erm_model, e['images'].to(device), e['labels'].to(device)) for e in envs])
        all_irm_accs.append([accuracy(irm_model, e['images'].to(device), e['labels'].to(device)) for e in envs])

        erm_feats, labels, colors = extract_features(erm_model, envs, device)
        irm_feats, _, _ = extract_features(irm_model, envs, device)

        idx = np.random.permutation(len(labels))
        s = int(len(labels) * 0.8)
        tr, te = idx[:s], idx[s:]
        erm_color_p.append(train_probe(erm_feats[tr], colors[tr], erm_feats[te], colors[te]))
        erm_digit_p.append(train_probe(erm_feats[tr], labels[tr], erm_feats[te], labels[te]))
        irm_color_p.append(train_probe(irm_feats[tr], colors[tr], irm_feats[te], colors[te]))
        irm_digit_p.append(train_probe(irm_feats[tr], labels[tr], irm_feats[te], labels[te]))

    erm_m, erm_s = np.mean(all_erm_accs, 0), np.std(all_erm_accs, 0)
    irm_m, irm_s = np.mean(all_irm_accs, 0), np.std(all_irm_accs, 0)

    probe_means = dict(erm_color=np.mean(erm_color_p), erm_digit=np.mean(erm_digit_p),
                       irm_color=np.mean(irm_color_p), irm_digit=np.mean(irm_digit_p))
    probe_stds = dict(erm_color=np.std(erm_color_p), erm_digit=np.std(erm_digit_p),
                      irm_color=np.std(irm_color_p), irm_digit=np.std(irm_digit_p))

    print('\n' + '='*60)
    print(f"{'Probe target':<22} | {'ERM features':<16} | {'IRM features':<16}")
    print('-'*60)
    for name, ek, ik in [('Color (spurious)', 'erm_color', 'irm_color'),
                          ('Digit (causal)',   'erm_digit', 'irm_digit')]:
        print(f'{name:<22} | {probe_means[ek]:.1%} ({probe_stds[ek]:.1%})'
              f'       | {probe_means[ik]:.1%} ({probe_stds[ik]:.1%})')
    print('='*60)

    plot_accuracy(erm_m, erm_s, irm_m, irm_s)
    plot_probes(probe_means, probe_stds)

    erm_hist_path = 'erm_history.npy'
    irm_hist_path = 'irm_history.npy'
    if os.path.exists(erm_hist_path) and os.path.exists(irm_hist_path):
        plot_dynamics(np.load(erm_hist_path), np.load(irm_hist_path))
    else:
        print('Skipping dynamics plot (re-run train.py to generate history files)')

    set_seeds(0)
    envs_0 = make_envs(seed=0)
    erm_0, irm_0 = MLP().to(device), MLP().to(device)
    erm_0.load_state_dict(torch.load('erm_model_seed0.pt', map_location=device))
    irm_0.load_state_dict(torch.load('irm_model_seed0.pt', map_location=device))
    plot_pca(erm_0, irm_0, envs_0, device)
