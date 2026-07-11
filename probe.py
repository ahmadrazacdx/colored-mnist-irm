"""Linear probing on frozen representations (color vs. digit class)"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from data import set_seeds, make_envs
from model import MLP
from train import accuracy


def extract_features_and_targets(model, envs, device):
    """Extract penultimate-layer features, digit labels, and colors."""
    model.eval()
    all_feats, all_labels, all_colors = [], [], []
    with torch.no_grad():
        for env in envs:
            imgs = env['images'].to(device)
            feats = model.features(imgs)
            
            all_feats.append(feats.cpu().numpy())
            all_labels.append(env['labels'].cpu().numpy())
            all_colors.append(env['colors'].cpu().numpy())
            
    return (
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_labels, axis=0).ravel(),
        np.concatenate(all_colors, axis=0).ravel()
    )


def train_probe(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

def generate_accuracy_plot(erm_means, erm_stds, irm_means, irm_stds):
    envs = ['Env1 (90% corr)', 'Env2 (80% corr)', 'Test (10% corr)']
    x = np.arange(len(envs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    eb_kwargs = dict(ecolor='black', capsize=4, alpha=0.7)

    rects1 = ax.bar(x - width/2, erm_means, width, yerr=erm_stds, label='ERM', color='#e74c3c', error_kw=eb_kwargs)
    rects2 = ax.bar(x + width/2, irm_means, width, yerr=irm_stds, label='IRM', color='#2ecc71', error_kw=eb_kwargs)

    ax.set_ylabel('Accuracy')
    ax.set_title('ERM vs. IRM Generalization Performance (3 Seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax.axhline(y=0.75, color='blue', linestyle=':', alpha=0.5, label='Shape Ceiling (75%)')
    ax.legend()

    def autolabel(rects, means):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{means[idx]:.1%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, erm_means)
    autolabel(rects2, irm_means)

    plt.tight_layout()
    plt.savefig('figures/erm_vs_irm_accuracy.png', dpi=150)
    plt.close()
    print("Saved figures/erm_vs_irm_accuracy.png")

def generate_probe_plot(probe_means, probe_stds):
    categories = ['Color Probe (Spurious)', 'Digit Probe (Causal)']
    x = np.arange(len(categories))
    width = 0.35

    erm_scores = [probe_means['erm_color'], probe_means['erm_digit']]
    irm_scores = [probe_means['irm_color'], probe_means['irm_digit']]
    
    erm_errs = [probe_stds['erm_color'], probe_stds['erm_digit']]
    irm_errs = [probe_stds['irm_color'], probe_stds['irm_digit']]
    eb_kwargs = dict(ecolor='black', capsize=4, alpha=0.7)

    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, erm_scores, width, yerr=erm_errs, label='ERM Representation', color='#e74c3c', error_kw=eb_kwargs)
    rects2 = ax.bar(x + width/2, irm_scores, width, yerr=irm_errs, label='IRM Representation', color='#2ecc71', error_kw=eb_kwargs)

    ax.set_ylabel('Probe Accuracy')
    ax.set_title('Representation Probing: What did the models learn? (3 Seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax.legend()

    def autolabel(rects, scores):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{scores[idx]:.1%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, erm_scores)
    autolabel(rects2, irm_scores)

    plt.tight_layout()
    plt.savefig('figures/probe_results.png', dpi=150)
    plt.close()
    print("Saved figures/probe_results.png")


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    SEEDS = [0, 1, 2]
    all_erm_accs = []
    all_irm_accs = []
    
    all_erm_color_probes = []
    all_erm_digit_probes = []
    all_irm_color_probes = []
    all_irm_digit_probes = []
    
    for seed in SEEDS:
        set_seeds(seed)
        envs = make_envs(seed=seed)
        erm_model = MLP().to(device)
        irm_model = MLP().to(device)
        
        erm_path = f'erm_model_seed{seed}.pt'
        irm_path = f'irm_model_seed{seed}.pt'
        
        if not os.path.exists(erm_path) or not os.path.exists(irm_path):
            raise FileNotFoundError(
                f"Missing checkpoints for seed {seed}. "
                f"Please run: python train.py --mode erm AND python train.py --mode irm"
            )
            
        erm_model.load_state_dict(torch.load(erm_path, map_location=device))
        irm_model.load_state_dict(torch.load(irm_path, map_location=device))
        erm_accs = [accuracy(erm_model, e['images'].to(device), e['labels'].to(device)) for e in envs]
        irm_accs = [accuracy(irm_model, e['images'].to(device), e['labels'].to(device)) for e in envs]
        all_erm_accs.append(erm_accs)
        all_irm_accs.append(irm_accs)
        
        erm_feats, labels, colors = extract_features_and_targets(erm_model, envs, device)
        irm_feats, _, _ = extract_features_and_targets(irm_model, envs, device)

        n_samples = len(labels)
        indices = np.random.permutation(n_samples)
        split = int(n_samples * 0.8)
        train_idx, test_idx = indices[:split], indices[split:]
        
        # Train Probes
        erm_color_acc = train_probe(erm_feats[train_idx], colors[train_idx], erm_feats[test_idx], colors[test_idx])
        erm_digit_acc = train_probe(erm_feats[train_idx], labels[train_idx], erm_feats[test_idx], labels[test_idx])
        
        irm_color_acc = train_probe(irm_feats[train_idx], colors[train_idx], irm_feats[test_idx], colors[test_idx])
        irm_digit_acc = train_probe(irm_feats[train_idx], labels[train_idx], irm_feats[test_idx], labels[test_idx])
        
        all_erm_color_probes.append(erm_color_acc)
        all_erm_digit_probes.append(erm_digit_acc)
        all_irm_color_probes.append(irm_color_acc)
        all_irm_digit_probes.append(irm_digit_acc)

    all_erm_accs = np.array(all_erm_accs)
    all_irm_accs = np.array(all_irm_accs)
    
    erm_means = all_erm_accs.mean(axis=0)
    erm_stds = all_erm_accs.std(axis=0)
    irm_means = all_irm_accs.mean(axis=0)
    irm_stds = all_irm_accs.std(axis=0)

    erm_color_mean, erm_color_std = np.mean(all_erm_color_probes), np.std(all_erm_color_probes)
    erm_digit_mean, erm_digit_std = np.mean(all_erm_digit_probes), np.std(all_erm_digit_probes)
    irm_color_mean, irm_color_std = np.mean(all_irm_color_probes), np.std(all_irm_color_probes)
    irm_digit_mean, irm_digit_std = np.mean(all_irm_digit_probes), np.std(all_irm_digit_probes)

    print("\n" + "="*70)
    print(f"{'PROBE TARGET':<25} | {'ERM FEATURES':<18} | {'IRM FEATURES':<18}")
    print("-"*70)
    print(f"{'Color (Spurious)':<25} | {erm_color_mean:.1%} ({erm_color_std:.1%})     | {irm_color_mean:.1%} ({irm_color_std:.1%})")
    print(f"{'Digit Label (Causal)':<25} | {erm_digit_mean:.1%} ({erm_digit_std:.1%})     | {irm_digit_mean:.1%} ({irm_digit_std:.1%})")
    print("="*70)

    generate_accuracy_plot(erm_means, erm_stds, irm_means, irm_stds)
    
    probe_means = {
        'erm_color': erm_color_mean,
        'erm_digit': erm_digit_mean,
        'irm_color': irm_color_mean,
        'irm_digit': irm_digit_mean
    }
    probe_stds = {
        'erm_color': erm_color_std,
        'erm_digit': erm_digit_std,
        'irm_color': irm_color_std,
        'irm_digit': irm_digit_std
    }
    generate_probe_plot(probe_means, probe_stds)
