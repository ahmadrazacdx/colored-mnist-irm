"""Causal and Representation Analysis for Invariant Risk Minimization (Colored MNIST).

Implements 4 clean Causal and Representation analyses:
1. Counterfactual Fairness & Do-Interventions (Pearl's SCM)
2. Subspace Alignment theta(S_shape, S_color)
3. Classifier Head Alignment in Feature Space (S_causal vs S_spurious)
4. Feature Covariance Spectrum & Effective Rank
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from data import set_seeds, make_envs
from model import MLP


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


# ==============================================================================
# 1. Counterfactual Fairness & Do-Interventions
# ==============================================================================
def analyze_counterfactual_fairness(erm_model, irm_model, envs, device):
    """Counterfactual Intervention do(Color = 1 - Color) holding Shape constant."""
    erm_model.eval()
    irm_model.eval()
    
    test_env = envs[2]
    images = test_env['images'].to(device)
    n = len(images)
    
    # Do-intervention: swap color channels (red <-> green)
    images_2d = images.reshape(n, 2, 14, 14)
    images_cf = torch.stack([images_2d[:, 1], images_2d[:, 0]], dim=1).reshape(n, -1)
    
    with torch.no_grad():
        erm_orig = erm_model(images).cpu().numpy().ravel()
        erm_cf = erm_model(images_cf).cpu().numpy().ravel()
        irm_orig = irm_model(images).cpu().numpy().ravel()
        irm_cf = irm_model(images_cf).cpu().numpy().ravel()
        
    erm_cfr = np.mean((erm_orig > 0) != (erm_cf > 0))
    irm_cfr = np.mean((irm_orig > 0) != (irm_cf > 0))
    
    erm_ces = np.mean(np.abs(erm_orig - erm_cf))
    irm_ces = np.mean(np.abs(irm_orig - irm_cf))
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    
    bars1 = axes[0].bar(['ERM', 'IRM'], [erm_cfr * 100, irm_cfr * 100], color=[ERM_COLOR, IRM_COLOR], width=0.4)
    axes[0].set_ylabel('Counterfactual Flip Rate (%)')
    axes[0].set_title(r'Flip Rate under $\text{do}(\text{Color})$', fontsize=10)
    axes[0].set_ylim(0, 100)
    for b in bars1:
        y = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2, y + 2, f'{y:.1f}%', ha='center', va='bottom', fontsize=8.5)
        
    bars2 = axes[1].bar(['ERM', 'IRM'], [erm_ces, irm_ces], color=[ERM_COLOR, IRM_COLOR], width=0.4)
    axes[1].set_ylabel(r'Logit Shift $|\Delta \text{Logit}|$')
    axes[1].set_title('Counterfactual Effect Size', fontsize=10)
    axes[1].set_ylim(0, max(erm_ces, irm_ces) * 1.25)
    for b in bars2:
        y = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8.5)
        
    plt.tight_layout()
    plt.savefig('figures/counterfactual_fairness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/counterfactual_fairness.png')
    return {'erm_cfr': erm_cfr, 'irm_cfr': irm_cfr, 'erm_ces': erm_ces, 'irm_ces': irm_ces}


# ==============================================================================
# 2. Subspace Orthogonality
# ==============================================================================
def analyze_subspace_orthogonality(erm_model, irm_model, envs, device):
    """Measure Angle theta(S_shape, S_color)."""
    erm_model.eval()
    irm_model.eval()
    
    def get_angle(model):
        feats_list, labels_list, colors_list = [], [], []
        with torch.no_grad():
            for env in envs[:2]:
                f = model.features(env['images'].to(device)).cpu().numpy()
                feats_list.append(f)
                labels_list.append(env['labels'].cpu().numpy().ravel())
                colors_list.append(env['colors'].cpu().numpy().ravel())
                
        X = np.concatenate(feats_list)
        Y_digit = np.concatenate(labels_list)
        Y_color = np.concatenate(colors_list)
        
        if len(X) > 5000:
            idx = np.random.choice(len(X), 5000, replace=False)
            X_sub, Y_digit_sub, Y_color_sub = X[idx], Y_digit[idx], Y_color[idx]
        else:
            X_sub, Y_digit_sub, Y_color_sub = X, Y_digit, Y_color
            
        w_digit = LogisticRegression(max_iter=200, tol=1e-3).fit(X_sub, Y_digit_sub).coef_[0]
        w_color = LogisticRegression(max_iter=200, tol=1e-3).fit(X_sub, Y_color_sub).coef_[0]
        
        cos_theta = np.dot(w_digit, w_color) / (np.linalg.norm(w_digit) * np.linalg.norm(w_color))
        return np.degrees(np.arccos(np.clip(abs(cos_theta), 0, 1)))
    
    erm_angle = get_angle(erm_model)
    irm_angle = get_angle(irm_model)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['ERM', 'IRM'], [erm_angle, irm_angle], color=[ERM_COLOR, IRM_COLOR], width=0.4)
    ax.set_ylabel(r'Subspace Angle $\theta(\mathcal{S}_{\text{shape}}, \mathcal{S}_{\text{color}})$ [deg]')
    ax.set_title(r'Subspace Angle $\theta(\mathcal{S}_{\text{shape}}, \mathcal{S}_{\text{color}})$')
    ax.set_ylim(0, 100)
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, y + 2, f'{y:.1f}°', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig('figures/subspace_geometry.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/subspace_geometry.png')
    return {'erm_angle': erm_angle, 'irm_angle': irm_angle}


# ==============================================================================
# 3. Classifier Head Alignment in Feature Space
# ==============================================================================
def analyze_head_trajectory(erm_model, irm_model, envs, device):
    """Compute classifier head alignment with causal (shape) and spurious (color) probes within each model's feature space."""
    erm_model.eval()
    irm_model.eval()
    
    def get_head_alignment(model):
        feats_list, labels_list, colors_list = [], [], []
        with torch.no_grad():
            for env in envs[:2]:
                imgs = env['images'].to(device)
                f = model.features(imgs).cpu().numpy()
                feats_list.append(f)
                labels_list.append(env['labels'].cpu().numpy().ravel())
                colors_list.append(env['colors'].cpu().numpy().ravel())
                
        X = np.concatenate(feats_list)
        Y_digit = np.concatenate(labels_list)
        Y_color = np.concatenate(colors_list)
        
        if len(X) > 5000:
            idx = np.random.choice(len(X), 5000, replace=False)
            X_sub, Y_digit_sub, Y_color_sub = X[idx], Y_digit[idx], Y_color[idx]
        else:
            X_sub, Y_digit_sub, Y_color_sub = X, Y_digit, Y_color
            
        w_digit = LogisticRegression(max_iter=200, tol=1e-3).fit(X_sub, Y_digit_sub).coef_[0]
        w_color = LogisticRegression(max_iter=200, tol=1e-3).fit(X_sub, Y_color_sub).coef_[0]
        
        w_head = model.head.weight.detach().cpu().numpy().ravel()
        
        s_causal = abs(np.dot(w_head, w_digit) / (np.linalg.norm(w_head) * np.linalg.norm(w_digit)))
        s_spurious = abs(np.dot(w_head, w_color) / (np.linalg.norm(w_head) * np.linalg.norm(w_color)))
        return s_causal, s_spurious

    s_causal_erm, s_spurious_erm = get_head_alignment(erm_model)
    s_causal_irm, s_spurious_irm = get_head_alignment(irm_model)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x - width/2, [s_causal_erm, s_causal_irm], width, label=r'Shape Probe (Causal $S_{\text{causal}}$)', color='#0072B2')
    bars2 = ax.bar(x + width/2, [s_spurious_erm, s_spurious_irm], width, label=r'Color Probe (Spurious $S_{\text{spurious}}$)', color='#D55E00')

    ax.set_ylabel(r'Head Alignment $|\cos(W_{\text{head}}, w_{\text{probe}})|$')
    ax.set_title('Classifier Head Feature Alignment')
    ax.set_xticks(x)
    ax.set_xticklabels(['ERM Head', 'IRM Head'])
    ax.set_ylim(0, max(s_causal_erm, s_causal_irm, s_spurious_erm, s_spurious_irm) * 1.3)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    for bars in [bars1, bars2]:
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=8.5)

    plt.tight_layout()
    plt.savefig('figures/head_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/head_trajectory.png')


# ==============================================================================
# 4. Feature Covariance Spectrum & Effective Rank
# ==============================================================================
def analyze_feature_spectrum(erm_model, irm_model, envs, device):
    """Compute SVD singular value spectrum and Effective Rank of representations."""
    erm_model.eval()
    irm_model.eval()
    
    with torch.no_grad():
        X_erm = torch.cat([erm_model.features(e['images'].to(device)) for e in envs[:2]]).cpu().numpy()
        X_irm = torch.cat([irm_model.features(e['images'].to(device)) for e in envs[:2]]).cpu().numpy()
        
    X_erm -= X_erm.mean(axis=0)
    X_irm -= X_irm.mean(axis=0)
    
    s_erm = np.linalg.svd(X_erm, compute_uv=False)[:30]
    s_irm = np.linalg.svd(X_irm, compute_uv=False)[:30]
    
    s_erm_norm = s_erm / s_erm[0]
    s_irm_norm = s_irm / s_irm[0]

    p_erm = s_erm / s_erm.sum()
    p_irm = s_irm / s_irm.sum()
    
    erank_erm = np.exp(-np.sum(p_erm * np.log(p_erm + 1e-12)))
    erank_irm = np.exp(-np.sum(p_irm * np.log(p_irm + 1e-12)))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, 31)
    ax.plot(x, s_erm_norm, label=f'ERM (Effective Rank: {erank_erm:.1f})', color=ERM_COLOR, linewidth=2)
    ax.plot(x, s_irm_norm, label=f'IRM (Effective Rank: {erank_irm:.1f})', color=IRM_COLOR, linewidth=2)
    
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel(r'Normalized Singular Value $\sigma_i / \sigma_1$')
    ax.set_title('Feature Covariance Singular Value Spectrum')
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/feature_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/feature_spectrum.png')
    return {'erank_erm': erank_erm, 'erank_irm': erank_irm}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('figures', exist_ok=True)
    
    set_seeds(0)
    envs = make_envs(seed=0)
    for env in envs:
        for key in env:
            env[key] = env[key].to(device)
    
    erm_model = MLP().to(device)
    irm_model = MLP().to(device)
    
    if os.path.exists('erm_model_seed0.pt'):
        erm_model.load_state_dict(torch.load('erm_model_seed0.pt', map_location=device))
    else:
        print('Warning: erm_model_seed0.pt not found. Run train.py first.')
        return
        
    if os.path.exists('irm_model_seed0.pt'):
        irm_model.load_state_dict(torch.load('irm_model_seed0.pt', map_location=device))
    else:
        print('Warning: irm_model_seed0.pt not found. Run train.py first.')
        return
        
    print('\n--- Running 4 Clean Analyses ---')
    res_cf = analyze_counterfactual_fairness(erm_model, irm_model, envs, device)
    res_sub = analyze_subspace_orthogonality(erm_model, irm_model, envs, device)
    analyze_head_trajectory(erm_model, irm_model, envs, device)
    res_spec = analyze_feature_spectrum(erm_model, irm_model, envs, device)
    
    print('\n================ Summary of Results ================')
    print(f'Counterfactual Flip Rate (CFR):  ERM = {res_cf["erm_cfr"]:.1%}  |  IRM = {res_cf["irm_cfr"]:.1%}')
    print(f'Counterfactual Effect Size (CES): ERM = {res_cf["erm_ces"]:.2f}   |  IRM = {res_cf["irm_ces"]:.2f}')
    print(f'Subspace Angle θ(Shape, Color):   ERM = {res_sub["erm_angle"]:.1f}°  |  IRM = {res_sub["irm_angle"]:.1f}°')
    print(f'Effective Feature Rank:          ERM = {res_spec["erank_erm"]:.1f}   |  IRM = {res_spec["erank_irm"]:.1f}')
    print('=====================================================\n')


if __name__ == '__main__':
    main()