"""Causal and Representation Analysis for Invariant Risk Minimization (Colored MNIST)."""

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

# 1. Counterfactual Fairness & Do-Interventions
def analyze_counterfactual_fairness(erm_model, irm_model, envs, device):
    erm_model.eval()
    irm_model.eval()
    
    test_env = envs[2]
    images = test_env['images'].to(device)
    n = len(images)
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
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    bars1 = axes[0].bar(['ERM', 'IRM'], [erm_cfr * 100, irm_cfr * 100], color=[ERM_COLOR, IRM_COLOR], width=0.4)
    axes[0].set_ylabel('Counterfactual Flip Rate (%)')
    axes[0].set_title('Flip Rate under do(Color)')
    axes[0].set_ylim(0, 100)
    for b in bars1:
        y = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2, y + 2, f'{y:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    bars2 = axes[1].bar(['ERM', 'IRM'], [erm_ces, irm_ces], color=[ERM_COLOR, IRM_COLOR], width=0.4)
    axes[1].set_ylabel('Logit Shift |Δ Logit|')
    axes[1].set_title('Counterfactual Effect Size (CES)')
    axes[1].set_ylim(0, max(erm_ces, irm_ces) * 1.25)
    for b in bars2:
        y = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2, y + 0.05, f'{y:.2f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('figures/counterfactual_fairness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/counterfactual_fairness.png')
    return {'erm_cfr': erm_cfr, 'irm_cfr': irm_cfr, 'erm_ces': erm_ces, 'irm_ces': irm_ces}


# 2. Subspace Orthogonality
def analyze_subspace_orthogonality(erm_model, irm_model, envs, device):
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
        
        w_digit = LogisticRegression(max_iter=1000).fit(X, Y_digit).coef_[0]
        w_color = LogisticRegression(max_iter=1000).fit(X, Y_color).coef_[0]
        
        cos_theta = np.dot(w_digit, w_color) / (np.linalg.norm(w_digit) * np.linalg.norm(w_color))
        return np.degrees(np.arccos(np.clip(abs(cos_theta), 0, 1)))
    
    erm_angle = get_angle(erm_model)
    irm_angle = get_angle(irm_model)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['ERM', 'IRM'], [erm_angle, irm_angle], color=[ERM_COLOR, IRM_COLOR], width=0.4)
    ax.axhline(90.0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Ideal Orthogonal (90°)')
    ax.set_ylabel('Subspace Angle θ(Shape, Color) [degrees]')
    ax.set_title('Subspace Orthogonality')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', framealpha=0.9)
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, y + 2, f'{y:.1f}°', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('figures/subspace_geometry.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/subspace_geometry.png')
    return {'erm_angle': erm_angle, 'irm_angle': irm_angle}


# 3. Environment Gradient Alignment Trajectory
def analyze_gradient_alignment(envs, args):
    device = args.device
    
    def track_alignment(is_irm):
        set_seeds(0)
        model = MLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        sims = []
        
        for step in range(1, args.steps + 1):
            grads = []
            for env in envs[:2]:
                model.zero_grad()
                logits = model(env['images'])
                loss = F.binary_cross_entropy_with_logits(logits, env['labels'])
                if is_irm and step >= args.anneal_steps:
                    scale = torch.ones(1, device=device, requires_grad=True)
                    penalty = torch.autograd.grad(F.binary_cross_entropy_with_logits(scale * logits, env['labels']), scale, create_graph=True)[0].pow(2).sum()
                    loss = loss + (args.penalty_weight * penalty) / args.penalty_weight
                loss.backward()
                
                g_vec = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
                grads.append(g_vec)
                
            sim = F.cosine_similarity(grads[0].unsqueeze(0), grads[1].unsqueeze(0)).item()
            sims.append(sim)
            
            loss_total = sum(F.binary_cross_entropy_with_logits(model(e['images']), e['labels']) for e in envs[:2]) / 2.0
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
        return sims

    erm_sims = track_alignment(is_irm=False)
    irm_sims = track_alignment(is_irm=True)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = np.arange(1, args.steps + 1)
    ax.plot(steps, erm_sims, label='ERM', color=ERM_COLOR, linewidth=2, alpha=0.8)
    ax.plot(steps, irm_sims, label='IRM', color=IRM_COLOR, linewidth=2, alpha=0.8)
    ax.axvline(190, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='IRM Penalty On (Step 190)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Cosine Similarity')
    ax.set_title('Environment Gradient Alignment Over Training')
    ax.set_ylim(-0.2, 1.05)
    ax.legend(framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('figures/gradient_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/gradient_alignment.png')


# 4. Classifier Head Trajectory in Phase Space
def analyze_head_trajectory(erm_model, irm_model, envs, device):
    feats_list, labels_list, colors_list = [], [], []
    with torch.no_grad():
        for env in envs[:2]:
            imgs = env['images'].to(device)
            f = erm_model.features(imgs).cpu().numpy()
            feats_list.append(f)
            labels_list.append(env['labels'].cpu().numpy().ravel())
            colors_list.append(env['colors'].cpu().numpy().ravel())
            
    X = np.concatenate(feats_list)
    Y_digit = np.concatenate(labels_list)
    Y_color = np.concatenate(colors_list)
    
    w_digit = LogisticRegression(max_iter=1000).fit(X, Y_digit).coef_[0]
    w_color = LogisticRegression(max_iter=1000).fit(X, Y_color).coef_[0]
    w_digit /= np.linalg.norm(w_digit)
    w_color /= np.linalg.norm(w_color)
    
    w_erm_head = erm_model.head.weight.detach().cpu().numpy().ravel()
    w_irm_head = irm_model.head.weight.detach().cpu().numpy().ravel()
    
    s_causal_erm = abs(np.dot(w_erm_head, w_digit) / np.linalg.norm(w_erm_head))
    s_spurious_erm = abs(np.dot(w_erm_head, w_color) / np.linalg.norm(w_erm_head))
    
    s_causal_irm = abs(np.dot(w_irm_head, w_digit) / np.linalg.norm(w_irm_head))
    s_spurious_irm = abs(np.dot(w_irm_head, w_color) / np.linalg.norm(w_irm_head))
    
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter([s_causal_erm], [s_spurious_erm], color=ERM_COLOR, s=120, label='ERM Head', zorder=5)
    ax.scatter([s_causal_irm], [s_spurious_irm], color=IRM_COLOR, s=120, label='IRM Head', zorder=5)
    
    ax.annotate('ERM (latching on color)', (s_causal_erm + 0.02, s_spurious_erm), fontsize=9, color=ERM_COLOR)
    ax.annotate('IRM (aligned with shape)', (s_causal_irm + 0.02, s_spurious_irm), fontsize=9, color=IRM_COLOR)
    
    ax.set_xlabel('Alignment with Shape Probe S_causal')
    ax.set_ylabel('Alignment with Color Probe S_spurious')
    ax.set_title('Classifier Head Alignment Phase Space')
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('figures/head_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/head_trajectory.png')


# 5. Feature Covariance Spectrum & Effective Rank
def analyze_feature_spectrum(erm_model, irm_model, envs, device):
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
    
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, 31)
    ax.plot(x, s_erm_norm, label=f'ERM (Effective Rank: {erank_erm:.1f})', color=ERM_COLOR, linewidth=2)
    ax.plot(x, s_irm_norm, label=f'IRM (Effective Rank: {erank_irm:.1f})', color=IRM_COLOR, linewidth=2)
    
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Normalized Singular Value σ_i / σ_1')
    ax.set_title('Feature Covariance Singular Value Spectrum')
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9)
    
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
        
    class Args:
        pass
    args = Args()
    args.device = device
    args.steps = 500
    args.lr = 1e-3
    args.penalty_weight = 1e4
    args.anneal_steps = 190

    res_cf = analyze_counterfactual_fairness(erm_model, irm_model, envs, device)
    res_sub = analyze_subspace_orthogonality(erm_model, irm_model, envs, device)
    analyze_gradient_alignment(envs, args)
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
