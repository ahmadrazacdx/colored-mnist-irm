"""Train ERM or IRM on Colored MNIST."""

import argparse
import torch
import torch.nn.functional as F

from data import set_seeds, make_envs
from model import MLP

def accuracy(model, images, labels):
    with torch.no_grad():
        preds = (model(images) > 0).float()
        return preds.eq(labels).float().mean().item()


def irm_penalty(logits, labels):
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(scale * logits, labels)
    grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
    return grad.pow(2).sum()


# ERM
def train_erm(envs, args):
    model = MLP().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        loss = 0.0
        for env in envs[:2]:
            logits = model(env['images'])
            loss += F.binary_cross_entropy_with_logits(logits, env['labels'])
        loss /= 2.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            accs = [accuracy(model, e['images'], e['labels']) for e in envs]
            print(f'  step {step:4d}  |  '
                  f'env1 {accs[0]:.1%}  env2 {accs[1]:.1%}  '
                  f'test {accs[2]:.1%}  |  loss {loss.item():.4f}')

    return model


# IRM

L2_REG = 1e-3

def train_irm(envs, args):
    model = MLP().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        losses, penalties = [], []
        for env in envs[:2]:
            logits = model(env['images'])
            losses.append(F.binary_cross_entropy_with_logits(logits, env['labels']))
            penalties.append(irm_penalty(logits, env['labels']))

        nll = torch.stack(losses).mean()
        penalty = torch.stack(penalties).mean()
        l2 = sum(p.pow(2).sum() for p in model.parameters())

        loss = nll + L2_REG * l2
        if step >= args.anneal_steps:
            loss += args.penalty_weight * penalty
            if args.penalty_weight > 1.0:
                loss /= args.penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            accs = [accuracy(model, e['images'], e['labels']) for e in envs]
            print(f'  step {step:4d}  |  '
                  f'env1 {accs[0]:.1%}  env2 {accs[1]:.1%}  '
                  f'test {accs[2]:.1%}  |  '
                  f'nll {nll.item():.4f}  penalty {penalty.item():.4f}')

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on Colored MNIST')
    parser.add_argument('--mode', choices=['erm', 'irm'], required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--penalty_weight', type=float, default=1e4,
                        help='IRM penalty weight (lambda)')
    parser.add_argument('--anneal_steps', type=int, default=190,
                        help='warmup steps before enabling IRM penalty')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seeds(args.seed)
    envs = make_envs(seed=args.seed)
    for env in envs:
        for key in env:
            env[key] = env[key].to(args.device)

    print(f'{args.mode.upper()} | device={args.device} '
          f'steps={args.steps} lr={args.lr} seed={args.seed}')
    if args.mode == 'irm':
        print(f'  penalty_weight={args.penalty_weight} '
              f'anneal_steps={args.anneal_steps} l2_reg={L2_REG}')

    train_fn = train_erm if args.mode == 'erm' else train_irm
    model = train_fn(envs, args)
    accs = [accuracy(model, e['images'], e['labels']) for e in envs]
    print(f'Final  |  env1 {accs[0]:.1%}  env2 {accs[1]:.1%}  test {accs[2]:.1%}')

    path = f'{args.mode}_model.pt'
    torch.save(model.state_dict(), path)
    print(f'Saved {path}')
