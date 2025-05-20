import matplotlib.pyplot as plt
import numpy as np
from environment.puppet import ListConsumer, ImpulseConsumer


def policy_plot(env, ax:plt.axes=None):
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(6,6))
    cmap = plt.get_cmap('Pastel1',len(env.c_list))
    bottom = 0
    for i, c in enumerate(env.c_list):
        if type(c) is ListConsumer:
            height = c.list_policy
            ax.bar(x=range(len(env.list_policy)), height=height, bottom=bottom, color=cmap(i),label=f'План {i}', edgecolor='black')
            bottom = bottom+height

    for i, c in enumerate(env.c_list):
        if type(c) is ImpulseConsumer:
            height = c.add_policy
            ax.bar(x=range(len(env.list_policy)), height=height, bottom=bottom, hatch='/',color=cmap(i), label=f'Импульс {i}', edgecolor='black')
            bottom = bottom+height
    ax.legend();
    ax.set_xlabel('Группа товаров');
    ax.set_ylabel('Объём покупок');
    ax.set_xticks(range(env.k_groups));
    return ax

def simulate(env, N, apply_to=None, discs=None):
    if type(discs) == type(None):
        x = np.linspace(0, 0.5, 20)
    else:
        x = discs
    y = []
    yerr = []
    s = []
    serr = []
    env.reset()
    for discount in x:
        action = np.zeros(env.k_groups)
        if isinstance(apply_to, list):
            action[apply_to] = discount
        else:
            action[:] = discount
        rewards = []
        sold = []
        for att in range(N):
            _, r0, _, info = env.step(action=action)
            env.reset()
            rewards.append(info['reward'])
            sold.append(info['total'])
        y.append(np.mean(rewards))
        yerr.append(np.std(rewards) / np.sqrt(N))  
        s.append(np.mean(sold))
        serr.append(np.std(sold) / np.sqrt(N))  
    return x, y, yerr, s, serr

def strategy_plot(env, N=200, apply_to=None, ax1=None, ax2=None, variants=None, discs=None):
    if type(variants) == type(None):
        variants = np.linspace(0.5, 5, 12)
    if not ax1:
        fig, ax = plt.subplots(1,2, figsize=(12,6))
    else:
        ax = (ax1,ax2)
    for v in variants:
        for c in env.c_list:
            if type(c) is ImpulseConsumer:
                c.impulse_func.coef = v
        x, y, yerr, s, serr = simulate(env, N, apply_to=apply_to, discs=discs)
        ax[0].errorbar(x,y,yerr, label=f"k = {v}")
        ax[1].errorbar(x, s, serr, label=f"k = {v:.2}")
    ax[0].set_xlabel('all_discount')
    ax[1].set_xlabel('all_discount')
    ax[0].set_ylabel(r'<Reward>')
    ax[1].set_ylabel(r'<Sold>')
    ax[1].legend()
    return ax