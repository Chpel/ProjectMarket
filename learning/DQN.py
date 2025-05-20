#
from .utils import Transition, ReplayMemory, TrainSession
#eps-greedy Explorer
from torch import no_grad,long,randint,rand
#DRL_optimizer   
from torch import tensor,bool,cat,zeros,nn
from torch.optim import Adam
#plots
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#utils
from torch import cuda,save,float32,manual_seed
from itertools import count
from tqdm import tqdm
from pathlib import Path
from IPython.display import clear_output

        

def select_action(state, env, Q, eps_threshold, device):
    sample = rand(1).item()
    if sample > eps_threshold:
        with no_grad():
            res = Q(state)
            res = res.argmax(-1)
    else:
        res = tensor(env.random_action, dtype=int).unsqueeze(0)
    return res
   

def optimize_model(policy_Q, target_Q, criterion, optimizer, memory, BATCH_SIZE, GAMMA, device):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch
    batch = Transition(*zip(*transitions))
    non_final_mask = tensor(tuple(map(lambda s: s is not None, \
                      batch.next_state)), device=device, dtype=bool)
    non_final_next_states = cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = cat(batch.state)
    action_batch = cat(batch.action).unsqueeze(-1)
    reward_batch = cat(batch.reward)

    #print(state_batch[0], action_batch[0])
    state_action_values = policy_Q(state_batch).gather(-1, action_batch).sum(1)
    next_state_values = zeros(BATCH_SIZE, device=device)
    with no_grad():
        n_f_n_s_rews = target_Q(non_final_next_states).max(-1).values
        next_state_values[non_final_mask] = n_f_n_s_rews.sum(-1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #print(state_action_values, expected_state_action_values)
    loss = criterion(state_action_values, expected_state_action_values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    nn.utils.clip_grad_value_(policy_Q.parameters(), 100)
    optimizer.step()
    
def plot_reward(rewards, resp_marks=[], resp_values=[], result=False, interactive=True):
    rews, codes = np.array(rewards, dtype=float).T
    plt.clf()
    plt.figure(figsize=(12,6))
    colors = ['red', 'black', 'blue', 'green']
    un_codes = [-2,-1, 1, 2]
    labels = ['complete fail', 'truncation', 'semi-success', 'complete success']
    x = np.arange(1, len(rews)+1)
    for i, c in enumerate(un_codes):
        plt.scatter(x[codes == c], rews[codes == c], c=colors[i], alpha=0.3, label=labels[i]) if (codes==c).sum() > 0 else None
    durations_t = tensor(rews, dtype=float)
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = cat((zeros(99).fill_(durations_t.mean()), means))
        plt.plot(means.numpy(), label='Mean')
    if result:
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Train History')
        plt.legend()
        plt.savefig('temp.png')
    else:
        plt.title('Training...')
    if len(resp_marks) > 0:
        for i, r in enumerate(resp_marks):
            plt.axvline(r, ymax=0.48)
            plt.axvline(r, ymin=0.52)
            _, _, ymin, ymax = plt.axis()
            plt.text(r, ymin + (ymax - ymin) * 0.5, str(resp_values[i]), rotation='vertical')
    if interactive:
        plt.draw()
        plt.gcf().canvas.flush_events()
        plt.pause(0.01)

def explore_rate_linear(x, e0, e1, e_decay):
    return e1 + (e0-e1) * np.maximum(1 - x / e_decay, 0)

def explore_rate_exp(x, e0, e1, e_decay):
    return e1 + (e0-e1) * np.exp(- x / e_decay)

def soft_update(t_Q, p_Q, tau):
    target_net_state_dict = t_Q.state_dict()
    policy_net_state_dict = p_Q.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
    t_Q.load_state_dict(target_net_state_dict)


#Trainer
def train(env, policy_Q, target_Q, criterion, optimizer, memory, device, params, rewards=None):
    plt.ion()
    plt.figure(figsize=(12,6))
    num_episodes = params['N_EPS']
    assert all(key in params for key in ['EPS_START','EPS_END','EPS_DECAY','GAMMA','TAU','BATCH_SIZE','REPORT']), 'Недостаточно'
    if not rewards:
        rewards = list()
    session = TrainSession(params.get('TRAIN', None))
    steps_done = 0
    stage = 0
    responsibility = np.round(1 - np.linspace(params['EPS_START'], 0,7),2)
    eps_marks = []
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get its state
        state = env.reset()
        ep_reward = 0
        ep_train_reward = 0
        state = tensor(state, dtype=float32, device=device).unsqueeze(0)
        eps_threshold = explore_rate_linear(i_episode, params['EPS_START'], params['EPS_END'], params['EPS_DECAY'])
        if (1 - eps_threshold > responsibility[stage]):
            stage += 1
            eps_marks.append(i_episode)

        for t in count():
            action = select_action(state, env, policy_Q, eps_threshold, device)
            steps_done += 1
            observation, reward, ep_code, info = env.step(action.tolist()[0])
            ep_train_reward += reward * params['GAMMA'] ** t
            ep_reward += info['reward'] * params['GAMMA'] ** t

            reward = tensor([reward], device=device)


            if ep_code != 0:
                next_state = None
            else:
                next_state = tensor(observation, dtype=float32, device=device).unsqueeze(0)

            # Store the transition in memory
            #print(action)
            memory.push(state, action, next_state, reward)
            # Move to the next state
            state = next_state
            
            if ep_code != 0:
                break
        
        # Perform one step of the optimization (on the policy network)
        optimize_model(policy_Q, target_Q, criterion, optimizer, memory, params['BATCH_SIZE'], params['GAMMA'], device)
        # Soft update of the target network's weights θ′ ← τ θ + (1 −τ )θ′
        soft_update(target_Q, policy_Q, params['TAU'])
        session.push(i_episode, ep_reward, ep_train_reward)
        if (i_episode) % params['REPORT'] == 0 and i_episode > 0:
            session.plots()

            
    print('Complete')
    res = params.copy()
    res['MODEL'] = target_Q.state_dict()
    save(res, res['VERSION']+'.pt')