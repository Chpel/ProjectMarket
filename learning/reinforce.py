import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from .utils import EpisodeSession, TrainSession
from torch import cat, manual_seed
from torch.distributions import Categorical

def get_action(policy, state, curiosity=0.75):
    with torch.no_grad():
        logits = policy(state)
        if torch.rand(1).item() < curiosity:
            probs = torch.softmax(logits, dim=-1)
            probs[probs < 0.05] = 0.05
            probs = torch.softmax(probs, dim=-1)
        else:
            probs = torch.full(logits.shape, 1/logits.shape[-1])
    d_action = Categorical(probs)
    #logging.debug(f"probs {probs}")
    # Выбор действия
    action = d_action.sample()
    #logging.debug(f"action {action}")
    return action

def generate_session(env, memory, policy, curiosity=0.5, t_max=1000):
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    #logging.debug(f"tensor state {state}")
    for t in range(t_max):
        # Получение вероятностей действий
        action = get_action(policy, state, curiosity)
        # Выполнение действия
        next_state, reward, done, _ = env.step(action.tolist()[0])
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        #logging.debug(f"tensor reward {reward}")
        memory.push(state, action, reward)
        state = next_state
        if done:
            break

def get_cumulative_rewards(rewards, gamma=0.99):
    G = np.zeros_like(rewards)
    G[-1] = rewards[-1]
    
    for t in reversed(range(len(rewards)-1)):
        G[t] = rewards[t] + gamma * G[t+1]
        
    return G

def gen_data(memory: EpisodeSession):
    batch = memory.values()
    # Transpose the batch
    state_batch = batch.state
    action_batch = batch.action
    reward_batch = batch.reward
    cumulative_rewards = get_cumulative_rewards(reward_batch)

    return cat(state_batch), \
            cat(action_batch).unsqueeze(-1), \
            cat(reward_batch), \
            torch.tensor(cumulative_rewards)


def train_on_session(env, memory: EpisodeSession, policy, optimizer):
    entropy_beta = 0.01
    states, actions, rewards, crewards = gen_data(memory)
    # Forward pass
    logits = policy(states)
    #logging.debug(f"train logits {logits}")
    probs = torch.softmax(logits, dim=-1)
    #logging.debug(f"train probs {probs}")
    log_probs = torch.log_softmax(logits, dim=-1)
    logging.debug(f"train log_probs {log_probs}")
    selected_log_probs = log_probs.gather(-1,actions)
    #logging.debug(f"train selected {selected_log_probs}")
    # Расчет потерь
    rew_loss = -(selected_log_probs * crewards).mean()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    logging.debug(f"train loss {rew_loss}")
    logging.debug(f"train entropy {entropy}")
    loss = rew_loss + entropy * entropy_beta 
    #logging.debug(f"train loss {loss}")
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()
    memory.clear()
    
    return rewards.numpy().sum()


def train(env, policy, optimizer, memory, device, cur_func: callable, params, seed=1234):
    manual_seed(seed)
    random.seed(seed)
    session = TrainSession(params['MONITOR'])
    n_episodes = params['N_EPS']
    n_epoches = params.get('N_EPC',1)
    # Основной цикл обучения
    for epoch in range(n_epoches):
        pbar = tqdm(range(n_episodes))
        for episode in pbar:
            curiosity = cur_func(episode / n_episodes)
            generate_session(env, memory, policy, curiosity)
            reward = train_on_session(env, memory, policy, optimizer)
            session.push(episode, reward)
            if episode % params['REPORT'] == 0 and episode > 0:
                session.plots()
    return session
