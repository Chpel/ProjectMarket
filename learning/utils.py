#MDP model (s, a, r, new_s)
import random
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
#MDP model (s, a, r, new_s)
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

Step = namedtuple('Step', ('state', 'action', 'reward'))
class EpisodeSession(object):
    def __init__(self):
        self.memory = list()
    
    def push(self, *args):
        self.memory.append(Step(*args))

    def clear(self):
        self.memory.clear()

    def values(self):
        return Step(*zip(*self.memory))
    
    def __len__(self):
        return len(self.memory)
    

Episode = namedtuple('Episode', ('id', 'reward', 'train_reward'))
class TrainSession(object):
    def __init__(self, capacity:int=None):
        if capacity:
            self.memory = deque(list(), maxlen=capacity)
        else: 
            self.memory = list()
        self.min_r = None
        self.max_r = None
    
    def push(self, *args):
        self.memory.append(Episode(*args))
        if not self.min_r or self.min_r > self.memory[-1].reward:
            self.min_r = self.memory[-1].reward
        if not self.max_r or self.max_r < self.memory[-1].reward:
            self.max_r = self.memory[-1].reward


    def clear(self):
        self.memory.clear()

    def values(self):
        return Episode(*zip(*self.memory))
    
    def __len__(self):
        return len(self.memory)
    
    def plots(self):
        clear_output()
        window = 100
        xy = self.values()
        plt.figure(figsize=(12,8))
        x, y, yt = torch.tensor(xy.id), torch.tensor(xy.reward), torch.tensor(xy.train_reward)
        plt.scatter(x,y, c='b', alpha=0.3)
        if y.shape[0] >= window:
            means = y.unfold(0, window, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(window-1).fill_(y.mean()), means))
            plt.plot(x, means.numpy(), label='Mean')
        plt.ylim(self.min_r, self.max_r)
        plt.show()
        plt.figure(figsize=(12,8))
        plt.scatter(x, yt, c='b', alpha=0.3)
        if y.shape[0] >= window:
            means = yt.unfold(0, window, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(window-1).fill_(yt.mean()), means))
            plt.plot(x, means.numpy(), label='Mean')
        plt.show()
        
    
