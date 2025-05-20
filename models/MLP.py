#DRL_Agent
from torch import nn,stack,cat,unflatten

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBrokerDRL(nn.Module):
    def __init__(self, k_groups: int = 5, k_actions:int = 6):
        super(SimpleBrokerDRL, self).__init__()
        
        # Проверяем корректность параметра
        if not 1 <= k_groups <= 10:
            raise ValueError("k_groups должен быть в диапазоне от 1 до 10")
        

        # Проверяем корректность параметра
        if not 4 <= k_actions <= 8:
            raise ValueError("k_actions должен быть в диапазоне от 4 до 8")

        # Определяем размеры слоев
        self.k_groups = k_groups
        self.k_actions = k_actions
        self.result_size = k_groups * k_actions  # [0, 0.5, k_act] на каждую группу 
        self.hidden_size = self.result_size * 4 # Дополнительный промежуточный слой
        self.intermediate_size = self.hidden_size * 8  # Дополнительный промежуточный слой
        
        # Создаем слои модели
        self.model = nn.Sequential(
            nn.Linear(k_groups, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.ReLU(),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.result_size)
            )        

    def forward(self, x):
        # Проверяем размерность входного тензора
        assert x.size(-1) == self.k_groups, \
            f"Входной вектор должен иметь размерность {self.k_groups}"
        x = self.model(x)
        x = x.unfold(-1, self.k_actions, self.k_actions) 
        return x       

