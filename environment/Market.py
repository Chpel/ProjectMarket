from base import AbstractEnvironment, AbstractPuppet
import numpy as np

class ListConsumer(AbstractPuppet):
    """
    list_policy: перечень продуктов к покупке - 0 если не планировали покупать, 
                                                иначе кол-во продукта, который планировали купить по полной стоимости
    add_policy: кол-во продукта, который могут купить дополнительно по полной стоимости
    limit: максимальная сумма, которую готовы оставить в магазине (или максимальная награда от потребителя)
    impulse_base: базовое значение импульсивности - вероятность купить по доп. списку
    impulse_coef: коэффициент привлечения к скидке - насколько увеличится вероятность купить продукт если увеличить скидку на 1%
    """
    def __init__(self, k_groups, k_cons_groups, k_add_groups, list_budget=1, add_budget=1, impulse_coef=0.1, impulse_base=0):
        
        list_policy = np.zeros(k_groups)
        list_policy[np.random.choice(k_groups, k_cons_groups)] = list_budget / k_cons_groups
        self.list_policy = list_policy
        add_policy = np.zeros(k_groups)
        add_policy[np.random.choice(k_groups, k_add_groups)] = add_budget / k_add_groups
        self.add_policy = add_policy
        self.limit = list_budget + add_budget
        self.impulse_coef = impulse_coef
        self.impulse_base = impulse_base
        
    def substep(self, prices):
        base_reward = (self.list_policy * prices).sum()
        remained = self.limit - base_reward
        chance = impulce_base + impulse_coef * (1-price)
        add_res = np.random.rand(len(self.add_policy))
        base_reward += min(remained, ((self.add_policy * prices)[add_res > chance]).sum())
        return base_reward
        
class SaleConsumer(AbstractPuppet):
    """
    prev_policy: перечень скидок на прошлый ход
    limit:
    """
    def __init__(self):
    
    def substep(self):
            
        

class MarketEnv(AbstractEnvironment):
    def __init__(self, k_groups, k_consumers):
        price_policy = np.ones(k_groups)
            