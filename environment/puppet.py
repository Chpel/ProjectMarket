from .base import AbstractEnvironment, AbstractPuppet
import numpy as np
import scipy.special as sp
import logging

class ListConsumer(AbstractPuppet):
    """
    k_groups: кол-во групп продуктов
    k_cons_groups: кол-во групп покупаемых товаров по плану
    list_budget: бюджет на плановые покупки
    limit: максимальная сумма, которую готовы оставить в магазине (или максимальная награда от потребителя)
    Результат:
    list_policy: перечень продуктов к покупке - 0 если не планировали покупать, 
                                                иначе кол-во продукта, который планировали купить по полной стоимости
    """
    def __init__(self, k_groups: int, k_cons_groups: int, list_budget=1):
        list_policy = np.zeros(k_groups)
        list_groups = np.random.choice(k_groups, k_cons_groups, replace=False)
        list_policy[list_groups] = list_budget / k_cons_groups
        self.list_policy = list_policy
        self.limit = list_budget

    @property
    def budget(self):
        return self.list_policy.sum()
        
    def substep(self, prices):
        base_reward = (self.list_policy * prices).sum()
        logging.info(f'bought {self.list_policy}')
        return base_reward, self.list_policy
    
class ImpulseFunc:
    def __init__(self, base, coef, level=None):
        self.base = base
        self.coef = coef
        self.level = level

    def __call__(self, prices):
        assert type(prices) is np.ndarray, "Неверный тип параметра цены"
        chance = self.base + self.coef * (1-prices)
        logging.debug(f"chance is {chance}")
        add_res = np.random.rand(len(prices))
        logging.debug(f"buy probs {add_res}")
        if type(self.level) is type(None):
            add_res = np.random.rand(len(prices))
        else:
            add_res = self.level
        return np.where(add_res < chance, 1, 0)
        
        
class ImpulseConsumer(AbstractPuppet):
    """
    k_groups: кол-во групп продуктов
    k_add_groups: кол-во групп покупаемых товаров по желанию
    add_budget: доп бюджет
    impulse_base: базовое значение импульсивности - вероятность купить по доп. списку
    impulse_coef: коэффициент привлечения к скидке - насколько увеличится вероятность купить продукт если увеличить скидку на 1%
    Результат:
    add_policy: кол-во продукта, который могут купить дополнительно по полной стоимости
    """
    def __init__(self, k_groups: int, k_add_groups: int, impulse_func:callable, add_budget=1):
        add_policy = np.zeros(k_groups)
        add_groups = np.random.choice(k_groups, k_add_groups, replace=False)
        add_policy[add_groups] = add_budget / k_add_groups
        self.add_policy = add_policy
        self.impulse_func = impulse_func

    @property
    def budget(self):
        return self.add_policy.sum()
        
    def substep(self, prices, limit=None):
        if not limit:
            limit = self.budget + 1
        buy_policy = self.add_policy * self.impulse_func(prices)
        add_reward = (buy_policy * prices).sum()
        logging.debug(add_reward)
        if add_reward > 0:
            add_bought = min(limit, add_reward) / add_reward
        else:
            add_bought = 0
        return add_reward * add_bought, buy_policy * add_bought
    
class MixedConsumer(AbstractPuppet):
    def __init__(self, l_cons:ListConsumer, i_cons:ImpulseConsumer, limit:float):
        assert l_cons.limit < limit, "Недостаточно средств для плановых покупок"
        self.l_cons = l_cons
        self.i_cons = i_cons
        self.limit = limit

    def substep(self, prices):
        l_rew, l_bought = self.l_cons.substep(prices, self.limit)
        remained = self.limit - l_rew
        i_rew, i_bought = self.i_cons.substep(prices, remained)
        return l_rew + i_rew, l_bought + i_bought

        
        

