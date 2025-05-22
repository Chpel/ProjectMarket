from .base import AbstractEnvironment, AbstractPuppet
import numpy as np
import scipy.special as sp
import logging
from .puppet import ListConsumer, ImpulseConsumer

class MarketEnv(AbstractEnvironment):
    """
    Окружение рынка с переменным объёмом товаров
    """
    consts = {'low_stock': 3, 'high_stock': 10,
              'over_stock': 2,
              'low_discount': 0, 'high_discount':0.5,
              'disc_step': 0.05}
    act_types = ['state', 'step', 'discount']
    rew_types = ['reward', 'diff', 'bonus', 'plan']

    def __init__(self, k_groups, k_actions=6, fix_stock=None, consumer_list=None, max_ep=None, action_type='state', reward_type='reward', base_state = 'zero', plan_reward=None):
        """
        k_groups: кол-во групп товаров
        fix_stock: фиксированный объём рынка
        low_budget: ограничение ресурсов для скидок (True - сумма процентов по всем товарам равна 100, False - на каждый товар отдельная скидка любой величины)
        consumer_list: список групп покупателей
        """
        self.k_groups = k_groups
        assert consumer_list, "No data about consumers"
        assert action_type in self.act_types, 'No valid action types'
        assert reward_type in self.rew_types, 'No valid reward types'
        self.c_list = consumer_list
        self.fix_stock = fix_stock
        self.k_actions = k_actions
        self.stock = None
        self.discount = None
        self.ep = None
        self.reward = None
        self.max_ep = max_ep
        self.action_type = action_type
        self.reward_type = reward_type

    @property
    def observation_space(self):
        return self.consts['low_discount'], self.consts['high_discount'], self.k_groups
    
    @property
    def action_space(self):
        if self.action_type in ['state', 'discount']:
            l = self.consts['low_discount']
            h = self.consts['high_discount']
            k = self.k_actions
            return l,h,k
        elif self.action_type == 'step':
            d = self.consts['disc_step']
            if self.k_actions == 3:
                return np.array([-d, 0, d]), self.k_groups
            elif self.k_actions == 2:
                return np.array([-d, d]), self.k_groups
        else:
            assert False, "Стратегия действия не известна"
    

    @property
    def discount_space(self):
        l = self.consts['low_discount']
        h = self.consts['high_discount']
        k = self.k_actions
        return l,h,k

    @property
    def state(self):
        if self.action_type == 'state':
            return np.array([self.stock])
        elif self.action_type in ['step', 'discount']:
            return np.array(self.discount)
        else:
            assert False, "Стратегия действия не известна"

    @property
    def list_policy(self):
        res = np.zeros(self.k_groups)
        for c in filter(lambda x: type(x) is ListConsumer, self.c_list):
            res += c.list_policy
        return res

    @property
    def add_policy(self):
        res = np.zeros(self.k_groups)
        for c in filter(lambda x: type(x) is ImpulseConsumer, self.c_list):
            res += c.add_policy
        return res
    
    @property
    def random_step(self):
        arr,k = self.action_space
        return np.random.choice(arr,k)
    
    @property
    def random_action(self):
        return np.random.choice(self.k_actions,self.k_groups)

    def get_step_discount(self, step):
        assert type(step) is list, "Действие не является массивом"
        arr, k = self.action_space
        return arr[step]
    
    def get_discount(self, action):
        assert type(action) is list, "Действие не является массивом"
        l,h,k = self.discount_space
        return np.linspace(l,h,k)[action]
     
    def reset(self):
        """
        Обновить состояние среды:
            - Восполнить склады товара на случайный stock_size
        """
        stock_size = self.fix_stock if self.fix_stock else np.random.randint(self.consts['low_stock'],self.consts['high_stock'])
        stock_size = float(stock_size)
        self.stock = np.full(self.k_groups, stock_size)
        self.ep = 0
        l,h,k = self.observation_space
        self.discount = np.full(k, np.round(np.random.random() * (h-l) + l, 1))
        self.reward, _, _ = self._reward_calc
        return self.state
    
    @property
    def _reward_calc(self):
        reward = 0
        prices = 1 - self.discount
        stop_ep = False
        total_sold = 0
        group_sold = np.array([0.]*self.k_groups)
        for c in self.c_list:
            reward0, sold = c.substep(prices)
            self.stock -= sold
            total_sold += sold.sum()
            group_sold += sold
            if np.all(self.stock > 0):
                reward += reward0
            else:
                #reward -= self.consts['over_stock'] * reward0
                stop_ep = True
        return reward, stop_ep, {'total': total_sold, 
                                'group': group_sold,
                                'reward': reward}
    
    def _bonus_calc(self, reward):
        bonus = 0
        all_list = self.list_policy.sum()
        all_add = self.add_policy.sum()
        if reward < all_list * 0.70:
            bonus -= 1
        if reward >= all_list * 0.90:
            bonus += 1
        if reward >= all_list + all_add * 0.1:
            bonus += 1
        if reward >= all_list + all_add * 0.3:
            bonus += 1
        if reward >= all_list + all_add * 0.5:
            bonus += 1
        if reward >= all_list + all_add * 0.7:
            bonus += 1
        return bonus
        
    def step(self, action):
        """
        action: 
            - Перечень скидок на следующий цикл
            - Награда за успешные покупки покупателей (обратная награда за покупки сверх склада)
            - Остановка эпизода при первом окончании продукта
        """
        self.ep += 1
        #get
        if self.action_type == 'state':
            self.discount = self.get_discount(action)
        elif self.action_type == 'step':
            step = self.get_step_discount(action)
            self.discount += step
            self.discount = np.clip(self.discount, self.consts['low_discount'], self.consts['high_discount'])
        elif self.action_type == 'discount':
            self.discount = action
        reward, stop_ep, info = self._reward_calc
        reward_diff = reward - self.reward
        self.reward = reward
        bonus = self._bonus_calc(reward)
        info['bonus'] = bonus
        if self.reward_type == 'reward':
            final = self.reward
        elif self.reward_type == 'diff':
            final = reward_diff
        elif self.reward_type == 'bonus':
            final = bonus
        elif self.reward_type == 'plan':
            final = self.reward - self.list_policy.sum()
        return self.state, final, stop_ep or self.ep == self.max_ep, info
           
            