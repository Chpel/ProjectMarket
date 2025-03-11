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
        chance = self.impulse_base + self.impulse_coef * (1-prices)
        add_res = np.random.rand(len(self.add_policy))
        add_reward = ((self.add_policy * prices)[add_res > chance]).sum()
        if add_reward > remained:
            add_reward = remained
        return base_reward + add_reward, self.list_policy + self.add_policy * add_reward / remained
        

class MarketEnv(AbstractEnvironment):
    consts = {'low_stock': 3,
                 'high_stock': 10,
                 'over_stock': 2}
    def __init__(self, k_groups, consumer_list=None):
        """
        Рынок:
            - k_groups групп товаров
            - список покупателей
        """
        self.k_groups = k_groups
        assert consumer_list, "No data about consumers"
        self.c_list = consumer_list
        self.stock = None
        
    def reset(self):
        """
        Обновить состояние среды:
            - Восполнить склады товара на случайный stock_size
        """
        stock_size = np.random.randint(self.consts['low_stock'],self.consts['high_stock'])
        self.stock = np.full(self.k_groups, stock_size)
        
    def step(self, action):
        """
        action: 
            - Перечень скидок на следующий цикл
            - Награда за успешные покупки покупателей (обратная награда за покупки сверх склада)
            - Остановка эпизода при первом окончании продукта
        """
        reward = 0
        prices = 1 - action
        stop_ep = False
        for c in self.c_list:
            reward0, sold = c.substep(prices)
            self.stock -= sold
            if np.all(self.stock > 0):
                reward += reward0
            else:
                reward -= self.consts['over_stock'] * reward0
                stop_ep = True
        return reward, self.stock, stop_ep
           
            