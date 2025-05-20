import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def non_dominated_sort(pop):
    """
    Функция для выполнения несопоставимой сортировки
    """
    n = len(pop)
    rank = np.zeros(n)
    front = []
    
    # Инициализация списков доминирования
    dominated_count = [0] * n
    dominated_list = [[] for _ in range(n)]
    
    # Проверка доминирования между всеми парами решений
    for i in range(n):
        for j in range(i+1, n):
            if dominates(pop[i], pop[j]):
                dominated_list[i].append(j)
                dominated_count[j] += 1
            elif dominates(pop[j], pop[i]):
                dominated_list[j].append(i)
                dominated_count[i] += 1
                
    # Формирование фронтов
    current_front = [i for i in range(n) if dominated_count[i] == 0]
    rank[current_front] = 1
    front.append(current_front)
    
    i = 1
    while len(front[i-1]) > 0:
        next_front = []
        for p in front[i-1]:
            for q in dominated_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
                    rank[q] = i + 1
        i += 1
        front.append(next_front)
        
    return rank, front

def dominates(sol1, sol2):
    """
    Проверка доминирования одного решения над другим
    """
    return all(sol1 <= sol2) and any(sol1 < sol2)

def crowding_distance(front, objectives):
    """
    Расчет расстояния скопления для заданного фронта
    """
    n = len(front)
    distance = np.zeros(n)
    
    for obj in range(objectives):
        # Сортировка решений по текущей целевой функции
        sorted_front = sorted(front, key=lambda x: x[obj])
        print(sorted_front)
        distance[sorted_front[0]] = distance[sorted_front[-1]] = float('inf')
        
        for i in range(1, n-1):
            distance[sorted_front[i]] = (sorted_front[i+1][obj] - sorted_front[i-1][obj])
            
    return distance

def selection(pop, rank, distance):
    """
    Селекция родителей
    """
    # Сортировка по рангу и расстоянию
    sorted_idx = np.argsort(rank + distance)
    return pop[sorted_idx]

def crossover(parent1, parent2):
    """
    Одноточечный кроссовер
    """
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutation(child):
    """
    Мутация с вероятностью 1/n
    """
    n = len(child)
    for i in range(n):
        if np.random.rand() < 1/n:
            child[i] = np.random.rand()
    return child


def nsga_ii(pop_size=100, generations=100, objectives=2):
    # Инициализация популяции
    population = np.random.rand(pop_size, objectives)
    
    for gen in range(generations):
        # Оценка целевых функций
        # Здесь должна быть функция оценки для вашей конкретной задачи
        
        # Несопоставимая сортировка
        rank, fronts = non_dominated_sort(population)
        
        # Расчет расстояния скопления
        distances = np.zeros(pop_size)
        for front in fronts:
            distances[front] = crowding_distance(population[front], objectives)
        
        # Формирование временного множества
        temp_population = []
        
        # Добавление первого фронта в новое множество
        i = 0
        while len(temp_population) + len(fronts[i]) <= pop_size:
            temp_population.extend(fronts[i])
            i += 1
        
        # Добавление части следующего фронта
        next_front = fronts[i]
        sorted_front = sorted(next_front, key=lambda x: distances[x], reverse=True)
        
        # Добавление решений с наибольшим расстоянием
        while len(temp_population) < pop_size:
            temp_population.append(sorted_front.pop(0))
        
        # Создание новой популяции
        new_population = []
        
        # Селекция родителей
        parents = selection(population, rank, distances)
        
        # Генерация потомства
        while len(new_population) < pop_size:
            # Выбор родителей
            parent1 = parents[np.random.randint(0, len(parents))]
            parent2 = parents[np.random.randint(0, len(parents))]
            
            # Кроссовер
            child1, child2 = crossover(parent1, parent2)
            
            # Мутация
            child1 = mutation(child1)
            child2 = mutation(child2)
            
            new_population.append(child1)
            new_population.append(child2)
        
        # Обновление популяции
        population = np.array(new_population[:pop_size])
    
    # Возврат финального Парето-фронта
    final_rank, final_fronts = non_dominated_sort(population)
    pareto_front = population[final_fronts[0]]
    
    return pareto_front
