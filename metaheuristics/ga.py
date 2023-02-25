import random
import time
from copy import copy
from typing import List, Callable, Dict, Tuple

import numpy as np
from dataset_utils import generate_mini_batch

from metaheuristics.population import Population


class GeneticAlgorithm(Population):
    """
        Genetic Algorithm

        Class Parameters
        ----------
        fitness_function : Callable
            Used for defining fitness function for each solution

        weight: float
            A weight parameter for the additional term in the fitness 
            function (NOTE: the function for defining the
            additional term needs to be defined by user, which 
            can be seen in "metaheuristics/fitness_fun.py")

        train_dataset : tuple
            train set (X,y)

        test_dataset : tuple
            test set (X,y) to be used for fitness evaluation

        operator_sequence : list
            containing search operators (crossover function, 
            mutation function) in order of activation

        probability_model : Probability Model
            Instance of ProbabilityModel 
            that contains prob_vec and noise_prob_vec in UMF
    """

    def __init__(self, dim: int, dim_range: List, num_obj: int, 
                 f: Callable, weight: float,
                 train_dataset: Tuple[np.ndarray, np.ndarray], 
                 test_dataset: Tuple[np.ndarray, np.ndarray]):
        # 使用super来初始化population中的变量
        super().__init__(dim, dim_range, num_obj)
        self.fitness_function = f
        self.weight = weight
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.operator_sequence = []
        self.add_search_operator(search_operator=
                                 GeneticAlgorithm.uniform_crossover, 
                                 parameters={"p_c": 1.0})
        self.add_search_operator(search_operator=
                                 GeneticAlgorithm.uniform_mutation, 
                                 parameters={"p_m": 1.0 / dim})
        # 变异(mutation)的概率是1/维度，也就是平均变异一位
        self.probability_model = None

    def add_search_operator(self, search_operator: Callable, parameters: Dict):
        self.operator_sequence.append((search_operator, parameters))

    def run_search_operation(self):
        self.create_temp_parent("random")
        # 深拷贝创建临时父代
        for operator, param in self.operator_sequence:
            operator(self.temp_parent, **param)
        # 执行交叉和变异两个操作
        self.create_offspring()
        # 将交叉和变异完成后的temp_parent赋值给offspring

    def evaluate_solution(self, attr, mini_train_dataset) -> float:
        # 评估每一个特征选择方案
        start_time = time.time()
        # 记录评估开始时间
        pop = getattr(self, attr)
        # print(len(pop))
        # getattr 用于返回一个对象属性值，后面调用的时候实际上是"offspring"
        self.fitness_function(population=pop, train_dataset=mini_train_dataset, 
                              test_dataset=self.test_dataset,
                              weight=self.weight)
        # 训练集上训练模型，测试集上评估结果
        return time.time() - start_time
        # 返回本论运行的时间

    def elitist_selection(self) -> None:
        entire_population = self.parent + self.offspring
        # tuple和list都可以直接相加来组合成新的tuplle或list
        entire_population.sort(key=lambda p: p.f_value[0], reverse=True)
        # sort the entire population (f_value is a list that holds fitness 
        # values across single/multi objective problems)
        new_parents = entire_population[:self.pop_size]
        # 保留前pop_size个
        self.set_new_parent(new_parents)

    @staticmethod # 返回函数的静态方法，无需实例化
    def uniform_crossover(population: List, p_c: float) -> None:
        for p1, p2 in zip(population[::2], population[1::2]):
            if random.random() < p_c:
                randvec = np.random.rand(len(p1))
                crosspnt = randvec <= 0.5
                # crosspnt是一个bool数组

                temp_p1 = copy(p1.arr[:])
                temp_p2 = copy(p2.arr[:])

                p1.arr[crosspnt] = temp_p2[crosspnt]
                p2.arr[crosspnt] = temp_p1[crosspnt]

    @staticmethod
    def uniform_mutation(population: List, p_m: float) -> None:
        for p1 in population:
            randvec = np.random.rand(len(p1))
            mutpt = randvec < p_m
            p1.arr[mutpt] = 1 - p1.arr[mutpt]


class GA_main(GeneticAlgorithm):
    """
        Genetic Algorithm (main object for calling GA)

        Class Parameters
        ----------
        total population size : int
            total population size

        gen: int
            Number of generations

        num_evaluation : int
            Store the total number of evaluations undertaken

        num_evaluation_list : list
            Store num_evaluation for each generation

        result : list
            Store the best fitness of each generation

        Class Methods
        ----------
        run() :
            Run the GA algorithm

        evaluate_task() :
            Evaluate the solutions and return the total evaluation cost

        generational_result() :
            Print result from every generation

    """

    def __init__(self, gen: int, total_pop_size: int, 
                 f: Callable, weight: float, train_dataset: tuple,
                 test_dataset: tuple, task_param):
        super().__init__(task_param["dim"], task_param["dim_range"], 
                         task_param["num_obj"], f, weight, 
                         train_dataset, test_dataset)
        self.total_pop_size = total_pop_size
        self.create_parent(total_pop_size, **task_param)
        self.gen = gen
        self.num_evaluation = 0
        self.num_evaluation_list = []
        self.result = []

    def run(self, print_result: bool) -> Tuple[List, List]:
        # 遗传算法运行
        
        self.num_evaluation += self.evaluate_task("parent")
        self.num_evaluation_list.append(self.num_evaluation)
        self.parent.sort(key=lambda p: p.f_value[0], reverse=True)
        self.result.append(self.parent[0].f_value[0])
        # 提前先演化一次，然后执行循环，因为第一代没有offspring，需要直接评估parent
        for g in range(2, self.gen + 1):
            self.run_search_operation()
            self.num_evaluation += self.evaluate_task("offspring")
            self.num_evaluation_list.append(self.num_evaluation)
            self.elitist_selection()
            self.result.append(self.parent[0].f_value[0])
            # self.generational_result(g, print_result=print_result) 
            # print the results at each generation

        self.generational_result(g, print_result=print_result)  
        # only print the results at the final generation
        return self.result, self.num_evaluation_list

    def evaluate_task(self, attr: str) -> float:
        # mini_train_dataset = \
        #     generate_mini_batch(train_dataset=self.train_dataset, 
        #                         mini_size=1.0, random_state=2)
        # 评估任务
        time_taken = [self.evaluate_solution(attr, self.train_dataset)]
        # 调用GA的评估方法，并返回时间
        return (sum(time_taken) / time_taken[-1]) * len(getattr(self, attr))
        return sum(time_taken)

    def generational_result(self, g: int, print_result: bool) -> None:
        # 输出运行的结果
        fitness_list = [p.f_value[0] for p in self.parent]
        avg_fitness = np.mean(fitness_list)
        if print_result:
            print("gen: {}, best: {}, mean: {}"
                  ", pop size: {}".format(g, self.parent[0].f_value[0],
                                          avg_fitness, self.total_pop_size))
        print("Total evaluation cost: {}".format(self.num_evaluation))
        print("")
