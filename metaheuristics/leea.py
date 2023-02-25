from cmath import sqrt
import random
from readline import append_history_file
import time
from copy import copy
from tkinter import N
from typing import List, Callable, Dict, Tuple
from typing_extensions import Self

import numpy as np
import matplotlib.pyplot as plt
from dataset_utils import generate_mini_batch

from metaheuristics.population import Population
from metaheuristics.ga import GeneticAlgorithm

class LEEA_main(GeneticAlgorithm):
    """
        Genetic Algorithm (main object for calling GA)

        Class Parameters
        ----------
        total population size : int
            total population size

        gen: int
            Number of generations
        
        mini_size: float
            Proportion from the main train set

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
        
        mini_size = 0.2
        error = []
        flag = 0
        update_list = []        
        miniset = \
            generate_mini_batch(train_dataset=self.train_dataset, 
                                    mini_size=mini_size)
        self.num_evaluation += self.evaluate_task("parent", mini_train_dataset=miniset)
        self.num_evaluation_list.append(self.num_evaluation)
        self.parent.sort(key=lambda p: p.f_value[0], reverse=True)
        temp = self.parent[0].f_value[0]
        self.fitness_function(population=[self.parent[0]], train_dataset=self.train_dataset, 
                          test_dataset=self.test_dataset,
                          weight=self.weight)
        update_list.append(1)
        self.result.append(self.parent[0].f_value[0])
        self.parent[0].f_value[0] = temp
        # 提前先演化一次，然后执行循环，因为第一代没有offspring，需要直接评估parent
        for g in range(2, self.gen + 1):
            self.run_search_operation()
            
            
            # 最初版本，每一代都重新选取小数据集
            # if g < self.gen * 0.20:
            #     mini_size = 0.2
            #     miniset = \
            #             generate_mini_batch(train_dataset=self.train_dataset, 
            #                             mini_size=mini_size)
            #     self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
            #     self.leea_elitist_selection(0.5, mini_train_dataset=miniset)
            # elif g < self.gen * 0.80:
            #     mini_size = 0.2 + g / self.gen
            #     miniset = \
            #             generate_mini_batch(train_dataset=self.train_dataset, 
            #                             mini_size=mini_size)
            #     self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
            #     self.leea_elitist_selection(0.5, mini_train_dataset=miniset)

            # else:
            #     miniset = self.train_dataset
            #     self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
            #     self.elitist_selection()
            
            
            
            mini_size = self.get_minisize(g)
            if mini_size == 1:
                miniset = self.train_dataset
                self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
                self.elitist_selection()
            else:
                if len(error) == 0:
                    miniset = \
                            generate_mini_batch(train_dataset=self.train_dataset, 
                            mini_size=mini_size)
                    self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
                    error.append(self.leea_elitist_selection(0.5, mini_train_dataset=miniset))
                    update_list.append(1)
                elif error[len(error)-1] > np.mean(error) or flag == 8: # 更新数据集
                    flag = 0
                    miniset = \
                            generate_mini_batch(train_dataset=self.train_dataset, 
                            mini_size=mini_size)
                    self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
                    error.append(self.leea_elitist_selection(0.5, mini_train_dataset=miniset))
                    update_list.append(1)
                else:
                    self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
                    self.elitist_selection()
                    update_list.append(0)
                    flag = flag + 1
            
             
            # mini_size = self.get_minisize(g)
            # if mini_size == 1:
            #     miniset = self.train_dataset
            #     self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
            #     self.elitist_selection()
            # else:
            #     if len(error) == 0:
            #         miniset = \
            #                 generate_mini_batch(train_dataset=self.train_dataset, 
            #                 mini_size=mini_size)
            #         self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
            #         error.append(self.leea_elitist_selection(0.5, mini_train_dataset=miniset))
            #     elif error[len(error)-1] > 0.11: # 更新数据集
            #         miniset = \
            #                 generate_mini_batch(train_dataset=self.train_dataset, 
            #                 mini_size=mini_size)
            #         self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
            #         error.append(self.leea_elitist_selection(0.5, mini_train_dataset=miniset))
            #     else:
            #         self.num_evaluation += self.evaluate_task("offspring", mini_train_dataset=miniset)
            #         self.elitist_selection()
            
            
            temp = self.parent[0].f_value[0]
            self.fitness_function(population=[self.parent[0]], train_dataset=self.train_dataset, 
                              test_dataset=self.test_dataset,
                              weight=self.weight)


            
            self.num_evaluation_list.append(self.num_evaluation)
            
            
            self.result.append(self.parent[0].f_value[0])
            # self.generational_result(g, print_result=print_result) 
            # print the results at each generation
            
            self.parent[0].f_value[0] = temp

        self.generational_result(g, print_result=print_result)  
        # only print the results at the final generation
        print(update_list)
        return self.result, self.num_evaluation_list
    
    def leea_elitist_selection(self, k : float, mini_train_dataset) -> None:
        pt = int(k * self.pop_size)
        self.parent = self.parent[:pt]
        error = []        
        for i in range(pt):
            error.append(self.parent[i].f_value[0])
        
        
        self.num_evaluation += self.evaluate_task("parent", mini_train_dataset)
        for i in range(pt):    
            error[i] -= self.parent[i].f_value[0]
        
        res = np.mean(np.abs(error))
        entire_population = self.parent + self.offspring
        entire_population.sort(key=lambda p: p.f_value[0], reverse=True)
        
        new_parents = entire_population[:self.pop_size]
        # 保留前pop_size个
        self.set_new_parent(new_parents)
        return res

    def get_minisize(self, g: int):
        if g / self.gen < 0.20:
            return 0.2
        elif g / self.gen < 0.8:
            return 0.2 + g / self.gen
        else:
            return 1
    
    
    def evaluate_task(self, attr: str, mini_train_dataset) -> float:
        # 评估任务
        time_taken1 = [self.evaluate_solution(attr, self.train_dataset)]
        time_taken0 = [self.evaluate_solution(attr, mini_train_dataset)]
        # 调用GA的评估方法，并返回时间
        return (sum(time_taken0) / time_taken1[-1]) * len(getattr(self, attr))
        return sum(time_taken0)

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
