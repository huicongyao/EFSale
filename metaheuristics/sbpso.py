import time
from typing import List, Callable, Dict, Tuple

import numpy as np

from metaheuristics.population import Population


class SBPSO(Population):
    """
        Sticky Binary PSO

        Class Parameters
        ----------
        fitness_function : Callable
            Used for defining fitness function for each solution

        weight: float
            A weight parameter for the additional term in the fitness function 
                (NOTE: the function for defining the additional term needs to be defined 
                by user, which can be seen in "metaheuristics/fitness_fun.py")

        train_dataset : tuple
            train set (X,y)

        test_dataset : tuple
            test set (X,y) to be used for fitness evaluation

        operator_sequence : list
            list containing search operators (crossover function, mutation function) in order of activation

        probability_model : ProbabilityModel
            Instance of ProbabilityModel that contains prob_vec and noise_prob_vec in UMF

        gbest : numpy.ndarray
            Global best solution

        gbest_score : float
            Fitness of the global best solution
    """

    def __init__(self, dim: int, dim_range: List, num_obj: int, f: Callable, weight: float,
                 train_dataset: Tuple[np.ndarray, np.ndarray], test_dataset: Tuple[np.ndarray, np.ndarray]):
        super().__init__(dim, dim_range, num_obj)
        self.operator_sequence = []
        self.fitness_function = f
        self.weight = weight
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.add_search_operator(search_operator=SBPSO.update_swarm,
                                 parameters={"i_m": 0.25, "i_p": 0.25, "i_g": 0.50, "max_life": 50})
        self.probability_model = None
        self.gbest = None
        self.gbest_score = 0.0

    def add_search_operator(self, search_operator: Callable, parameters: Dict):
        self.operator_sequence.append((search_operator, parameters))

    def run_search_operation(self):
        self.create_temp_parent("random")
        for operator, param in self.operator_sequence:
            operator(self.temp_parent, self.gbest, **param)
        self.create_offspring()

    def evaluate_solution(self, attr) -> float:
        start_time = time.time()
        pop = getattr(self, attr)
        self.fitness_function(population=pop, train_dataset=self.train_dataset, test_dataset=self.test_dataset, weight=self.weight)
        return time.time() - start_time

    def elitist_selection(self) -> None:
        self.set_new_parent(self.offspring[:])

    def update_gbest(self):
        self.parent.sort(key=lambda p: p.f_value[0], reverse=True)
        if self.parent[0].f_value[0] > self.gbest_score:
            self.gbest_score = self.parent[0].f_value[0]
            self.gbest = self.parent[0].arr[:]

    @staticmethod
    def update_swarm(population: List, gbest: np.ndarray, i_m: float, i_p: float, i_g: float, max_life: int):
        for p in population:
            p.stickiness = 1 - p.current_life / max_life
            p.flipping_prob = i_m * (1 - p.stickiness) + i_p * np.absolute(
                p.pbest - p.arr) + i_g * np.absolute(gbest - p.arr)
            bool_mask = np.random.rand(len(p.arr)) <= p.flipping_prob
            for d in range(p.current_life.shape[0]):
                if bool_mask[d]:
                    p.current_life[d] = 0
                else:
                    p.current_life[d] += 1

            p.arr = np.absolute(bool_mask - p.arr)


class SBPSO_main(SBPSO):
    """
        Sticky Binary PSO (main object for calling SBPSO)

        Class Parameters
        ----------
        total population size : int
            total population size

        gen: int
            Number of generations

        num_evaluation : int
            Store the total number of evaluation undertaken

        num_evaluation_list : list
            Store num_evaluation for each generation

        result : list
            Store the best fitness of each generation

        Class Methods
        ----------
        run() :
            Run the SBPSO algorithm

        evaluate_task() :
            Evaluate the solutions in each tasks and return the total evaluation cost

        generational_result() :
            Print result from every generation

    """

    def __init__(self, gen: int, total_pop_size: int, f: Callable, weight: float, train_dataset: tuple, test_dataset: tuple,
                 task_param):
        super().__init__(task_param["dim"], task_param["dim_range"], task_param["num_obj"], f, weight, train_dataset,
                         test_dataset)
        self.total_pop_size = total_pop_size
        self.create_parent(total_pop_size, **task_param)
        self.gen = gen
        self.num_evaluation = 0
        self.num_evaluation_list = []
        self.result = []

    def run(self, print_result: bool) -> Tuple[List, List]:
        self.num_evaluation += self.evaluate_task("parent")
        self.num_evaluation_list.append(self.num_evaluation)
        self.parent.sort(key=lambda p: p.f_value[0], reverse=True)
        self.update_gbest()
        self.result.append(self.parent[0].f_value[0])
        for g in range(2, self.gen + 1):
            self.run_search_operation()
            self.num_evaluation += self.evaluate_task("offspring")
            self.num_evaluation_list.append(self.num_evaluation)
            self.elitist_selection()
            self.update_gbest()
            self.result.append(self.parent[0].f_value[0])
            # self.generational_result(g, print_result=print_result)    # print the results at each generation

        self.generational_result(g, print_result=print_result)  # only print the results at the final generation
        return self.result, self.num_evaluation_list

    def evaluate_task(self, attr: str) -> float:
        time_taken = [self.evaluate_solution(attr)]
        return (sum(time_taken) / time_taken[-1]) * len(getattr(self, attr))

    def generational_result(self, g: int, print_result: bool) -> None:
        fitness_list = [p.f_value[0] for p in self.parent]
        avg_fitness = np.mean(fitness_list)
        if print_result:
            print("gen: {}, best: {}, mean: {}, pop size: {}".format(g, self.parent[0].f_value[0], avg_fitness,
                                                                     self.total_pop_size))
        print("Total evaluation cost: {}".format(self.num_evaluation))
        print("")
