import decimal
import operator
from functools import reduce
from math import ceil
from typing import List, Callable, Tuple

import numpy as np

from metaheuristics.ga import GeneticAlgorithm
from metaheuristics.individual import Individual
from metaheuristics.population import Population
from metaheuristics.prob_model import ProbabilityModel


class MEML_GA:
    """
        Multi-task Evolutionary Meta-Learning based on GA (using GA as the base algorithm)

        Class Parameters
        ----------
        total_population_size : int
            total population size

        tasks : list
            List of tasks (actually, each task is a GA object here)

        weights_matrix : numpy.ndarray
            Weights from mixture models

        ti : int
            Transfer interval

        lr : float
            Learning rate

        num_tasks : int
            Number of tasks (one main task plus several helper tasks)

        gen: int
            Number of generations

        ind_param : dict
            Parameters needed for initialising individual class

        num_evaluation : int
            Store the total number of evaluations undertaken by each task

        num_evaluation_list : list
            Store num_evaluation for each generation

        result : list
            Store the best fitness of each generation

        Class Methods
        ----------
        run() :
            Run the MEML-GA algorithm

        evaluate_task() :
            Evaluate the solutions in each task and return the total evaluation cost

        update_population_size() :
            Update the population size of each task based on the weights from mixture models

        update_weight_list() :
            Update the weight matrix after based on the weights from mixture models

        expectation_maximization() :
            The well-known EM algorithm

        sample_from_mixture_model() :
            Sample new solutions from the mixture model for each task

        sample_from_umf() :
            Sample from the probability vector based on UMF

        generational_result() :
            Print result from every generation

        univarate_marginal_distribution() :
            Create probability vectors based on UMF

        likelihood_from_umda() :
            Calculating the likelihood of a sample coming from each task

        likelihood_main_pop_loov() :
            Calculating the likelihood of a main task's sample coming from its probability distribution (UMF)

    """

    def __init__(self, num_tasks: int, gen: int, lr: float, ti: int, total_pop_size: int, f: Callable, weight: float,
                 train_datasets: List, test_datasets: tuple, task_param):

        self.total_pop_size = total_pop_size
        self.tasks = [GeneticAlgorithm(task_param["dim"], task_param["dim_range"], task_param["num_obj"],
                                       f, weight, train_datasets[i], test_datasets) for i in range(num_tasks)]

        self.weights_matrix = np.array(
            [np.array([1.0 / num_tasks for _ in range(num_tasks)]) for _ in range(num_tasks)])
        for idx, task in enumerate(self.tasks):
            task.create_parent(int(total_pop_size * self.weights_matrix[-1, idx]), **task_param)
        self.ti = ti
        self.lr = lr
        self.num_tasks = num_tasks
        self.gen = gen
        self.ind_param = task_param
        self.num_evaluation = 0
        self.num_evaluation_list = []
        self.result = []

    def run(self, print_result: bool) -> Tuple[List, List]:
        self.num_evaluation += self.evaluate_tasks("parent")
        self.num_evaluation_list.append(self.num_evaluation)
        for task in self.tasks:
            task.parent.sort(key=lambda p: p.f_value[0], reverse=True)
        self.result.append(self.tasks[-1].parent[0].f_value[0])
        for g in range(2, self.gen + 1):
            if self.num_tasks == 1 or g % self.ti != 0:
                for idx, task in enumerate(self.tasks):
                    task.run_search_operation()
            else:
                for idx, task in enumerate(self.tasks):
                    task.probability_model = self.univarate_marginal_distribution(task.parent)
                temp_weights_matrix = self.expectation_maximization()
                # print("raw weights: {}".format(temp_weights_matrix[-1, :]))
                self.update_weight_matrix(temp_weights_matrix)
                # print("Transfer No.{} - weights: {}".format(int(g / self.ti), self.weights_matrix[-1, :]))
                self.update_population_size()
                self.sample_from_mixture_model()

            self.num_evaluation += self.evaluate_tasks("offspring")
            self.num_evaluation_list.append(self.num_evaluation)
            for task in self.tasks:
                task.elitist_selection()

            self.result.append(self.tasks[-1].parent[0].f_value[0])
            # self.generational_result(g, print_result=print_result)    # print the results at each generation

        self.generational_result(g, print_result=print_result)  # only print the results at the final generation
        return self.result, self.num_evaluation_list

    def evaluate_tasks(self, attr: str) -> float:
        time_taken_per_task = [task.evaluate_solution(attr) for task in self.tasks]
        return (sum(time_taken_per_task) / time_taken_per_task[-1]) * len(getattr(self.tasks[-1], attr))

    def update_population_size(self):
        for idx, task in enumerate(self.tasks):
            task.pop_size = int(ceil(self.total_pop_size * self.weights_matrix[-1, :][idx]))

    def update_weight_matrix(self, temp_weights_matrix):
        self.weights_matrix[-1] = self.weights_matrix[-1, :] + self.lr * (
                temp_weights_matrix[-1, :] - self.weights_matrix[-1, :])
        self.weights_matrix[:-1] = temp_weights_matrix[:-1, :]

    def expectation_maximization(self) -> np.ndarray:
        weights_list = []
        for target_task in self.tasks:
            l_matrix = []
            for source_task in self.tasks:
                if target_task is source_task:
                    l_vec = self.likelihood_main_pop_loov(target_task.parent)
                    l_matrix.append(l_vec)
                else:
                    l_vec = self.likelihood_from_umda(target_task.parent, source_task.probability_model.noise_prob_vec)
                    l_matrix.append(l_vec)

            l_matrix = np.stack(l_matrix)
            iteration = 100
            weights = np.array([decimal.Decimal(1.0 / self.num_tasks) for _ in range(self.num_tasks)])
            for _ in range(iteration):
                marginal_vec = weights.dot(l_matrix)
                for j in range(self.num_tasks):
                    weights[j] = np.sum((weights[j] * l_matrix[j, :]) / marginal_vec) / l_matrix.shape[1]

            modified_weights = np.array(weights, dtype=np.float32) + np.random.normal(0, 0.01, len(weights))
            for i in range(len(modified_weights)):
                modified_weights[i] = max(modified_weights[i], 0)
            psum = sum(modified_weights)
            if psum == 0:
                new_weights = np.zeros(len(modified_weights))
                new_weights[-1] = 1.0
            else:
                new_weights = modified_weights / psum

            weights_list.append(new_weights)

        return np.array(weights_list)

    def sample_from_mixture_model(self):
        transferred_solutions_list = []
        for idx, task in enumerate(self.tasks):
            task_sample_size = np.ceil(task.pop_size * self.weights_matrix[:, idx]).astype(np.int32)
            transferred_solutions = []
            for sample_size in task_sample_size:
                transferred_solutions.append(
                    self.sample_from_umf(sample_size, task.probability_model.prob_vec))
            transferred_solutions_list.append(transferred_solutions)

        transferred_solutions_list = list(map(list, zip(*transferred_solutions_list)))
        for idx, task in enumerate(self.tasks):
            task.create_temp_parent("transfer",
                                    transferred_solutions=reduce(operator.add, transferred_solutions_list[idx]))
            task.create_offspring()

    def sample_from_umf(self, num_sample: int, prob_vec: np.ndarray):
        self.ind_param["init_distribution"] = Population.univariate_marginal_frequency
        self.ind_param["prob_vec"] = prob_vec
        return [Individual(**self.ind_param) for _ in range(num_sample)]

    def generational_result(self, g: int, print_result: bool) -> None:
        for idx, task in enumerate(self.tasks):
            fitness_list = [p.f_value[0] for p in task.parent]
            avg_fitness = np.mean(fitness_list)
            if print_result:
                print("Task-{} ---> gen: {}, best: {}, mean: {}, pop size: {}".format(idx, g, task.parent[0].f_value[0],
                                                                                      avg_fitness, task.pop_size))
        print("Total evaluation cost: {}".format(self.num_evaluation))
        print("")

    @staticmethod
    def univarate_marginal_distribution(parent):
        solutions = np.array([p.arr for p in parent])
        prob_vec = np.average(solutions, axis=0)
        noise_prob_vec = np.average(solutions, axis=0)
        for i in range(len(noise_prob_vec)):
            if noise_prob_vec[i] != 0:
                noise_prob_vec[i] += np.random.normal(0, 0.01)
        noise_prob_vec[noise_prob_vec <= 0.01] = 0.01
        noise_prob_vec[noise_prob_vec >= 0.99] = 0.99
        return ProbabilityModel(prob_vec=prob_vec, noise_prob_vec=noise_prob_vec)

    @staticmethod
    def likelihood_from_umda(pop, prob_vec):
        constant = decimal.Decimal('1.0')
        likelihood_vec = np.array([constant for _ in range(len(pop))], dtype=np.dtype(decimal.Decimal))
        for counter, ind in enumerate(pop):
            for i in range(len(prob_vec)):
                if ind.arr[i] == 0:
                    p = 1.0 - prob_vec[i]
                else:
                    p = prob_vec[i]

                likelihood_vec[counter] *= decimal.Decimal(p)

        return likelihood_vec

    @staticmethod
    def likelihood_main_pop_loov(pop):
        if len(pop) == 1:
            likelihood_vec = []
            prob_model = MEML_GA.univarate_marginal_distribution(pop)
            likelihood_vec.extend(MEML_GA.likelihood_from_umda(pop, prob_model.prob_vec))
        else:
            k_folds = len(pop)  # leave one out cross validation
            likelihood_vec = []
            for i in range(k_folds):
                batch_pop = [pop[i]]
                if i != k_folds - 1:
                    model_pop = pop[:i] + pop[i + 1:]
                else:
                    model_pop = pop[:i]

                prob_model = MEML_GA.univarate_marginal_distribution(model_pop)
                likelihood_vec.extend(MEML_GA.likelihood_from_umda(batch_pop, prob_model.prob_vec))
            # prob_model = MEML_GA.univarate_marginal_distribution([pop])
        return likelihood_vec
