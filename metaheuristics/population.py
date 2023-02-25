import random
from copy import deepcopy
from typing import List

import numpy as np

from metaheuristics.individual import Individual


class Population:
    """
        A population of solutions (instance of Individual class) 
        to be used in any population-based EC algorithms

        Class Parameters
        ----------
        offpsring : list (default=None)
            Offspring of the current generation

        parent : list (default=None)
            Parent of the current generation

        temp_parent : list (default=None)
            Temp parent are parents selected to go through 
            search operation

        pop_size : int (default=None)
            Size of the population in the current generation

        dim : int
            Dimension of the solution

        dim_range : list
            Range for each dimension of the solution

        num_obj : int
            Number of objectives in the current problem set
    """

    def __init__(self, dim: int, dim_range: List, num_obj: int):
        self.offspring = None
        self.parent = None
        self.temp_parent = None
        self.pop_size = None
        self.dim = dim
        self.dim_range = dim_range
        self.num_obj = num_obj

    def create_parent(self, pop_size: int, **kwargs):
        self.parent = [Individual(**kwargs) for _ in range(pop_size)]
        self.pop_size = pop_size

    def create_temp_parent(self, ttype: str, transferred_solutions=None):
        if ttype == "random":
            self.temp_parent = random.sample(deepcopy(self.parent),
                                             len(self.parent))
        # sample(population, k, *, counts=None)
        # choose k unique random elements from a population sequence or set
        elif ttype == "transfer":
            self.temp_parent = transferred_solutions
        else:
            raise ValueError('Wrong type entered')
            # ValueError is raised when a user gives an invalid value
            # to a function but is of a valid argument

    def create_offspring(self):
        self.offspring = self.temp_parent

    def remove_offspring(self):
        self.offspring = None
        self.temp_parent = None

    def set_new_parent(self, new_parents):
        self.parent = new_parents
        self.remove_offspring()

    @staticmethod # 均匀分布
    def uniform_distribution(dim: int, dim_range: List, 
                             discrete: bool) -> np.ndarray:
        if discrete:
            return np.rint( (dim_range[1] - dim_range[0])\
                * np.random.rand(dim) + dim_range[0] )
            # np.rind(): round elements of the array to the nearest integer
        else:
            return (dim_range[1] - dim_range[0]) * \
                np.random.rand(dim) + dim_range[0]

    @staticmethod # 边缘频率分布
    def univariate_marginal_frequency(dim: int, 
                                      dim_range: List, discrete: bool,
                                      prob_vec: np.ndarray):
        p = np.random.rand(dim)
        bool_vec = p < prob_vec
        return bool_vec.astype(int)
