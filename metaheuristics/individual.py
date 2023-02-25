from typing import Callable

import numpy as np


class Individual:
    """
        Each solution in a population is an instance 
        of the Individual class.

        Class Parameters
        ----------
        arr : numpy.ndarray
            The encoded solution

        f_value : numpy.ndarray
            f_value is a list that holds fitness 
            values across single/multi objective problems

        pbest : numpy.ndarray (default=None)
            Personal best for each solution (used 
            for PSO algorithms)

        pbest_score : float (default=0.0)
            Personal best score for each solution (
                used for PSO algorithms)

        transfer_id : str
            Can be used to track transferred solution (
                Not used in this version)

    """

    def __init__(self, num_obj: int, 
                 init_distribution: Callable, **kwargs):
        # 类型和变量注解只是提供了一种提示，对于运行实际上没有任何影响
        # 有一些库，如 mypy，可一检查出脚本中不符合类型注解的调用情况
        # **kwarg = dim: int, dim_range: List, discrete: bool
        self.arr = init_distribution(**kwargs)
        self.f_value = np.zeros(num_obj)
        self.pbest_score = 0.0
        self.pbest = None
        self.transfer_id = None
        self.current_life = np.array([0 for _ in range(kwargs["dim"])])
        self.flipping_prob = None
        self.stickiness = None


    def __len__(self):
        return self.arr.size
