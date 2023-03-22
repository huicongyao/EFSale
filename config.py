from dataset_utils import *

"""
    The configuration file that stores the experiment settings for running a specifc algorithm (GA, SBPSO, MEML-GA and MEML-SBPSO).

    Parameters
    ----------
    dataset : Callable
         Function to retrieve/generate dataset
         
    output_file : str
        file_path for the output file
        
    num_trials: int
        Number of times to run a specific algorithm
        
    total_pop_size: int
        Total population size used in a specific algorithm
        
    gen: int
        Number of generations used in a specific algorithm
        
    weight: float
        A weight parameter for the additional term in the fitness function 
        (NOTE: the function for defining the additional term needs to be defined by user, 
            which can be seen in "metaheuristics/fitness_fun.py")
    
    num_tasks: int
        Number of tasks (ONLY used in MEML-GA and MEML-SBPSO)
        
    lr: float
        Learning rate for mixture modelling (ONLY used in MEML-GA and MEML-SBPSO)
        
    ti: int
        Transfer interval (ONLY used in MEML-GA and MEML-SBPSO)
        
    evol_algo: str
        Name for the base algorithm in MEML (ONLY used for plotting results)
    
"""

dataset = read_credit_card_default
output_file = "result.json"
num_trials = 10
total_pop_size = 100
gen = 100
weight = 0.5

    

#### ONLY used for plotting results
evol_algo = "GA"

