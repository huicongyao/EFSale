import config
from dataset_utils import *
from metaheuristics.ga import GA_main
from metaheuristics.leea import LEEA_main
from metaheuristics.leeav0 import LEEA_main_v0
from metaheuristics.sbpso import SBPSO_main
from metaheuristics.population import Population
from metaheuristics.fitness_fun import *
from metaheuristics.meml_ga import MEML_GA
from metaheuristics.meml_sbpso import MEML_SBPSO
import time

if __name__ == '__main__':
    """
        This is the main script to run a specific feature selection algorithm (GA, SBPSO, MEML-GA and MEML-SBPSO). Most hyperparameters are located in config.py

        Parameters
        ----------
        list_of_fitness : list
            Store the best fitness of every generation from each trial
        
        list_of_num_evaluation: list
            Store the number of evaluations of every generation from each trial
            
        X : array of shape [n_samples, n_features]
            Instances of the dataset

        y : array of shape [n_samples]
            The integer labels for class membership of each instance.

        dim: int
            Dimensionality of the samples

        file_string: str
            Name of the dataset
            
        train_datasets: list
            list of train set (X,y) assigned accordingly for each task
            
        test_dataset: tuple
            test set (X,y) to be used for fitness evaluation
            
        task_param: dict
            parameter for initialising individuals in each task
        
        experiment_meta: dict
            Contain the hyperparameter and information of the experiment, to be stored together with the results in the output
    """
    alg_names = []
    alg_names.append("LEEA")
    alg_names.append("LEEAv0")
    alg_names.append("GA")
    alg_names.append("SBPSO")
    # alg_names.append("meml_GA")
    # alg_names.append("meml_SBPSO")

    X, y, dim, dataset_name = config.dataset()  # import dataset from the configuration file

    for alg_name in alg_names:
        
        if alg_name == "GA":
            
            print("\nBASIC_GENETIC_ALGORITHM\n\n")
            list_of_fitness = []
            list_of_num_evaluation = []
            time_cost = 0
            for i in range(1, config.num_trials + 1):
                print("=============================== Trial No.{} ===================================".format(i))
                # parameters needed for initialising each task (GENERALLY, user do not need to modify them)
                task_param = {"init_distribution": Population.uniform_distribution, 
                            "discrete": True, "dim": dim, "dim_range": [0, 1],
                            "num_obj": 1}    # the "init_distribution" denotes 
                                            # the function of encoding solutions

                    # split the used dataset to train set and test set
                train_X, train_y, test_X, test_y = \
                    train_test_split_stratified(X=X, y=y, test_size=0.2, random_state=i)
                train_dataset, test_dataset = (train_X, train_y), (test_X, test_y)
                alg = GA_main(gen=config.gen, total_pop_size=config.total_pop_size, 
                                f=fitness_function, weight=config.weight,
                                train_dataset=train_dataset, 
                                test_dataset=test_dataset, 
                                task_param=task_param)
                start_time = time.time()
                fitness_list, num_evaluation_list = alg.run(print_result=True)
                time_cost = time_cost + time.time() - start_time
                list_of_fitness.append(fitness_list)
                list_of_num_evaluation.append(num_evaluation_list)

            experiment_meta = {"dataset_name": dataset_name, "pop_size": config.total_pop_size, "gen": config.gen,
                            "algorithm": alg_name, "num_trial": config.num_trials,
                            "json_filename": config.output_file, "fitness_list": list_of_fitness,
                            "num_evaluation_list": list_of_num_evaluation}
            output_to_json(**experiment_meta)
        
            print("Total time cost: {},"
                " average time cost: {}".format(time_cost, time_cost / config.num_trials))
        
        if alg_name == "LEEA":
            print("\nLEAST_EVALUATION_EVOLUTION_ALGORITHM\n\n")
            list_of_fitness = []
            list_of_num_evaluation = []
            time_cost = 0
            
            for i in range(1, config.num_trials + 1):
                print("=============================== Trial No.{} ===================================".format(i))
                # parameters needed for initialising each task (GENERALLY, user do not need to modify them)
                task_param = {"init_distribution": Population.uniform_distribution, 
                            "discrete": True, "dim": dim, "dim_range": [0, 1],
                            "num_obj": 1}    # the "init_distribution" denotes 
                                            # the function of encoding solutions

                    # split the used dataset to train set and test set
                train_X, train_y, test_X, test_y = \
                    train_test_split_stratified(X=X, y=y, test_size=0.2, random_state=i)
                train_dataset, test_dataset = (train_X, train_y), (test_X, test_y)
                alg = LEEA_main(gen=config.gen, total_pop_size=config.total_pop_size, 
                                f=fitness_function, weight=config.weight,
                                train_dataset=train_dataset, 
                                test_dataset=test_dataset, 
                                task_param=task_param)
                start_time = time.time()
                fitness_list, num_evaluation_list = alg.run(print_result=True)
                time_cost = time_cost + time.time() - start_time
                list_of_fitness.append(fitness_list)
                list_of_num_evaluation.append(num_evaluation_list)

            experiment_meta = {"dataset_name": dataset_name, "pop_size": config.total_pop_size, "gen": config.gen,
                            "algorithm": alg_name, "num_trial": config.num_trials,
                            "json_filename": config.output_file, "fitness_list": list_of_fitness,
                            "num_evaluation_list": list_of_num_evaluation}
            output_to_json(**experiment_meta)
        
            print("Total time cost: {},"
                " average time cost: {}".format(time_cost, time_cost / config.num_trials))
        
        if alg_name == "LEEAv0":
            print("\nLEAST_EVALUATION_EVOLUTION_ALGORITHM--0\n\n")
            list_of_fitness = []
            list_of_num_evaluation = []
            time_cost = 0
            
            for i in range(1, config.num_trials + 1):
                print("=============================== Trial No.{} ===================================".format(i))
                # parameters needed for initialising each task (GENERALLY, user do not need to modify them)
                task_param = {"init_distribution": Population.uniform_distribution, 
                            "discrete": True, "dim": dim, "dim_range": [0, 1],
                            "num_obj": 1}    # the "init_distribution" denotes 
                                            # the function of encoding solutions

                    # split the used dataset to train set and test set
                train_X, train_y, test_X, test_y = \
                    train_test_split_stratified(X=X, y=y, test_size=0.2, random_state=i)
                train_dataset, test_dataset = (train_X, train_y), (test_X, test_y)
                alg = LEEA_main_v0(gen=config.gen, total_pop_size=config.total_pop_size, 
                                f=fitness_function, weight=config.weight,
                                train_dataset=train_dataset, 
                                test_dataset=test_dataset, 
                                task_param=task_param)
                start_time = time.time()
                fitness_list, num_evaluation_list = alg.run(print_result=True)
                time_cost = time_cost + time.time() - start_time
                list_of_fitness.append(fitness_list)
                list_of_num_evaluation.append(num_evaluation_list)

            experiment_meta = {"dataset_name": dataset_name, "pop_size": config.total_pop_size, "gen": config.gen,
                            "algorithm": alg_name, "num_trial": config.num_trials,
                            "json_filename": config.output_file, "fitness_list": list_of_fitness,
                            "num_evaluation_list": list_of_num_evaluation}
            output_to_json(**experiment_meta)
        
            print("Total time cost: {},"
                " average time cost: {}".format(time_cost, time_cost / config.num_trials))

        
        if alg_name == "SBPSO":
            print("\nSTICKY BINARY PARTICLE SWARM OPTIMIZATION\n\n")
            list_of_fitness = []
            list_of_num_evaluation = []
            time_cost = 0

            for i in range(1, config.num_trials + 1):
                print("=============================== Trial No.{} ===================================".format(i))
                # parameters needed for initialising each task (GENERALLY, user do not need to modify them)
                task_param = {"init_distribution": Population.uniform_distribution, 
                            "discrete": True, "dim": dim, "dim_range": [0, 1],
                            "num_obj": 1}    # the "init_distribution" denotes 
                                            # the function of encoding solutions
                                            
                train_X, train_y, test_X, test_y = train_test_split_stratified(X=X, y=y, test_size=0.2, random_state=i)
                train_dataset, test_dataset = (train_X, train_y), (test_X, test_y)
                alg = SBPSO_main(gen=config.gen, total_pop_size=config.total_pop_size, 
                                f=fitness_function, weight=config.weight,
                                train_dataset=train_dataset, 
                                test_dataset=test_dataset, 
                                task_param=task_param)
                start_time = time.time()
                fitness_list, num_evaluation_list = alg.run(print_result=True)
                time_cost = time_cost + time.time() - start_time
                list_of_fitness.append(fitness_list)
                list_of_num_evaluation.append(num_evaluation_list)
            
            
            experiment_meta = {"dataset_name": dataset_name, "pop_size": config.total_pop_size, "gen": config.gen,
                            "algorithm": alg_name, "num_trial": config.num_trials,
                            "json_filename": config.output_file, "fitness_list": list_of_fitness,
                            "num_evaluation_list": list_of_num_evaluation}
            output_to_json(**experiment_meta)
        
            print("Total time cost: {},"
                " average time cost: {}".format(time_cost, time_cost / config.num_trials))

        if alg_name == "meml_GA":
            print("\nMEML_GA\n\n")
            list_of_fitness = []
            list_of_num_evaluation = []
            time_cost = 0

            for i in range(1, config.num_trials + 1):
                print("=============================== Trial No.{} ===================================".format(i))
                # parameters needed for initialising each task (GENERALLY, user do not need to modify them)
                task_param = {"init_distribution": Population.uniform_distribution, 
                            "discrete": True, "dim": dim, "dim_range": [0, 1],
                            "num_obj": 1}    # the "init_distribution" denotes 
                                            # the function of encoding solutions
                                            
                train_datasets, test_datasets = split_datasets_for_MEML(X=X, y=y, test_size=0.2,
                                                                    k_folds=config.num_tasks - 1, random_state=i)
                alg = MEML_GA(num_tasks=config.num_tasks, gen=config.gen, lr=config.lr, ti=config.ti,
                          total_pop_size=config.total_pop_size, f=fitness_function, weight=config.weight, train_datasets=train_datasets,
                          test_datasets=test_datasets, task_param=task_param)
                start_time = time.time()
                fitness_list, num_evaluation_list = alg.run(print_result=True)
                time_cost = time_cost + time.time() - start_time
                list_of_fitness.append(fitness_list)
                list_of_num_evaluation.append(num_evaluation_list)
            
            
            experiment_meta = {"dataset_name": dataset_name, "pop_size": config.total_pop_size, "gen": config.gen,
                            "algorithm": alg_name, "num_trial": config.num_trials,
                            "json_filename": config.output_file, "fitness_list": list_of_fitness,
                            "num_evaluation_list": list_of_num_evaluation}
            output_to_json(**experiment_meta)
        
            print("Total time cost: {},"
                " average time cost: {}".format(time_cost, time_cost / config.num_trials))
        
        
        if alg_name == "meml_SBPSO":
            print("\nMEML_SBPSO\n\n")
            list_of_fitness = []
            list_of_num_evaluation = []
            time_cost = 0

            for i in range(1, config.num_trials + 1):
                print("=============================== Trial No.{} ===================================".format(i))
                # parameters needed for initialising each task (GENERALLY, user do not need to modify them)
                task_param = {"init_distribution": Population.uniform_distribution, 
                            "discrete": True, "dim": dim, "dim_range": [0, 1],
                            "num_obj": 1}    # the "init_distribution" denotes 
                                            # the function of encoding solutions
                                            
                train_datasets, test_datasets = split_datasets_for_MEML(X=X, y=y, test_size=0.2,
                                                                    k_folds=config.num_tasks - 1, random_state=i)
                alg = MEML_SBPSO(num_tasks=config.num_tasks, gen=config.gen, lr=config.lr, ti=config.ti,
                             total_pop_size=config.total_pop_size, f=fitness_function, weight=config.weight, train_datasets=train_datasets,
                             test_datasets=test_datasets, task_param=task_param)
                start_time = time.time()
                fitness_list, num_evaluation_list = alg.run(print_result=True)
                time_cost = time_cost + time.time() - start_time
                list_of_fitness.append(fitness_list)
                list_of_num_evaluation.append(num_evaluation_list)
            
            
            experiment_meta = {"dataset_name": dataset_name, "pop_size": config.total_pop_size, "gen": config.gen,
                            "algorithm": alg_name, "num_trial": config.num_trials,
                            "json_filename": config.output_file, "fitness_list": list_of_fitness,
                            "num_evaluation_list": list_of_num_evaluation}
            output_to_json(**experiment_meta)
        
            print("Total time cost: {},"
                " average time cost: {}".format(time_cost, time_cost / config.num_trials))

