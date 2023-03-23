import json

import matplotlib.pyplot as plt
import numpy as np

import config

if __name__ == '__main__':
    """
        This script is used to compare results between Single task EA and MEML through a fitness plot

        Parameters
        ----------
        dataset_identifier : str
            Name of the dataset to be used for comparison

        list_of_identifiers: list
            The algorithms used in comparison

        graph lines : list
            Key characters for defining graphlines in matplotlib

        graph color : list
            Key characters for defining color in matplotlib

        labels: list
            Label for each algorithm in the plot

        out_file: str
            The file that stores the output of the result ("result.json")
    """

    _, _, _, dataset_identifier = config.dataset()

    list_of_identifiers = ["_GA","_SBPSO", "_ALEEFS"]
    print(list_of_identifiers)

    graph_lines = ['y^--', 'rv--', 'b*--']
    graph_color = ['y', 'r', 'b']

    labels = ["GA","SBPSO", "EFSale"]
    out_file = config.output_file

    with open(out_file, "r") as read_file:
        new_data = json.load(read_file)
    

    # print(new_data.keys())

    plt.xlabel('Number of Function Evaluation')
    plt.ylabel('Fitness Value')

    print("\n\n")
    print(dataset_identifier)
    time_cost = []
    for i in range(len(list_of_identifiers)):
        result = new_data[dataset_identifier + list_of_identifiers[i]]
        fn_eval_array = np.array(result['num_evaluation_list'])
        fitness_array = np.array(result['fitness_list'])
        time_cost.append(result["time_cost"])
        print(result.keys())
        avg_fn_eval_list = np.mean(fn_eval_array, axis=0)
        if i == 1:
            avg_fn_eval_list = avg_fn_eval_list * time_cost[1] / time_cost[0]
        avg_fitness_list = np.mean(fitness_array, axis=0)

        std_fitness_list = np.std(fitness_array, axis=0)

        plt.plot(avg_fn_eval_list, avg_fitness_list, graph_lines[i],
                 label=labels[i])
        plt.fill_between(avg_fn_eval_list, avg_fitness_list - std_fitness_list / 2.0,
                         avg_fitness_list + std_fitness_list / 2.0,
                         color=graph_color[i], alpha=0.2)

    axes = plt.gca()
    # axes.set_xlim([0, 10000])
    # axes.set_ylim([0.78, 0.83])
    # axes.set_ylim([0.92, 0.94])
    plt.legend(loc="lower right")
    plt.show()
