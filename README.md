Please install python 3.5 and above

### Dependencies:

```bash
pip install -r requirements.txt
```

Basic Instructions:

Add the used dataset in the "datasets" folder, and add a read dataset function in "dataset_utils" file.

Put the experiment settings for running a specific algorithm in "config.py" file.

Execute "mian.py" to run different feature selection algorithms (GA, SBPSO and EFSale).

EXecute "plot_result.py" to show the difference among feature selection algorithms

NOTE: 
1. In the current version, only the classifier accuracy is used as the fitness function in different algorithms. For users who want to include an additional term in the fitness function, they can add the codes for defining the additional term in the get_additonal_term() function in  "metaheuristics/fitness_fun.py" file.
2. You may find running EFSale usually cost more time, that is because we have compete evert feature in all training set first to transform the computing cost, and then cover it with mini train data set.