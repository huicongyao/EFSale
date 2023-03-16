import json
import os
from tkinter.messagebox import NO
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def train_test_split_stratified(X, y, test_size, random_state=None):
    """
        Split dataset into stratified train and test set

        Parameters
        ----------
        train_X : array of shape [n_samples, n_features]
            Train set instances

        train_y : array of shape [n_samples]
            The integer labels for class membership of each train instance.

        test_X : array of shape [n_samples, n_features]
            Test set instances

        test_y : array of shape [n_samples]
            The integer labels for class membership of each test instance.
    """

    train_X, train_y, test_X, test_y = None, None, None, None
    if random_state:
        skf = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    else:
        skf = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_idx, test_idx in skf.split(X, y):
        train_X, train_y = X[train_idx], y[train_idx]
        test_X, test_y = X[test_idx], y[test_idx]

    return train_X, train_y, test_X, test_y

# 定义用来分化minibatch的函数

def generate_mini_batch(train_dataset, mini_size, random_state = None):
    train_X, train_y, X, y = None, None, train_dataset[0], train_dataset[1]
    # idx = np.random.choice(len(X), int(mini_size * len(X)), replace=False) # 这里没有使用分层抽样
    # train_X, train_y = X[idx], y[idx]
    if random_state:
        skf = StratifiedShuffleSplit(n_splits=1, test_size=mini_size, random_state=random_state)
    else:
        skf = StratifiedShuffleSplit(n_splits=1, test_size=mini_size)
    
    for train_idx, test_idx in skf.split(X, y):
        train_X, train_y = X[test_idx], y[test_idx]    # 使用了分层抽样的结果
    return train_X, train_y

        

def k_fold_split(X, y, k_folds):
    """
        Split train dataset into smaller evenly-sized dataset for each auxiliary task
    """

    k_folds_X = []
    k_folds_y = []
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    for train_idx, test_idx in skf.split(X, y):
        k_folds_X.append(X[test_idx])
        k_folds_y.append(y[test_idx])

    return list(zip(k_folds_X, k_folds_y))



def output_to_json(**kwargs):
    """
        Output results and experiment details to json file
    """

    if os.path.exists(kwargs["json_filename"]):
        with open(kwargs["json_filename"]) as json_file:
            json_decoded = json.load(json_file)
        print("file exist")
    else:
        json_decoded = {}
        print("file does not exist")
    print(kwargs["dataset_name"])
    print(kwargs["algorithm"])
    json_decoded[kwargs["dataset_name"] + "_" + kwargs["algorithm"]] = kwargs

    with open(kwargs["json_filename"], "w") as write_file:
        json.dump(json_decoded, write_file)


# this is a toy example
def read_sklearn_toy_dataset_50k() -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
        Create toy dataset using sklearn api

        Parameters
        ----------

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The generated samples.

        y : array of shape [n_samples]
            The integer labels for class membership of each sample.

        dim : int
            Dimensionality of the samples

        file_string : str
            Name of this dataset
    """

    file_string = "sklearn_toy_dataset_50"

    X, y = make_classification(n_samples=50000, n_features=100, n_classes=2,
                               n_informative=10, n_redundant=30, n_repeated=30, random_state=1)

    dim = X.shape[1]

    return X, y, dim, file_string


def read_spam_database_data() -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
        Read your dataset from .xls/.csv/.txt/.data files

        Parameters
        ----------

        Returns
        -------
        X : array of shape [n_samples, n_features]
            Instances of the dataset

        y : array of shape [n_samples]
            The integer labels for class membership of each instance.

        dim : int
            Dimensionality of the samples

        file_string : str
            Name of this dataset
    """

    file_path = 'datasets/spambase.data'
    df_source = pd.read_csv(file_path, sep=',', encoding='utf-8', header=None)

    df_X = df_source.iloc[:, :-1]
    df_y = df_source.iloc[:, -1]

    X = df_X.values
    y = df_y.values
    dim = X.shape[1]
    file_string = "spam_database"

    return X, y, dim, file_string


def read_credit_card_default() -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
        Read your dataset from .xls/.csv/.txt/.data files

        Parameters
        ----------

        Returns
        -------
        X : array of shape [n_samples, n_features]
            Instances of the dataset

        y : array of shape [n_samples]
            The integer labels for class membership of each instance.

        dim : int
            Dimensionality of the samples

        file_string : str
            Name of this dataset
    """
    file_path = 'datasets/default_of_credit_card_clients.csv'
    df_source = pd.read_csv(file_path, sep=',', encoding='utf-8', header=0)

    df_X = df_source.iloc[:, :-1]
    df_y = df_source.iloc[:, -1]

    X = df_X.values
    y = df_y.values
    dim = X.shape[1]
    file_string = "credit_card_default"

    return X, y, dim, file_string






def read_ida2016() -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
        Read your dataset from .xls/.csv/.txt/.data files

        Parameters
        ----------

        Returns
        -------
        X : array of shape [n_samples, n_features]
            Instances of the dataset

        y : array of shape [n_samples]
            The integer labels for class membership of each instance.

        dim : int
            Dimensionality of the samples

        file_string : str
            Name of this dataset
    """
    file_path = 'datasets/aps_failure_training_set_0.csv'
    df_train = pd.read_csv(file_path, sep=',', encoding='utf-8', header=0)
    file_path = 'datasets/aps_failure_test_set_0.csv'
    df_test = pd.read_csv(file_path, sep=',', encoding='utf-8', header=0)
    df_source = pd.concat([df_train, df_test], ignore_index=True)
    df_source['class'].replace(to_replace=['neg', 'pos'], value=[0, 1], inplace=True)

    df_X, df_y = df_source.iloc[:, 1:], df_source.iloc[:, 0]
    df_X.replace(to_replace='na', value=np.nan, inplace=True)
    df_X = df_X.apply(pd.to_numeric)
    df_X.fillna(df_X.mean(), inplace=True)

    X = df_X.values
    y = df_y.values
    dim = X.shape[1]
    file_string = "ida_2016"

    return X, y, dim, file_string




def read_LOL_Master_Ranked_Games():
    
    
    # file_path = "datasets/LOL_dataset/Challenger_Ranked_Games.csv"
    # file_path = "datasets/LOL_dataset/GrandMaster_Ranked_Games.csv"
    file_path = "datasets/LOL_dataset/Master_Ranked_Games.csv"
    df_source = pd.read_csv(file_path, sep=',', encoding="utf-8", header=0)
    lol_data = df_source.copy()
    X = lol_data.filter(like="blue")
    gameDuration = lol_data["gameDuraton"]
    X.insert(0, "gameduraton", gameDuration, True)
    y = X.pop("blueWins").values
    X = X.values
    
    dim = X.shape[1]
    file_string = "LOL_Master_Ranked_Games"
    return X, y, dim, file_string


def read_LOL_GrandMaster_Ranked_Games():
    
    
    file_path = "datasets/LOL_dataset/GrandMaster_Ranked_Games.csv"
    df_source = pd.read_csv(file_path, sep=',', encoding="utf-8", header=0)
    lol_data = df_source.copy()
    X = lol_data.filter(like="blue")
    gameDuration = lol_data["gameDuraton"]
    X.insert(0, "gameduraton", gameDuration, True)
    y = X.pop("blueWins").values
    X = X.values
    
    dim = X.shape[1]
    file_string = "LOL_GrandMaster_Ranked_Games"
    return X, y, dim, file_string



def read_LOL_Challenger_Ranked_Games():
    
    
    file_path = "datasets/LOL_dataset/Challenger_Ranked_Games.csv"
    df_source = pd.read_csv(file_path, sep=',', encoding="utf-8", header=0)
    lol_data = df_source.copy()
    X = lol_data.filter(like="blue")
    gameDuration = lol_data["gameDuraton"]
    X.insert(0, "gameduraton", gameDuration, True)
    y = X.pop("blueWins").values
    X = X.values
    
    dim = X.shape[1]
    file_string = "LOL_Challenger_Ranked_Games"
    return X, y, dim, file_string