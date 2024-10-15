"""
This script is supposed to create multiple make_classification datasets and then run a hyperparameter optimization on them.
The results are then stored in a csv file.

"""

import datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import utils
import itertools
import random
from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd
import os
import ast
import numpy as np

def hp_search_svm(dataset):
    X_train, X_test, y_train, y_test = dataset["X_train"], dataset["X_test"], dataset["y_train"], dataset["y_test"]

    estimator = SVC(kernel="rbf", class_weight=utils.class_weight_dict(y_train))
    param_grid = dict(gamma=[1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100], C=[1e-4, 1e-2, 1e0, 1e2, 1e4])

    clf = GridSearchCV(estimator, param_grid)
    clf.fit(X_train, y_train.ravel())

    best_gamma = clf.best_params_["gamma"]
    best_C = clf.best_params_["C"]
    return best_gamma, best_C

def hp_search_knn(dataset):
    X_train, X_test, y_train, y_test = dataset["X_train"], dataset["X_test"], dataset["y_train"], dataset["y_test"]

    estimator = KNN()
    param_grid = dict(n_neighbors=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
    grid = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train.ravel())
    best_n_neighbors = grid.best_params_['n_neighbors']
    return best_n_neighbors


def get_save_file_path():
    save_dir = "results/make_classification/"
    save_file = "hp_opt_results.csv"
    return save_dir, save_file

def init_save_file(make_classification_kwargs):
    # try to load the save file
    try:
        result_df = load_save_file()
        return result_df
    except:
        # init empty dataframe with the columns of make_classification_kwargs and the hyperparameters
        result_df = pd.DataFrame(columns=list(make_classification_kwargs.keys()) + ["best_gamma", "best_C", "best_k"])
        return result_df

def load_save_file():
    save_dir, save_file = get_save_file_path()
    # if file exists, load it
    if not os.path.exists(save_dir + save_file):
        return None
    else:
        df = pd.read_csv(save_dir + save_file)
    return df

def save_results(result_dict):
    save_dir, save_file = get_save_file_path()
    # if dictionary does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if no save file exists, create a new one
    if not os.path.exists(save_dir + save_file):
        result_df = pd.DataFrame(columns=list(result_dict.keys()))
        result_df.to_csv(save_dir + save_file, index=False)
    # load the previous save file
    df = load_save_file()
    # make new row from result_dict
    new_row = pd.DataFrame([result_dict])
    # add the new row to the dataframe
    df = pd.concat([df, new_row])
    # save the new dataframe to the save file
    df.to_csv(save_dir + save_file, index=False)
    print("Saved new result to save file.")
    return

def check_if_in_df(df, class_weight, n_clusters_per_class, ratio_informative):
    if df is None:
        return False
    if len(df) > 0:

        already_in_df = df_result[
            (ast.literal_eval(df_result["weights"][0])[0] == class_weight) &
            (df_result["n_clusters_per_class"] == n_clusters_per_class) &
            (df_result["n_informative"] == ratio_informative)
        ].shape[0] > 0
    else:
        already_in_df = False
    return already_in_df

if __name__ == "__main__":
    # fix random seed for reproducibility
    random.seed(0)

    class_weight_candidates = [0.5, 0.6, 0.7, 0.8, 0.9]
    n_clusters_per_class_candidates = [2, 3, 4, 5]
    ratio_informative_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # define a grid of hyperparameters
    grid = list(itertools.product(
        class_weight_candidates,
        n_clusters_per_class_candidates,
        ratio_informative_candidates
    ))

    # iterate over the grid
    for class_weight, n_clusters_per_class, ratio_informative in grid:
        n_features = 20
        n_informative = int(n_features * ratio_informative)
        n_redundant = n_features - n_informative

        make_classification_kwargs = dict(
            n_samples=1000,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            flip_y=0.1,
            weights=[class_weight, 1 - class_weight],
            random_state=0
        )

        # load the save file
        df_result = load_save_file()

        # check if the values for class_weight, n_clusters_per_class and ratio_informative are already in the dataframe
        already_in_df = check_if_in_df(df_result, class_weight, n_clusters_per_class, n_informative)
        if already_in_df:
            print("Already in df. Skipping.")
            continue
        try:
            dataset, _ = datasets.provide_dataset("sklearn_make_classification", flipping_set_size=100,
                                                  opt_arguments=make_classification_kwargs)
        except ValueError:
            print("Dataset parameters incompatible. Skipping.")
            continue

        best_gamma, best_C = hp_search_svm(dataset)
        best_k = hp_search_knn(dataset)

        # make a resulting dict from make_classification_kwargs and the hyperparameters
        result_dict = make_classification_kwargs
        result_dict["best_gamma"] = best_gamma
        result_dict["best_C"] = best_C
        result_dict["best_k"] = best_k

        # add a new row to the dataframe
        save_results(result_dict)

    print("stop")