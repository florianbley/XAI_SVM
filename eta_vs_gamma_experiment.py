import numpy as np
import os
import pandas as pd
import random
import datasets
import ast
from sklearn.svm import SVC
from feature_flipping import train_kde, run_feature_flipping
from neuralised_svm import neuralised_svm
from hp_opt_make_classification import hp_search_svm
import utils
from copy import deepcopy

def extract_make_classification_params_from_df_row(row):
    make_classification_kwargs = dict(
        n_samples=1000,
        n_features=row["n_features"],
        n_informative=row["n_informative"],
        n_redundant=row["n_redundant"],
        n_clusters_per_class=row["n_clusters_per_class"],
        flip_y=0.1,
        weights=ast.literal_eval(row[["weights"]][0]),
        random_state=row["random_state"]
    )
    return make_classification_kwargs


def neural_explanations(model, dataset):
    explanation_dict = {}
    neural_svm = neuralised_svm(model)
    X_flipping = dataset["X_flipping"]
    for eta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        R_eta = neural_svm.explain(X_flipping, first_rule="hybrid", eta=eta)
        explanation_dict["R_eta={}".format(eta)] = R_eta
    return explanation_dict

def dataset_selection_for_experiment(df_dataset_params):
    unique_gamma, reverse_inds = np.unique(df_dataset_params[["best_gamma"]].values, return_inverse=True)
    print("Done")

    df_out = pd.DataFrame(columns=df_dataset_params.columns)
    for ind, gamma in enumerate(unique_gamma):
        gamma_inds = np.where(reverse_inds == ind)[0]
        n_unique = len(gamma_inds)
        if n_unique == 0:
            continue
        gamma_inds = np.random.choice(gamma_inds, min(n_unique, 5), replace=False)
        for row_ind in gamma_inds:
            row = df_dataset_params.iloc[row_ind:row_ind + 1]
            df_out = pd.concat([df_out, row])
        # if n_unique is smaller than 5, vary the random state of a random entry
        while n_unique < 5:
            # random gamma index
            random_gamma_ind = np.random.choice(gamma_inds)
            row = deepcopy(df_dataset_params.iloc[random_gamma_ind:random_gamma_ind + 1])
            row["random_state"] = random.randint(0, 1000)
            df_out = pd.concat([df_out, row])
            n_unique += 1
    return df_out

def auc_column_names():
    return ["R_eta=0", "R_eta=0.1", "R_eta=0.2", "R_eta=0.3", "R_eta=0.4", "R_eta=0.5", "R_eta=0.6", "R_eta=0.7",
            "R_eta=0.8", "R_eta=0.9", "R_eta=1"]

def dataset_column_names():
    return ["n_samples", "n_features", "n_informative", "n_redundant", "n_clusters_per_class", "flip_y", "weights",
            "random_state", "best_gamma", "best_C", "best_k"]

def auc_results_exist(row):
    auc_exists = False
    for key in auc_column_names():
        if row[key] is not None:
            auc_exists = True
            break
    return auc_exists

def main():
    # fix random seed for reproducibility
    random.seed(0)

    # 1) load the file with the make_classification hyperparameters
    make_classification_result = pd.read_csv("results/make_classification/hp_opt_results.csv")

    # 2) for each unique value of gamma, select 5 rows in make_classification_result
    experiment_df = dataset_selection_for_experiment(make_classification_result)
    # initialise auc_columns
    for key in auc_column_names():
        experiment_df[key] = None

    # 3) for each row extract the dataset parameters, create the dataset, and train the model
    for row_ind in range(len(experiment_df)):
        row = experiment_df.iloc[row_ind]

        if auc_results_exist(row): continue

        dataset_params_dict = extract_make_classification_params_from_df_row(row)

        # create dataset
        dataset, data_dict = datasets.provide_dataset("sklearn_make_classification", flipping_set_size=100,
                                                      opt_arguments=dataset_params_dict)

        # pixelflipping assumes a -1, 1 binary label. remap all labels in dataset
        dataset["y_train"] = np.where(dataset["y_train"] == 0, -1, 1)
        dataset["y_test"] = np.where(dataset["y_test"] == 0, -1, 1)

        # train the model
        model = SVC(kernel='rbf', class_weight=utils.class_weight_dict(dataset["y_train"]))
        model.set_params(C=row["best_C"], gamma=row["best_gamma"])
        model.fit(dataset["X_train"], dataset["y_train"])
        score = model.score(dataset["X_test"], dataset["y_test"])

        neural_explantion_dict = neural_explanations(model, dataset)
        kde = train_kde(dataset, 0)
        results_dict = run_feature_flipping(model, kde, dataset, neural_explantion_dict)

        auc_dict = {}
        for key in results_dict.keys():
            auc_dict[key] = results_dict[key]["AUC"]

        # if experiment df has no columns like the auc_dict keys, add them
        for key in auc_dict.keys():
            if key not in experiment_df.columns:
                experiment_df[key] = None

        # add the auc values to the experiment df
        for key in auc_dict.keys():
            col_ind = experiment_df.columns.get_loc(key)
            experiment_df.iloc[row_ind, col_ind] = auc_dict[key]

        # save the experiment df
        experiment_df.to_csv("results/make_classification/eta_vs_gamma_experiment.csv", index=False)

    print("Stop")

if __name__ == "__main__":
    main()

    print("Done")




    """row = make_classification_result.iloc[row_ind]
            dataset_params_dict = extract_make_classification_params_from_df_row(row)
            svm_hps = {"gamma": gamma, "C": row["best_C"]}
            print("Stop")
            
            dataset_params_dict = extract_make_classification_params_from_df_row(row)
            print("Stop")   """



    # 4) run pixelflipping on the model and dataset, record the auc values

    # 5) save the results to a file, make sure dataset parameters are unique in this entry