import numpy as np
from sklearn.svm import SVC, LinearSVC

import torch
from copy import deepcopy

from kde import KDE
from neuralised_svm import neuralised_svm
from neuralised_knn import neuralised_knn

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN

from typing import Literal, List

import pandas as pd
import pickle
import random
import os
import svm_explanations


from datasets import provide_dataset


def train_svm(dataset: dict, data_dict):
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    estimator = SVC(kernel="rbf", class_weight=data_dict["class_weight_dicts"])

    param_grid = dict(gamma=[1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100], C=[1e-4, 1e-2, 1e0, 1e2, 1e4])

    clf = GridSearchCV(estimator, param_grid)
    clf.fit(X_train, y_train)

    best_gamma = clf.best_params_["gamma"]
    best_C = clf.best_params_["C"]
    best_score = clf.best_score_
    best_svc = SVC(kernel="rbf", gamma=best_gamma, C=best_C, class_weight=data_dict["class_weight_dicts"])
    best_svc.fit(X_train, y_train)
    score = best_svc.score(X_test, y_test)

    data_dict["best_gamma"] = best_gamma
    data_dict["best_C"] = best_C
    data_dict["best_score"] = score
    data_dict["intercept"] = best_svc.intercept_
    origin_prediction = best_svc.decision_function(np.zeros_like(X_test[:1]))
    origin_prediction_without_bias = origin_prediction - best_svc.intercept_
    data_dict["origin_prediction"] = origin_prediction
    data_dict["origin_prediction_without_bias"] = origin_prediction_without_bias

    sup_ind = best_svc.support_
    alpha = best_svc.dual_coef_
    K = sklearn.metrics.pairwise.rbf_kernel(X_train, X_train, gamma=best_gamma)
    K = K[sup_ind][:, sup_ind]
    sq_norm = np.abs(alpha) @ K @ np.abs(alpha.T)
    width = 1 / sq_norm ** .5
    data_dict["width"] = width

    return best_svc, data_dict


def train_knn(dataset: dict, data_dict: dict):
    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]
    knn = KNN()
    param_grid = dict(n_neighbors=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train.ravel())
    best_n_neighbors = grid.best_params_['n_neighbors']
    data_dict["best_n_neighbors"] = best_n_neighbors

    best_knn = KNN(n_neighbors=best_n_neighbors)
    best_knn.fit(X_train, y_train.ravel())
    best_score = best_knn.score(X_test, y_test.ravel())
    data_dict["best_knn_score"] = best_score

    # get average distance to neighbors of each data point
    neighbour_dists, neighbour_inds = best_knn.kneighbors(X_test, return_distance=True)
    avg_neighbour_dists = neighbour_dists.sum(1).mean(axis=0)
    data_dict["avg_neighbour_dists"] = avg_neighbour_dists

    return best_knn, data_dict


def train_lin_svm(dataset: dict, data_dict: dict):
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    lin_svm = SVC(kernel="linear")
    param_grid = dict(C=[1e-4, 1e-2, 1e0, 1e2])
    clf = GridSearchCV(lin_svm, param_grid)
    clf.fit(X_train, y_train)
    best_lin_C = clf.best_params_["C"]
    best_lin_svm = SVC(kernel="linear", C=best_lin_C)
    best_lin_svm.fit(X_train, y_train)

    best_lin_score = best_lin_svm.score(X_test, y_test)

    data_dict["best_lin_score"] = best_lin_score

    return lin_svm, data_dict


def explanations_for_svms(svm, neural_svm, dataset: dict):
    X_flipping = dataset["X_flipping"]
    X_train = dataset["X_train"]

    explanation_dict = {}
    explanation_dict["GI"] = svm_explanations.gradient_times_input(svm, X_flipping)
    explanation_dict["IG"] = svm_explanations.integrated_gradients(svm, X_flipping)
    grads = svm_explanations.svm_gadient(svm, X_flipping)
    explanation_dict["grads"] = grads
    sensitivities = grads ** 2
    explanation_dict["sensitivities"] = sensitivities * svm.predict(X_flipping)[:, None]
    R_var = X_train.var(0)[None, :] * np.ones_like(X_flipping) * svm.predict(X_flipping)[:, None]
    explanation_dict["R_var"] = R_var

    explanation_dict["neural_IG"] = neural_svm.integrated_gradients(X_flipping)
    explanation_dict["random"] = np.random.standard_normal(X_flipping.shape)

    explanation_dict["R_centroid"] = svm_explanations.centroid_diff(svm, X_flipping, X_train)

    eta_values = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    neural_svn = neuralised_svm(svm)
    for eta in eta_values:
        R_eta = neural_svn.explain(X_flipping, first_rule="hybrid", eta=eta)
        explanation_dict["R_eta={}".format(eta)] = R_eta

    explanation_dict["R_occ"] = svm_explanations.occlusion(svm, X_flipping)



    return explanation_dict


def explanations_for_knn(knn, neural_knn, dataset):
    X_flipping = dataset["X_flipping"]
    X_train = dataset["X_train"]

    explanation_dict = {}
    R_var = X_train.var(0)[None, :] * np.ones_like(X_flipping) * knn.predict(X_flipping)[:, None]
    explanation_dict["R_var"] = R_var

    explanation_dict["neural_IG"] = neural_knn.integrated_gradients(X_flipping)
    explanation_dict["random"] = np.random.standard_normal(X_flipping.shape)
    explanation_dict["Occ"] = svm_explanations.occlusion(knn, X_flipping)

    #kappa_values = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    """kappa_values = [0, 3, 7, 11, 17, 21, 49]
    for kappa in kappa_values:
        R_kappa = neural_knn.explain(X_flipping, first_rule="hybrid", eta=.5, kappa=kappa)
        explanation_dict["R_kappa={}".format(kappa)] = R_kappa"""
    eta_values = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    for eta in eta_values:
        R_eta = neural_knn.explain(X_flipping, first_rule="hybrid", eta=eta)
        explanation_dict["R_eta={}".format(eta)] = R_eta

    explanation_dict["R_centroid"] = svm_explanations.centroid_diff(knn, X_flipping, X_train)

    return explanation_dict


def gather_explanations(model, neural_model, dataset: dict, model_type: str):
    if model_type == "SVM":
        return explanations_for_svms(model, neural_model, dataset)
    elif model_type == "KNN":
        return explanations_for_knn(model, neural_model, dataset)
    else:
        raise NotImplementedError


def train_kde(dataset: dict, adjustment_gamma: float = 0):
    X_train = dataset["X_train"]
    if len(X_train)<5000:
        X_kde = X_train
    else:
        X_kde = X_train
        random.shuffle(X_kde)
        X_kde = X_kde[:5000]
    kde_model = KDE(adjustment_gamma=adjustment_gamma)
    kde_model.fit(X_kde)
    return kde_model


def compute_flipping_curves(model, kde_model, dataset: dict, explanation_list: List[np.array], flip_interval: int):
    flipping_curves = []
    X_flipping, y_flipping = dataset["X_flipping"], dataset["y_flipping"]
    for i in range(len(X_flipping)):
        sample = X_flipping[i:i + 1]
        #original_prediction = svc.forward(sample).item()
        original_prediction = model.predict(sample).item()

        explanation = explanation_list[i]
        # get indices in the order of the explanation values starting with highest relevance
        flipping_order = np.argsort(explanation)[::-1]

        # if the original prediction is negative, we want to flip the order
        explained_sign = 1 if original_prediction > 0 else -1
        if explained_sign == -1:
            flipping_order = flipping_order[::-1]

        working_sample = deepcopy(sample)[0]
        working_mask = torch.ones(sample.shape[1]).bool()
        # start perturbation loop
        sample_flipping_curve_list = [explained_sign * model.predict(sample).item()]
        for i, ind in enumerate(flipping_order):
            working_mask[ind] = False
            if i % flip_interval == 0:
                resamples = kde_model.conditional_sample(
                    working_sample, working_mask, n_samples=50).detach().cpu().numpy()
                resample_preds = explained_sign * model.predict(resamples)
                sample_flipping_curve_list.append(resample_preds.mean())
        flipping_curves.append(sample_flipping_curve_list)
    flipping_curves = np.array(flipping_curves)
    return flipping_curves


def compute_auc(flipping_curve):
    return np.trapz(flipping_curve) / np.trapz(np.ones_like(flipping_curve) * flipping_curve[0])


def run_feature_flipping(model, kde_model, dataset, XAI_dict):
    # infer the flipping interval
    n_d = dataset["X_train"].shape[1]
    if n_d > 10:
        flip_interval = 2
    if n_d > 50:
        flip_interval = 5
    if n_d <= 10:
        flip_interval = 1

    results_dict = {}
    for explanation_key, explanations in XAI_dict.items():
        explanation_result_dict = {}

        curves = compute_flipping_curves(model, kde_model, dataset, explanations, flip_interval)
        auc_list = [compute_auc(curve) for curve in curves]
        # auc = compute_reverse_auc(mean_curve)
        mean_auc = np.mean(auc_list)
        explanation_result_dict["AUC"] = mean_auc
        explanation_result_dict["curves"] = curves
        explanation_result_dict["R"] = explanations
        results_dict[explanation_key] = explanation_result_dict
    return results_dict


def provide_model(dataset: dict, data_dict: dict, model_type : str):
    if model_type == "SVM":
        return train_svm(dataset, data_dict)
    elif model_type == "KNN":
        return train_knn(dataset, data_dict)
    else :
        raise NotImplementedError


def neuralise_model(model, model_type: str):

    if model_type == "SVM":
        return neuralised_svm(model)
    elif model_type == "KNN":
        return neuralised_knn(model)
    else :
        raise NotImplementedError

def save_results(save_dict, dataset_name, model_type, local_adjustment_gamma: float = 0.0):

    if local_adjustment_gamma == 0.0:
        global_local_folder = "global"
    else:
        global_local_folder = "local_{}".format(local_adjustment_gamma)

    save_string = "pf_results/{}/{}/{}.pkl".format(global_local_folder, model_type, dataset_name)
    if not os.path.exists("pf_results/{}/{}".format(global_local_folder, model_type)):
        os.makedirs("pf_results/{}/{}".format(global_local_folder, model_type))
    with open(save_string, "wb") as f:
        pickle.dump(save_dict, f)

def main(dataset_names: List[str], model_type: str, adjustment_gamma: float = 0.0):
    for dataset_name in dataset_names:
        # load data
        dataset, data_dict = provide_dataset(dataset_name, flipping_set_size=100)

        # train models
        model, data_dict = provide_model(dataset, data_dict, model_type)
        lin_svm, data_dict = train_lin_svm(dataset, data_dict)

        # neuralised model
        neural_model = neuralise_model(model, model_type)

        # compute all explanations
        explanation_dict = gather_explanations(model, neural_model, dataset, model_type)

        kde = train_kde(dataset, adjustment_gamma)

        results_dict = run_feature_flipping(model, kde, dataset, explanation_dict)

        save_results({"data_dict": data_dict, "results_dict": results_dict}, dataset_name, model_type, local_adjustment_gamma)

        print("Dataset {} done".format(dataset_name))
    print("Done")

if __name__ == "__main__":
    dataset_names = [#"wine_quality", "concrete", "breast_cancer", "rice", "raisin",
                     "diabetes", "car_evaluation", "real_estate", "catalyst"]
    model_type = "KNN"
    local_adjustment_gamma = 10
    main(dataset_names, model_type, local_adjustment_gamma)