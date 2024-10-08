import copy

from sklearn.linear_model import LogisticRegressionCV
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.svm import SVC, LinearSVC
from matplotlib import colors
import torch
from copy import deepcopy


from kde import KDE
import matplotlib.pyplot as plt
from neuralised_svm import neuralised_svm
from scipy.special import softmax as softmax

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import sklearn
import scipy
from sklearn.decomposition import PCA
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import pickle
import random
import os
from reduced_sets_svm import ReducedSetSVM
from pixelflipping import train_test_split, standardise_data, flipping_procedure, compute_auc, make_kde, \
    shuffle_data, svm_GI, svm_IG, resample_balanced_classes, manual_svm_gadient, \
    provide_opposite_class_kde_models, provide_kde_models, reverse_flipping_procedure, compute_reverse_auc
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
from datasets import load_concrete_dataset
from sklearn.neighbors import KNeighborsClassifier as KNN
from neuralised_knn import neuralised_knn
from scipy.spatial.distance import cdist

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans


def gaussian_kernel(x_grid, y_grid, x_data, y_data, z_data, gamma):
    # Calculate distances between all grid points and data points
    distances = cdist(np.column_stack((x_grid.flatten(), y_grid.flatten())),
                      np.column_stack((x_data, y_data)))

    # Compute Gaussian kernel weights
    kernel_weights = np.exp(- gamma * distances ** 2)

    # Normalize kernel weights for each grid point
    kernel_weights /= np.sum(kernel_weights, axis=1, keepdims=True)

    # Calculate weighted average of z values
    z_grid = np.dot(kernel_weights, z_data)

    # Reshape z_grid to match the shape of x_grid and y_grid
    z_grid = z_grid.reshape(x_grid.shape)

    return z_grid

def smooth_grid_average(x_data, y_data, z_data, gamma=1):
  # Example z values
    # Define grid points
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)

    # Generate grid points within data boundaries
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Compute the Gaussian kernel average of z values on the grid
    z_grid = gaussian_kernel(x_grid, y_grid, x_data, y_data, z_data, gamma)

    plt.figure(figsize=(10, 8))

    #cmap seismic
    #colornorm = TwoSlopeNorm(vmin=z_data.min(), vcenter=0, vmax=z_data.max())


    colornorm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    plt.contourf(x_grid, y_grid, z_grid, cmap='seismic', norm=colornorm)
    #plt.scatter(x_data, y_data, c=z_data, cmap='viridis', edgecolors='k')
    plt.colorbar(label='z values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Kernel Density Estimation')
    plt.show()

def grouped_bar_plot(list_of_ages, list_of_age_explanations, LR_age_dual):
    # create 6 bins for the ages with equal or similar number of samples in each bin
    unique_ages, inds, reverse, counts = np.unique(list_of_ages, return_index=True, return_inverse=True,
                                                   return_counts=True)

    group_tuples = [[1, 3], [7], [14], [28], [56], [90, 91], [100, 120], [180, 270, 360, 365]]
    group_unique_inds = [(0, 1), (2,), (3,), (4,), (5,), (6, 7), (8, 9), (10, 11, 12, 13)]

    # create a list of lists of indices such that each list contains the indices of the samples in the same age group
    group_reverses = []
    for group in group_unique_inds:
        group_reverses.append(np.where(np.isin(reverse, group))[0])

    # for each age group calculate the median explanation
    grouped_explanations = []
    for group in group_reverses:
        mean = np.mean(list_of_age_explanations[group])
        grouped_explanations.append(mean)

    grouped_explanations = np.array(grouped_explanations)
    grouped_age_explanations = np.array(grouped_explanations)  # [:, -2]

    positions = []
    widths = []

    for i in range(len(group_tuples)):
        start = group_tuples[i][0]
        if i < len(group_tuples) - 1:
            end = group_tuples[i + 1][0]
        else:
            end = group_tuples[i][-1] + 1  # Use the last value of the current tuple if there is no next tuple
        positions.append(start)
        widths.append(end - start)

        # Create the bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i in range(len(group_tuples)):
        if i == 0:
            ax.bar(positions[i], grouped_age_explanations[i], width=widths[i], align='edge', edgecolor='black',
                   color="red", label="KNN Explanation")

        else:
            ax.bar(positions[i], grouped_age_explanations[i], width=widths[i], align='edge', edgecolor='black',
                   color="red")

    # plot a dotted line with y = LR_age_dual * x
    x = np.linspace(0, 365, 100)
    y = LR_age_dual * (x - list_of_ages.mean())
    axes2 = plt.twinx()
    axes2.plot(x, y, linestyle='dotted', color='orange', label="LR Explanation")
    axes2.set_ylim(-1800, 1800)
    axes2.set_ylabel('Line plot')
    # Labeling the plot
    ax.set_xlabel('Ages')
    ax.set_ylabel('Average Age Relevance')
    greatest_abs_value = np.max(np.abs(grouped_age_explanations))
    ax.set_ylim(-greatest_abs_value, greatest_abs_value)
    ax.set_title('Bar Plot with Manually Set Age Groups')
    ax.grid(True)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axes2.get_legend_handles_labels()
    axes2.legend(lines + lines2, labels + labels2, loc=0)

    # Display the plot
    plt.show()

def combined_relevance_scatter(df_data, explanation_df, gamma):
    save_string = "wine/combined_relevance_scatter_gamma={}_white.png".format(gamma)
    fig, ax = plt.subplots(11, 11, figsize=(30, 30))
    for i, column in enumerate(df_data.columns):
        for j, column2 in enumerate(df_data.columns):
            cmap = plt.get_cmap('seismic')
            exp = explanation_df[column].values + explanation_df[column2].values
            #smooth_grid_average(df_data[column], df_data[column2], exp, gamma=1)
            max_abs_exp_value = max(np.abs(exp))
            norm = TwoSlopeNorm(vmin=-max_abs_exp_value, vcenter=0, vmax=max_abs_exp_value)
            #norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)

            ax[j, i].scatter(df_data[column], df_data[column2], c=exp, cmap=cmap,
                             norm=norm,
                             label="XAI_{}".format(column), s=1)
            ax[j, i].set_xlabel(column)
            ax[j, i].set_ylabel(column2)
            ax[j, i].legend()
    plt.tight_layout()
    plt.savefig(save_string)
    plt.close()

def combined_relevance_contour(df_data, explanation_df, gamma):
    save_string = "wine/combined_relevance_contour_gamma={}.png".format(gamma)
    fig, ax = plt.subplots(11, 11, figsize=(30, 30))
    for i, column in enumerate(df_data.columns):
        for j, column2 in enumerate(df_data.columns):
            exp = explanation_df[column].values + explanation_df[column2].values
            smooth_grid_average(df_data[column], df_data[column2], exp, gamma=1)
            max_abs_exp_value = max(np.abs(exp))
            norm = TwoSlopeNorm(vmin=-max_abs_exp_value, vcenter=0, vmax=max_abs_exp_value)

            x_min, x_max = np.min(df_data[column]), np.max(df_data[column])
            y_min, y_max = np.min(df_data[column2]), np.max(df_data[column2])

            # Generate grid points within data boundaries
            x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

            # Compute the Gaussian kernel average of z values on the grid
            z_grid = gaussian_kernel(x_grid, y_grid, df_data[column], df_data[column2], exp, gamma)

            ax[j, i].contourf(x_grid, y_grid, z_grid, cmap="seismic", norm=norm)
            ax[j, i].set_xlabel(column)
            ax[j, i].set_ylabel(column2)
            ax[j, i].legend()
    plt.tight_layout()
    plt.savefig(save_string)
    plt.close()


def folder_train_explain(X, y, gamma, C, X_optional=None, y_optional=None):
    """
    Partition the data into folds. Always explain one fold and train on the rest.
    Gather the explanations and return them.

    :param X:
    :param y:
    :return:
    """
    explained_cluster_data_list = []
    cluster_explanation_list = []
    kf = KFold(n_splits=5)  # Number of folds based on the dataset size
    fold_index = 0
    for train_index, explain_index in kf.split(X):
        fold_index += 1
        X_train, X_explain = X[train_index], X[explain_index]
        y_train, y_explain = y[train_index], y[explain_index]

        if X_optional is not None and y_optional is not None:
            # concatenate X_train and X_optional
            X_train = np.concatenate((X_train, X_optional), axis=0)
            y_train = np.concatenate((y_train, y_optional), axis=0)

        # Model training
        svc = SVC(kernel='rbf', class_weight={0.0: 0.6, 1.0: 0.4}, C=C, gamma=gamma)
        svc.fit(X_train, y_train)
        neural_svm = neuralised_svm(svc)

        # Explanation
        eta = min(0.3, max(0, np.log10(gamma) * 0.3 + 0.3))
        R_eta = neural_svm.explain(X_explain, first_rule="hybrid", eta=eta, reweight_explanation=False)
        R_eta = R_eta / np.sum(np.abs(R_eta), axis=1, keepdims=True)

        # Append explained data and explanations to lists
        explained_cluster_data_list.append(X_explain)
        cluster_explanation_list.append(R_eta)

    explained_cluster_data = np.concatenate(explained_cluster_data_list, axis=0)
    cluster_explanation = np.concatenate(cluster_explanation_list, axis=0)
    return explained_cluster_data, cluster_explanation


if __name__ == "__main__":
    # fix seed
    np.random.seed(0)
    #matplotlib.use('Agg')

    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features.values
    y = wine_quality.data.targets.values

    # only keep samples with color white
    X = X[np.where(wine_quality.data.original["color"].values=="white")]
    y = y[np.where(wine_quality.data.original["color"].values=="white")]

    # if larger then 6 set to 1, else set to -1
    y = np.where(y > 5, 1, 0)

    # data (as pandas dataframes)
    df_X = wine_quality.data.features.iloc[np.where(wine_quality.data.original["color"].values=="white")]
    # drop features index 5 and 6
    df_y = wine_quality.data.targets.iloc[np.where(wine_quality.data.original["color"].values=="white")]

    # shuuffle X, y, df_X, df_y
    shuffle_inds = np.random.permutation(len(y))
    X = X[shuffle_inds]
    y = y[shuffle_inds]
    df_X = df_X.iloc[shuffle_inds]
    df_y = df_y.iloc[shuffle_inds]
    df = pd.concat([df_X, df_y], axis=1)
    X = df_X.values

    #y = df_y.values
    X_standardised = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # summarise statistics of df_X
    df_X.describe()

    D = sklearn.metrics.pairwise.euclidean_distances(X_standardised, X_standardised) ** 2
    # compute median of D
    median = np.median(D)
    X_standardised = X_standardised / median ** (1 / 2)

    # set strongest 10 percent to 1, rest to 0
    y_binary = y
        #combined_relevance_scatter(df_explain, explanation_df, gamma)
    """best_accs = []
    gamma_list = [0.001, 0.1, 1, 3, 10, 30, 100]
    C_list = [1e-4, 1e-2, 1e-1, 1e1, 1e3]
    best_param_comb_list = []
    for gamma in gamma_list:
        X_train, X_test = X_standardised, X_standardised
        y_train, y_test = y_binary, y_binary
        # do grid search for best C
        acc_C_list = []
        for C in C_list:
            svc = SVC(kernel='rbf', gamma=gamma, C=C, class_weight={0.0: 0.6, 1.0: 0.4})
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)
            acc = sklearn.metrics.accuracy_score(y_test, y_pred)
            acc_C_list.append(acc)
        best_C = C_list[np.argmax(acc_C_list)]
        best_param_comb_list.append([gamma, best_C])
        best_acc = max(acc_C_list)
        best_accs.append(best_acc)"""

    best_param_comb_list = [[0.001, 1000.0], [0.1, 1000.0], [1, 1000.0], [3, 1000.0], [10, 1000.0], [30, 1000.0], [100, 1000.0]]

    n_neighbours = 20
    # perform k-means clustering of X_standardised creating 5 clusters
    kmeans = KMeans(n_clusters=n_neighbours, random_state=0).fit(X_standardised)
    cluster_labels = kmeans.labels_

    """# make list of data arrays containing the data points of each cluster
    cluster_data_list = []
    cluster_target_list = []
    for i in range(n_neighbours):
        cluster_data_list.append(X_standardised[np.where(cluster_labels == i)])
        cluster_target_list.append(y_binary[np.where(cluster_labels == i)])"""

    """dfs_explain = []
    for k, X_cluster in enumerate(cluster_data_list):"""
        """y_cluster = cluster_target_list[k]"""
    for best_params in best_param_comb_list:
        # for k in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]:
        gamma = best_params[0]
        C = best_params[1]

        # X_optional are the data points that are not in the cluster
        X_optional = np.concatenate([X_standardised[np.where(cluster_labels != k)]])
        y_optional = np.concatenate([y_binary[np.where(cluster_labels != k)]])

        explained_cluster_data, cluster_explanation = \
            folder_train_explain(X_cluster, y_cluster, gamma, C, X_optional,y_optional)

        cluster_explanation_df = pd.DataFrame(data=cluster_explanation, columns=df_X.columns)
        cluster_explanation_df.to_csv("wine/explanation_df_{}_white_cluster_{}.csv".format(gamma, k), index=False)

        explained_cluster_data_df = pd.DataFrame(data=explained_cluster_data, columns=df_X.columns)
        explained_cluster_data_df.to_csv("wine/explained_df_white_cluster_{}.csv".format(k), index=False)

    explanation_dfs = [pd.read_csv("wine/explanation_df_{}_white_cluster_{}.csv".format(1, k)) for k in range(n_neighbours)]
    dfs_explain = [pd.read_csv("wine/explained_df_white_cluster_{}.csv".format(k)) for k in range(n_neighbours)]
    fig, ax = plt.subplots(3, 11, figsize=(30, 15))
    for k, explanation_df in enumerate(explanation_dfs[:3]):
        df_explain = dfs_explain[k]
        cmap = plt.get_cmap('seismic')
        norm = TwoSlopeNorm(vmin=-.8, vcenter=0, vmax=.8)
        # make scatter plots of df_explain values for each column with explanation_df values for each columns
        for i, column in enumerate(df_explain.columns):
            ax[k, i].scatter(df_explain[column], explanation_df[column], alpha=1, c=explanation_df[column], cmap=cmap, norm=norm, s=0.1)
            ax[k, i].set_title(column)
            # turn off x and y labels
            ax[k, i].set_yticklabels([])
            ax[k, i].set_xticklabels([])

            # vertical line at x = 0
            ax[k, i].axvline(x=0, color='black', linestyle='--')

            # show x only from -1 to 1
            ax[k, i].set_xlim(-1, 1)
    plt.tight_layout()
    plt.show()
    print("Done")

    for gamma in gamma_list:
        explanation_df = pd.read_csv("wine/explanation_df_{}_white.csv".format(gamma))
        combined_relevance_scatter(df_explain, explanation_df, gamma)
    print("Done")


