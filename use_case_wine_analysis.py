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


def find_best_Cs(gamma_list):
    best_accs = []

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
        best_accs.append(best_acc)
    return best_param_comb_list

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

def combined_relevance_scatter(df_data, explanation_df, gamma, type):
    save_string = "wine/combined_relevance_scatter_gamma={}_{}.png".format(gamma, type)
    fig, ax = plt.subplots(11, 11, figsize=(30, 30))
    for i, column in enumerate(df_data.columns):
        for j, column2 in enumerate(df_data.columns):
            cmap = plt.get_cmap('seismic')
            exp = explanation_df[column].values# + explanation_df[column2].values
            #smooth_grid_average(df_data[column], df_data[column2], exp, gamma=1)
            max_abs_exp_value = max(np.abs(exp))
            percentile_1 = np.percentile(exp, 5)
            percentile_99 = np.percentile(exp, 95)
            norm = TwoSlopeNorm(vmin=percentile_1, vcenter=0, vmax=percentile_99)
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


def folder_train_explain(X, y, gamma, C, X_optional=None, y_optional=None, eta=None, n_splits=5, normal_type="l1"):
    """
    Partition the data into folds. Always explain one fold and train on the rest.
    Gather the explanations and return them.

    :param X:
    :param y:
    :return:
    """
    explained_data_list = []
    explanation_list = []
    kf = KFold(n_splits=n_splits)  # Number of folds based on the dataset size
    fold_index = 0
    explained_indices = []
    for train_index, explain_index in kf.split(X):
        fold_index += 1
        X_train, X_explain = X[train_index], X[explain_index]
        y_train, y_explain = y[train_index], y[explain_index]
        explained_indices.append(explain_index)

        if X_optional is not None and y_optional is not None:
            # concatenate X_train and X_optional
            X_train = np.concatenate((X_train, X_optional), axis=0)
            y_train = np.concatenate((y_train, y_optional), axis=0)

        # Model training
        svc = SVC(kernel='rbf', class_weight={0.0: 0.6, 1.0: 0.4}, C=C, gamma=gamma)
        svc.fit(X_train, y_train)
        neural_svm = neuralised_svm(svc)

        # Explanation
        if eta is None:
            eta = min(0.3, max(0, np.log10(gamma) * 0.3 + 0.3))
        R_eta = neural_svm.explain(X_explain, first_rule="hybrid", eta=eta, reweight_explanation=False)
        #R_eta = R_eta / np.sum(np.abs(R_eta), axis=1, keepdims=True)
        neural_pred = neural_svm.forward(X_explain)
        R_eta = normalise_explanation(R_eta, preds=neural_pred, const = 1e-10, type=normal_type)

        #R_eta = R_eta / np.abs(np.sum(R_eta, axis=1, keepdims=True))

        # Append explained data and explanations to lists
        explained_data_list.append(X_explain)
        explanation_list.append(R_eta)

    explained_data = np.concatenate(explained_data_list, axis=0)
    explanation = np.concatenate(explanation_list, axis=0)
    explained_inds = np.concatenate(explained_indices, axis=0)
    return explained_data, explanation, explained_inds

def get_data(type="white"):
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features.values
    y = wine_quality.data.targets.values
    units = \
        {"fixed_acidity": "g/dm³", "volatile_acidity": "g/dm³", "citric_acid": "g/dm³", "residual_sugar": "g/dm³",
            "chlorides": "g/dm³", "free_sulfur_dioxide": "mg/dm³", "total_sulfur_dioxide": "mg/dm³", "density": "g/cm³",
            "pH": "1", "sulphates": "g/dm³", "alcohol": "%vol"}
    # only keep samples with color white
    X = X[np.where(wine_quality.data.original["color"].values == type)]
    y = y[np.where(wine_quality.data.original["color"].values == type)]

    # if larger then 6 set to 1, else set to -1
    y = np.where(y > 5, 1, 0)

    # data (as pandas dataframes)
    df_X = wine_quality.data.features.iloc[np.where(wine_quality.data.original["color"].values == type)]
    # drop features index 5 and 6
    df_y = wine_quality.data.targets.iloc[np.where(wine_quality.data.original["color"].values == type)]

    # shuuffle X, y, df_X, df_y
    shuffle_inds = np.random.permutation(len(y))
    X = X[shuffle_inds]
    y = y[shuffle_inds]
    df_X = df_X.iloc[shuffle_inds]
    df_y = df_y.iloc[shuffle_inds]
    df = pd.concat([df_X, df_y], axis=1)
    X = df_X.values

    # y = df_y.values
    X_standardised = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # summarise statistics of df_X
    df_X.describe()

    D = sklearn.metrics.pairwise.euclidean_distances(X_standardised, X_standardised) ** 2
    # compute median of D
    median = np.median(D)
    X_standardised = X_standardised / median ** (1 / 2)

    y_binary = y

    return X_standardised, y_binary, df_X, units

def fill_fig_rows(df_explain, explanation_df, ax, row_ind):
    cmap = plt.get_cmap('seismic')
    norm = TwoSlopeNorm(vmin=-.8, vcenter=0, vmax=.8)
    # make scatter plots of df_explain values for each column with explanation_df values for each columns
    for i, column in enumerate(df_explain.columns):
        ax[row_ind, i].scatter(df_explain[column], explanation_df[column], alpha=1, color="black", s=0.1)
        # turn off x and y labels
        ax[row_ind, i].set_yticklabels([])
        ax[row_ind, i].set_xticklabels([])

        # remove tick
        ax[row_ind, i].set_yticks([])
        ax[row_ind, i].set_xticks([])

        # vertical line at x = 0
        #ax[row_ind, i].axvline(x=0, color='black', linestyle='--')

        # show x only from -1 to 1
        ax[row_ind, i].set_xlim(-1, 1)

def low_ph_scatter(df_explain, explanation_df, df_explain_unstandardised, units, type):
    ind_below_zero_ph = np.where(df_explain["pH"] < -0.2)[0]
    #combined_relevance_scatter(df_explain.iloc[ind_below_zero_ph], explanation_df.iloc[ind_below_zero_ph], 0.1, type)

    df_explain_low_ph = df_explain.loc[ind_below_zero_ph]
    df_explain_unstandardised_low_ph = df_explain_unstandardised.iloc[ind_below_zero_ph]
    explanation_df_low_ph = explanation_df.loc[ind_below_zero_ph]
    ph_explanations = explanation_df_low_ph["pH"].values
    top_99_percentile = np.percentile(ph_explanations, 99)
    low_1_percentile = np.percentile(ph_explanations, 1)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    exp = explanation_df_low_ph["pH"].values
    cmap = plt.get_cmap('seismic')
    norm = TwoSlopeNorm(vmin=low_1_percentile, vcenter=0, vmax=top_99_percentile)
    ax[0].scatter(df_explain_unstandardised_low_ph["residual_sugar"], df_explain_unstandardised_low_ph["pH"], c=exp, cmap=cmap,
                  norm=norm,
                  label="XAI_{}".format("pH"), s=3)
    ax[0].set_ylabel("pH")
    ax[0].set_xlabel("Residual Sugar [g/dm³]")
    ax[1].scatter(df_explain_unstandardised_low_ph["alcohol"], df_explain_unstandardised_low_ph["pH"], c=exp, cmap=cmap,
                  norm=norm,
                  label="XAI_{}".format("pH"), s=3)
    ax[1].set_ylabel("pH")
    ax[1].set_xlabel("Alcohol [%vol]")
    plt.tight_layout()
    plt.savefig("wine/low_ph_relevance_scatter_{}.png".format(type))
    return

def low_sugar_scatter(df_explain, explanation_df, df_explain_unstandardised, units, type):
    ind_below_zero_sugar = np.where(df_explain["residual_sugar"] < -0.21)[0]
    #combined_relevance_scatter(df_explain.iloc[ind_below_zero_sugar], explanation_df.iloc[ind_below_zero_sugar], 0.1, type)

    df_explain_low_sugar = df_explain.loc[ind_below_zero_sugar]
    df_explain_unstandardised_low_sugar = df_explain_unstandardised.iloc[ind_below_zero_sugar]
    explanation_df_low_sugar = explanation_df.loc[ind_below_zero_sugar]
    sugar_explanations = explanation_df_low_sugar["residual_sugar"].values
    top_99_percentile = np.percentile(sugar_explanations, 99)
    low_1_percentile = np.percentile(sugar_explanations, 1)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    exp = explanation_df_low_sugar["residual_sugar"].values
    cmap = plt.get_cmap('seismic')
    norm = TwoSlopeNorm(vmin=low_1_percentile, vcenter=0, vmax=top_99_percentile)
    ax[0].scatter(df_explain_unstandardised_low_sugar["chlorides"], df_explain_unstandardised_low_sugar["residual_sugar"], c=exp, cmap=cmap,
                  norm=norm,
                  label="XAI_{}".format("residual_sugar"), s=3)
    ax[0].set_ylabel("Residual Sugar [g/dm³]")
    ax[0].set_xlabel("chlorides")
    ax[1].scatter(df_explain_unstandardised_low_sugar["alcohol"], df_explain_unstandardised_low_sugar["residual_sugar"], c=exp, cmap=cmap,
                  norm=norm,
                  label="XAI_{}".format("residual_sugar"), s=3)
    ax[1].set_ylabel("Residual Sugar [g/dm³]")
    ax[1].set_xlabel("alcohol [%vol]")
    plt.tight_layout()
    plt.savefig("wine/low_sugar_relevance_scatter_{}.png".format(type))
    return


def normalise_explanation(R, preds=None, const=1e-3, type="l1"):

    if type == "l1":
        R = R / np.sum(np.abs(R), axis=1, keepdims=True)
    if type == "pred_normalised":
        R = R / (np.abs(preds)[:, None]+const)
    return R

def zoomed_in_sugar_chloride_scatter_plots(df_explain_unstandardised, explanation_df):
    residual_sugar_data = df_explain_unstandardised["residual_sugar"]
    residual_sugar_relevance = explanation_df["residual_sugar"]
    perc_99 = np.percentile(residual_sugar_relevance, 99)
    perc_1 = np.percentile(residual_sugar_relevance, 1)
    high_chloride_inds = df_explain_unstandardised["chlorides"].values > 0.10
    residual_sugar_low_chloride = residual_sugar_data[~high_chloride_inds]
    residual_sugar_low_chloride_exp = residual_sugar_relevance[~high_chloride_inds]
    residual_sugar_high_chloride = residual_sugar_data[high_chloride_inds]
    residual_sugar_high_chloride_exp = residual_sugar_relevance[high_chloride_inds]
    # norm = TwoSlopeNorm(vmin=perc_5, vcenter=0, vmax=perc_95)
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 2.5))
    ax.scatter(residual_sugar_low_chloride, residual_sugar_low_chloride_exp, s=0.3, color='teal')
    ax.set_ylim(perc_1, perc_99)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 10, 20])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=.5)
    ax.set_title("Chlorides < 10g/L", fontsize=8)
    ax.set_xlabel("Residual sugar", fontsize=8)
    ax.set_ylabel("Relevance", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("wine/sugar_relevance_scatter_low_chloride.png", bbox_inches="tight", dpi=600)
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 2.5))
    ax.scatter(residual_sugar_high_chloride, residual_sugar_high_chloride_exp, s=0.3, color='orange')
    ax.set_ylim(perc_1, perc_99)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 10, 20])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=.5)
    ax.set_title("Chlorides > 10g/L", fontsize=8)
    ax.set_xlabel("Residual sugar", fontsize=8)
    ax.set_ylabel("Relevance", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("wine/sugar_relevance_scatter_high_chloride.png", bbox_inches="tight", dpi=600)

def zoomed_in_ph_sugar_scatter_plots(df_explain_unstandardised, explanation_df):
    ph_data = df_explain_unstandardised["pH"]
    ph_relevance = explanation_df["pH"]
    perc_99 = np.percentile(ph_relevance, 99)
    perc_1 = np.percentile(ph_relevance, 1)
    high_residual_sugar_inds = df_explain_unstandardised["residual_sugar"].values > 10
    ph_low_sugar = ph_data[~high_residual_sugar_inds]
    ph_low_sugar_exp = ph_relevance[~high_residual_sugar_inds]
    ph_high_sugar = ph_data[high_residual_sugar_inds]
    ph_high_sugar_exp = ph_relevance[high_residual_sugar_inds]
    # norm = TwoSlopeNorm(vmin=perc_5, vcenter=0, vmax=perc_95)
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 2.5))
    ax.scatter(ph_low_sugar, ph_low_sugar_exp, s=0.3, color='teal')
    ax.set_ylim(perc_1, perc_99)
    ax.set_xlim(2.5, 4)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=.5)
    ax.set_title("Residual sugar < 10 g/L", fontsize=8)
    ax.set_xlabel("pH", fontsize=8)
    # disable y ticks
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticks([2.5, 3, 3.5, 4])
    ax.set_ylabel("Relevance", fontsize=8)
    plt.tight_layout()
    plt.savefig("wine/ph_relevance_scatter_low_sugar.png", bbox_inches="tight", dpi=600)
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 2.5))
    ax.scatter(ph_high_sugar, ph_high_sugar_exp, s=0.3, color='orange')
    ax.set_ylim(perc_1, perc_99)
    ax.set_xlim(2.5, 4)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=.5)
    ax.set_title("Residual sugar > 10 g/L", fontsize=8)
    ax.set_xlabel("pH", fontsize=8)
    # disable y ticks
    ax.set_yticks([])
    ax.set_ylabel("Relevance", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticks([2.5, 3, 3.5, 4])
    plt.tight_layout()
    plt.savefig("wine/ph_relevance_scatter_high_sugar.png", bbox_inches="tight", dpi=600)

def feature_relevance_scatter_plots(df_explain_unstandardised, type, normal_type):
    fig, ax = plt.subplots(2, 11, figsize=(9.5, 3.5))
    explanation_df_linear = pd.read_csv(
        "wine/explanation_df_gamma_{}_eta_{}_{}_{}.csv".format(0.000001, 0, type, normal_type))
    fill_fig_rows(df_explain_unstandardised, explanation_df_linear, ax, 0)
    ax[0, 0].set_ylabel("Relevance \n $\gamma = 1e-5$, $\eta = 0$", fontsize=7)
    explanation_df = pd.read_csv(
        "wine/explanation_df_gamma_{}_eta_{}_{}_{}.csv".format(0.1, 0.05, type, normal_type))
    fill_fig_rows(df_explain_unstandardised, explanation_df, ax, 1)
    ax[1, 0].set_ylabel("Relevance \n $\gamma = 0.1$,  $\eta = 0.05$", fontsize=7)
    # iterate over all columns
    # ax[1, 0] should display the x-axis ticks and values
    # an xlim should be set from the 10% to 90% quantile
    # ticks should be set at 25%, 50% and 75% quantiles
    for i, column in enumerate(df_explain.columns):
        # standard deviation of the column
        std = np.std(df_explain_unstandardised[column])
        min = np.min(df_explain_unstandardised[column])
        mean = np.mean(df_explain_unstandardised[column])
        perc_1 = np.percentile(df_explain_unstandardised[column], 0.5)
        perc_10 = np.percentile(df_explain_unstandardised[column], 10)
        perc_90 = np.percentile(df_explain_unstandardised[column], 90)
        perc_50 = np.percentile(df_explain_unstandardised[column], 50)
        perc_99 = np.percentile(df_explain_unstandardised[column], 99.5)
        # thin vertical line at mean
        ax[0, i].axvline(x=mean, color='black', linestyle='--', linewidth=0.5)
        min_lim = perc_1 - .5 * std
        max_lim = perc_99 + .5 * std
        ax[0, i].set_xlim(perc_1 - .5 * std, perc_99 + .5 * std)
        ax[1, i].set_xlim(min_lim, max_lim)
        first_tick = min_lim + 0.25 * (max_lim - min_lim)
        second_tick = min_lim + 0.75 * (max_lim - min_lim)
        ax[1, i].set_xticks([first_tick, second_tick])
        # Calculate mean and symmetrical range around zero
        mean_exp = np.mean(explanation_df_linear.values)
        max_offset = np.percentile(np.abs(explanation_df_linear.values - mean_exp), 99)
        # Set y-limits to center zero
        ax[0, i].set_ylim(-max_offset, max_offset)
        mean_exp = np.mean(explanation_df.values)
        max_offset = np.percentile(np.abs(explanation_df.values - mean_exp), 99)
        # Set y-limits to center zero
        ax[1, i].set_ylim(-max_offset, max_offset)
        if column == "density" or column == "chlorides":
            ax[1, i].set_xticklabels([round(first_tick, 2), round(second_tick, 2)], fontsize=7)
        else:
            ax[1, i].set_xticklabels([round(first_tick, 1), round(second_tick, 1)], fontsize=7)
        # thin vertical line at mean
        ax[1, i].axvline(x=mean, color='black', linestyle='--', linewidth=.5)
        ax[0, i].axhline(y=0, color='black', linestyle='--', linewidth=.5)
        ax[1, i].axhline(y=0, color='black', linestyle='--', linewidth=.5)
        ax[0, i].set_title(columns_strings[i], fontsize=7)
        # vertical line at x = 0
        # ax[1, i].axvline(x=0, color='black', linestyle='--')
        # show x only from -1 to 1
        # ax[1, i].set_xlim(perc_005, perc_995)
    # plt.tight_layout()
    plt.savefig("wine/input_relevance_scatter_{}_{}.png".format(type, normal_type), bbox_inches="tight", dpi=600)
    print("Done")

if __name__ == "__main__":
    # fix seed
    np.random.seed(0)
    #matplotlib.use('Agg')

    type = "white"
    X_standardised, y_binary, df_X, units = get_data(type=type)

    normal_type = "none"

    gamma_list = [0.000001, 0.1]
        #combined_relevance_scatter(df_explain, explanation_df, gamma)

    best_param_comb_list = find_best_Cs(gamma_list)
    #best_param_comb_list = [[0.001, 1000.0], [0.1, 1000.0], [1, 1000.0], [3, 1000.0], [10, 1000.0], [30, 1000.0], [100, 1000.0]]
    eta_list = [0, 0.05]
    for eta in eta_list:
        for best_params in best_param_comb_list:
            # for k in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]:
            gamma = best_params[0]
            C = best_params[1]

            explained_data, explanation, explained_inds = \
                folder_train_explain(X_standardised, y_binary, gamma, C, eta=eta, n_splits=20, normal_type=normal_type)

            explained_data_df = pd.DataFrame(data=explained_data, columns=df_X.columns)
            explained_data_df.to_csv("wine/explained_df_{}.csv".format(type), index=False)

            explanation_df = pd.DataFrame(data=explanation, columns=df_X.columns)
            explanation_df.to_csv("wine/explanation_df_gamma_{}_eta_{}_{}_{}.csv".format(gamma, eta, type, normal_type), index=False)


    df_explain = explained_data_df
    df_explain_unstandardised = df_X.iloc[explained_inds]

    columns_strings = []
    # iterate over all columns, replace _ with space and add to list
    # if column is longer than 15 characters, replace first _ with \n
    for column in df_explain.columns:
        unit = units[column]
        #if len(column) > 10:
            #column = column.replace("_", "\n", 1)
        column = column.replace("_", "\n")
        # capitalize first letter of each word
        column = column.title()

        # except for Ph append unit in square brackets
        if not column == "Ph":
            column = column + " \n[" + unit + "]"
        if column == "Ph":
            column = "PH" + " \n[1]"

        columns_strings.append(column)


    feature_relevance_scatter_plots(df_explain_unstandardised, type, normal_type)

    zoomed_in_ph_sugar_scatter_plots(df_explain_unstandardised, explanation_df)

    zoomed_in_sugar_chloride_scatter_plots(df_explain_unstandardised, explanation_df)

    low_ph_scatter(df_explain, explanation_df, df_explain_unstandardised, units, type)

    low_sugar_scatter(df_explain, explanation_df, df_explain_unstandardised, units, type)

    low_high_chloride_sugar_relevance_scatter(df_explain_unstandardised, explanation_df)



    for gamma in gamma_list:
        explanation_df = pd.read_csv("wine/explanation_df_{}_{}.csv".format(gamma, type))

    print("Done")


