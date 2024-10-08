from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import torch
import torchvision

from medmnist import ChestMNIST, PneumoniaMNIST
import sklearn

def load_rice_dataset():
    # fetch dataset
    rice_cammeo_and_osmancik = fetch_ucirepo(id=545)

    # data (as pandas dataframes)
    X = rice_cammeo_and_osmancik.data.features.values
    y = rice_cammeo_and_osmancik.data.targets.values
    # set "Osmancik" to 0 and "Cammeo" to 1
    y = np.where(y == 'Osmancik', -1, 1)
    return X, y

def load_diabetes_dataset():
    # fetch dataset
    early_stage_diabetes_risk_prediction = fetch_ucirepo(id=529)

    # data (as pandas dataframes)
    X = early_stage_diabetes_risk_prediction.data.features
    gender_map = {"Male":1.0, "Female":0.0}
    X["gender"] = X["gender"].map(gender_map)

    X = X.replace({"Yes": 1.0, "No": 0.0})
    X = X.values

    y = early_stage_diabetes_risk_prediction.data.targets.values
    y = np.where(y=='Positive', 1, -1)
    return X, y


def load_breast_cancer_dataset():
    # data (as pandas dataframes)
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    X = breast_cancer_wisconsin_diagnostic.data.features.values
    y = breast_cancer_wisconsin_diagnostic.data.targets.values
    y = np.where(y=='B', 1, -1)
    return X, y


def load_taiwanese_bankruptcy_dataset():
    # fetch dataset
    taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572)

    # data (as pandas dataframes)
    X = taiwanese_bankruptcy_prediction.data.features.values
    y = taiwanese_bankruptcy_prediction.data.targets.values
    y = np.where(y == 0, -1, 1)
    return X, y


def load_polish_bankruptcy_dataset():
    # fetch dataset
    polish_companies_bankruptcy = fetch_ucirepo(id=365)

    X = polish_companies_bankruptcy.data.features.values
    y = polish_companies_bankruptcy.data.targets.values
    y = np.where(y == 0, -1, 1)

    # remove colums with nan
    is_nan = np.isnan(X).any(axis=1)
    X = X[~is_nan]
    return X, y

def load_fertility_dataset():
    fertility = fetch_ucirepo(id=244)

    # data (as pandas dataframes)
    X = fertility.data.features.values
    y = fertility.data.targets.values
    y = np.where(y == "N", -1, 1)

    return X, y


def load_wine_quality_dataset():
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features.values
    y = wine_quality.data.targets.values

    # if larger then 6 set to 1, else set to -1
    y = np.where(y > 5, 1, -1)

    return X, y

def load_wine_quality_multiclass():
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features.values
    y = wine_quality.data.targets.values

    # drop class y==3, y==9
    drop_indices = np.where((y == 3) | (y == 9))
    X = np.delete(X, drop_indices, axis=0)
    y = np.delete(y, drop_indices, axis=0)

    return X, y


def load_concrete_dataset():
    concrete_compressive_strength = fetch_ucirepo(id=165)

    # data (as pandas dataframes)
    X = concrete_compressive_strength.data.features.values
    y = concrete_compressive_strength.data.targets.values

    # if larger then 90th percentile set to 1, else set to -1
    perc_90 = np.percentile(y, 90)
    y = np.where(y > perc_90, 1, -1)

    return X, y


def load_catalyst_dataset(return_feature_names=False):
    # Load the CSV file
    dataset = pd.read_csv('df_tree.csv')
    dataset = dataset.drop(columns=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'GLa'])
    data_top = dataset.keys()

    feature_names = data_top[1:-1]  # Assuming the first column is non-numeric and the last column is the label
    X = dataset.iloc[:, 1:-1].values  # Only select the columns that should be numeric
    # to float
    X = X.astype(float)
    y = dataset.iloc[:, -1].values  # The label column is assumed to be the last column

    # Ensure y is in integer format for the following operations
    y = y.astype(int)
    y = np.where(y == 0, -1, 1)

    if return_feature_names:
        return X, y, feature_names
    else:
        return X, y

def load_superconductivity_dataset():
    superconductivty_data = fetch_ucirepo(id=464)

    X = superconductivty_data.data.features.values
    y = superconductivty_data.data.targets.values
    y_median = statistics.median(y)

    y = np.where(y > y_median, 1, -1)

    return X, y

def load_superconductivity_dataset_percentiles():
    superconductivty_data = fetch_ucirepo(id=464)

    X = superconductivty_data.data.features.values
    y = superconductivty_data.data.targets.values

    # calculate percentiles
    y_percentiles = np.percentile(y, [20, 40, 60, 80])

    # digitize y
    y = np.digitize(y, y_percentiles)

    return X, y


def load_pneumonia_mnist():
    dataset_train = PneumoniaMNIST(split="train", download=True, size=64)
    X_train, y_train = dataset_train.imgs, dataset_train.labels
    y_train = np.where(y_train == 0, -1, 1)

    dataset_test = PneumoniaMNIST(split="test", download=True, size=64)
    X_test, y_test = dataset_test.imgs, dataset_test.labels
    y_test = np.where(y_test == 0, -1, 1)

    return X_train, y_train, X_test, y_test


def load_raisin():
    # fetch dataset
    raisin = fetch_ucirepo(id=850)

    # data (as pandas dataframes)
    X = raisin.data.features.values
    y = raisin.data.targets

    class_names = np.unique(y)

    # class names contains two strings, if it is the former 1, else -1
    y = np.where(y == class_names[0], 1, -1)

    return X, y


def load_car_evaluation():
    # fetch dataset
    car_evaluation = fetch_ucirepo(id=19)

    # data (as pandas dataframes)
    X = car_evaluation.data.features.values
    uniques_1col = np.unique(X[:, 0])
    # vhigh - 4, high - 3, med - 2, low - 1
    X[:, 0] = np.where(X[:, 0] == "vhigh", 4, X[:, 0])
    X[:, 0] = np.where(X[:, 0] == "high", 3, X[:, 0])
    X[:, 0] = np.where(X[:, 0] == "med", 2, X[:, 0])
    X[:, 0] = np.where(X[:, 0] == "low", 1, X[:, 0])
    uniques_2col = np.unique(X[:, 1])
    # same
    X[:, 1] = np.where(X[:, 1] == "vhigh", 4, X[:, 1])
    X[:, 1] = np.where(X[:, 1] == "high", 3, X[:, 1])
    X[:, 1] = np.where(X[:, 1] == "med", 2, X[:, 1])
    X[:, 1] = np.where(X[:, 1] == "low", 1, X[:, 1])

    uniques_3col = np.unique(X[:, 2])
    # 5more - 5
    X[:, 2] = np.where(X[:, 2] == "5more", 5, X[:, 2])
    uniques_4col = np.unique(X[:, 3])
    # more - 6
    X[:, 3] = np.where(X[:, 3] == "more", 6, X[:, 3])
    uniques_5col = np.unique(X[:, 4])
    # small - 1, med - 2, big - 3
    X[:, 4] = np.where(X[:, 4] == "small", 1, X[:, 4])
    X[:, 4] = np.where(X[:, 4] == "med", 2, X[:, 4])
    X[:, 4] = np.where(X[:, 4] == "big", 3, X[:, 4])
    uniques_6col = np.unique(X[:, 5])
    # low - 1, med - 2, high - 3
    X[:, 5] = np.where(X[:, 5] == "low", 1, X[:, 5])
    X[:, 5] = np.where(X[:, 5] == "med", 2, X[:, 5])
    X[:, 5] = np.where(X[:, 5] == "high", 3, X[:, 5])

    # cast to float
    X = X.astype(float)

    y = car_evaluation.data.targets

    # make unacc one class and the rest the other
    y = np.where(y == "unacc", 1, -1)
    return X, y

def load_car_evaluation_multiclass():
    # fetch dataset
    car_evaluation = fetch_ucirepo(id=19)

    # data (as pandas dataframes)
    X = car_evaluation.data.features.values
    uniques_1col = np.unique(X[:, 0])
    # vhigh - 4, high - 3, med - 2, low - 1
    X[:, 0] = np.where(X[:, 0] == "vhigh", 4, X[:, 0])
    X[:, 0] = np.where(X[:, 0] == "high", 3, X[:, 0])
    X[:, 0] = np.where(X[:, 0] == "med", 2, X[:, 0])
    X[:, 0] = np.where(X[:, 0] == "low", 1, X[:, 0])
    uniques_2col = np.unique(X[:, 1])
    # same
    X[:, 1] = np.where(X[:, 1] == "vhigh", 4, X[:, 1])
    X[:, 1] = np.where(X[:, 1] == "high", 3, X[:, 1])
    X[:, 1] = np.where(X[:, 1] == "med", 2, X[:, 1])
    X[:, 1] = np.where(X[:, 1] == "low", 1, X[:, 1])

    uniques_3col = np.unique(X[:, 2])
    # 5more - 5
    X[:, 2] = np.where(X[:, 2] == "5more", 5, X[:, 2])
    uniques_4col = np.unique(X[:, 3])
    # more - 6
    X[:, 3] = np.where(X[:, 3] == "more", 6, X[:, 3])
    uniques_5col = np.unique(X[:, 4])
    # small - 1, med - 2, big - 3
    X[:, 4] = np.where(X[:, 4] == "small", 1, X[:, 4])
    X[:, 4] = np.where(X[:, 4] == "med", 2, X[:, 4])
    X[:, 4] = np.where(X[:, 4] == "big", 3, X[:, 4])
    uniques_6col = np.unique(X[:, 5])
    # low - 1, med - 2, high - 3
    X[:, 5] = np.where(X[:, 5] == "low", 1, X[:, 5])
    X[:, 5] = np.where(X[:, 5] == "med", 2, X[:, 5])
    X[:, 5] = np.where(X[:, 5] == "high", 3, X[:, 5])

    # cast to float
    X = X.astype(float)

    y = car_evaluation.data.targets
    unique_labels = np.unique(y)
    # encode each label as an integer
    for i, label in enumerate(unique_labels):
        y = np.where(y == label, i, y)
    return X, y

def load_real_estate_evaluation():
    # fetch dataset
    real_estate_valuation = fetch_ucirepo(id=477)

    # data (as pandas dataframes)
    X = real_estate_valuation.data.features.values
    y = real_estate_valuation.data.targets.values

    # plot histogram of y
    #plt.hist(y, bins=20)
    # compute median of y
    y_median = statistics.median(y)
    # make y binary
    y = np.where(y > y_median, 1, -1)

    return X, y

def load_real_estate_percentiles():
    # fetch dataset
    real_estate_valuation = fetch_ucirepo(id=477)

    # data (as pandas dataframes)
    X = real_estate_valuation.data.features.values
    y = real_estate_valuation.data.targets.values

    # compute percentiles of y
    y_percentiles = np.percentile(y, [20, 40, 60, 80])
    # make y multiclass
    y = np.digitize(y, y_percentiles)

    return X, y

def load_mnist_49():
    # fetch dataset
    train_dataset = torchvision.datasets.MNIST(train=True, root="data", download=True)
    X, y = train_dataset.data, train_dataset.targets
    # only 4 and 9
    mask = (y == 4) | (y == 9)
    X, y = X[mask], y[mask]
    y = np.where(y == 4, 1, -1)
    X = X.numpy()
    X = X.reshape(X.shape[0], -1)


    return X, y


def load_mnist():
    # fetch dataset
    train_dataset = torchvision.datasets.MNIST(train=True, root="data", download=True)
    X, y = train_dataset.data, train_dataset.targets
    X = X.numpy()
    X = X.reshape(X.shape[0], -1)
    y = y.numpy()[:, None]
    return X, y


def load_cover_type():
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features.values
    y = covertype.data.targets.values

    return X, y

def load_beans():
    dry_bean = fetch_ucirepo(id=602)

    # data (as pandas dataframes)
    X = dry_bean.data.features.values
    y = dry_bean.data.targets.values

    unique_classes = np.unique(y)
    # enumerate classes 1, 2, 3, 4, 5, ...
    for i, c in enumerate(unique_classes):
        y = np.where(y == c, i, y)
    return X, y

def load_mines():
    # fetch dataset
    land_mines = fetch_ucirepo(id=763)

    # data (as pandas dataframes)
    X = land_mines.data.features.values
    y = land_mines.data.targets.values

    return X, y


def shuffle_data(X, y):
    shuffle_inds = np.random.permutation(len(X))
    X = X[shuffle_inds]
    y = y[shuffle_inds]
    return X, y


def train_test_split(X, y, test_size=0.2):
    split_ind = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_ind], X[split_ind:]
    y_train, y_test = y[:split_ind], y[split_ind:]
    return X_train, X_test, y_train, y_test

def standardise_data(X_train, X_test):
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # if train_std == 0 set all values to 0
    X_train[:, train_std == 0] = 0
    X_test[:, train_std == 0] = 0
    return X_train, X_test

def distance_normalization(X_train, X_test):
    # compute squared pairwise distance matrix D
    D = sklearn.metrics.pairwise.euclidean_distances(X_train, X_train) ** 2
    # compute median of D
    median = np.median(D)

    # divide X_train, X_test by median
    X_train_dnormed = X_train / median ** (1 / 2)
    X_test_dnormed = X_test / median ** (1 / 2)

    D = sklearn.metrics.pairwise.euclidean_distances(X_train_dnormed, X_train_dnormed) ** 2
    median = np.median(D)

    assert np.isclose(median, 1)

    return X_train_dnormed, X_test_dnormed

def preprocessing(raw_dataset: dict):
    np.random.seed(0)
    X = raw_dataset["X"]
    y = raw_dataset["y"]

    class_counts = np.unique(y, return_counts=True)[1]
    class_ratio = np.max(class_counts) / class_counts.sum()
    data_dict = {"n_data_points": X.shape[0], "class_ratio": class_ratio, "n_features": X.shape[1]}

    # check if y is 1D
    if len(y.shape) == 1:
        y = y[:, None]
    # shuffle
    X, y = shuffle_data(X, y)

    # make sure data is not too large
    X, y = X[:5000], y[:5000]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 0,1 standardisation
    X_train, X_test = standardise_data(X_train, X_test)

    # make sure median of data distance is one
    X_train, X_test = distance_normalization(X_train, X_test)

    class_weight_dicts = {
        1: (y_train == -1).sum() / len(y_train),
        -1: (y_train == 1).sum() / len(y_train)
}
    data_dict["class_weight_dicts"] = class_weight_dicts

    dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    return dataset, data_dict


def provide_dataset(dataset_name: str, flipping_set_size: int = 100):
    X, y = None, None
    if dataset_name == "concrete":
        X, y = load_concrete_dataset()
    elif dataset_name == "breast_cancer":
        X, y = load_breast_cancer_dataset()
    elif dataset_name == "rice":
        X, y = load_rice_dataset()
    elif dataset_name == "raisin":
        X, y = load_raisin()
    elif dataset_name == "diabetes":
        X, y = load_diabetes_dataset()
    elif dataset_name == "car_evaluation":
        X, y = load_car_evaluation()
    elif dataset_name == "real_estate":
        X, y = load_real_estate_evaluation()
    elif dataset_name == "wine_quality":
        X, y = load_wine_quality_dataset()
    elif dataset_name == "catalyst":
        X, y = load_catalyst_dataset()
    else:
        raise NotImplementedError("Dataset not implemented")

    raw_dataset = {"X": X, "y": y}

    # preprocessing
    processed_dataset, data_dict = preprocessing(raw_dataset)

    # create flipping dataset
    # limit the size of the flipping set to 100
    X_test = processed_dataset["X_test"]
    y_test = processed_dataset["y_test"]
    X_flipping = X_test[:min(flipping_set_size, len(X_test))]
    y_flipping = y_test[:min(flipping_set_size, len(X_test))]
    processed_dataset["X_flipping"] = X_flipping
    processed_dataset["y_flipping"] = y_flipping

    return processed_dataset, data_dict
