from typing import Literal
import os
import pickle
import pandas as pd
import numpy as np

def get_global_local_string(local_adjustment_gamma : float) -> str:
    if local_adjustment_gamma == 0:
        return "global"
    else:
        return "local_{}".format(local_adjustment_gamma)

def load_result_dicts(local_adjustment_gamma : float, model_type : Literal["SVM", "KNN"]) -> dict:
    global_local_folder = get_global_local_string(local_adjustment_gamma)

    folder = "pf_results/{}/{}".format(global_local_folder, model_type)

    list_of_result_dicts = {}
    # load the pkl files inside the folder
    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            file_name = file.split(".")[0]
            with open(folder + "/" + file, 'rb') as f:
                loaded_dict = pickle.load(f)
                list_of_result_dicts[file_name] = loaded_dict
    return list_of_result_dicts

#         eta_ast = round(max(0, best_log_gamma*.3 + .3), 1)

def get_auc_dict(result_dict) -> dict:
    pf_result_keys = result_dict.keys()
    dataset_auc_dict = {}
    for d, dataset in enumerate(pf_result_keys):
        explanation_auc_dict = {}
        for key, value in result_dict[dataset]["results_dict"].items():
            explanation_auc_dict[key] = value["AUC"]
        dataset_auc_dict[dataset] = explanation_auc_dict
    return dataset_auc_dict

def get_k_dict(result_dict, model_type):
    dataset_k_dict = {}
    for d, dataset in enumerate(result_dict):
        if model_type == "KNN":
            k = result_dict[dataset]["data_dict"]["best_n_neighbors"]
            dataset_k_dict[dataset] = k
    return dataset_k_dict

def get_gamma_dict(result_dict, model_type) -> dict:
    dataset_gamma_dict = {}
    for d, dataset in enumerate(result_dict):
        if model_type == "SVM":
            log_gamma = np.log10(result_dict[dataset]["data_dict"]["best_gamma"])
            dataset_gamma_dict[dataset] = log_gamma
        elif model_type == "KNN":
            dataset_gamma_dict[dataset] = 0.5
    return dataset_gamma_dict

def get_eta_prime_dict(gamma_dict : dict, model_type) -> dict:
    eta_prime_dict = {}
    if model_type == "SVM":
        for dataset in gamma_dict.keys():
            log_gamma = gamma_dict[dataset]
            eta_prime = round(max(0, log_gamma * .3 + .3), 1)
            eta_prime_dict[dataset] = eta_prime
    return eta_prime_dict

def filter_eta(auc_dict: dict, eta_prime_dict: dict) -> dict:
    # from the auc values for R_eta=... only keep eta=0, eta=1, and eta=eta_prime
    new_dataset_auc_dict = {}
    for dataset in auc_dict.keys():
        new_explanation_auc_dict = {}
        for explanation, auc in auc_dict[dataset].items():
            if explanation.startswith("R_eta="):
                eta = float(explanation.split("=")[1])
                if eta == 0 or eta == 1:
                    new_explanation_auc_dict[explanation] = auc
                if eta == eta_prime_dict[dataset]:
                    new_explanation_auc_dict["R_eta=prime"] = auc
            else:
                new_explanation_auc_dict[explanation] = auc
        new_dataset_auc_dict[dataset] = new_explanation_auc_dict
    return new_dataset_auc_dict


def generate_latex_table(model_type, model_strings, naive_columns, neuralized_columns):
    # Determine the number of columns in the Naive and Neuralized sections
    num_naive_columns = naive_columns.shape[1]
    num_neuralized_columns = neuralized_columns.shape[1]

    # Start the LaTeX table
    latex_str = "\\begin{table*}[t]\n\\centering\n"
    latex_str += "\\begin{tabular}{l|" + "c" * num_naive_columns + "|" + "c" * num_neuralized_columns + "}\n"
    latex_str += "& \\multicolumn{" + str(num_naive_columns) + "}{c|}{Naive}  & \\multicolumn{" + str(num_neuralized_columns) + "}{c}{Neuralized (ours)}\\\\\n"
    latex_str += "Dataset \hspace{2cm} Model  & " + " & ".join(naive_columns.columns) + " & " + " & ".join(neuralized_columns.columns) + "\\\\\n"
    latex_str += "\\midrule\n"

    dataset_names = naive_columns.index.tolist()
    # Add rows for each dataset and corresponding model string

    all_values = pd.concat([naive_columns, neuralized_columns], axis=1)

    for i, dataset_name in enumerate(dataset_names):
        model_string = model_strings.loc[dataset_name].tolist()
        value_list = all_values.loc[dataset_name].tolist()

        # Add the row to the LaTeX table
        latex_str += f"{dataset_name} \hfill \\footnotesize "+ str(model_string[0]) + " & "

        # map "-" to np.inf
        value_list_numerical = [np.inf if value == "-" else value for value in value_list]
        # get the indices of the smallest and second-smallest values
        min_index = value_list_numerical.index(min(value_list_numerical))
        second_min_index = value_list_numerical.index(sorted(value_list_numerical)[1])
        for ind, value in enumerate(value_list):
            if value == "-":
                latex_str += "- "
            elif ind == min_index:
                latex_str += "\\textbf{" + "{:.3f}".format(value) + "} "
            elif ind == second_min_index:
                latex_str += "\\underline{" + "{:.3f}".format(value) + "} "
            else:
                latex_str += "{:.3f}".format(value) + " "
            if ind < len(value_list) - 1:
                latex_str += "& "
        latex_str += "\\\\\n"

    # End the table
    latex_str += "\\bottomrule\n\\end{tabular}\n"
    latex_str += "\\caption{AUC results of the pixelflipping experiment. Global KDE, both classes}\n"
    latex_str += "\\label{table:pf}\n"
    latex_str += "\\end{table*}"

    return latex_str

def get_gamma_string(df_log_gamma):
    gamma_strings = 10 ** df_log_gamma
    # round to 3 decimal places
    gamma_strings = gamma_strings.round(3)
    # transform values to string by prefixing with model_type
    gamma_strings = gamma_strings.applymap(lambda x:"$\\gamma=" + str(x)+"$")
    return gamma_strings


def prepare_auc_df_SVM(model_type, pf_result_dict, dataset_auc_dict):
    gamma_dict = get_gamma_dict(pf_result_dict, model_type)
    df_gamma = pd.DataFrame.from_dict(gamma_dict, orient="index")

    # use gamma to determine eta_prime per a formula
    eta_prime_dict = get_eta_prime_dict(gamma_dict, model_type)
    # filter the auc_dict to only keep eta=0, eta=1, and eta=eta_prime
    new_dataset_auc_dict = filter_eta(dataset_auc_dict, eta_prime_dict)
    auc_df = pd.DataFrame.from_dict(new_dataset_auc_dict, orient="index")

    # join with gamma values
    auc_df = auc_df.join(df_gamma)
    # rename created column "0" to "log_gamma"
    auc_df = auc_df.rename(columns={0: "log_gamma"})
    # sort by log_gamma from smallest to largest
    #auc_df = auc_df.sort_values(by="log_gamma")
    # sort alphabetically by index
    auc_df = auc_df.sort_index()
    # add average row
    auc_df.loc['Average'] = auc_df.mean(axis=0)

    # in the column and index names prefix all "_" with "\" to make them LaTeX compatible
    auc_df.columns = auc_df.columns.str.replace("_", "\_")
    auc_df.index = auc_df.index.str.replace("_", "\_")

    gamma_strings = get_gamma_string(auc_df[["log\\_gamma"]])

    return auc_df, gamma_strings


def compute_eta_prime_for_KNN(dataset_auc_dict):
    """
    Compute the eta_prime value for KNN models by finding the eta value with the lowest AUC on average.
    :param dataset_auc_dict:
    :return:
    """
    auc_df = pd.DataFrame.from_dict(dataset_auc_dict, orient="index")
    auc_df.loc['Average'] = auc_df.mean(axis=0)
    lowest_auc = np.inf
    best_col = None
    for eta_col in auc_df.columns:
        if eta_col.startswith("R_eta="):
            if auc_df.loc['Average', eta_col] < lowest_auc:
                lowest_auc = auc_df.loc['Average', eta_col]
                best_col = eta_col
    eta_prime = float(best_col.split("=")[1])
    return eta_prime

def prepare_auc_df_KNN(model_type, pf_result_dict, dataset_auc_dict):

    eta_prime = compute_eta_prime_for_KNN(dataset_auc_dict)

    # filter the auc_dict to only keep eta=0, eta=1, and eta=eta_prime
    new_dataset_auc_dict = filter_eta(dataset_auc_dict, {dataset: eta_prime for dataset in dataset_auc_dict.keys()})
    auc_df = pd.DataFrame.from_dict(new_dataset_auc_dict, orient="index")

    k_dict = get_k_dict(pf_result_dict, model_type)
    df_k = pd.DataFrame.from_dict(k_dict, orient="index")

    # join with gamma values
    auc_df = auc_df.join(df_k)
    # rename created column "0" to "log_gamma"
    auc_df = auc_df.rename(columns={0: "k"})
    auc_df = auc_df.sort_index()
    # add average row
    auc_df.loc['Average'] = auc_df.mean(axis=0)

    # in the column and index names prefix all "_" with "\" to make them LaTeX compatible
    auc_df.columns = auc_df.columns.str.replace("_", "\_")
    auc_df.index = auc_df.index.str.replace("_", "\_")

    k_strings = auc_df[["k"]].applymap(lambda x: "$k=" + str(x) + "$")

    # if there is a column "Occ", rename to "R\\_occ"
    if "Occ" in auc_df.columns:
        auc_df = auc_df.rename(columns={"Occ": "R\\_occ"})
    return auc_df, k_strings


def select_columns_with_defaults(df, columns):
    """
    Select specified columns from the dataframe.
    If a column is missing, add it with default values set to '-'.

    Parameters:
    df (pd.DataFrame): The original dataframe
    columns (list): List of columns to extract

    Returns:
    pd.DataFrame: A new dataframe with the specified columns (or default columns if missing)
    """
    # Create a new dataframe with the selected columns if they exist
    result_df = pd.DataFrame()
    # init the same indices as the original dataframe
    result_df.index = df.index

    for col in columns:
        if col in df.columns:
            result_df[col] = df[col]
        else:
            # Add a column with all values set to '-'
            result_df[col] = '-'

    return result_df

def main(local_adjustment_gamma : float = 0, model_type : str = "SVM", naive_explanations=None, neural_explanations=None):
    pf_result_dict = load_result_dicts(local_adjustment_gamma, model_type)
    dataset_auc_dict = get_auc_dict(pf_result_dict)

    if model_type == "SVM":
        auc_df, param_strings = prepare_auc_df_SVM(model_type, pf_result_dict, dataset_auc_dict)
    elif model_type == "KNN":
        auc_df, param_strings = prepare_auc_df_KNN(model_type, pf_result_dict, dataset_auc_dict)
    else:
        raise ValueError("Model type must be either 'SVM' or 'KNN'")

    model_strings = param_strings.applymap(lambda x: model_type + " " + x)

    naive_columns = select_columns_with_defaults(auc_df, naive_explanations)
    neuralized_columns = select_columns_with_defaults(auc_df, neural_explanations)

    # Generate LaTeX table
    latex_table = generate_latex_table(model_type, model_strings, naive_columns, neuralized_columns)
    print(latex_table)

    return


if __name__ == "__main__":

    local_adjustment_gamma = 0
    model_type = "KNN"

    naive_explanations = ["GI", "IG", "sensitivities", "R\\_occ"]
    neural_explanations = ["R\\_eta=0", "neural\\_IG", "R\\_eta=prime"]

    main(local_adjustment_gamma, model_type, naive_explanations, neural_explanations)

    print("Done")