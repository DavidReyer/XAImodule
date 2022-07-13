import xai
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, datetime
from scipy.stats import spearmanr


def timestamp():
    return date.strftime(datetime.now(), '_%Y-%m-%d_%H-%M-%S-%f')


def processCategorical(X):
    # Process categories. Lime only accepts categories when processed into int
    # https://github.com/marcotcr/lime/issues/346
    cat_features = []
    cat_names = {}
    for i, dtype in enumerate(X.dtypes):
        if dtype.name == "category":
            # Save index to tell lime where the categorical columns are
            cat_features.append(i)
            # Process the corresponding column
            # Get all unique values in the category column
            unique_values = X.iloc[:, i].unique()
            # Add the values to the names dictionary required by lime
            cat_names.update({i: list(unique_values.categories.values)})
            # Replace values in dataframe
            X.iloc[:, i] = X.iloc[:, i].map({val: i for i, val in enumerate(unique_values.categories.values)})
        if dtype.name == "bool":
            cat_features.append(i)
            unique_values = [False, True]
            cat_names.update({i: unique_values})
            X.iloc[:, i] = X.iloc[:, i].map({val: i for i, val in enumerate(unique_values)})

    # fill nan values. Fill it with 9999... with the length of the max number of digits in the df + 1
    fill_value = int("9" * (len(str(int(X.max().max()))) + 1))
    for col in X.columns:
        if X[col].dtype.name == "category":
            # category cols require special handling
            X[col] = X[col].cat.add_categories(fill_value).fillna(fill_value)
        else:
            X[col] = X[col].fillna(fill_value)

    return X, cat_names, cat_features


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def compile_html(image_plots, path, dataset_name):
    with open(path + "dataset_analysis_for_" + dataset_name + timestamp() + ".html", "w") as output_file:
        output_file.write(f"<h1> Dataset analysis of {dataset_name} </h1>\n\n<div>")
        for filename in image_plots:
            with open(filename, "r") as file:
                output_file.write(f"<h1> {filename} </h1>\n\n")
                output_file.write(file.read())
                output_file.write("\n\n")
                file.close()
        output_file.write("</div>")
        output_file.close()


def makePlot(df, colname, image_plot_path):
    plt.clf()
    # Sturge’s rule
    bins = (int(np.ceil(np.log2(len(df))) + 1))
    if df[colname].dtype.name == "category" or df[colname].dtype.name == "bool":
        # bins is either determined by sturges rule or the number of unique values, whatever is less
        # this is to not make plots of categoricals with 2-3 values not stupid
        df[colname].value_counts().plot(kind='bar')
    else:
        df[colname].plot(kind='hist')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)
    plt.title(colname)
    filename = image_plot_path + "_" + colname + "_plot" + timestamp() + ".svg"
    plt.savefig(filename)
    return filename


def makeFeatureImbalancePlot(df, first_colname, second_colname, categorical_columns, image_plot_path):
    plt.clf()
    plt.figure(figsize=(12, 6))
    xai.imbalance_plot(df, first_colname, second_colname, categorical_cols=categorical_columns)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)
    filename = image_plot_path + "_" + first_colname + "_vs_" + second_colname + "_feature_imbalance.svg"
    plt.savefig(filename)
    return filename


def removePlots(plots):
    for plot_path in plots:
        os.remove(plot_path)


def explainXAIToolbox(X, Y, target, dataset_name, remove_plots=True):
    plt.ioff()
    X[target] = Y
    X_org = X.copy()
    proc_X, cat_names, cat_features = processCategorical(X)
    image_plot_path = "plots/image_plots/" + dataset_name
    output_path = "plots/output/"

    for path in [image_plot_path, output_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    categorical_columns = list(X.select_dtypes(['category']).columns) + list(X.select_dtypes(['bool']).columns)
    image_plot_filenames = []

    for col in list(X.columns):
        filename = makePlot(X_org, col, image_plot_path)
        image_plot_filenames.append(filename)


    plt.clf()
    plt.rcParams['figure.figsize'] = [16, 16]
    feature_correlation_plot = xai.correlations(proc_X,
                                                include_categorical=True,
                                                categorical_cols=categorical_columns,
                                                plot_type="matrix")
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.5)
    filename = image_plot_path + "_correlation_matrix.svg"
    plt.savefig(filename)
    image_plot_filenames.append(filename)

    # get correlations using spearmanr
    corr = spearmanr(proc_X).correlation
    # get indices of correlations ordered desc
    indices = largest_indices(corr, (len(proc_X.columns) * len(proc_X.columns)))

    plotted_indices = []
    for first_col_idx, second_col_index in zip(indices[0], indices[1]):
        # if the correlation isnt a col with itself or has already been plotted the other way around -> plot it
        if first_col_idx != second_col_index and [second_col_index, first_col_idx] not in plotted_indices:
            filename = makeFeatureImbalancePlot(X_org, X_org.columns[first_col_idx], X_org.columns[second_col_index], categorical_columns, image_plot_path)
            image_plot_filenames.append(filename)
            plotted_indices.append([first_col_idx, second_col_index])
        # only plot top 5
        if len(plotted_indices) == 5:
            break

    compile_html(image_plot_filenames, output_path, dataset_name)
    if remove_plots:
        removePlots(image_plot_filenames)