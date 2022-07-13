import shap
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime


def make_html_force_plot(base_value, shap_values, X, path, filename_detail):
    filename = path + f"/shap_force_plot_{filename_detail}.html"
    shap.save_html(filename, shap.force_plot(base_value, shap_values, X))
    return filename


def make_svg_waterfall_plot(base_value, shap_values, X, path, filename_detail):
    filename = path + f"/waterfall_{filename_detail}.svg"
    shap.waterfall_plot(
        shap.Explanation(values=shap_values, base_values=base_value,
                         data=X, feature_names=X.index.tolist()),
        max_display=50)
    plt.savefig(filename)
    plt.clf()
    return filename


def make_svg_beeswarm_plot(base_value, shap_values, X, path, filename_detail):
    filename = path + f"/beeswarm_{filename_detail}.svg"
    shap.plots.beeswarm(shap.Explanation(values=shap_values,
                                         base_values=base_value,
                                         data=X,
                                         feature_names=X.columns.tolist()))
    plt.savefig(filename)
    plt.clf()
    return filename


def make_svg_summary_plot(shap_values, X, path):
    filename = path + "/summary_bar.svg"
    shap.summary_plot(shap_values=shap_values, features=X, plot_type='bar')
    plt.savefig(filename)
    plt.clf()
    return filename


def compile_html(html_plots, image_plots, automl_name, path, ml_task):
    with open(path + "/shap_" + automl_name + "_" + ml_task + ".html", "w") as output_file:
        output_file.write(f"<h1> SHAP output of {automl_name} </h1>\n\n")
        for filename in html_plots:
            with open(filename, "r") as shap_file:
                output_file.write(f"<h1> {filename} </h1>\n\n")
                output_file.write(shap_file.read())
                output_file.write("\n\n")
                shap_file.close()
        for filename in image_plots:
            with open(filename, "r") as shap_file:
                output_file.write(f"<h1> {filename} </h1>\n\n")
                output_file.write(shap_file.read())
                output_file.write("\n\n")
                shap_file.close()
        output_file.close()


def removePlots(plots):
    for plot_path in plots:
        os.remove(plot_path)


def getShapExplainer(automl, X, ml_task):
    def prediction_probability(X, cols=X.columns, dtypes=X.dtypes):
        df = pd.DataFrame(data=X, columns=cols)
        df = df.astype(dtype=dict(zip(cols, dtypes.values)))
        return automl.predict_proba(df)

    def predict(X, cols=X.columns, dtypes=X.dtypes):
        df = pd.DataFrame(data=X, columns=cols)
        df = df.astype(dtype=dict(zip(cols, dtypes.values)))
        return automl.predict(df)

    if ml_task == "classification":
        return shap.KernelExplainer(prediction_probability, X)
    if ml_task == "regression":
        return shap.KernelExplainer(predict, X)


def explainShap(automl, X, Y, automl_name, ml_task, remove_plots=True, number_of_samples=5):
    shap.initjs()

    html_plot_path = "plots/html_plots/" + automl_name
    image_plot_path = "plots/image_plots/" + automl_name
    output_path = "plots/output/"

    for path in [html_plot_path, image_plot_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    X_set = X.iloc[0:number_of_samples, :]
    explainer = getShapExplainer(automl, X_set, ml_task)
    shap_values = explainer.shap_values(X_set)
    predictions = automl.predict(X_set)

    html_plot_filenames = []
    image_plot_filenames = []

    if ml_task == "classification":
        for class_idx in Y.unique():
            row_idx = Y[Y == class_idx].index[0]
            # make prediction (class_idx is the true value)
            prediction = predictions[row_idx]

            filename = make_html_force_plot(base_value=explainer.expected_value[class_idx], shap_values=shap_values[class_idx][row_idx], X=X_set.iloc[row_idx], path=html_plot_path, filename_detail=f"classification_rowidx{row_idx}_classidx{class_idx}_truth")
            html_plot_filenames.append(filename)
            filename = make_html_force_plot(base_value=explainer.expected_value[int(prediction)], shap_values=shap_values[int(prediction)][row_idx], X=X_set.iloc[row_idx], path=html_plot_path, filename_detail=f"classification_rowidx{row_idx}_classidx{int(prediction)}_prediction")
            html_plot_filenames.append(filename)

            filename = make_svg_waterfall_plot(base_value=explainer.expected_value[class_idx], shap_values=shap_values[class_idx][row_idx], X=X_set.iloc[row_idx], path=html_plot_path, filename_detail=f"classification_rowidx{row_idx}_classidx{class_idx}_truth")
            image_plot_filenames.append(filename)
            filename = make_svg_waterfall_plot(base_value=explainer.expected_value[int(prediction)], shap_values=shap_values[int(prediction)][row_idx], X=X_set.iloc[row_idx], path=html_plot_path, filename_detail=f"classification_rowidx{row_idx}_classidx{int(prediction)}_prediction")
            image_plot_filenames.append(filename)
            filename = make_svg_beeswarm_plot(base_value=explainer.expected_value[class_idx], shap_values=shap_values[class_idx], X=X_set, path=html_plot_path, filename_detail=f"classification_classidx{class_idx}_truth")
            image_plot_filenames.append(filename)
            filename = make_svg_beeswarm_plot(base_value=explainer.expected_value[int(prediction)], shap_values=shap_values[int(prediction)], X=X_set, path=html_plot_path, filename_detail=f"classification_classidx{class_idx}_prediction")
            image_plot_filenames.append(filename)

    if ml_task == "regression":
        row_idx = 0
        filename = make_html_force_plot(base_value=explainer.expected_value, shap_values=shap_values[row_idx], X=X_set.iloc[row_idx], path=html_plot_path, filename_detail=f"regression_rowidx{row_idx}")
        html_plot_filenames.append(filename)

        filename = make_svg_waterfall_plot(base_value=explainer.expected_value,
                                           shap_values=shap_values[row_idx], X=X_set.iloc[row_idx],
                                           path=html_plot_path,
                                           filename_detail=f"regression_rowidx{row_idx}")
        image_plot_filenames.append(filename)


    filename = make_svg_summary_plot(shap_values, X_set, image_plot_path)
    image_plot_filenames.append(filename)

    compile_html(html_plot_filenames, image_plot_filenames, automl_name, output_path, ml_task)
    if remove_plots:
        removePlots(html_plot_filenames)
        removePlots(image_plot_filenames)