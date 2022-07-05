import shap
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime


def timestamp():
    return date.strftime(datetime.now(), '_%Y-%m-%d_%H-%M-%S-%f')


def make_html_force_plot(explainer, shap_values, X, class_idx, row_idx, path, filename_detail):
    filename = path + f"shap_force_plot_class_{class_idx}_row_{row_idx}_{filename_detail}_{timestamp()}.html"
    shap.save_html(filename,
                   shap.force_plot(explainer.expected_value[class_idx],
                                   shap_values[class_idx][row_idx],
                                   X.iloc[row_idx]))
    return filename


def make_svg_waterfall_plot(explainer, shap_values, X, class_idx, row_idx, path, filename_detail):
    filename = path + f"waterfall_class_{class_idx}_row_{row_idx}_{filename_detail}_{timestamp()}.svg"
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[class_idx][row_idx], base_values=explainer.expected_value[class_idx],
                         data=X.iloc[row_idx], feature_names=X.columns.tolist()),
        max_display=50)
    plt.savefig(filename)
    plt.clf()
    return filename


def make_svg_beeswarm_plot(shap_values, explainer, X, class_idx, path, filename_detail):
    filename = path + f"beeswarm_class_{class_idx}_{filename_detail}_{timestamp()}.svg"
    shap.plots.beeswarm(shap.Explanation(values=shap_values[class_idx],
                                         base_values=explainer.expected_value[class_idx],
                                         data=X,
                                         feature_names=X.columns.tolist()))
    plt.savefig(filename)
    plt.clf()
    return filename


def make_svg_summary_plot(shap_values, X, path):
    filename = path + "summary_bar" + timestamp() + ".svg"
    shap.summary_plot(shap_values=shap_values, features=X, plot_type='bar')
    plt.savefig(filename)
    plt.clf()
    return filename


def compile_html(html_plots, image_plots, automl_name, path):
    with open(path + "shap_" + automl_name + timestamp() + ".html", "w") as output_file:
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


def explainShap(automl, X, Y, automl_name, remove_plots=True, number_of_samples=5):

    def prediction_probability(X, cols=X.columns, dtypes=X.dtypes):
        df = pd.DataFrame(data=X, columns=cols)
        df = df.astype(dtype=dict(zip(cols, dtypes.values)))
        return automl.predict_proba(df)

    shap.initjs()

    html_plot_path = "plots/html_plots/" + automl_name
    image_plot_path = "plots/image_plots/" + automl_name
    output_path = "plots/output/"

    for path in [html_plot_path, image_plot_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    X_set = X.iloc[0:number_of_samples, :]


    explainer = shap.KernelExplainer(prediction_probability, X)
    shap_values = explainer.shap_values(X_set)

    predictions = automl.predict(X_set)

    html_plot_filenames = []
    image_plot_filenames = []

    for class_idx in Y.unique():
        row_idx = Y[Y == class_idx].index[0]
        # make prediction (class_idx is the true value)
        prediction = predictions[row_idx]

        filename = make_html_force_plot(explainer, shap_values, X_set, class_idx, row_idx, html_plot_path, "truth")
        html_plot_filenames.append(filename)
        filename = make_html_force_plot(explainer, shap_values, X_set, int(prediction), row_idx, html_plot_path, "prediction")
        html_plot_filenames.append(filename)

        filename = make_svg_waterfall_plot(explainer, shap_values, X_set, class_idx, row_idx, image_plot_path, "truth")
        image_plot_filenames.append(filename)
        filename = make_svg_waterfall_plot(explainer, shap_values, X_set, int(prediction), row_idx, image_plot_path, "prediction")
        image_plot_filenames.append(filename)
        filename = make_svg_beeswarm_plot(shap_values, explainer, X_set, class_idx, image_plot_path, "truth")
        image_plot_filenames.append(filename)
        filename = make_svg_beeswarm_plot(shap_values, explainer, X_set, int(prediction), image_plot_path, "prediction")
        image_plot_filenames.append(filename)

    filename = make_svg_summary_plot(shap_values, X_set, image_plot_path)
    image_plot_filenames.append(filename)

    compile_html(html_plot_filenames, image_plot_filenames, automl_name, output_path)
    if remove_plots:
        removePlots(html_plot_filenames)
        removePlots(image_plot_filenames)