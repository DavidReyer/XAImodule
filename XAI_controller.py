import shap
import lime
import lime.lime_tabular
import numpy as np


def explainShap(automl, X):
    shap.initjs()

    explainer = shap.KernelExplainer(automl.predict_proba, X)
    shap_values = explainer.shap_values(X)

    shap.save_html('force_plot.html', shap.force_plot(explainer.expected_value[0], shap_values[0], X))



def explainLime(automl, X):
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

    X = X.fillna(0)

    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X), feature_names=X.columns, discretize_continuous=True,
                                                       categorical_features=cat_features, categorical_names=cat_names)

    i = np.random.randint(0, X.shape[0])
    exp = explainer.explain_instance(np.array(X)[i], automl.predict_proba, num_features=2, top_labels=1)

    exp.save_to_file('./lime_out', labels=None, predict_proba=True, show_predicted_value=True)