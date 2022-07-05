import pandas as pd
import shap
import lime
import lime.lime_tabular
import numpy as np
import xai
import traceback
import matplotlib.pyplot as plt


def executeAutoML(function, name, *func_args):
    print(f"\n\n\n#################### {name} ####################\n\n\n")
    try:
        function(*func_args)
    except:
        traceback.print_exc()


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


    X = X.fillna(0)
    return X, cat_names, cat_features









def explainLime(automl, X):
    X, cat_names, cat_features = processCategorical(X)

    def prediction_probability(X, cols=X.columns, dtypes=X.dtypes):
        df = pd.DataFrame(data=X, columns=cols)
        df = df.astype(dtype=dict(zip(cols, dtypes.values)))
        return automl.predict_proba(df)

    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X), feature_names=X.columns, discretize_continuous=True,
                                                       categorical_features=cat_features, categorical_names=cat_names)

    i = np.random.randint(0, X.shape[0])
    exp = explainer.explain_instance(np.array(X)[i], prediction_probability, num_features=2, top_labels=1)

    exp.save_to_file('lime_out.html', labels=None, predict_proba=True, show_predicted_value=True)


