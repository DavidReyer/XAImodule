import shap
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from predict_time_sources import DataType, SplitMethod, feature_preparation

with open('datasets/titanic_working_configuration.json') as file:
    config_json = json.load(file)

target = config_json["tabular_configuration"]["target"]["target"]
features = config_json["tabular_configuration"]["features"].items()
X = pd.read_csv('datasets/titanic_working_tabular_classification.csv').drop(target, axis=1, errors='ignore')

# convert all object columns to categories, because autosklearn only supports numerical, bool and categorical features
X[X.select_dtypes(['object']).columns] = X.select_dtypes(['object']).apply(lambda x: x.astype('category'))


X = feature_preparation(X, features)

with open('models/pytorch-export/model_pytorch.p', 'rb') as file:
    automl = pickle.load(file)

predicted_y = automl.predict(X)
predicted_y = np.reshape(predicted_y, (-1, 1))
pd.DataFrame(data=predicted_y, columns=["predicted"]).to_csv("predictions.csv")

shap.initjs()

explainer = shap.KernelExplainer(automl.predict_proba, X)
shap_values = explainer.shap_values(X)


fig=plt.gcf()
shap.save_html('test.html', shap.force_plot(explainer.expected_value[0], shap_values[0], X))

