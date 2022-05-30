import os.path
import shap
import sys
import pickle
import json
from predict_time_sources import feature_preparation, DataType, SplitMethod
from autogluon.tabular import TabularDataset, TabularPredictor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plty

with open('datasets/titanic_working_configuration.json') as file:
    config_json = json.load(file)

target = config_json["tabular_configuration"]["target"]["target"]
features = config_json["tabular_configuration"]["features"].items()
X = pd.read_csv('datasets/titanic_working_tabular_classification.csv').drop(target, axis=1, errors='ignore')


X = feature_preparation(X, features)

# load model and make predictions
print("starting to load model")
automl = TabularPredictor.load('./models/gluon-export/model_gluon.gluon')

predicted_y = automl.predict(X, as_pandas=False)
predicted_y = np.reshape(predicted_y, (-1, 1))

shap.initjs()

explainer = shap.KernelExplainer(automl.predict_proba, X)
shap_values = explainer.shap_values(X)


fig=plt.gcf()
shap.save_html('test.html', shap.force_plot(explainer.expected_value[0], shap_values[0], X))

