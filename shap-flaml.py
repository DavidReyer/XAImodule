import shap
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from predict_time_sources import DataType, SplitMethod, feature_preparation

with open('datasets/titanic_working_configuration_flaml.json') as file:
    config_json = json.load(file)

target = config_json["tabular_configuration"]["target"]["target"]
features = config_json["tabular_configuration"]["features"].items()
X = pd.read_csv('datasets/titanic_working_tabular_classification.csv').drop(target, axis=1, errors='ignore')

# convert all object columns to categories, because autosklearn only supports numerical, bool and categorical features
X[X.select_dtypes(['object']).columns] = X.select_dtypes(['object']).apply(lambda x: x.astype('category'))


X = feature_preparation(X, features)

with open('models/flaml-export/model_flaml.p', 'rb') as file:
    automl = pickle.load(file)

predicted_y = automl.predict(X)
predicted_y = np.reshape(predicted_y, (-1, 1))
pd.DataFrame(data=predicted_y, columns=["predicted"]).to_csv("predictions.csv")


shap_output = automl.predict(X)
proba = automl.predict_proba(X)
#shap_values = shap_output[:, :-1]
#expected_value = shap_output[0, -1]

#explainer = shap.Explainer(automl)

explainer = shap.KernelExplainer(automl.predict_proba, X)
shap_values = explainer.shap_values(X)


fig=plt.gcf()
shap.save_html('test.html', shap.force_plot(explainer.expected_value[0], shap_values[0], X))
#shap_values = explainer(X)
fig.savefig('forceplot.png')

#shap.plots.waterfall(shap_values[0])
#shap.plots.beeswarm(shap_values)
#shap.plots.heatmap(shap_values)
shap.summary_plot(shap_values, X)
#shap.plots.scatter(shap_values)
