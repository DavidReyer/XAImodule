from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_adapters.shap_adapter import explainShap


X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "models/sklearn-export/titanic_working_configuration.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/sklearn-export/model_sklearn.p")

explainShap(automl, X, Y, "auto_sklearn", remove_plots=True, number_of_samples=50, ml_task="classification")



X, Y, target, features = loadDataset("datasets/college_working_tabular_regression.csv", "models/regression/sklearn-export/college_configuration.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/regression/sklearn-export/model_sklearn.p")

explainShap(automl, X, Y, "auto_sklearn", remove_plots=True, number_of_samples=50, ml_task="regression")