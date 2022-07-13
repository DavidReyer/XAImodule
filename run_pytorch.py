from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_adapters.shap_adapter import explainShap
"""
X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "datasets/titanic_working_configuration.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/titanic_with_categorical/pytorch-export/model_pytorch.p")

explainShap(automl, X, Y, "auto_pytorch_withcat_", remove_plots=True, number_of_samples=50)
"""

X, Y, target, features = loadDataset("datasets/college_working_tabular_regression.csv", "models/regression/sklearn-export/college_configuration.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/regression/pytorch-export/model_pytorch.p")

explainShap(automl, X, Y, "auto_pytorch_regression", remove_plots=True, number_of_samples=50, ml_task="regression")