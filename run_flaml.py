from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_adapters.shap_adapter import explainShap

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "datasets/titanic_working_configuration_flaml.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/titanic_with_categorical/flaml-export/model_flaml.p")

explainShap(automl, X, Y, "flaml", remove_plots=True, number_of_samples=5)