from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_adapters.shap_adapter import explainShap

print("\n\n\n\n ########## Category ########## \n\n\n\n")

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "datasets/titanic_working_configuration.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/titanic_with_categorical/pytorch-export/model_pytorch.p")

explainShap(automl, X, Y, "auto_pytorch_withcat_", remove_plots=True, number_of_samples=50)

print("\n\n\n\n ########## No category ########## \n\n\n\n")

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "datasets/titanic_working_configuration_nocategories.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/titanic_without_categorical/pytorch-export/model_pytorch.p")

explainShap(automl, X, Y, "auto_pytorch_nocat_", remove_plots=True, number_of_samples=50)
