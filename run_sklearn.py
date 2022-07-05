from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_adapters.shap_adapter import explainShap

print("\n\n\n\n ########## Category ########## \n\n\n\n")

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "datasets/titanic_working_configuration.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/titanic_with_categorical/sklearn-export/model_sklearn.p")

explainShap(automl, X, Y, "auto_sklearn_withcat", remove_plots=True, number_of_samples=1)

print("\n\n\n\n ########## No category ########## \n\n\n\n")

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "datasets/titanic_working_configuration_nocategories.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/titanic_without_categorical/sklearn-export/model_sklearn.p")

explainShap(automl, X, Y, "auto_sklearn_nocat", remove_plots=True, number_of_samples=1)