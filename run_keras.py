from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_adapters.shap_adapter import explainShap

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "datasets/titanic_working_configuration_nocategories.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("models/titanic_without_categorical/keras-export/model_keras.p")

explainShap(automl, X, Y, "auto_keras", remove_plots=True, number_of_samples=50)