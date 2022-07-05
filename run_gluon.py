from util import loadDataset
from autogluon.tabular import TabularDataset, TabularPredictor
from XAI_adapters.shap_adapter import explainShap

print("\n\n\n\n ########## Category ########## \n\n\n\n")

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "datasets/titanic_working_configuration.json")
automl = TabularPredictor.load('./models/titanic_with_categorical/gluon-export/model_gluon.gluon')

explainShap(automl, X, Y, "auto_gluon_withcat", remove_plots=True, number_of_samples=5)

print("\n\n\n\n ########## No category ########## \n\n\n\n")

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "datasets/titanic_working_configuration_nocategories.json")
automl = TabularPredictor.load('./models/titanic_without_categorical/gluon-export/model_gluon.gluon')

explainShap(automl, X, Y, "auto_gluon_nocat", remove_plots=True, number_of_samples=5)