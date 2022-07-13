from util import loadDataset
from autogluon.tabular import TabularDataset, TabularPredictor
from XAI_adapters.shap_adapter import explainShap

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "models/gluon-export/titanic_working_configuration.json")
automl = TabularPredictor.load('./models/gluon-export/model_gluon.gluon')

explainShap(automl, X, Y, "auto_gluon", remove_plots=True, number_of_samples=50)