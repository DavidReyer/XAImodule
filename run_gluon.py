import traceback
from util import loadDataset
from XAI_controller import explainLime, explainXAIToolbox, executeAutoML
from autogluon.tabular import TabularDataset, TabularPredictor
from XAI_adapters.shap_adapter import explainShap

X, Y, target, features = loadDataset("titanic_working_tabular_classification.csv", "titanic_working_configuration.json")
automl = TabularPredictor.load('./models/gluon-export/model_gluon.gluon')

explainShap(automl, X, Y, "auto_gluon")
executeAutoML(explainLime, "LIME", *[automl, X])
executeAutoML(explainXAIToolbox, "XAI Toolbox", *[automl, X, Y])