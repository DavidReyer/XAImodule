import traceback
from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_controller import explainLime, explainXAIToolbox, executeAutoML
from XAI_adapters.shap_adapter import explainShap

X, Y, target, features = loadDataset("titanic_working_tabular_classification.csv", "titanic_working_configuration_flaml.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("flaml-export/model_flaml.p")

explainShap(automl, X, Y, "flaml")
executeAutoML(explainLime, "LIME", *[automl, X])
executeAutoML(explainXAIToolbox, "XAI Toolbox", *[automl, X, Y])