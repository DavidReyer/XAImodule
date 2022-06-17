import traceback
from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_controller import explainLime, explainXAIToolbox, executeAutoML
from XAI_adapters.shap_adapter import explainShap

X, Y, target, features = loadDataset("titanic_working_tabular_classification.csv", "titanic_working_configuration.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("mljar-export/mljar-model.p")

explainShap(automl, X, Y, "mljar")
executeAutoML(explainLime, "LIME", *[automl, X])
executeAutoML(explainXAIToolbox, "XAI Toolbox", *[automl, X, Y])