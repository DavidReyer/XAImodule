import traceback
from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_controller import explainLime, explainXAIToolbox, executeAutoML
from XAI_adapters.shap_adapter import explainShap

X, Y, target, features = loadDataset("titanic_working_tabular_classification.csv", "titanic_working_configuration_nocategories.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("sklearn-export/model_sklearn.p")

explainShap(automl, X, Y, "auto_sklearn")
executeAutoML(explainLime, "LIME", *[automl, X])
executeAutoML(explainXAIToolbox, "XAI Toolbox", *[automl, X, Y])