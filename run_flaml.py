import traceback
from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_controller import explainShap, explainLime

X, target, features = loadDataset("titanic_working_tabular_classification.csv", "titanic_working_configuration_flaml.json")
X = convertObjectColumnDatatypesToCategory(X)
automl = loadAutoml("flaml-export/model_flaml.p")

print("\n\n\n#################### SHAP ####################\n\n\n")
try:
    explainShap(automl, X)
except:
    traceback.print_exc()

print("\n\n\n#################### LIME ####################\n\n\n")
try:
    explainLime(automl, X)
except:
    traceback.print_exc()