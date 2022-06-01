import traceback
from util import loadDataset
from XAI_controller import explainShap, explainLime
from autogluon.tabular import TabularDataset, TabularPredictor

X, target, features = loadDataset("titanic_working_tabular_classification.csv", "titanic_working_configuration.json")
automl = TabularPredictor.load('./models/gluon-export/model_gluon.gluon')

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