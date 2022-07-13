from util import loadDataset, loadAutoml, convertObjectColumnDatatypesToCategory
from XAI_adapters.shap_adapter import explainShap
from supervised.automl import AutoML
import os

X, Y, target, features = loadDataset("datasets/titanic_working_tabular_classification.csv", "models/mljar-export/titanic_working_configuration.json")
X = convertObjectColumnDatatypesToCategory(X)
#automl = loadAutoml("models/mljar-export/mljar-model.p")
automl = AutoML(results_path=os.path.join(os.path.dirname(__file__), "models/mljar-export/Models"))

explainShap(automl, X, Y, "mljar", remove_plots=True, number_of_samples=50)