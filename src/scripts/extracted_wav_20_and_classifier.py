from src.const import metrics, scripts
from src.const.paths import EX_DNN_20
from src.code.Model import Model
from src.code.Classification import ClassificationSystem

METRIC = metrics.EER
DIR_FEATURES = EX_DNN_20
SCRIPT = scripts.EX_DNN_20
COUNT_SPEAKERS = 20
MODELS = [Model(id, SCRIPT) for id in range(COUNT_SPEAKERS)]
models = ClassificationSystem(MODELS, metric=METRIC)
x_test, y_test, models = models.x_test, models.y_test, models.train_models
score = ClassificationSystem.testing(METRIC, models, x_test, y_test)
print(score)
