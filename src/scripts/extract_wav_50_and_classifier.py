from src.const import metrics, classifiers, scripts
from src.const.paths import EX_DNN_50
from src.code.Model import Model
from src.code.Classification import ClassificationSystem

METRIC = metrics.EER
CLASSIFIER = classifiers.SVM_LINEAR
DIR_FEATURES = EX_DNN_50
COUNT_SPEAKERS = 20
COUNT_RECORDS = 50
NORMALIZE = True
MODELS = [Model(id, scripts.EX_DNN_50) for id in
          range(COUNT_SPEAKERS)]
models = ClassificationSystem(MODELS, metric=METRIC, records_train=40, records_test=10, is_normalize=NORMALIZE)
x_test, y_test, models = models.x_test, models.y_test, models.train_models
score = ClassificationSystem.testing(METRIC, models, x_test, y_test)
print(score)  # (0.5059210526315789, 0.051945745982116896) для 40/10
