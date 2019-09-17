from src.const import metrics, classifiers, scripts
from src.code.Classification import ClassificationSystem

if __name__ == '__main__':
    METRIC = metrics.EER
    SCRIPT = scripts.EX_DNN_20
    CLASSIFIER = classifiers.SVM_LINEAR
    data = ClassificationSystem(script=SCRIPT, metric=METRIC,
                                dim=None, is_balance=None, is_mix=False, is_normalize=True,
                                records_train=40, records_test=10)
    ClassificationSystem.training(data)
    score = ClassificationSystem.testing(METRIC, data.train_models, data.x_test, data.y_test)
    print(score)  # (0.5059210526315789, 0.051945745982116896) для 40/10
