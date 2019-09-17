from src.code.ItemStacking import ItemStacking
from src.utils.visualization.visualization_classification import error_visualization
from src.const import metrics, classifiers, scripts

if __name__ == '__main__':
    METRIC = metrics.ACCURACY
    CLASSIFIER = classifiers.SVM_LINEAR
    TITLE = f'{METRIC}_{CLASSIFIER.name}'
    SCRIPT = scripts.HTK_20
    item = ItemStacking(CLASSIFIER, SCRIPT, METRIC, is_mix=False, is_balance=None, is_normalize=True, dim=None)
    error_visualization(item.score_train, item.score_test, metric=METRIC, title=TITLE).show()
