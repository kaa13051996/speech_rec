from src.const import classifiers, metrics, scripts
from src.code.Classification import ClassificationSystem
from src.utils.save import save_models

from numpy import array
import logging

logging.basicConfig(level=logging.DEBUG,
                    format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s')


class ItemStacking:
    def __init__(self, classifier, script, metric=metrics.PROBA, nb_train=40, nb_test=10, is_normalize=True,
                 is_mix=False, is_balance=False, dim=None):
        self.classifier = classifier
        self.metric = metric
        script.model = classifier
        self.models = ClassificationSystem(script, self.metric, records_train=nb_train,
                                           records_test=nb_test, is_normalize=is_normalize, is_mix=is_mix,
                                           is_balance=is_balance, dim=dim)
        ClassificationSystem.training(self.models) if self.models.train_models is None else None
        # save_models(self.models.train_models, name=f'{self.classifier.name}')
        self.score_train = ClassificationSystem.testing(self.metric, self.models.train_models, self.models.x_train,
                                                        self.models.y_train
                                                        )
        self.score_test = ClassificationSystem.testing(self.metric, self.models.train_models, self.models.x_test,
                                                       self.models.y_test
                                                       )


def get_features(items_bagging):
    x_train, x_test = [], []

    for item in items_bagging:
        x_train.append(array(item.score_train))
        x_test.append(array(item.score_test))
    x_train = array(x_train).transpose()
    x_test = array(x_test).transpose()
    return x_train, x_test


def get_labels(items_bagging):
    y_train = array(items_bagging[0].models.y_train).ravel()
    y_test = array(items_bagging[0].models.y_test).ravel()
    return y_train, y_test


def training(x_train, y_train, classifier):
    logging.info(f'Обучение классификатора {classifier.name}...')
    model = classifier.classifier.fit(x_train, y_train)
    return array([model])


if __name__ == '__main__':
    METRIC = metrics.EER
    SCRIPT = scripts.HTK_20
    CLASSIFIERS = [classifiers.SVM_LINEAR, classifiers.SVM_POLY]
    END_CLASSIFIER = classifiers.SVM_LINEAR
    items_bagging = []
    for classifier in CLASSIFIERS:
        items_bagging.append(ItemStacking(classifier, SCRIPT))
    x_train, x_test = get_features(items_bagging)
    y_train, y_test = get_labels(items_bagging)
    model = training(x_train, y_train, classifiers.LDA)
    score = ClassificationSystem.testing(METRIC, model, array([x_test]), y_test)
    print(score)
    print('Success!')
