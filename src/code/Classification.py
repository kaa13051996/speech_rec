import logging
from copy import copy
from sys import stdout

from numpy import array

from src.code.Model import Model
from src.const import classifiers, metrics, scripts
from src.const.cfg import COUNT_RECORDS_TRAINING, COUNT_RECORDS_TEST
from src.utils.load import load_model
from src.utils.preprocess_data import split_data, shuffle_data, balance_data, expansion, \
    normalize
from src.utils.scores import metric_accuracy, metric_proba, metric_balance_acc, predict, metric_eer

logging.basicConfig(level=logging.DEBUG,
                    format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s')


class ClassificationSystem:
    def __init__(self, script, metric=metrics.EER,
                 dim=None, is_balance=None, is_mix=False, is_normalize=True,
                 records_train=COUNT_RECORDS_TRAINING, records_test=COUNT_RECORDS_TEST):
        self.proportion = (records_train, records_test)
        self.script = script

        if self.script.model.optimal_settings is None:
            self.balance = (self.proportion[0] * is_balance[0], self.proportion[0] * is_balance[1]) \
                if is_balance is not None else None
            self.mix = is_mix
            self.normalize = is_normalize
        else:
            self.balance = (self.proportion[0] * self.script.model.optimal_settings.balance[0],
                            self.proportion[0] * self.script.model.optimal_settings.balance[1]) \
                if is_balance is not None else None
            self.mix = self.script.model.optimal_settings.mix
            self.normalize = self.script.model.optimal_settings.normalize

        self.metric = metric  # надо убрать
        self.classifier = self.script.model
        self.__recovery_model = self.classifier.path_recovery
        self.__models = [Model(id, script=self.script) for id in range(self.script.count_speakers)]
        self.__features = self.__models[0].features
        self.__labels = array([model.labels for model in self.__models])
        del self.__models

        self.x_train, self.x_test, self.y_train, self.y_test = self.get_data()
        self.train_models = load_model(
            path_recovery=self.__recovery_model) if self.__recovery_model is not None else None

    def get_data(self):
        final_x_train, final_x_test, final_y_train, final_y_test = [], [], [], []
        features = self.__features

        # if self.dim is not None and type(features) == ndarray:
        #     features = dimensionality_reduction(self.dim, features)

        if self.normalize:
            features = normalize(features)

        if type(features) == list:
            features = expansion(self.__features)
            features = features.reshape(features.shape[0], features.shape[1] * features.shape[2])

        x_train, x_test = split_data(features, nb_train=self.proportion[0], nb_test=self.proportion[1])
        del self.__features
        final_x_train = [x_train] * self.script.count_speakers
        final_x_test = [x_test] * self.script.count_speakers

        for model in range(self.script.count_speakers):
            y_train, y_test = split_data(self.__labels[model], nb_train=self.proportion[0],
                                         nb_test=self.proportion[1])
            if self.balance is not None:
                final_x_train[model], y_train = balance_data(final_x_train[model], y_train,
                                                             max_true=self.balance[0],
                                                             max_false=self.balance[1])
                # final_x_test[model], y_test = balance_data(final_x_test[model], y_test,
                #                                            max_true=self.balance[1][0],
                #                                            max_false=self.balance[1][1])

            if self.mix:
                final_x_train[model], y_train = shuffle_data(final_x_train[model], y_train)
                # final_x_test[model], y_test = shuffle_data(final_x_test[model], y_test)

            final_y_train.append(y_train)
            final_y_test.append(y_test)

        del self.__labels
        return array(final_x_train), array(final_x_test), array(final_y_train), array(final_y_test)

    @staticmethod
    def training(data):
        print(f'Обучение моделей на {data.classifier.name}:')
        # logging.info(f'Обучение классификатора {data.classifier.name}...')
        train_models = []
        for speaker in range(len(data.x_train)):
            stdout.write(f'\r{speaker}/{len(data.x_train)}.')
            model = data.classifier.classifier.fit(data.x_train[speaker], data.y_train[speaker])
            train_models.append(copy(model))
            # print(f'Обучена {speaker}/{len(data.x_train)} модель.=', flush=True, end='>')
            # print(f'=', flush=True, sep='*', end='>')
            # logging.info(f'Обучена {speaker} модель.')
            stdout.flush()
        data.train_models = copy(array(train_models))
        return array(train_models)

    @staticmethod
    def testing(metric, train_models, x_test, y_test):
        METRICS_MAP = {
            metrics.ACCURACY: metric_accuracy,
            metrics.PROBA: metric_proba,
            metrics.EER: metric_eer,
            metrics.BALANCE_ACC: metric_balance_acc,
            metrics.PREDICT: predict
        }
        scores = METRICS_MAP.get(metric)(train_models, x_test, y_test)
        return scores


if __name__ == '__main__':
    METRIC = metrics.EER
    SCRIPT = scripts.HTK_20
    CLASSIFIER = classifiers.SVM_LINEAR
    data = ClassificationSystem(script=SCRIPT, metric=METRIC)
    models = ClassificationSystem.training(data)
    # save_models(models, name='svmL20_40_10_nm')
    score = ClassificationSystem.testing(data.metric, data.train_models, data.x_test, data.y_test)
    print(score)
