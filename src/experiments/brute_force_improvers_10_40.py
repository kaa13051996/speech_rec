from src.code.ItemStacking import ItemStacking
from src.code.Classification import ClassificationSystem
from src.utils.visualization.visualization_classification import error_visualization
from src.const import metrics, classifiers, scripts
from src.utils.optimal_settings import OptimalSettings

from numpy import mean
from datetime import datetime
import winsound


def display_and_out(title):
    print(title)
    f = open('scores', 'a')
    f.write(title)
    f.close()


PROPORTION = [40, 10]
LIST_PAR = [
    OptimalSettings(normalize=False, mix=False, balance=None),
    OptimalSettings(normalize=True, mix=False, balance=None),
    OptimalSettings(normalize=False, mix=True, balance=None),
    OptimalSettings(normalize=False, mix=False, balance=1),
    OptimalSettings(normalize=True, mix=True, balance=None),
    OptimalSettings(normalize=True, mix=False, balance=1),
    OptimalSettings(normalize=False, mix=True, balance=1),
    OptimalSettings(normalize=True, mix=True, balance=1),
]
DICT_BALANCE = {
    classifiers.SVM_LINEAR: [1, 15],
    classifiers.SVM_POLY: [1, 2],
    classifiers.SVM_RBF: [1, 13],
    classifiers.LDA: [1, 13],
    classifiers.BAYES: [1, 1],
    classifiers.TREE: [1, 1],
    classifiers.ADABOOST: [1, 6]
}

if __name__ == '__main__':
    '''
    Скрипт для перебора дополнительных функций по улучшению при различном соотношении данных (10/40 и 40/10).
    '''
    METRICS = [metrics.EER, metrics.ACCURACY]
    CLASSIFIERS = [classifiers.SVM_POLY, classifiers.SVM_RBF, classifiers.LDA,
                   classifiers.BAYES, classifiers.TREE,
                   classifiers.ADABOOST]
    SCRIPT = scripts.HTK_20
    display_and_out(f'\n{datetime.now()} htk-6 {SCRIPT.recovery_features}\n\n')
    for classifier in CLASSIFIERS:
        display_and_out(f'{classifier.name}\n')
        for par in LIST_PAR:
            if par.balance == 1:
                par.balance = DICT_BALANCE[classifier]
            print(f'\nnorm: {par.normalize}, mix: {par.mix}, bal: {par.balance}\n')
            SCRIPT.model = classifier
            data = ClassificationSystem(script=SCRIPT, is_normalize=par.normalize, is_mix=par.mix,
                                        is_balance=par.balance,
                                        records_train=PROPORTION[0],
                                        records_test=PROPORTION[1])

            EER, ACC = [], []
            for meas in range(1):
                print(f'{meas}')
                # data = ClassificationSystem(script=SCRIPT, is_normalize=par.normalize, is_mix=par.mix,
                #                             is_balance=par.balance,
                #                             records_train=PROPORTION[0],
                #                             records_test=PROPORTION[1])
                ClassificationSystem.training(data)
                eer = ClassificationSystem.testing(metrics.EER, data.train_models, data.x_test, data.y_test)
                print(eer)
                acc = mean(ClassificationSystem.testing(metrics.ACCURACY, data.train_models, data.x_test, data.y_test))
                print(acc)
                EER.append(eer)
                ACC.append(acc)
            display_and_out('\t'.join(map(str, EER)) + '\n')
            display_and_out('\t'.join(map(str, ACC)) + '\n')
            # for metric in METRICS:
            #     score = ClassificationSystem.testing(metric, data.train_models, data.x_test, data.y_test)
            #     if metric != metrics.EER:
            #         score = mean(score)
            #     display_and_out(f'{metric}:{score}\n')
    print('Succes!')
    duration = 1000  # millisecond
    freq = 430  # Hz
    winsound.Beep(freq, duration)
    # METRIC = metrics.EER
    # CLASSIFIER = classifiers.SVM_LINEAR
    # TITLE = f'{METRIC}_{CLASSIFIER.name}'
    # SCRIPT = scripts.HTK_20
    # item = ItemBagging(CLASSIFIER, SCRIPT, METRIC)
    # error_visualization(item.score_train, item.score_test, metric=METRIC, title=TITLE).show()
