from src.code.ItemBagging import ItemBagging, get_features, get_labels, training
from src.code.Classification import ClassificationSystem
from src.const import metrics, classifiers
from src.const.scripts import HTK_20, EX_DNN_50
# from src.experiments.brute_force_improvers_10_40 import Parameters
from src.utils.optimal_settings import OptimalSettings
from numpy import array, mean
from datetime import datetime
import winsound
from src.utils.preprocess_data import shuffle_data
from random import seed


def display_and_out(title):
    print(title)
    f = open('scores_stacking', 'a')
    f.write(title)
    f.close()


def bagging_10_40():
    PROPORTION = [10, 40]
    CLASSIFIERS = [classifiers.SVM_LINEAR, classifiers.SVM_RBF]
    # PAR = [Parameters(True, True, [PROPORTION[0] * 2, PROPORTION[1] * 2], None),
    #        Parameters(True, False, None, None)]
    PAR = [OptimalSettings(True, True, None),
           OptimalSettings(True, False, None)]
    return PROPORTION, CLASSIFIERS, PAR


def bagging_40_10():
    PROPORTION = [40, 10]
    CLASSIFIERS = [classifiers.SVM_LINEAR, classifiers.SVM_RBF, classifiers.LDA]
    PAR = [OptimalSettings(True, False, None),
           OptimalSettings(True, False, None),
           OptimalSettings(False, False, None)]
    return PROPORTION, CLASSIFIERS, PAR


if __name__ == '__main__':
    METRICS = [metrics.EER, metrics.ACCURACY]
    END_CLASSIFIERS = [classifiers.SVM_LINEAR, classifiers.SVM_POLY, classifiers.SVM_RBF, classifiers.LDA,
                       classifiers.TREE, classifiers.BAYES, classifiers.ADABOOST]
    SCRIPT = HTK_20
    BAGGING = [bagging_40_10]  # bagging_40_10,
    display_and_out(f'\n{datetime.now()} {SCRIPT.recovery_features}\n\n')
    for script_bagging in BAGGING:
        PROPORTION, CLASSIFIERS, PAR = script_bagging()
        display_and_out(f'\nprop: {PROPORTION[0]}/{PROPORTION[1]}\n')
        items_bagging = []
        for item in range(len(CLASSIFIERS)):
            display_and_out(
                f'classifier: {CLASSIFIERS[item].name}, norm: {PAR[item].normalize}, mix: {PAR[item].mix}, bal: '
                f'{PAR[item].balance}\n')
            items_bagging.append(
                ItemBagging(CLASSIFIERS[item], SCRIPT, metric=metrics.PROBA, nb_train=PROPORTION[0],
                            nb_test=PROPORTION[1], is_normalize=PAR[item].normalize, is_mix=PAR[item].mix,
                            is_balance=PAR[item].balance))
            # save_models(items_bagging[item], name=f'bag_prop{PROPORTION[0]}{PROPORTION[1]}_{CLASSIFIERS[item].name}')
        x_train, x_test = get_features(items_bagging)
        y_train, y_test = get_labels(items_bagging)
        x_train, y_train = shuffle_data(x_train, y_train)
        x_test, y_test = shuffle_data(x_test, y_test)
        for end_classifier in END_CLASSIFIERS:
            # display_and_out(f'end classifier: {end_classifier.name}\n')
            EER, ACC = [], []
            for mes in range(1):
                print(mes)
                model = training(x_train, y_train, end_classifier)
                eer = ClassificationSystem.testing(metrics.EER, model, array([x_test]), array([y_test]))
                print(eer)
                acc = mean(ClassificationSystem.testing(metrics.ACCURACY, model, array([x_test]), array([y_test])))
                print(acc)
                EER.append(eer)
                ACC.append(acc)
                del model
            display_and_out('\t'.join(map(str, EER)) + '\n')
            display_and_out('\t'.join(map(str, ACC)) + '\n')

            # model = training(x_train, y_train, end_classifier)
            # for metric in METRICS:
            #     score = ClassificationSystem.testing(metric, model, array([x_test]), array([y_test]))
            #     if metric != metrics.EER:
            #         score = score[0]
            #     display_and_out(f'metric: {metric}, score: {score}\n')
    print('Succes!')
    duration = 1000  # millisecond
    freq = 430  # Hz
    winsound.Beep(freq, duration)
