from src.code.Classification import ClassificationSystem
from src.const import metrics, classifiers, scripts
import winsound


def display_and_out(title):
    print(title)
    f = open('eer_balance', 'a')
    f.write(title)
    f.close()


if __name__ == '__main__':
    METRIC = metrics.EER
    CLASSIFIERS = [classifiers.SVM_LINEAR, classifiers.SVM_POLY, classifiers.SVM_RBF, classifiers.LDA, classifiers.TREE,
                   classifiers.BAYES, classifiers.ADABOOST]
    SCRIPT = scripts.EX_DNN_50
    PROPORTION = [40, 10]
    BALANCE = [[1, 1],
               [1, 2],
               [1, 3],
               [1, 4],
               [1, 5],
               [1, 6],
               [1, 7],
               [1, 8],
               [1, 9],
               [1, 10],
               [1, 11],
               [1, 12],
               [1, 13],
               [1, 14],
               [1, 15],
               [1, 16],
               [1, 17],
               [1, 18],
               [1, 19]]

    for classifier in CLASSIFIERS:
        display_and_out(f'{classifier.name}\n')
        for balance in BALANCE:
            print(balance)
            # display_and_out(f'{balance}\n')
            # for classifier in CLASSIFIERS:
            SCRIPT.model = classifier
            data = ClassificationSystem(script=SCRIPT, metric=METRIC, is_normalize=True, is_mix=False,
                                        is_balance=balance, dim=None,
                                        records_train=PROPORTION[0],
                                        records_test=PROPORTION[1])
            ClassificationSystem.training(data)
            score = ClassificationSystem.testing(data.metric, data.train_models, data.x_test, data.y_test)
            display_and_out(f'{score}\n')
    print('Succes!')
    duration = 1000  # millisecond
    freq = 430  # Hz
    # winsound.Beep(freq, duration)
    # METRIC = metrics.EER
    # CLASSIFIER = classifiers.SVM_LINEAR
    # TITLE = f'{METRIC}_{CLASSIFIER.name}'
    # SCRIPT = scripts.HTK_20
    # item = ItemBagging(CLASSIFIER, SCRIPT, METRIC)
    # error_visualization(item.score_train, item.score_test, metric=METRIC, title=TITLE).show()
