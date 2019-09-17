# from const import cfg, metrics
# from const.classifiers import SVM_LINEAR
# from main import get_features, generate_labels, training, testing, plot_save
from matplotlib import pyplot as plt
from src.const.metrics import EER
from numpy import mean, std

from src.utils.save import save_plots


def error_visualization(scores, classifier, metric):
    dpi = 80
    fig = plt.figure(dpi=dpi, figsize=(512 / dpi, 384 / dpi))
    plt.title(f'{classifier.name} {metric}')
    ax = plt.axes()
    ax.yaxis.grid(True, zorder=1)

    data_names = ['10/40', '20/30', '30/20', '40/10']
    train = [score['train'] for score in scores]
    test = [score['test'] for score in scores]
    if metric != EER:
        train = [mean(score) for score in train]
        test = [mean(score) for score in test]
        plt.ylim(0.9, 1)

    xs = range(len(data_names))

    plt.bar([x + 0.05 for x in xs], train,
            width=0.2, color='red', alpha=0.7, label='train',
            zorder=2)
    plt.bar([x + 0.3 for x in xs], test,
            width=0.2, color='blue', alpha=0.7, label='test',
            zorder=2)
    plt.xticks(xs, data_names)
    plt.legend(loc='upper right')
    return plt


if __name__ == '__main__':
    bla = [{'train': 123, 'test': 456}, {'train': 321, 'test': 654}, {'train': 8975, 'test': 4356789}]
    temp = [i['train'] for i in bla]
    temp2 = [i['test'] for i in bla]
    blaaa = 1
#     classifier = SVM_LINEAR
#     metric = metrics.ACCURACY
#     FEATURED_DIRECTORY = r"..\data\signs_htk" if cfg.IS_HTK else r"..\data\signs_csv"
#     scores = []
#     for count in range(10, 50, 10):
#         cfg.COUNT_RECORDS_TRAINING = count
#         cfg.COUNT_RECORDS_TEST = 50 - count
#         features_train = get_features(dir_name=FEATURED_DIRECTORY, start=1, stop=cfg.COUNT_RECORDS_TRAINING)  # 40*20
#         labels_learning = generate_labels(records=cfg.COUNT_RECORDS_TRAINING)  # 20*800
#
#         models = training(features_train, labels_learning, classifier)
#         scores_train = testing(models, features_train, labels_learning, metric=metric)
#
#         features_test = get_features(dir_name=FEATURED_DIRECTORY, start=cfg.COUNT_RECORDS_TRAINING + 1,
#                                      stop=cfg.COUNT_RECORDS_TRAINING + cfg.COUNT_RECORDS_TEST)  # 10*20
#         labels_test = generate_labels(records=cfg.COUNT_RECORDS_TEST)  # 20*200
#
#         scores_test = testing(models, features_test, labels_test, metric=metric)
#         scores.append({'train': scores_train, 'test': scores_test})
#     error_visualization(scores, classifier, metric)
