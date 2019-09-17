from src.const import metrics, classifiers, scripts
from src.code.ItemStacking import ItemStacking
from src.utils.visualization.plot_curve.plot_train_test import error_visualization
from src.utils.save import save_plots

if __name__ == '__main__':
    '''
    Скрипт сохраняет картинку того, как разное соотношение данных на обучение и тест влияет на значение метрики. 
    Изначальное соотношение 10/40, потом изменение с шагом в 10 до соотношения 40/10.
    '''
    classifier = classifiers.SVM_LINEAR
    metric = metrics.ACCURACY
    script = scripts.HTK_20
    scores = []
    for count in range(10, 50, 10):
        nb_train = count
        nb_test = 50 - count
        item = ItemStacking(classifier, script, metric, nb_train=nb_train, nb_test=nb_test, is_normalize=True,
                            is_balance=None, is_mix=True, dim=None)
        scores.append({'train': item.score_train, 'test': item.score_test})
    plt = error_visualization(scores, classifier, metric)
    plt.show()
    # save_plots(plt, name=f'{script.count_speakers}_{metric}_{classifier.name}')
