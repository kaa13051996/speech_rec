from numpy import array
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.utils.tDCF_python_v1.eval_metrics import compute_eer

'''
Данный модуль содержит функции для рассчета метрик, где:
- models (list): список моделей;
- X (numpy array): матрица признаков, где строка == наблюдение == аудиозапись, столбец == признак;
- y (numpy array): одномерная матрица метрок принадлежности аудиозаписи диктору.
Подробнее о каждой метрике (кроме EER) можно почитать в документации:
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
Функция "metric_eer_n" считает EER для нейронной сети.
'''


def metric_accuracy(models, X, y):
    scores = []
    for i in range(len(models)):
        predicted = models[i].predict(X[i])
        score = accuracy_score(y_true=y[i], y_pred=predicted)
        scores.append(score)
    return scores


def metric_proba(models, X, y):
    scores = []
    for model, _x in zip(models, X):
        proba = model.predict_proba(_x)  # апостериорные вероятности
        for record in proba:
            scores.append(record[1])
    return scores


def metric_balance_acc(models, X, y):
    scores = []
    for i in range(len(models)):
        predicted = models[i].predict(X[i])
        scores.append(balanced_accuracy_score(y[i], predicted))
    return scores


def predict(models, X, y):
    scores = []
    for model, _x in zip(models, X):
        scores.append(model.predict(_x))  # предсказанный класс
    return scores


def metric_eer(models, X, y):
    scores = metric_proba(models, X, y)
    ones_scores, zeros_scores = separated_scores(y, scores)
    scores = compute_eer(ones_scores, zeros_scores)
    return scores[0]


def separated_scores(y, scores_predict):
    ones_scores, zeros_scores = [], []
    y = y.ravel()
    for index in range(len(y)):
        score = scores_predict[index]
        ones_scores.append(score) if y[index] == 1 else zeros_scores.append(score)
    return array(ones_scores), array(zeros_scores)


def metric_eer_n(models, X, y):
    scores = models.predict(X).ravel()
    ones_scores, zeros_scores = separated_scores(y, scores)
    scores = compute_eer(ones_scores, zeros_scores)
    return scores[0]
