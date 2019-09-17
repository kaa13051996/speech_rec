from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.utils.optimal_settings import OptimalSettings
from src.const.paths import MODELS_DIR

__TOL = 0.000001
__MAX_ITER = 100000


class Classifier:
    def __init__(self, name, classifier, path_recovery=None, optimal_settings=None):
        self.name = name
        self.classifier = classifier
        self.path_recovery = path_recovery
        self.optimal_settings = optimal_settings


SVM_LINEAR = Classifier(
    'svm-linear',
    classifier=SVC(kernel='linear', max_iter=__MAX_ITER, tol=__TOL, probability=True),
    optimal_settings=None)  # ,OptimalSettings(normalize=True, mix=True, balance=[1, 5])
# path_recovery=MODELS_DIR / 'svm-linear.joblib')  # 'svmL20_40_10_nm.joblib'

SVM_RBF = Classifier(
    'svm-rbf',
    classifier=SVC(kernel='rbf', max_iter=__MAX_ITER, tol=__TOL, probability=True),
    optimal_settings=None)
# path_recovery=MODELS_DIR / 'svm-rbf.joblib')

SVM_POLY = Classifier(
    'svm-poly',
    classifier=SVC(kernel='poly', max_iter=__MAX_ITER, gamma='auto', tol=__TOL, probability=True),
    optimal_settings=None)

LDA = Classifier(
    'lda',
    classifier=LinearDiscriminantAnalysis(tol=__TOL),
    optimal_settings=None)
# path_recovery=MODELS_DIR / 'lda.joblib')

BAYES = Classifier(
    'bayes',
    classifier=GaussianNB(),
    optimal_settings=None)

TREE = Classifier(
    'tree',
    classifier=DecisionTreeClassifier(random_state=0),
    optimal_settings=None)

ADABOOST = Classifier(
    'adaboost',
    classifier=AdaBoostClassifier(),
    optimal_settings=None)
