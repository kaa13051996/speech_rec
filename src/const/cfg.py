from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datetime import datetime
from os.path import join
from keras.callbacks import TensorBoard, CSVLogger

from src.const import classifiers, paths
# from src.utils.load import load_model, load_features
from src.const.scripts import HTK_20, HTK_30, MATLAB, CSV, WAV_20, WAV_30, EX_DNN_20, EX_DNN_50

# настройки
COUNT_SPEAKERS = 20  # 50  # 20  # 30
ALL_RECORDS_SPEAKER = 50  # 10  # 50
COUNT_RECORDS_TRAINING = 40  # 7  # 40
COUNT_RECORDS_VAL = 10
COUNT_RECORDS_TEST = ALL_RECORDS_SPEAKER - COUNT_RECORDS_TRAINING
DIMENSIONALITY = None
TOL = 0.000001
IS_NORMALIZE = True
IS_BALANCE = False
IS_MIX = True
# DIR_FEATURES = paths.WAV

# логирование
NAME_LOG = str(datetime.strftime(datetime.today(), '%H.%M.%S_%d-%m-%y')) + '.log'
CSV_LOGGER = CSVLogger(join(paths.CSV_LOG, NAME_LOG))
TENSORBOARD_LOG = TensorBoard(log_dir=paths.TENSORBOARD_LOG, histogram_freq=1, write_images=True)

# выбор папки, формата признаков, списка дикторов, файл восстановления признаков
# CHOISE = {
#     scripts.HTK_20: HTK_20,
#     scripts.HTK_30: HTK_30,
#     scripts.MATLAB: MATLAB,
#     scripts.CSV: CSV,
#     scripts.WAV_20: WAV_20,
#     scripts.WAV_30: WAV_30,
#     scripts.EX_DNN_20: EX_DNN_20,
#     scripts.EX_DNN_50: EX_DNN_50
# }

# LIST_SPEAKERS = {
#     paths.HTK: [i for i in range(16, 26)] + [j for j in range(41, 51)],
#     paths.MATLAB: [i for i in range(1, COUNT_SPEAKERS + 1)],
#     paths.CSV: [i for i in range(16, 26)] + [j for j in range(41, 51)],
#     paths.WAV: [i for i in range(16, 26)] + [j for j in range(41, 51)],
#     paths.SPECTR: [i for i in range(16, 26)] + [j for j in range(41, 51)]
# }
# FORMAT_FEATURES = {
#     paths.HTK: 'htk',
#     paths.MATLAB: 'csv',
#     paths.CSV: 'csv',
#     paths.WAV: 'signs_wav',
#     paths.SPECTR: 'png'
# }
#
# MODEL_RECOVERY_MAP = {
#     recovery.RECOVERY_MODELS: load_model,
#     recovery.RECOVERY_FEATURES: load_features,
#     recovery.RECOVERY_NONE: []
# }

# CLASSIFIER_MAP = {
#     classifiers.SVM_LINEAR: SVC(kernel='linear', max_iter=100000, tol=TOL, probability=True),
#     classifiers.SVM_RBF: SVC(kernel='rbf', max_iter=5000, tol=TOL, probability=True),
#     classifiers.SVM_POLY: SVC(kernel='poly', max_iter=5000, gamma='auto', tol=TOL, probability=True),
#     classifiers.LDA: LinearDiscriminantAnalysis(tol=TOL),
#     classifiers.BAYES: GaussianNB(),
#     classifiers.TREE: DecisionTreeClassifier(random_state=0),
#     classifiers.ADABOOST: AdaBoostClassifier()
# }
