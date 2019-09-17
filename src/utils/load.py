from os.path import join, exists

from joblib import load
from pickle import load as load_pickle
from keras.models import load_model as lm
from csv import reader, QUOTE_NONNUMERIC
from os.path import splitext

from src.const import paths
from src.const.extensions import F_CSV, F_NPY, F_PIC
from numpy import array, load as ld


def load_network(dir=paths.NETWORKS_DIR, file='network.h5', path_recovery=None):
    '''
    Загружает нс.
    :param dir: каталог, откуда загружать.
    :param file: имя файла.
    :param path_recovery: ну или подать путь целиком.
    :return: если путь и файл существует, то вернуть нс.
    '''
    full_name = path_recovery if path_recovery is not None else join(dir, file)
    return lm(full_name) if check_exist(full_name) else 'Нет файла по данному пути!'


def load_model(dir=paths.MODELS_DIR, file='model_0.joblib', path_recovery=None):
    '''
    Загружает модель.
    :param dir: каталог, откуда загружать.
    :param file: имя файла.
    :param path_recovery: ну или подать путь целиком.
    :return: загруженная модель.
    '''
    full_name = path_recovery if path_recovery is not None else join(dir, file)
    return load(full_name) if check_exist(full_name) else 'Нет файла по данному пути!'


def load_features(dir=paths.READY_TO_USE_FEAT, file='1000.csv'):
    features = []
    full_name = join(dir, file)
    extension = splitext(full_name)[1]
    if not check_exist(full_name):
        raise OSError(f'Файла {full_name} не существует!')
    if extension == F_CSV:
        with open(full_name, 'r', newline='') as csv_file:
            rd = reader(csv_file, delimiter=',', quoting=QUOTE_NONNUMERIC)
            for row in rd:
                features.append(row)
        features = array(features)
    elif extension == F_NPY:
        features = ld(full_name)
    elif extension == F_PIC:
        with open(full_name, 'rb') as file:
            features = load_pickle(file)
    else:
        raise Exception('Нет алгоритма загрузки файла с данным расширением!')
    return features


def check_exist(path):
    '''
    Проверка на существование в ОС.
    :param path: путь до файла.
    :return: да/нет.
    '''
    return exists(path)


if __name__ == '__main__':
    network = load_network()
    model = load_model()
    print(network, model, sep='\n')
