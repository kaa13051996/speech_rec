from sklearn.decomposition import PCA
from keras.utils import normalize as norm_keras
from sklearn.preprocessing import normalize as norm_skl
from numpy import asarray, concatenate, transpose, column_stack, array, isnan, amin, amax, vstack, ndarray
from random import sample
from numpy.random import shuffle
from itertools import chain

from matplotlib import pyplot as plt


# from src.const.cfg import COUNT_SPEAKERS, COUNT_RECORDS_TRAINING, COUNT_RECORDS_TEST


def dimensionality_reduction(dim, features):
    pca = PCA(n_components=dim)
    # pca = PCA(n_components=cfg.DIMENSIONALITY, svd_solver='full')
    reduced = pca.fit_transform(features)

    return reduced


def concatenated_data(x, y):
    '''
    Объедененные данные для обучения ИЛИ тестирования, где в последнем столбце матрицы признаков находятся метки
    присутствия диктора.
    :param x: матрица признаков 20*800|200*6373.
    :param y: матрица меток 20*800|200*1.
    :return: общая матрица данных.
    '''
    data = []
    concatenate(x, transpose([y]), axis=1)
    return asarray(data)


def concatenated_features(features_20, features_30, proportion_20=(30, 10, 10), proportion_30=(40, 10, 0)):
    '''
    Объединяет признаки разных наборов данных с разными дикторами (пример для 20 и 30 дикторов,
    результат - количество наблюдений для всех 50 дикторов):
        train: 20*30+30*40=600+1200=1800;
        val: 20*10+30*10=200+300=500;
        extract: 20*(30+10+10)=1000;
        test: 20*10=200.
    :param features: список признаков для 20 и 30 дикторов.
    :param proportion: пропорции для обучения, валидации, тестирования.
    :return: общий список признаков для 50 дикторов.
    '''
    count_speakers = int(len(features_20) / sum(proportion_20))
    x_train_20, x_test_20 = split_data_wav(features_20,
                                           nb_train=proportion_20[0] + proportion_20[1],
                                           nb_test=proportion_20[2],
                                           count_speakers=count_speakers)
    x_train_20, x_val_20 = split_data_wav(x_train_20,
                                          nb_train=proportion_20[0],
                                          nb_test=proportion_20[1],
                                          count_speakers=count_speakers)

    count_speakers = int(len(features_30) / sum(proportion_30))
    x_train_30, x_val_30 = split_data_wav(features_30,
                                          nb_train=proportion_30[0],
                                          nb_test=proportion_30[1],
                                          count_speakers=count_speakers)

    x_train, x_val, x_test = x_train_20 + x_train_30, x_val_20 + x_val_30, change_order_obs(20, x_train_20, x_val_20,
                                                                                            x_test_20)
    del x_train_20, x_train_30, x_val_20, x_val_30, x_test_20
    # return x_train, x_val, x_test_20
    return x_train, x_val, x_test


def change_order_obs(count_speakers=20, *data):
    separation, grouped = [], []
    for sample in data:
        mass = []
        size = len(sample)
        records = int(size / count_speakers)
        # l = [i for i in range(size)]
        for i in range(0, size, records):
            mass.append(sample[i:i + records])
        separation.append(mass)
    del mass, sample, data

    if type(separation[0][0]) is list:
        for i in range(count_speakers):
            grouped.append(separation[0][i] + separation[1][i] + separation[2][i])
    elif type(separation[0][0]) is ndarray:
        for i in range(count_speakers):
            grouped.append(vstack((separation[0][i], separation[1][i], separation[2][i])))
    else:
        raise Exception('Неизвестный тип для группирования входных данных!')
    del separation

    merged = list(chain.from_iterable(grouped))
    del grouped
    return merged


def balance_data(x, y, max_true=None, max_false=None):
    '''
    Балансировка данных, чтобы не было большого разброса 0 и 1.
    :param x: массив признаков.
    :param y: массив меток 20 дикторов.
    :param max_true: макс количество наблюдений, где присутствует диктор.
    :param max_false: макс количество наблюдений, где диктор отсутствует.
    :var index_true: индексы элементов меток, где диктор присутствует.
    :var index_false: индексы элементов меток, где диктор отсутствует.
    :var short_index_true: из index_true берутся max_true значений (индексов).
    :var short_index_false: из index_false берутся max_false значений (индексов).
    :var final_x: берутся строки из матрицы признаков с индексами short_index_true и short_index_false.
    :var final_y: генерируются массивы 1 и 0 и объединяются, размеров max_true и max_false соответственно.
    :return: ndarray final_x, ndarray final_y
    '''
    index_true = [i for i, e in enumerate(y) if e == 1]
    index_false = [i for i, e in enumerate(y) if e == 0]
    MAX_TRUE = len(index_true) if max_true is None or max_true > len(index_true) else max_true
    MAX_FALSE = len(index_false) if max_false is None or max_false > len(index_false) else max_false
    short_index_true = sample(index_true, MAX_TRUE)
    short_index_false = sample(index_false, MAX_FALSE)
    final_x = ([x[_i] for _i in short_index_true] + [x[_i] for _i in short_index_false])
    final_y = ([1] * MAX_TRUE + [0] * MAX_FALSE)
    final_x, final_y = shuffle_data(array(final_x), array(final_y))
    return final_x, final_y


def mixing_data(x, y):
    '''
    Перемешивание строк (наблюдений) в пределах одного диктора.
    :param y: метки наличия диктора.
    :param x: массив с признаками для 20 дикторов.
    :return: перемешенные признаки (final_x) и соответствующие метки (final_y).
    '''
    combined = column_stack((x, transpose(y)))
    shuffle(combined)
    final_x = (combined[:, :-1])
    final_y = (transpose(combined[:, -1:].astype(int))[0])
    return asarray(final_x), asarray(final_y)


def shuffle_data(x, y):
    type_features = type(x)
    numbered_x = list(enumerate(x))
    shuffle(numbered_x)
    ind, shuffle_x = zip(*numbered_x)
    shuffle_y = []
    for obs in ind:
        shuffle_y.append(y[obs])
    shuffle_y = array(shuffle_y)

    if type_features == list:
        shuffle_x = list(shuffle_x)
    elif type_features == ndarray:
        shuffle_x = array(shuffle_x)
    else:
        raise Exception('Нет алгоритма перемешивания для заданных входных данных.')
    return shuffle_x, shuffle_y


def normalize(features):
    type_mass = type(features)
    axis = features[0].shape
    if type_mass == ndarray and len(axis) == 2:
        features = norm_keras(features, axis=1)
    elif type_mass == ndarray and len(axis) == 1:
        features = norm_skl(features, axis=0)
    elif type_mass == list and len(axis) == 2:
        features = normalize_3D_features(features)
    else:
        raise Exception('Поданы данные для которых нет алгоритма обработки!')
    return features


def normalize_3D_features(features):
    '''
    Нормализация низкоуровневых признаков (частоты по времени). Итерирование по столбцам наблюдений.
    :param features: список признаков.
    :return: нормализованные признаки.
    '''
    x_max, x_min = max_coef(features), min_coef(features)
    sub_max_min = x_max - x_min
    for rec in features:
        for column in range(rec.shape[1]):
            rec[:, column] = ((rec[:, column] - x_min) / sub_max_min) - 0.5
    return features


def max_coef(features):
    '''
    Изначально находится максимум по строкам (например: 1500 наблюдений, 129 частот и разная длина фразы;
    нормализация по частотам). На выходе: матрица наблюдений*количество частот (например: 1500*129).
    После ищется максимум по строкам для всех наблюдений (например, после матрицы 1500*129 на выходе
    получится вектор 129).
    :param features: список признаков.
    :return: вектор нормализации (максимумы).
    '''
    maximum = []
    for rec in features:
        maximum.append(amax(rec, axis=1))

    return array(maximum).max(axis=0)


def min_coef(features):
    minimum = []
    for rec in features:
        minimum.append(amin(rec, axis=1))

    return array(minimum).min(axis=0)


def split_data_wav(data, nb_train=40, nb_test=10, count_speakers=20):
    train, test = [], []
    all_records = nb_train + nb_test
    temp = [data[i:i + all_records] for i in range(0, len(data), all_records)]
    for speaker in range(count_speakers):
        train.append(temp[speaker][:nb_train])
        test.append(temp[speaker][-nb_test:])
    train = list(chain(*train))
    test = list(chain(*test))
    return train, test


def split_data(data, nb_train=40, nb_test=10, count_speakers=20):
    train, test = [], []
    count_axis = len(data.shape)
    all_records = nb_train + nb_test

    # для меток
    if count_axis <= 1:
        temp_data = data.reshape(count_speakers, all_records)
        for speaker in range(count_speakers):
            train.append(temp_data[speaker][:nb_train])
            test.append(temp_data[speaker][-nb_test:])
        train, test = array(train), array(test)
        train = train.reshape(count_speakers * nb_train)
        test = test.reshape(count_speakers * nb_test)
    # для матрицы
    elif count_axis == 2:
        temp_data = data.reshape(count_speakers, all_records, data.shape[1])
        for speaker in range(count_speakers):
            train.append(temp_data[speaker][:nb_train])
            test.append(temp_data[speaker][-nb_test:])
        train, test = array(train), array(test)
        train = train.reshape(count_speakers * nb_train, data.shape[1])
        test = test.reshape(count_speakers * nb_test, data.shape[1])
    # для картинок
    elif count_axis == 3:
        temp_data = data.reshape(count_speakers, all_records, data.shape[1], data.shape[2])
        for speaker in range(count_speakers):
            train.append(temp_data[speaker][:nb_train])
            test.append(temp_data[speaker][-nb_test:])
        train, test = array(train), array(test)
        train = train.reshape(count_speakers * nb_train, data.shape[1], data.shape[2])
        test = test.reshape(count_speakers * nb_test, data.shape[1], data.shape[2])

    else:
        raise Exception('Количество осей больше 3!')

    return train, test


def expansion(lst):
    shape = len(lst[0].shape)
    size_x = [item.shape[1] for item in lst]
    max_x = max(size_x)
    index_no_max = [index for index, item in enumerate(lst) if item.shape[1] != max_x]

    result = []
    for index, item in enumerate(lst):
        if index in index_no_max:
            difference = array([[0] * (max_x - item.shape[1])] * item.shape[0])
            join_array = concatenate((item, difference), axis=1)
            result.append(join_array)
        else:
            result.append(item)
    return array(result)


def expansion_1D(lst):
    shape = len(lst[0].shape)
    size_x = [item.shape[0] for item in lst]
    max_x = max(size_x)
    index_no_max = [index for index, item in enumerate(lst) if item.shape[0] != max_x]

    result = []
    for index, item in enumerate(lst):
        if index in index_no_max:
            difference = array([0] * (max_x - item.shape[0]))
            join_array = concatenate((item, difference), axis=0)
            result.append(join_array)
        else:
            result.append(item)
    return result


def expansion_3D(lst):
    size_x = [item.shape[2] for item in lst]
    max_x = max(size_x)
    index_no_max = [index for index, item in enumerate(lst) if item.shape[2] != max_x]

    result = []
    for index, item in enumerate(lst):
        if index in index_no_max:
            difference = array([[[0] * (max_x - item.shape[2])] * item.shape[1]] * item.shape[0])
            join_array = concatenate((item, difference), axis=2)
            result.append(join_array)
        else:
            result.append(item)
    # if len(set(item.shape[1] for item in lst)) > 1:
    #     result = temp(result)
    return result


def contraction(lst):
    size_x = [item.shape[1] for item in lst]
    min_x = min(size_x)
    index_no_min = [index for index, item in enumerate(lst) if item.shape[1] != min_x]

    result = []
    for index, item in enumerate(lst):
        if index in index_no_min:
            array = item[:, :min_x]
            result.append(array)
        else:
            result.append(item)
    return result


def contraction_1D(lst):
    size_x = [item.shape[0] for item in lst]
    min_x = min(size_x)
    index_no_min = [index for index, item in enumerate(lst) if item.shape[0] != min_x]

    result = []
    for index, item in enumerate(lst):
        if index in index_no_min:
            array = item[:min_x]
            result.append(array)
        else:
            result.append(item)
    return result


def contraction_3D(lst):
    size_x = [item.shape[2] for item in lst]
    min_x = min(size_x)
    index_no_min = [index for index, item in enumerate(lst) if item.shape[2] != min_x]

    result = []
    for index, item in enumerate(lst):
        if index in index_no_min:
            array = item[:, :, :min_x]
            result.append(array)
        else:
            result.append(item)
    return result
