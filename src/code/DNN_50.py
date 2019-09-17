from random import randint

from keras.utils import to_categorical
from numpy import vstack, hstack
from numpy import zeros

from src.code.DNN import DNN
from src.const.scripts import WAV_20, WAV_30
# from src.const.paths import WAV_20
from src.const.settings_for_generator import Options as GenSettings
from src.utils.load import load_features
from src.utils.preprocess_data import concatenated_features, change_order_obs
from src.utils.preprocess_data import split_data, shuffle_data, normalize

'''
- для создания архитектуры в файле src/const/network_architectures.py был изменен COUNT_SPEAKER с 20 на 30;
'''


class DNN_50(DNN):
    def __init__(self, callback, architecture, picture_width=200, epochs=10, batch_size=10, steps=None,
                 count_extract_features=1000, left=10):
        self.proportion_20 = (30, 10, 10)
        self.proportion_30 = (40, 10, 0)
        self.mix = True
        self.recovery_model = False
        self.count_speakers = 50
        self.callback = None
        self.features_to_extract = None
        self.x_train, self.x_val, self.x_test = self.get_features()
        self.y_train, self.y_val, self.y_test = self.get_labels(50)

        self.axis = len(self.x_train[0].shape)
        self.shape = self.get_shape(picture_width)

        if self.mix:
            self.x_train, self.y_train = shuffle_data(self.x_train, self.y_train)
            self.x_val, self.y_val = shuffle_data(self.x_val, self.y_val)
            self.x_test, self.y_test = shuffle_data(self.x_test, self.y_test)

        self.epochs = epochs
        self.batch_size = batch_size
        self.steps = steps
        self.gen_param_train = GenSettings(len(self.x_train), self.epochs, self.batch_size, self.steps)  # opt_train
        self.gen_param_val = GenSettings(len(self.x_val), self.epochs, self.batch_size, self.steps)  # opt_val
        self.gen_param_test = GenSettings(len(self.x_test), self.epochs, self.batch_size, self.steps)  # opt_test

        self.left_border = randint(0, max([item.shape[1] for item in self.x_train])) if left is None else left
        self.right_border = self.left_border + picture_width
        # self.architecture = architecture
        self.count_extract_features = count_extract_features
        self.model = self.get_model(architecture)

    def get_labels(self, records=50):
        '''
        Генерирует метки для 20 и 30 дикторов (разделяя их для обучения, валидации, тестирования),
        добавляет нулевые столбцы до 50 дикторов.
        :param records: ненужный параметры, для совместимости с классом DNN.
        :return: y_train, y_val, y_test
        '''
        speakers_20, speakers_30 = 20, 30
        labels_20 = formation_labels(speakers_20, records)
        labels_20 = hstack((labels_20, zeros((len(labels_20), speakers_30), dtype=int)))
        y_train_20, y_test_20 = split_data(labels_20, nb_train=40, nb_test=10, count_speakers=speakers_20)
        y_train_20, y_val_20 = split_data(y_train_20, nb_train=30, nb_test=10, count_speakers=speakers_20)

        labels_30 = formation_labels(speakers_30, records)
        labels_30 = hstack((zeros((len(labels_30), speakers_20), dtype=int), labels_30))
        y_train_30, y_val_30 = split_data(labels_30, nb_train=40, nb_test=10, count_speakers=30)

        y_train = vstack((y_train_20, y_train_30))
        y_val = vstack((y_val_20, y_val_30))
        # y_test = vstack((y_train_20, y_val_20, y_test_20))
        y_test = change_order_obs(20, y_train_20, y_val_20, y_test_20)

        return y_train, y_val, y_test

    def get_features(self):
        features_20 = load_features(file=WAV_20.recovery_features)
        features_30 = load_features(file=WAV_30.recovery_features)
        all_features = features_20 + features_30
        all_features = normalize(all_features)
        normalize_20 = all_features[:1000]
        normalize_30 = all_features[1000:]
        self.features_to_extract = normalize_20
        x_train, x_val, x_test = concatenated_features(normalize_20, normalize_30, self.proportion_20,
                                                       self.proportion_30)
        return x_train, x_val, x_test


def formation_labels(count_speakers, records=50):
    labels = []
    for speaker in range(count_speakers):
        for record in range(records):
            labels.append(speaker)
    return to_categorical(labels, count_speakers, dtype=int)
