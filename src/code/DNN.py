from random import randint

from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from numpy import array, newaxis, concatenate, zeros, ndarray

from src.code.Model import Model
from src.const.network_architectures import PNN, CNN_2D, CNN_MNIST, CNN_1D, pnn, cnn_2D, cnn_mnist, cnn_1D
from src.const.settings_for_generator import Options as GenSettings
from src.utils.preprocess_data import split_data, split_data_wav, shuffle_data, normalize
from src.utils.spectrogram import expansion


class DNN(Model):
    def __init__(self, script, picture_width, callback=None,
                 is_mix=True, is_normalize=True, architecture=PNN,
                 count_extract_features=1000, is_preprocessing_list=False, left=10):
        super().__init__(id=0, script=script)
        self.mix = is_mix
        self.normalize = is_normalize
        self.type_features = type(self.features)
        self.preprocessing_list = is_preprocessing_list
        self.recovery_model = False
        self.callback = callback

        if self.normalize:
            self.features = normalize(self.features)
        if self.preprocessing_list and self.type_features == list:
            self.features = DNN.list_to_array(self.features)
            self.type_features = type(self.features)

        self.x_train, self.x_val, self.x_test = self.get_data(self.features)
        self.y_train, self.y_val, self.y_test = self.get_data(self.labels)

        self.axis = len(self.x_train[0].shape)
        self.shape = self.get_shape(picture_width)
        # if self.type_features == ndarray:
        #     self.x_train, self.x_val, self.x_test = list(self.add_dimension(self.x_train, self.x_val, self.x_test))

        if self.mix:
            self.x_train, self.y_train = shuffle_data(self.x_train, self.y_train)
            self.x_val, self.y_val = shuffle_data(self.x_val, self.y_val)
            self.x_test, self.y_test = shuffle_data(self.x_test, self.y_test)

        self.epochs = 10
        self.batch_size = 10
        self.steps = None
        self.gen_param_train = GenSettings(len(self.x_train), self.epochs, self.batch_size, self.steps)  # opt_train
        self.gen_param_val = GenSettings(len(self.x_val), self.epochs, self.batch_size, self.steps)  # opt_val
        self.gen_param_test = GenSettings(len(self.x_test), self.epochs, self.batch_size, self.steps)  # opt_test

        self.left_border = randint(0, max([item.shape[1] for item in self.x_train])) if left is None else left
        self.right_border = self.left_border + picture_width

        self.count_extract_features = count_extract_features
        self.model = self.get_model(architecture)

        del self.features, self.labels

    def get_shape(self, picture_width):
        picture_height = len(self.x_train[0])
        if self.axis == 2:
            picture_width = picture_width if picture_width is not None else \
                max([item.shape[1] for item in self.features])
            shape = (picture_height, picture_width)
        elif self.axis == 1:
            shape = (picture_height,)
        else:
            raise Exception('Необработанное количество измерений.')
        return shape

    def get_labels(self, records=50, COUNT_SPEAKERS=20):
        labels = []
        for speaker in range(COUNT_SPEAKERS):
            for record in range(50):
                labels.append(speaker)
        labels = to_categorical(labels, COUNT_SPEAKERS, dtype=int)
        return labels

    def get_data(self, data, COUNT_RECORDS_TRAINING=40, COUNT_RECORDS_TEST=10, COUNT_RECORDS_VAL=10):
        new_count_train = COUNT_RECORDS_TRAINING - COUNT_RECORDS_VAL
        if type(data) == list:
            train, test = split_data_wav(data, nb_train=COUNT_RECORDS_TRAINING, nb_test=COUNT_RECORDS_TEST)
            train, val = split_data_wav(train, nb_train=new_count_train, nb_test=COUNT_RECORDS_VAL)
        elif type(data) == ndarray:
            train, test = split_data(data, nb_train=COUNT_RECORDS_TRAINING, nb_test=COUNT_RECORDS_TEST)
            train, val = split_data(train, nb_train=new_count_train, nb_test=COUNT_RECORDS_VAL)
        else:
            raise Exception('Нет алгоритма обработки входных данных.')
        return train, val, test

    def get_model(self, architecture):
        if self.recovery_model:
            pass
        else:
            NETWORK_ARCHITECTURE_MAP = {
                CNN_2D: cnn_2D,
                CNN_MNIST: cnn_mnist,
                CNN_1D: cnn_1D,
                PNN: pnn
            }
            self.model = NETWORK_ARCHITECTURE_MAP.get(architecture)(self.shape, self.count_extract_features,
                                                                    self.count_speakers)
            self.compilation()
            return self.model

    def compilation(self):
        '''
        Компиляция сети.
        :param model:
        :return:
        '''
        opt = Adadelta()  # Adam()  # Adadelta()  # SGD()
        self.model.compile(loss=categorical_crossentropy,
                           optimizer=opt,
                           metrics=['accuracy'])

    def training(self):
        self.x_train, self.x_val = list(self.add_dimension(self.x_train, self.x_val))
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=self.epochs,
            verbose=1,
            batch_size=self.batch_size,
            validation_data=(self.x_val, self.y_val),
            shuffle=self.mix,
            callbacks=self.callback
        )
        self.model = history.model
        return history

    def trainig_gen(self):
        history = self.model.fit_generator(
            # generator=self.gen_train(self.x_train, self.y_train),
            generator=self.generator_data(self.x_train, self.y_train, self.gen_param_train),
            epochs=self.epochs,
            steps_per_epoch=self.gen_param_train.steps,
            validation_steps=self.gen_param_val.steps,
            # validation_data=self.gen_val_test(self.x_val, self.y_val, 10, 10 + self.__picture_width,
            #                                   self.gen_param_val.init_interval),
            validation_data=(self.x_val, self.y_val),
            shuffle=self.mix,
            callbacks=self.callback
        )
        self.model = history.model
        return history

    def testing(self):
        self.x_test = list(self.add_dimension(self.x_test))
        loss, acc = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size, callbacks=self.callback)
        return loss, acc

    def testing_gen(self):
        loss, acc = self.model.evaluate_generator(
            # generator=self.gen_val_test(self.x_test, self.y_test, 10, 10 + self.picture_width,
            #                             self.gen_param_test.init_interval),
            generator=self.generator_data(self.x_train, self.y_train, self.gen_param_test, 0, self.shape[1]),
            steps=self.gen_param_test.steps  # , callbacks=self.callback
        )
        return loss, acc

    def generator_data(self, features, labels, param, left_border=None, right_border=None):
        for epoch in range(self.epochs):
            features, labels = shuffle_data(features, labels)
            for step in range(param.steps):
                selected_features = features[param.init_interval[0]:param.init_interval[1]]
                batch_labels = array(labels[param.init_interval[0]:param.init_interval[1]])

                self.check_list_to_array(selected_features)
                batch_features = array(self.cut_features(selected_features, left_border, right_border))

                # if self.normalize:
                #     batch_features = normalize(batch_features)
                batch_features = batch_features[:, :, :, newaxis]

                param.init_interval[0] = param.init_interval[0] + param.batch_size
                param.init_interval[1] = param.init_interval[1] + param.batch_size

                # for _ in range(self.batch_size):
                #     yield batch_features, batch_labels
                yield batch_features, batch_labels
            param.init_interval = [0, param.batch_size]

    def cut_features(self, features, left_border=None, right_border=None):
        '''
        Вырезает разные временные промежутки заданной длины.
        '''
        batch_features = []
        target_weight = self.shape[1]  # self.picture_width
        picture_height = self.shape[0]
        for observation in features:
            original_weight = observation.shape[1]
            if left_border is None and right_border is None:
                left_border = randint(0, original_weight)
                right_border = left_border + target_weight
            if right_border > original_weight:
                difference = zeros((picture_height, right_border - original_weight))
                temp = concatenate((observation, difference), axis=1)
                batch_features.append(temp[:, left_border:right_border])
            else:
                batch_features.append(observation[:, left_border:right_border])
        return batch_features

    @staticmethod
    def reduction_to_size(features, left_border, right_border):
        batch_features = []
        for observation in features:
            height, weight = observation.shape[0], observation.shape[1]
            if right_border > weight:
                difference = array([[0] * (right_border - weight)] * height)
                temp = concatenate((observation, difference), axis=1)
                batch_features.append(temp[:, left_border:right_border])
            else:
                batch_features.append(observation[:, left_border:right_border])
        batch_features = array(batch_features)
        return batch_features

    @staticmethod
    def check_list_to_array(features):
        uniq_height = set([obs.shape[0] for obs in features])
        if len(uniq_height) > 1:
            raise Exception('Не преобразуется в numpy array, т.к. размерность картинки (высота) не одинаковая.')

    @staticmethod
    def list_to_array(arr):
        return array(expansion(arr))

    def extract_features(self, features):
        self.model.pop()
        x_test = array(expansion(features))[:, :, :, newaxis]
        x_test = DNN.reduction_to_size(x_test, 10, 210)
        features = self.model.predict(x_test)
        return features

    def add_dimension(self, *data):
        for ind in range(len(data)):
            if self.axis == 1:
                yield data[ind][:, :, newaxis]
            elif self.axis == 2:
                yield data[ind][:, :, :, newaxis]
            else:
                yield data[ind]

