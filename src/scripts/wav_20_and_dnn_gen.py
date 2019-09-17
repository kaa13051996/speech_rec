from src.code.DNN import DNN
from src.const.network_architectures import CNN_2D
from src.utils.scores import metric_eer_n as metric_eer
from src.const.scripts import WAV_20
from src.const.cfg import TENSORBOARD_LOG
from src.utils.spectrogram import expansion
from src.utils.preprocess_data import shuffle_data

from numpy import array, newaxis

# SPEAKERS = 20, информация в script
WIDTH = 200
ARCH = CNN_2D
CALLBACKS = [TENSORBOARD_LOG]
if __name__ == '__main__':
    # dnn = DNN(dir_features=DIR_FEATURES, architecture=CNN_2D)
    dnn = DNN(script=WAV_20, picture_width=WIDTH, architecture=ARCH, is_preprocessing_list=False, callback=CALLBACKS)

    '''
    Если не использовать генератор для валидации, то нужно предварительно:
    - выровнять матрицу;
    - обрезать матрицу признаков;
    - добавить измерение. 
    Также нужно изменить строчку в функции training_gen.
    '''
    dnn.x_val = DNN.list_to_array(dnn.x_val)
    dnn.x_val = DNN.reduction_to_size(dnn.x_val, 0, WIDTH)
    dnn.x_val = list(dnn.add_dimension(dnn.x_val))[0]

    history = dnn.trainig_gen()

    '''
    Анологично для тестирования.
    Для тестирования лучше использовать не генератор, т.к. для вычисления eer все равно нужен будет array.
    '''
    dnn.x_test = DNN.list_to_array(dnn.x_test)
    dnn.x_test = DNN.reduction_to_size(dnn.x_test, 0, WIDTH)
    dnn.x_test = list(dnn.add_dimension(dnn.x_test))[0]
    loss, acc = dnn.testing()
    # loss, acc = dnn.testing_gen()
    eer = metric_eer(dnn.model, dnn.x_test, dnn.y_test)
    print(f'EER: {eer[0]}\tLOSS: {loss}\tACC: {acc}')
