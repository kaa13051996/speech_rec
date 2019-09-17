from src.const.network_architectures import CNN_2D, CNN_MNIST
from src.const.paths import WAV_30, WAV
from src.code.DNN_50 import DNN_50
from src.code.DNN import DNN
from src.utils.scores import metric_eer_n as metric_eer
from src.utils.save import save_features

if __name__ == '__main__':
    CALLBACS = None  # WAV_20  # [WAV_20, WAV_30]
    WIDTH = 200
    EPOCH = 10
    BATCH_SIZE = 20
    dnn = DNN_50(callback=CALLBACS, architecture=CNN_2D, picture_width=WIDTH,
                 epochs=EPOCH, batch_size=BATCH_SIZE)

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
