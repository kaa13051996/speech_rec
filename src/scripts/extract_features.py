from numpy import array, newaxis
from src.code.DNN_50 import DNN_50
from src.code.DNN import DNN
from src.const.network_architectures import CNN_2D
from src.utils.preprocess_data import expansion
from src.utils.save import save_features, save_networks
from src.utils.visualization.visualization_nn import run as visualization
from src.utils.save import save_structure_network


def display_and_out(title):
    print(title)
    f = open('brute_batch', 'a')
    f.write(title)
    f.close()


if __name__ == '__main__':
    CALLBACS = None  # WAV_20  # [WAV_20, WAV_30]
    for _ in range(1):
        WIDTH = 200
        EPOCH = 10
        BATCH_SIZE = 300
        EXTACT_FEAT = 1000
        LEFT = None
        dnn = DNN_50(callback=CALLBACS, architecture=CNN_2D, picture_width=WIDTH,
                     epochs=EPOCH, batch_size=BATCH_SIZE, count_extract_features=EXTACT_FEAT, left=LEFT)

        '''
        Если не использовать генератор для валидации, то нужно предварительно:
        - выровнять матрицу;
        - обрезать матрицу признаков;
        - добавить измерение.
        Также нужно изменить строчку в функции training_gen.
        '''
        dnn.x_val = DNN.list_to_array(dnn.x_val)
        dnn.x_val = DNN.reduction_to_size(dnn.x_val, dnn.left_border, dnn.right_border)
        dnn.x_val = list(dnn.add_dimension(dnn.x_val))[0]

        history = dnn.trainig_gen()
        acc = history.history['val_acc']
        max_acc = max(acc)
        head = f'{WIDTH}w_{EPOCH}e_{BATCH_SIZE}b_{EXTACT_FEAT}ex'
        display_and_out(head + '\n')
        display_and_out('\t'.join(map(str, acc)) + '\n')

        # visualization(history.history)
        # save_structure_network(dnn.model, name='extract1000_model.png')
        # save_networks(history.model, name='train_NN_for_extract')
        if max_acc > 0.03:
            features = dnn.extract_features(dnn.features_to_extract)
            save_features(features, name=f'extract_big_neur_{acc}val_acc.pickle')
