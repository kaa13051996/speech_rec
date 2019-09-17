from src.const.network_architectures import CNN_2D
from src.code.DNN_50 import DNN_50
from src.code.DNN import DNN
from src.utils.scores import metric_eer_n as metric_eer
from src.utils.save import save_features

if __name__ == '__main__':
    CALLBACS = None
    WIDTH = 200
    EPOCH = 10
    BATCH_SIZE = 20
    EXTACT_FEAT = 1000
    LEFT = None
    dnn = DNN_50(callback=CALLBACS, architecture=CNN_2D, picture_width=WIDTH,
                 epochs=EPOCH, batch_size=BATCH_SIZE, count_extract_features=EXTACT_FEAT, left=LEFT)

    dnn.x_train = DNN.list_to_array(dnn.x_train)
    dnn.x_train = DNN.reduction_to_size(dnn.x_train, dnn.left_border, dnn.right_border)

    dnn.x_val = DNN.list_to_array(dnn.x_val)
    dnn.x_val = DNN.reduction_to_size(dnn.x_val, dnn.left_border, dnn.right_border)

    history = dnn.training()
    features = dnn.extract_features(dnn.features_to_extract)
    save_features(features, name='extract_not_gen.pickle')
