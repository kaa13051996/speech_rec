from src.code.DNN import DNN
from src.const.network_architectures import CNN_2D
from src.utils.scores import metric_eer_n as metric_eer
from src.const.scripts import WAV_30

'''
Использует низкоуровневые признаки 20 дикторов для обучения нейронной сети.
Генератор не используется (picture_width=None), список расширяется.
'''

# SPEAKERS = 30
if __name__ == '__main__':
    dnn = DNN(script=WAV_30, picture_width=None, architecture=CNN_2D,
              is_preprocessing_list=True)
    history = dnn.training()
    loss, acc = dnn.testing()
    eer = metric_eer(dnn.model, dnn.x_test, dnn.y_test)
    print(f'EER: {eer[0]}\tLOSS: {loss}\tACC: {acc}')
