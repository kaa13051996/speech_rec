from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.models import Sequential
from src.const.cfg import COUNT_SPEAKERS
from keras import regularizers

CNN_2D = 'convolutional neural network train'
CNN_MNIST = 'for check mnist'
CNN_1D = 'для высокоуровневых htk признаков'
PNN = 'fully connected neural network (perceptron neural network)'


def pnn(shape, extract_features=None, nb_classes=20):
    # count_axis = len(shape)
    # if count_axis > 2:
    #     raise Exception('Это архитектура только для 2D данных (не картинок)!')
    # input_dim = (shape[1], ) if count_axis == 2 else (shape[1], shape[2],)
    model = Sequential()
    COUNT_NEURONS_LAYER = 64
    # NB_CLASSES = COUNT_SPEAKERS
    model.add(Dense(units=COUNT_NEURONS_LAYER, input_dim=shape[1], activation='relu'))
    model.add(Dropout(0.5))
    layer = Flatten() if extract_features is None else Dense(units=extract_features, activation='relu')
    model.add(layer)
    model.add(Dense(units=nb_classes, activation='softmax'))
    model.summary()
    return model


def cnn_2D(shape=(129, 200), extract_features=None, nb_classes=20):
    # count_axis = len(shape)
    # if count_axis < 2:
    #     raise Exception('Эта архитектура только для 3D данных (не обычных матриц)!')
    model = Sequential()
    COUNT_NEURONS_LAYER = 64
    NB_CLASSES = COUNT_SPEAKERS
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(shape[0], shape[1], 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())  # 46080
    # kernel_regularizer=regularizers.l2(0.01)
    model.add(Dense(units=extract_features, activation='relu')) if extract_features is not None else None
    model.add(Dense(units=nb_classes, activation='softmax'))
    model.summary()
    return model


def cnn_1D(shape=(6373,), extract_features=None, nb_classes=20):
    model = Sequential()
    COUNT_NEURONS_LAYER = 64
    # NB_CLASSES = COUNT_SPEAKERS
    # model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(count_features, )))
    # model.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
    model.add(Conv1D(100, 10, activation='relu', input_shape=(shape[0], 1)))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    layer = Flatten() if extract_features is None else Dense(units=extract_features, activation='relu')
    model.add(layer)
    model.add(Dense(nb_classes, activation='softmax'))
    print(model.summary())
    return model


def cnn_mnist(shape=(28, 28), extract_features=None, nb_classes=10):
    model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(shape[0], shape[1], 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=nb_classes, activation='softmax'))
    model.summary()
    return model

# def cnn_test(shape=(600, 200)):
#     count_axis = len(shape)
#     if count_axis < 2:
#         raise Exception('Эта архитектура только для 3D данных (не обычных матриц)!')
#     model = Sequential()
#     COUNT_NEURONS_LAYER = 64
#     NB_CLASSES = COUNT_SPEAKERS
#     model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(shape[0], shape[1], 1), activation='relu'))
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(units=NB_CLASSES, activation='softmax'))
#     model.summary()
#     return model
