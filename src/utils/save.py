from src.const import paths
from src.const.extensions import F_NPY, F_PIC, F_CSV
from joblib import dump
from keras.utils import plot_model
from os.path import join, splitext
from csv import writer, QUOTE_NONNUMERIC
from numpy import save
from pickle import dump as dump_pickle


def save_models(clf, dir=paths.MODELS_DIR, name='model'):
    '''
    Сораняет модели (предполагается, что обученные).
    :param clf: модель.
    :param dir: каталог, куда сохранять.
    :param name: имя модели.
    '''
    dump(clf, f'{dir}/{name}.joblib')


def save_networks(network, dir=paths.NETWORKS_DIR, name='network'):
    '''
    Сохраняет нейронные сети (предполагается, что обученные).
    :param network: нс.
    :param dir: каталог, куда сохранять.
    :param name: имя нс.
    '''
    network.save(f'{dir}/{name}.h5')


def save_plots(plt, dir=paths.PLOTS_DIR, name='picture'):
    '''
    Сохраняет графики.
    :param plt: график.
    :param dir: каталог, куда сохранять.
    :param name: имя графика.
    '''
    plt.savefig(f'{dir}/{name}.png', bbox_inches='tight')


def save_structure_network(nn, dir=paths.PLOTS_DIR, name='model.png'):
    plot_model(nn, to_file=join(dir, name), show_shapes=True, show_layer_names=True)


def save_features(feat, dir=paths.READY_TO_USE_FEAT, name='1000_htk.csv'):
    extension = splitext(name)[1]
    full_name = join(dir, name)
    if extension == F_CSV:
        with open(full_name, 'w', newline='', encoding='utf-8') as csv_file:
            wr = writer(csv_file, delimiter=',', quoting=QUOTE_NONNUMERIC)
            wr.writerows(feat)
    elif extension == F_PIC:
        with open(full_name, 'wb') as file:
            dump_pickle(feat, file)
    elif extension == F_NPY:
        save(full_name, feat)
    else:
        raise Exception('Данное расширение не поддерживается!')


if __name__ == '__main__':
    print('Imports success!')
