from csv import reader
from struct import unpack
# from src.utils.preprocess_data import expansion_3D  # run as get_spectrogram
# from src.utils.spectrogram import get_spectrogram
from numpy import array
from math import log10
from scipy import signal
from scipy.io import wavfile


def read_csv(files):
    '''
    Считывает файлы-наблюдения формата csv из заданной папки.
    :param files: пути до имени файлов-наблюдений, содержащий 6373 признака в каждом.
    :return: матрица признаков.
    '''
    features = []
    for file in files:
        with open(file, newline='') as csvfile:
            reader_csv = reader(csvfile, delimiter=';')
            next(reader_csv, None)
            for row in reader_csv:
                features.append([float(n) for n in row[2:]])
    return features


def read_matlab(files):
    '''
    Считывает файлы-наблюдения matlab из заданной папки.
    :param files: пути до имени файлов-наблюдений, содержащий 6373 признака в каждом.
    :return: матрица признаков.
    '''
    features = []
    for file in files:
        with open(file, newline='') as csvfile:
            reader_csv = reader(csvfile, delimiter=',')
            for row in reader_csv:
                features.append([float(n) for n in row[0:]])
    return features


def read_htk(files):
    '''
    Считывает файлы-наблюдения формата htk из заданной папки.
    :param files: пути до имени файлов-наблюдений, содержащий 6373 признака в каждом.
    :return: матрица признаков.
    '''
    features = []
    for file in files:
        fin = open(file, 'rb')
        data_bin = fin.read()
        nframes, frate, nbytes, feakind = unpack('>iihh', data_bin[:12])
        ndim = int(nbytes / 4)  # feature dimension (4 bytes per value)
        features.append(unpack('>' + 'f' * ndim, data_bin[12:]))
    return features


def read_spectrogram(files):
    result = []
    # files = [join(WAV, file) for file in listdir(WAV)]
    for file in files:
        sample_rate, data = wavfile.read(file)
        frequencies, times, spectrogram = signal.spectrogram(data, sample_rate, window=('tukey', 0.025))

        temp = []
        for row in spectrogram:
            height = []
            for elem in row:
                height.append(log10(elem) if elem != 0 else 0)
            temp.append(array(height))
            # temp.append(array(list(map(log10, row))))

        result.append(array(temp))
    # return get_spectrogram(files)
    return result
