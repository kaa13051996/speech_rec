from matplotlib import pyplot as plt
from matplotlib.pyplot import specgram
from math import log10
from scipy import signal
from scipy.io import wavfile
from os import listdir
from os.path import join
from os.path import basename
from numpy import array, amax, amin, isnan

from src.const.paths import WAV
from src.utils.save import save_plots
from src.utils.preprocess_data import expansion, contraction, normalize_3D_features as normalize

'''http://www.dsplib.ru/content/win/win.html
- было принято решение не исопльзовать добивание и укорачивание для создания модели;
'''


def get_spectrogram(files, is_visual=False, is_save=False):
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
        # result.append(spectrogram)

        if not is_visual and is_save:
            raise Warning('Сохранения не будет, т.к. is_visual==False!')

        if is_visual:
            # plt.subplot(211)
            # plt.title('Spectrogram of a signs_wav file with piano music')
            # plt.plot(data)
            # plt.xlabel('Sample')
            # plt.ylabel('Amplitude')

            plt.subplot(212)
            # ax = plt.axes()
            # ax.set_axis_off()
            spectrum, freqs, t, im = plt.specgram(data, Fs=sample_rate, xextent=None, scale_by_freq=False)
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            # plt.axis('off')

            # plt.pcolormesh(times, frequencies, spectrogram)
            # plt.imshow(spectrogram)
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')

            # plt.show()
            save_plots(plt, name=basename(file)[:-4]) if is_save else None
            plt.clf()
            plt.close()
    return result  # asarray(result, dtype=float32)


def run(files, is_visual=False, is_save=False, is_expansion=True):
    '''
    Может стоит реализовать какую-то более сложную логику, иначе перенести в Speaker.
    :param files:
    :param is_visual:
    :param is_save:
    :param is_expansion:
    :return:
    '''
    data = get_spectrogram(files, is_visual=is_visual, is_save=is_save)
    result = expansion(data) if is_expansion else contraction(data)
    return array(result)


if __name__ == '__main__':
    # cap_1 = array([[1, 2], [99, 5], [6, 7]])
    # cap_2 = array([[22, 11, 33], [44, 55, 66], [77, 88, 99]])
    # cap_3 = array([[111, 222, 555, 444], [555, 222, 333, 9], [111, 222, 333, 444]])
    # cap_4 = array([[6666666, 222, 333, 444], [111, 2123422, 333, 444], [111, 222, 333, 444]])
    # cap_5 = array([[1, 35462], [4, 5], [-10, 7]])
    # ls = [cap_1, cap_2, cap_3, cap_4, cap_5]
    # data_1 = array(expansion(ls), dtype=float)
    # print(data_1)
    #
    # cap_6 = array([[9, 10, 11], [8, 9, 10], [10, 11, 12]])
    # cap_7 = array([[99, 1010, 1111, 1212], [99, 1010, 1111, 1212], [99, 1010, 1111, 1212]])
    # cap_8 = array(
    #     [[999, 101010, 111111, 888, 1212], [999, 101010, 111111, 888, 1212], [999, 101010, 111111, 888, 1212]])
    # cap_9 = array(
    #     [[999, 101010, 111111, 888, 1212], [999, 101010, 111111, 888, 1212], [999, 101010, 111111, 888, 1212]])
    # cap_10 = array([[9, 10, 1212], [8, 9, 1212], [10, 11, 1212]])
    # ls_2 = [cap_6, cap_7, cap_8, cap_9, cap_10]
    # data_2 = array(expansion(ls_2), dtype=float)
    #
    # data_3D = [data_1, data_2]
    # data_3D = normalize(data_3D)
    # print(data_3D)

    files = get_spectrogram([join(WAV, file) for file in listdir(WAV)], True, True)
    # data = run(files)
    # print(data)
