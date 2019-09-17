from os.path import join
# from csv import reader
# from struct import unpack
from numpy import array
from itertools import chain

# from src.const.cfg import CHOISE, FORMAT_FEATURES, LIST_SPEAKERS
from src.const.paths import HTK, WAV, WAV_30, READY_TO_USE_FEAT
# from src.const.recovery import CSV as F_CSV, BIN as F_BIN
# from src.const.cfg import CHOISE, DIR_FEATURES as DF  # , F_CSV, F_BIN
from src.utils.save import save_features as sf
from src.utils.preprocess_data import expansion_3D, contraction_3D
from src.const.extensions import F_CSV, F_PIC, F_NPY
from src.const import scripts


class Speaker:

    def __init__(self, id, script):
        self.id = id
        self.script = script
        self.__dir_features = self.script.directory
        self.__features_format = self.script.format
        self.features = self.extract_features()

    def extract_features(self):
        if self.script.is_extract_features:
            list_files = self.list_files()
            features = self.script.def_read(list_files)
            try:
                features = array(features)
            except ValueError:
                pass
        else:
            raise Exception(
                f'Признаки из данной папки ({self.__dir_features}) предназначены только для чтения, '
                f'а не для извлечения!')
        return features

    def list_files(self, start=1, stop=50):
        files = []
        for record in range(start, stop + 1):
            files.append(join(self.__dir_features, f'{self.id} ({record}).{self.__features_format}'))
        return files


def list_to_array(speakers, is_expansion=True):
    features_before = [speaker.features for speaker in speakers]
    features_after = array(expansion_3D(features_before)) if is_expansion else array(contraction_3D(features_before))
    for speaker, item in zip(speakers, features_after):
        speaker.features = item
    return array([speaker.features for speaker in speakers])


def save_features(speakers, path=READY_TO_USE_FEAT, name='6450n_wav_nn', format=F_NPY):
    '''
    Для стегонограммы 4D, иначе 3D.
    '''
    features = list(chain(*[speaker.features for speaker in speakers]))

    if format == F_NPY:
        features = array(features)
        size = features.shape
        if len(size) == 3:
            features = features.reshape((size[0] * size[1], size[2]))
        elif len(size) == 4:
            features = features.reshape((size[0] * size[1], size[2], size[3]))
        else:
            raise Exception('Нет алгоритма уплощения!')
    elif format in (F_CSV, F_PIC):
        pass
    else:
        raise Exception('Для данного формата нет алгоритма сохранеия!')

    sf(features, path, name + format)


if __name__ == '__main__':
    IS_SAVE = False
    SCRIPT = scripts.HTK_20
    id_speakers = SCRIPT.list_speakers
    speakers = [Speaker(id, SCRIPT) for id in id_speakers]
    # if DIR_FEATURES == WAV:
    #     change_shape_features(speakers)
    save_features(speakers, name='wav_1500_list_log', format=F_PIC) if IS_SAVE else None
    print('Success!')
