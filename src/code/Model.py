from src.code.Speaker import Speaker, list_to_array
from src.const.cfg import IS_NORMALIZE, DIMENSIONALITY, COUNT_SPEAKERS, ALL_RECORDS_SPEAKER
from src.const.paths import HTK, WAV, READY_TO_USE_FEAT
# from src.const.recovery import BIN_FLAT_WAV, BIN_FLAT_HTK
from src.utils.load import load_features
from src.const import scripts

from numpy import array
from sklearn.preprocessing import normalize as normalize_sl
from keras.utils import normalize as normalize_kr
from itertools import chain


class Model:
    def __init__(self, id, script, records=ALL_RECORDS_SPEAKER):
        self.id = id
        self.all_records_speaker = records
        self.script = script
        self.count_speakers = self.script.count_speakers
        self.dir_features = self.script.directory
        self.__recovery_features = self.script.recovery_features
        self.__list_id_speakers = self.script.list_speakers
        self.features = self.get_features()
        self.labels = self.get_labels()

    def get_features(self):
        if self.__recovery_features is None:
            features = [Speaker(id, self.script).features for id in self.__list_id_speakers]
            if type(features[0]) == list:
                features = list(chain(*features))
            else:
                features = array(features)
                features = features.reshape((features.shape[0] * features.shape[1], features.shape[2]))
        else:
            features = load_features(file=self.__recovery_features)
        return features

    def get_labels(self):
        speakers = self.count_speakers
        records = self.all_records_speaker
        ones = [0] * (speakers * records - 1)
        ones[self.id * records:self.id * records + records - 1] = [1] * records
        return array(ones)


if __name__ == '__main__':
    SCRIPT = scripts.WAV_20
    id_models = [id for id in range(SCRIPT.count_speakers)]
    models = [Model(model, script=SCRIPT) for model in id_models]
    print('Success!')
