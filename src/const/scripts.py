from src.const.paths import READY_TO_USE_FEAT, HTK as htk, CSV as csv, MATLAB as matlab, WAV as wav, MODELS_DIR
from src.utils.read_features import read_csv, read_htk, read_matlab, read_spectrogram
from src.utils.load import load_features
from src.const import classifiers

# файлы с признаками
__BIN_FLAT_HTK = '1000_htk_nn.npy'
__PIC_FLAT_LIST_WAV = 'wav_1000_list_original.pickle'
__PIC_FLAT_LIST_WAV_30 = 'wav_30.pickle'
__PIC_FLAT_ARR_DNN = 'extract_20.pickle'
__PIC_FLAT_ARR_DNN_50 = 'extract_50.pickle'

__ALL_RECORDS_SPEAKER = 50
__COUNT_RECORDS_TRAINING = 40
__COUNT_RECORDS_VAL = 10
__COUNT_RECORDS_TEST = __ALL_RECORDS_SPEAKER - __COUNT_RECORDS_TRAINING


class Structure:
    def __init__(self, directory, format, list_speakers, def_read, model=classifiers.SVM_LINEAR,
                 extract_features=True, recovery_features=None):
        self.directory = directory
        self.format = format
        self.is_extract_features = extract_features
        self.list_speakers = list_speakers
        self.count_speakers = len(self.list_speakers)
        self.model = model
        self.recovery_features = recovery_features
        self.def_read = def_read


CSV = Structure(
    directory=csv,
    format='csv',
    list_speakers=[i for i in range(16, 26)] + [j for j in range(41, 51)],
    def_read=read_csv,
    recovery_features=None
)

MATLAB = Structure(
    directory=matlab,
    format='csv',
    list_speakers=[i for i in range(1, 21)],
    def_read=read_matlab,
    recovery_features=None
)

HTK_20 = Structure(
    directory=htk,
    format='htk',
    list_speakers=[i for i in range(16, 26)] + [j for j in range(41, 51)],
    def_read=read_htk,
    model=classifiers.SVM_LINEAR,
    recovery_features=READY_TO_USE_FEAT / __BIN_FLAT_HTK
)

HTK_30 = Structure(
    directory=htk,
    format='htk',
    list_speakers=[i for i in range(1, 16)] + [j for j in range(26, 41)],
    def_read=read_htk,
    recovery_features=None
)

WAV_20 = Structure(
    directory=wav,
    format='wav',
    list_speakers=[i for i in range(16, 26)] + [j for j in range(41, 51)],
    def_read=read_spectrogram,
    recovery_features=READY_TO_USE_FEAT / __PIC_FLAT_LIST_WAV
)

WAV_30 = Structure(
    directory=wav,
    format='wav',
    list_speakers=[i for i in range(1, 16)] + [j for j in range(26, 41)],
    def_read=read_spectrogram,
    recovery_features=READY_TO_USE_FEAT / __PIC_FLAT_LIST_WAV_30
)

EX_DNN_20 = Structure(
    directory=READY_TO_USE_FEAT,
    format='pickle',
    extract_features=False,
    list_speakers=[i for i in range(1, 21)],
    def_read=load_features(file=__PIC_FLAT_ARR_DNN),
    recovery_features=READY_TO_USE_FEAT / __PIC_FLAT_ARR_DNN
)

EX_DNN_50 = Structure(
    directory=READY_TO_USE_FEAT,
    format='pickle',
    extract_features=False,
    list_speakers=[i for i in range(1, 21)],
    def_read=load_features(file=__PIC_FLAT_ARR_DNN_50),
    recovery_features=READY_TO_USE_FEAT / __PIC_FLAT_ARR_DNN_50
)
