from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

# каталоги с исходными признаками
CSV = ROOT_DIR / 'data' / 'signs_csv'
MATLAB = ROOT_DIR / 'data' / 'signs_matlab'
HTK = ROOT_DIR / 'data' / 'signs_htk'
# SPECTR = ROOT_DIR / 'data' / 'signs_spectrogram'
WAV = ROOT_DIR / 'data' / 'signs_wav'
WAV_30 = ROOT_DIR / 'data' / 'signs_wav'  # 'ready_to_use' / 'wav_30.pickle'  # ROOT_DIR / 'data' / 'signs_wav'
EX_DNN_20 = ROOT_DIR / 'data' / 'ready_to_use' / 'extract_feat.pickle'
EX_DNN_50 = ROOT_DIR / 'data' / 'ready_to_use' / 'extract_50_new_1000.pickle'

# каталог с записанными признаками
READY_TO_USE_FEAT = ROOT_DIR / 'data' / 'ready_to_use'

# каталоги для сохранения
PLOTS_DIR = ROOT_DIR / 'src' / 'save' / 'plots'
MODELS_DIR = ROOT_DIR / 'src' / 'save' / 'models'
NETWORKS_DIR = ROOT_DIR / 'src' / 'save' / 'networks'
TENSORBOARD_LOG = ROOT_DIR / 'src' / 'logs' / 'tensorboard'
CSV_LOG = ROOT_DIR / 'src' / 'logs'

BALANCE_FOR_STAT = ROOT_DIR / 'src' / 'experiments' / 'balance_for_stat.csv'

# tensorboard --logdir=logs\tensorboard
