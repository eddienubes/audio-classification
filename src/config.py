import math

LOG_LEVEL = 'DEBUG'

DEFAULT_SAMPLE_RATE = 22050
AUDIO_FRAME_SIZE_SAMPLES = 2048

LIBRARY_AUDIO_FEATURE_SEARCH_WINDOW_LENGTH_SEC = 1
# First number of frames to look for characteristics. Later on we'll filter out files
# whose data doesn't meet the criteria in the first number of frames
# Formula: search_window_in_sec / (hop_length / sample_rate)
# hop_length reference link: https://librosa.org/doc/main/generated/librosa.stft.html#librosa-stft
LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH = math.ceil(LIBRARY_AUDIO_FEATURE_SEARCH_WINDOW_LENGTH_SEC / (
        (AUDIO_FRAME_SIZE_SAMPLES // 4) / DEFAULT_SAMPLE_RATE))
LIBRARY_AUDIO_FILE_MAX_DURATION_SEC = 5
LIBRARY_AUDIO_MIN_RMS = 0.02
LIBRARY_SOURCE_SIZE_LIMIT = 1000
LIBRARY_SOURCE_PATH = '/Volumes/Samsung SSD 980 PRO 2TB/Sample Packs'
LIBRARY_DATAFRAME_PATH = './data/dataframe.pkl'
LIBRARY_HDF5_PATH = './data/library.h5'
# ['snare', 'kick', 'hat', 'tom', 'clap', 'rim', 'open', 'ride', 'crash', 'snap', 'bongo', 'shaker']
CATEGORIES = {
    'snare': ['snare'],
    'kick': ['kick'],
    'hi-hat': ['hi', 'hat'],
    'tom': ['tom'],
    'clap': ['clap'],
    'rim': ['rim'],
    'open-hat': ['hat', 'open'],
    'ride': ['ride'],
    'crash': ['crash'],
    'snap': ['snap'],
    'bongo': ['bongo'],
    'shaker': ['shaker']
}
