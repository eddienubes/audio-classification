import math

LOG_LEVEL = 'DEBUG'

SAMPLE_RATE = 22050
AUDIO_FRAME_SIZE_SAMPLES = 2048
AUDIO_FEATURE_SEARCH_WINDOW_DURATION_SEC = 1
# Window of 5ms
AUDIO_ONSET_DETECTION_WINDOW_SEC = (AUDIO_FEATURE_SEARCH_WINDOW_DURATION_SEC / 1000) * 5
# Sometimes librosa doesn't identify onsets in the beginning of the audio file
AUDIO_SILENCE_SEC = 1.0

# First number of frames to look for characteristics. Later on we'll filter out files
# whose data doesn't meet the criteria in the first number of frames
# Formula: search_window_in_sec / (hop_length / sample_rate)
# hop_length reference link: https://librosa.org/doc/main/generated/librosa.stft.html#librosa-stft
LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH = math.ceil(AUDIO_FEATURE_SEARCH_WINDOW_DURATION_SEC / (
        (AUDIO_FRAME_SIZE_SAMPLES // 4) / SAMPLE_RATE))
LIBRARY_AUDIO_FILE_MAX_DURATION_SEC = 5
LIBRARY_AUDIO_MIN_RMS = 0.02
LIBRARY_SOURCE_SIZE_LIMIT = 1000
LIBRARY_SOURCE_PATH = '/Volumes/Samsung SSD 980 PRO 2TB/Sample Packs'
LIBRARY_DATAFRAME_PATH = './data/dataframe.pkl'
LIBRARY_HDF5_PATH = './data/library.h5'

CATEGORIES = {
    'snare': {
        'include': ['snare'],
        'exclude': []
    },
    'kick': {
        'include': ['kick'],
        'exclude': []
    },
    'hi-hat': {
        'include': ['hi', 'hat'],
        'exclude': []
    },
    'tom': {
        'include': ['tom'],
        'exclude': []
    },
    'clap': {
        'include': ['clap'],
        'exclude': []
    },
    'rim': {
        'include': ['rim'],
        'exclude': ['grime']
    },
    'open-hat': {
        'include': ['hat', 'open'],
        'exclude': []
    },
    'ride': {
        'include': ['ride'],
        'exclude': []
    },
    'crash': {
        'include': ['crash'],
        'exclude': []
    },
    'snap': {
        'include': ['snap'],
        'exclude': []
    },
    'bongo': {
        'include': ['bongo'],
        'exclude': []
    },
    'shaker': {
        'include': ['shaker'],
        'exclude': []
    }
}
