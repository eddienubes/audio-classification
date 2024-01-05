import numpy as np

import config
import librosa
import librosa.feature
import librosa.onset
import utils

logger = utils.get_logger('audio-utils')


def get_max_rms(raw_audio: np.ndarray) -> float:
    """
    Gets max RMS of first N frames of a give raw_audio
    :param raw_audio:
    :return:
    """
    rms_arr = get_rms(raw_audio)
    max_rms = max(rms_arr[:config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH])

    return max_rms


def get_rms(raw_audio: np.ndarray) -> np.ndarray:
    """
    Get RMS array for selected frames
    :param raw_audio:
    :return: A 1-dimensional array with RMS over frame length
    """
    frame_length = min(config.AUDIO_FRAME_SIZE_SAMPLES, len(raw_audio))
    # Short-Time Fourier Transform
    stft = librosa.stft(y=raw_audio, n_fft=frame_length)
    magnitude, phase = librosa.magphase(stft)
    # docs reference: https://librosa.org/doc/main/generated/librosa.stft.html#librosa-stft
    rms_tuple = librosa.feature.rms(S=magnitude, frame_length=frame_length, hop_length=frame_length // 4)
    rms_arr = rms_tuple[0]
    return rms_arr


def get_onsets(raw_audio: np.ndarray) -> np.ndarray:
    """
    Onset is the point where the energy of the starts to increase
    :param raw_audio
    :return
    """
    hop_length = int(librosa.time_to_samples(config.AUDIO_ONSET_DETECTION_WINDOW_SEC, sr=config.SAMPLE_RATE))
    onsets = librosa.onset.onset_detect(y=raw_audio, sr=config.SAMPLE_RATE, hop_length=hop_length, units='time')

    return onsets


def shift_audio_by_sec(raw_audio: np.ndarray, sec: int) -> np.ndarray:
    shifted_audio = np.append(np.zeros(int(sec * config.SAMPLE_RATE)), raw_audio)
    return shifted_audio




