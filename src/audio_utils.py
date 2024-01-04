import numpy as np
from numpy import ndarray

import config
import librosa
import librosa.feature
import utils

logger = utils.get_logger('audio-utils')


def get_rms(raw_audio: np.ndarray) -> float:
    """
    Gets RMS of first N frames of a give raw_audio
    :param raw_audio:
    :return:
    """
    frame_length = min(config.AUDIO_FRAME_SIZE_SAMPLES, len(raw_audio))
    # Short-Time Fourier Transform
    stft = librosa.stft(y=raw_audio, n_fft=frame_length)
    magnitude, phase = librosa.magphase(stft)
    # docs reference: https://librosa.org/doc/main/generated/librosa.stft.html#librosa-stft
    # Seems like librosa doesn't
    rms_arr = librosa.feature.rms(S=magnitude, frame_length=frame_length, hop_length=frame_length // 4)
    max_rms = max(rms_arr[0][:config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH])

    return max_rms
