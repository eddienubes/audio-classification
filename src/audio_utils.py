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


def get_mpeg7_features(raw_audio: np.ndarray) -> dict:
    features = dict()
    frame_length_samples = 512

    rms_per_frame = get_rms(raw_audio)
    peak_idx, rms_indices_above_threshold = get_mpeg7_rms_indices(rms_per_frame)
    log_attack_time_sec = get_log_attack_time(peak_idx, rms_indices_above_threshold, frame_length_samples)
    temporal_centroid, temporal_centroid_duration = get_temporal_centroid_and_duration(rms_per_frame,
                                                                                       rms_indices_above_threshold,
                                                                                       frame_length_samples)
    release_samples = get_release(frame_length_samples, rms_indices_above_threshold, peak_idx)

    # Log Attack Time
    features['log_attack_time'] = log_attack_time_sec
    features['temporal_centroid'] = temporal_centroid
    features['temporal_centroid_duration'] = temporal_centroid_duration
    # Ration between Log Attack Time and Temporal Centroid
    features['lat_tc_ratio'] = log_attack_time_sec / temporal_centroid if temporal_centroid > 0 else np.NaN
    features['release'] = release_samples

    return features


def get_log_attack_time(peak_idx: int, rms_indices_above_threshold: np.ndarray, frame_length: int) -> float:
    """
    Calculates and returns log attack time for a given distribution of RMS
    :param frame_length number of frames
    :param peak_idx
    :param rms_indices_above_threshold
    :return
    """

    # Get the first element to find out where the 2% of the rms hits for the first time.
    first_index_above_threshold = rms_indices_above_threshold[0]

    # Get frame length in seconds for a given number of samples.
    # We use this window size particularly for calculating the attack time
    frame_length_sec = frame_length / config.SAMPLE_RATE
    attack_time_sec = peak_idx * frame_length_sec - first_index_above_threshold * frame_length_sec

    # There are cases where attack_time_sec is 0. Since we can't take the log of it let's assume that the attack time
    # is half of the frame size.
    log_attack_time_sec = np.log10(attack_time_sec) if attack_time_sec > 0 else np.log10(frame_length_sec / 2.0)

    return log_attack_time_sec


def get_temporal_centroid_and_duration(rms_per_frame: np.ndarray, rms_indices_above_threshold: np.ndarray,
                                       frame_length: int) -> [float,
                                                              float]:
    """
    :param rms_per_frame
    :param rms_indices_above_threshold
    :param frame_length
    :return: tuple of temporal centroid (in seconds) and temporal_centroid_duration (in samples)
    """
    # Temporal centroid is calculated via mean squared amplitude
    rms_per_frame_squared = rms_per_frame ** 2
    first_rms_above_threshold_index = rms_indices_above_threshold[0]
    last_rms_above_threshold_index = rms_indices_above_threshold[-1]

    # We need frames only from the first hit above 2% threshold to the last hit above this threshold
    temp_centroid_rms_span = rms_per_frame_squared[first_rms_above_threshold_index: last_rms_above_threshold_index + 1]
    temporal_centroid = np.sum(temp_centroid_rms_span * np.linspace(0.0, 1.0, len(temp_centroid_rms_span))) / np.sum(
        temp_centroid_rms_span) \
        if np.sum(temp_centroid_rms_span) > 0 else np.NaN

    temporal_centroid_duration = frame_length * len(temp_centroid_rms_span)

    return temporal_centroid, temporal_centroid_duration


def get_release(frame_length: int, rms_indices_above_threshold: np.ndarray, peak_idx: int) -> float:
    """
    Returns release of an audio file in samples
    :param frame_length
    :param rms_indices_above_threshold
    :param peak_idx
    :return:
    """

    # Release is a difference in samples between last RMS above threshold index and peak index
    return frame_length * (rms_indices_above_threshold[-1] - peak_idx)


def get_mpeg7_rms_indices(rms_per_frame: np.ndarray) -> (int, np.ndarray):
    """
    MPEG7 features rely only on the frames where RMS is at least above 2% of the peak.
    Follow this paper for more: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=90371bbd7e2fc82ee01182b43435ca31926ede9b
    :param rms_per_frame
    :return tuple where 0 - peak index, 1 - array of indices
    """
    peak_idx = np.argmax(rms_per_frame)

    # Here we filter our rms values below this 2%
    # rms_above_threshold is a boolean array
    rms_above_threshold = rms_per_frame >= 0.02 * rms_per_frame[peak_idx]
    # np.where returns a tuple with 1 element. This element is an array of indices where the condition above is True.
    rms_indices_above_threshold = np.where(rms_above_threshold)[0]

    return peak_idx, rms_indices_above_threshold
