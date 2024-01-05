import numpy as np
import config
import librosa
import librosa.feature


class AudioAnalyzer:
    def __init__(self, raw_audio: np.ndarray):
        self.raw_audio = raw_audio
        self.magnitude = None
        self.original_duration = len(raw_audio) / float(config.SAMPLE_RATE)

        self.rms = self.get_rms()
        self.max_rms = self.get_max_rms()

    def get_rms(self) -> np.ndarray:
        """
        Get RMS array for selected frames
        :return: A 1-dimensional array with RMS over frame length
        """
        frame_length = min(config.AUDIO_FRAME_SIZE_SAMPLES, len(self.raw_audio))
        # Short-Time Fourier Transform
        stft = librosa.stft(y=self.raw_audio, n_fft=frame_length)
        magnitude, phase = librosa.magphase(stft)

        self.magnitude = magnitude

        # docs reference: https://librosa.org/doc/main/generated/librosa.stft.html#librosa-stft
        rms_tuple = librosa.feature.rms(S=magnitude, frame_length=frame_length, hop_length=frame_length // 4)
        rms_arr = rms_tuple[0]
        return rms_arr

    def get_onsets(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        Onset is the point where the energy of the starts to increase
        :return
        """
        hop_length = int(librosa.time_to_samples(config.AUDIO_ONSET_DETECTION_WINDOW_SEC, sr=config.SAMPLE_RATE))
        onsets = librosa.onset.onset_detect(y=raw_audio, sr=config.SAMPLE_RATE, hop_length=hop_length,
                                            units='time')

        return onsets

    def shift_audio_by_sec(self, sec: int) -> np.ndarray:
        shifted_audio = np.append(np.zeros(int(sec * config.SAMPLE_RATE)), self.raw_audio)
        return shifted_audio

    def get_max_rms(self) -> float:
        """
        Gets max RMS of first N frames of a give raw_audio
        :return:
        """
        max_rms = max(self.rms[:config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH])

        return max_rms

    def get_start_end_time(self) -> [float, float | None]:
        shifted_audio = self.shift_audio_by_sec(config.AUDIO_SILENCE_SEC)

        onset_timings_shift_sec = config.AUDIO_SILENCE_SEC + 0.01
        end = None

        onsets = self.get_onsets(shifted_audio)

        if len(onsets) == 0:
            return [0, None]
        if len(onsets) > 1:
            end = onsets[1] - onset_timings_shift_sec

        start = max(onsets[0] - onset_timings_shift_sec, 0.0)
        return [start, end]
