import numpy as np
import librosa
import librosa.feature
import config


class AudioMiscFeaturesAnalyzer:
    def __init__(self, raw_audio: np.ndarray, raw_rms: np.ndarray):
        self.raw_audio = raw_audio
        self.raw_rms = raw_rms

        loud_rms, loud_rms_indices = self.get_loud_rms(raw_rms)
        self.loud_rms = loud_rms
        self.loud_rms_indices = loud_rms_indices

        self.log_rms = np.log10(self.loud_rms)

    def get_features(self) -> dict:
        mean_rms, std_rms, max_rms = self.get_rms_features()
        g_mean_rms, g_std_rms, g_zcr_rms = self.get_rms_gradient_features()
        mean_zcr, std_zcr, zcr_at_rms_peak = self.get_zcr_features()

        return {
            'g_mean_rms': g_mean_rms,
            'g_std_rms': g_std_rms,
            'g_zcr_rms': g_zcr_rms,
            'mean_rms': mean_rms,
            'std_rms': std_rms,
            'max_rms': max_rms,
            'mean_zcr': max_rms,
            'std_zcr': std_zcr,
            'zcr_at_rms_peak': zcr_at_rms_peak,
            'crest_factor': self.get_crest_factor(max_rms, mean_rms)
        }

    def get_crest_factor(self, rms_max: float, rms_mean: float) -> float:
        return rms_max / rms_mean

    def get_rms_features(self) -> [float, float, float]:
        mean_rms = np.mean(self.log_rms)
        # Standard deviation of the RMS
        std_rms = np.std(self.log_rms)
        max_rms = np.max(self.log_rms)

        return mean_rms, std_rms, max_rms

    def get_rms_gradient_features(self) -> [float, float, float]:
        # We need more than 1 frame for gradient
        long_enough = len(self.log_rms) > 1
        if not long_enough:
            return np.NaN, np.NaN, np.NaN

        rms_gradient = np.gradient(self.log_rms)
        mean_rms = np.mean(rms_gradient)
        std_rms = np.std(self.log_rms)
        # Zero Crossing Rate
        # https://stackoverflow.com/questions/15415271/how-to-compute-zero-crossing-rate-of-signal
        zcr_rms = len(np.where(np.diff(np.sign(rms_gradient)))[0]) / float(len(rms_gradient))

        return mean_rms, std_rms, zcr_rms

    def get_zcr_features(self) -> [float, float, float]:
        hop_length = config.AUDIO_FRAME_SIZE_SAMPLES // 4
        zcr = librosa.feature.zero_crossing_rate(self.raw_audio, frame_length=config.AUDIO_FRAME_SIZE_SAMPLES,
                                                 hop_length=hop_length)
        zcr = zcr[0][:config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH][self.loud_rms_indices]
        rms_peak_index = np.argmax(self.loud_rms)

        mean_zcr = np.mean(zcr)
        std_zcr = np.std(zcr)
        zcr_at_rms_peak = zcr[rms_peak_index]

        return mean_zcr, std_zcr, zcr_at_rms_peak

    def get_loud_rms(self, rms: np.ndarray) -> [np.ndarray, np.ndarray]:
        rms = rms[:config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH]

        # Some algorithms perform badly on sounds that are too quiet
        loud_rms_indices = rms >= config.LIBRARY_AUDIO_MIN_RMS

        if loud_rms_indices <= 0:
            raise Exception('Not enough loud RMS frames for analysis')

        loud_rms = rms[loud_rms_indices]

        return loud_rms, loud_rms_indices
