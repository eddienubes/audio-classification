import numpy as np
import librosa
import librosa.feature
import config


class AudioSpectralFeaturesAnalyzer:
    def __init__(self, raw_audio: np.ndarray, spectral_magnitude: np.ndarray, loud_rms_indices: np.ndarray,
                 long_enough_for_gradient: bool, loudest_rms_frame_index: int):
        self.spectral_magnitude = spectral_magnitude
        self.frame_length = min(config.AUDIO_FRAME_SIZE_SAMPLES, len(raw_audio))
        self.loud_rms_indices = loud_rms_indices
        self.long_enough_for_gradient = long_enough_for_gradient
        self.loudest_rms_frame_index = loudest_rms_frame_index

        hop_length = self.frame_length // 4

        # Spectral Centroid
        spectral_centroid_t = librosa.feature.spectral_centroid(
            S=self.spectral_magnitude, n_fft=self.frame_length, hop_length=hop_length)
        spectral_centroid = spectral_centroid_t[0][:config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH][
            self.loud_rms_indices]
        log_spectral_centroid = np.log10(spectral_centroid)

        self.spectral_centroid = spectral_centroid
        self.log_spectral_centroid = log_spectral_centroid

        # Spectral Bandwidth
        spectral_bandwidth_t = librosa.feature.spectral_bandwidth(S=self.spectral_magnitude, n_fft=self.frame_length,
                                                                  hop_length=hop_length)
        spectral_bandwidth = spectral_bandwidth_t[0][:config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH][self.loud_rms_indices]
        log_spectral_bandwidth = np.log10(spectral_bandwidth)

        self.spectral_bandwidth = spectral_bandwidth
        self.log_spectral_bandwidth = log_spectral_bandwidth

        spec_flat_t = librosa.feature.spectral_flatness(
            S=self.spectral_magnitude, n_fft=self.frame_length, hop_length=hop_length)
        spectral_flatness = spec_flat_t[0][:config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH][self.loud_rms_indices]
        self.spectral_flatness = spectral_flatness

        self.spectral_rollofs_percenatges = [.15, .85]
        self.spectral_rollofs = []
        for roll_percent in self.spectral_rollofs_percenatges:
            spec_rolloff_t = librosa.feature.spectral_rolloff(
                S=self.spectral_magnitude, roll_percent=roll_percent, n_fft=self.frame_length, hop_length=hop_length
            )
            spec_rolloff = spec_rolloff_t[0][:config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH][self.loud_rms_indices]
            self.spectral_rollofs.append(spec_rolloff)

    def get_features(self):
        log_spec_cent_mean, log_spec_cent_std, log_spec_cent_peak = self.get_log_spectral_centroid_features()
        log_spec_cent_g_mean, leg_spec_cent_g_std, log_spec_cent_g_zcr = self.get_log_spectral_centroid_gradient_features()
        spec_band_mean, spec_band_std, spec_band_peak, spec_band_g_mean = self.get_log_spectral_bandwidth_features()
        spec_flat_mean, spec_flat_max, spec_flat_min, spec_flat_std, spec_flat_peak, spec_flat_g_mean = self.get_spectral_flatness_features()
        spec_folloff_features = self.get_spectral_rolloff_features()
        mfcc_features = self.get_mfcc_features()

        return {
            'log_spec_cent_mean': log_spec_cent_mean,
            'log_spec_cent_std': log_spec_cent_std,
            'log_spec_cent_peak': log_spec_cent_peak,
            'log_spec_cent_g_mean': log_spec_cent_g_mean,
            'leg_spec_cent_g_std': leg_spec_cent_g_std,
            'log_spec_cent_g_zcr': log_spec_cent_g_zcr,
            'spec_band_mean': spec_band_mean,
            'spec_band_std': spec_band_std,
            'spec_band_peak': spec_band_peak,
            'spec_band_g_mean': spec_band_g_mean,
            'spec_flat_mean': spec_flat_mean,
            'spec_flat_max': spec_flat_max,
            'spec_flat_min': spec_flat_min,
            'spec_flat_std': spec_flat_std,
            'spec_flat_peak': spec_flat_g_mean,
            **spec_folloff_features,
            **mfcc_features
        }

    def get_log_spectral_centroid_features(self) -> [float, float, float]:
        log_spectral_centroid_mean = np.mean(self.log_spectral_centroid)
        log_spectral_centroid_std = np.std(self.log_spectral_centroid)
        log_spectral_centroid_peak = self.log_spectral_centroid[self.loudest_rms_frame_index]

        return log_spectral_centroid_mean, log_spectral_centroid_std, log_spectral_centroid_peak

    def get_log_spectral_centroid_gradient_features(self) -> [float, float, float]:
        if not self.long_enough_for_gradient:
            return np.NaN, np.NaN, np.NaN

        log_spectral_gradient_centroid = np.gradient(self.log_spectral_centroid)
        log_spectral_centroid_g_mean = np.mean(log_spectral_gradient_centroid)
        log_spectral_centroid_g_std = np.std(log_spectral_centroid_g_mean)
        log_spectral_centroid_g_zcr = len(np.where(np.diff(np.sign(log_spectral_gradient_centroid)))[0]) / float(
            len(log_spectral_gradient_centroid))

        return log_spectral_centroid_g_mean, log_spectral_centroid_g_std, log_spectral_centroid_g_zcr

    def get_log_spectral_bandwidth_features(self):
        spec_band_mean = np.mean(self.log_spectral_bandwidth)
        spec_band_std = np.std(self.log_spectral_bandwidth)
        spec_band_peak = np.max(self.log_spectral_bandwidth)
        spec_band_g_mean = np.mean(
            np.gradient(self.log_spectral_bandwidth)) if self.long_enough_for_gradient else np.NaN

        return spec_band_mean, spec_band_std, spec_band_peak, spec_band_g_mean

    def get_spectral_flatness_features(self) -> [float, float, float, float, float, float]:
        spec_flat_mean = np.mean(self.spectral_flatness)
        spec_flat_max = np.max(self.spectral_flatness)
        spec_flat_min = np.min(self.spectral_flatness)
        spec_flat_std = np.std(self.spectral_flatness)
        spec_flat_peak = self.spectral_flatness[self.loudest_rms_frame_index]
        spec_flat_g_mean = np.mean(np.gradient(self.spectral_flatness)) if self.long_enough_for_gradient else np.NaN

        return spec_flat_mean, spec_flat_max, spec_flat_min, spec_flat_std, spec_flat_peak, spec_flat_g_mean

    def get_spectral_rolloff_features(self) -> dict:
        features = dict()
        for index, percentage in enumerate(self.spectral_rollofs_percenatges):
            roll_percent_int = int(100 * percentage)
            spectral_rolloff = self.spectral_rollofs[index]
            features[f'log_spec_rolloff_{roll_percent_int}_peak'] = np.log10(
                spectral_rolloff[self.loudest_rms_frame_index]) \
                if spectral_rolloff[self.loudest_rms_frame_index] > 0.0 else np.NaN
            # For some reason some sounds give random 0.0s for the spectral rolloff of certain frames.
            # After log these are -inf and need to be filtered before taking the min
            log_spec_rolloff = np.log10(spectral_rolloff[spectral_rolloff != 0.0])
            features[f'log_spec_rolloff_{roll_percent_int}_max'] = np.max(log_spec_rolloff) \
                if len(log_spec_rolloff) > 0 else np.NaN
            features[f'log_spec_rolloff_{roll_percent_int}_min'] = np.min(log_spec_rolloff) \
                if len(log_spec_rolloff) > 0 else np.NaN

        return features

    def get_mfcc_features(self):
        features = dict()
        n_mfcc = 13

        # We the first mfcc value because it's just a volume
        mfccs = librosa.feature.mfcc(S=self.spectral_magnitude, n_mfcc=n_mfcc)[1:,
                :config.LIBRARY_AUDIO_MAX_FRAMES_FOR_SEARCH][:, self.loud_rms_indices]
        n_mfcc -= 1

        # Compute once because it's faster
        transformed_mfcc = {
            'mean': np.mean(mfccs, axis=1),
            'loudest': mfccs[:, self.loudest_rms_frame_index]
        }

        for n in range(n_mfcc):
            # std wasn't found to contribute anything
            for op in ['mean', 'loudest']:
                features[f'mfcc_{n}_{op}'] = transformed_mfcc[op][n]

            features[f'mfcc_{n}_d_mean'] = np.NaN

        if self.long_enough_for_gradient:
            mfcc_g_mean = np.mean(np.gradient(mfccs, axis=1), axis=1)
            for n in range(n_mfcc):
                features[f'mfcc_{n}_d_mean'] = mfcc_g_mean[n]

        return features
