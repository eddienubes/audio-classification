import numpy as np
import pandas
from audio_mpeg_features_analyzer import *
from audio_misc_features_analyzer import *
from audio_spectral_features_analyzer import *
from audio_analyzer import *

from pathlib import Path
from utils import exc_to_message, get_logger


# TODO: Extract some of the methods to a new class AudioAnalyzer
class AudioLibraryReader:
    def __init__(self):
        self.logger = get_logger(AudioLibraryReader.__name__)

    def read(self, input_dir_path: Path) -> pandas.DataFrame:
        self.logger.info(f'Searching for audio within {input_dir_path}')
        dataframe_rows = []
        for input_file in input_dir_path.glob('**/*.*'):
            if len(dataframe_rows) >= config.LIBRARY_SOURCE_SIZE_LIMIT:
                break
            if not self.apply_file_filter(input_file):
                continue

            # Check whether filename contains required category, otherwise it's not suitable for training data
            category = self.get_category_by_path(input_file)
            if category is None:
                continue

            absolute_file_path = input_file.resolve().as_posix()
            self.logger.info(f'Trying to analyze file: {absolute_file_path}')

            # Audio is PCM data right here
            audio = self.load_raw_audio(absolute_file_path, fast=True)
            if audio is None:
                continue

            audio_analyzer = AudioAnalyzer(audio)

            is_accepted = self.apply_audio_file_filter(audio_analyzer.original_duration,
                                                       audio_analyzer.max_rms)
            if not is_accepted:
                continue

            start_time, end_time = audio_analyzer.get_start_end_time()

            mpeg_analyzer = AudioMpegFeaturesAnalyzer(audio, audio_analyzer.rms)
            mpeg7_features = mpeg_analyzer.get_features()

            misc_analyzer = AudioMiscFeaturesAnalyzer(audio, audio_analyzer.rms)
            misc_features = misc_analyzer.get_features()

            spectral_analyzer = AudioSpectralFeaturesAnalyzer(audio, audio_analyzer.magnitude,
                                                              misc_analyzer.loud_rms_indices,
                                                              misc_analyzer.long_enough_for_gradient,
                                                              misc_analyzer.loudest_rms_frame_index)
            spectral_features = spectral_analyzer.get_features()

            properties = {
                'audio_file_path': absolute_file_path,
                'filename': Path(absolute_file_path).stem,
                'original_duration': audio_analyzer.original_duration,
                'rms': audio_analyzer.max_rms,
                'category': category,
                **mpeg7_features,
                **misc_features,
                **spectral_features,
                # We fill up these later
                'start_time': start_time,
                'end_time': end_time,
            }

            dataframe_rows.append(properties)

        return pandas.DataFrame(dataframe_rows)

    def load_raw_audio(self, absolute_file_path: str, sample_rate: int = config.SAMPLE_RATE, offset=0,
                       duration=None, fast=False):
        try:
            time_series, sr = librosa.load(absolute_file_path, sr=sample_rate, offset=offset, duration=duration,
                                           res_type=('kaiser_fast' if fast else 'kaiser_best'))
        except BaseException as error:
            self.logger.warning(f'Cannot read raw audio path: {absolute_file_path}, error: {exc_to_message()}')
            return None

        return time_series
        # if (duration is None and time_series.shape[0] > 0) or (duration is not None and time_series.shape[0].)

    def apply_audio_file_filter(self, original_duration: float, max_rms: float) -> bool:
        """
        Apply filters to an audio file to decide wether to load them save them in the dataframe
        :param original_duration: 
        :param max_rms:
        :return
        """
        # Exclude all files with a duration longer than required
        if original_duration > config.LIBRARY_AUDIO_FILE_MAX_DURATION_SEC:
            return False

        # Exclude all files quieter than required
        if max_rms < config.LIBRARY_AUDIO_MIN_RMS:
            return False

        return True

    def get_category_by_path(self, input_path: Path) -> str | None:
        # Try to find required category in the audio file name
        for category in config.CATEGORIES:
            filename = input_path.stem
            keywords = config.CATEGORIES[category]

            negative_matches = 0
            for keyword in keywords['exclude']:
                if keyword.lower() in filename.lower():
                    negative_matches += 1
                    break

            # Ignore files with blacklisted keywords in the filename
            if negative_matches != 0:
                continue

            positive_matches = 0
            for keyword in keywords['include']:
                if keyword.lower() in filename.lower():
                    positive_matches += 1

            # Basically, if all keywords matched with a filename
            if positive_matches >= len(keywords['include']):
                return category

        return None

    def apply_file_filter(self, input_file: Path) -> bool:
        if not input_file.is_file():
            return False

        for keyword in config.LIBRARY_FILE_KEYWORD_BLACKLIST:
            if keyword.lower() in input_file.resolve().as_posix().lower():
                return False

        whitelist_keyword_matches = 0
        for keyword in config.LIBRARY_FILE_KEYWORD_WHITELIST:
            if keyword.lower() in input_file.resolve().as_posix().lower():
                whitelist_keyword_matches += 1

        if whitelist_keyword_matches < len(config.LIBRARY_FILE_KEYWORD_WHITELIST):
            return False

        return True
