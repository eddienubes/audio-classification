import numpy as np
import pandas
from audio_mpeg_features_analyzer import *
from audio_misc_features_analyzer import *

from pathlib import Path
from utils import exc_to_message, get_logger
from audio_utils import *


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
            if not input_file.is_file():
                continue

            absolute_file_path = input_file.resolve().as_posix()
            self.logger.info(f'Trying to analyze file: {absolute_file_path}')

            # Check whether filename contains required category, otherwise it's not suitable for training data
            category = self.get_category_by_path(input_file)
            if category is None:
                continue

            # Audio is PCM data right here
            audio = self.load_raw_audio(absolute_file_path, fast=True)
            if audio is None:
                continue

            prepared_audio = self.prepare_audio(audio)

            is_accepted, features = self.apply_audio_file_filter(prepared_audio)
            if not is_accepted:
                continue

            start_time, end_time = self.get_start_end_time(prepared_audio)

            mpeg_analyzer = AudioMpegFeaturesAnalyzer(prepared_audio)
            mpeg7_features = mpeg_analyzer.get_features()

            misc_analyzer = AudioMiscFeaturesAnalyzer(prepared_audio, mpeg_analyzer.rms_per_frame)
            misc_features = misc_analyzer.get_features()

            properties = {
                'audio_file_path': absolute_file_path,
                'filename': Path(absolute_file_path).stem,
                'original_duration': features['original_duration'],
                'rms': features['rms'],
                'category': category,
                **mpeg7_features,
                **misc_features,
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

    def apply_audio_file_filter(self, raw_audio: np.ndarray) -> (bool, dict):
        """
        Apply filters to an audio file to decide wether to load them save them in the dataframe
        :param input_path
        :param raw_audio PCM audio data
        :return
        """
        # Exclude all files with a duration longer than required
        original_duration = len(raw_audio) / float(config.SAMPLE_RATE)
        if original_duration > config.LIBRARY_AUDIO_FILE_MAX_DURATION_SEC:
            return False, {}

        # Exclude all files quieter than required
        rms = get_max_rms(raw_audio)
        if rms < config.LIBRARY_AUDIO_MIN_RMS:
            return False, {}

        features = {
            'original_duration': original_duration,
            'rms': rms
        }

        return True, features

    def get_category_by_path(self, input_path: Path) -> str | None:
        # Try to find required category in the audio file name
        for category in config.CATEGORIES:
            filename = input_path.stem
            keywords = config.CATEGORIES[category]

            negative_matches = 0
            for keyword in keywords['exclude']:
                if keyword in filename.lower():
                    negative_matches += 1
                    break

            # Ignore files with blacklisted keywords in the filename
            if negative_matches != 0:
                continue

            positive_matches = 0
            for keyword in keywords['include']:
                if keyword in filename.lower():
                    positive_matches += 1

            # Basically, if all keywords matched with a filename
            if positive_matches >= len(keywords['include']):
                return category

        return None

    def get_start_end_time(self, raw_audio: np.ndarray) -> [float, float | None]:
        shift_sec = config.AUDIO_SILENCE_SEC + 0.01
        end = None

        onsets = get_onsets(raw_audio)

        if len(onsets) == 0:
            return [0, None]
        if len(onsets) > 1:
            end = onsets[1] - shift_sec

        start = max(onsets[0] - shift_sec, 0.0)
        return [start, end]

    def prepare_audio(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        Conduct operations to prepare audio for feature extraction
        :param raw_audio:
        :return:
        """
        return shift_audio_by_sec(raw_audio, config.AUDIO_SILENCE_SEC)
