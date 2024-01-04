import pandas

from pathlib import Path
from utils import exc_to_message, get_logger
from audio_utils import *


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

            is_accepted, features = self.apply_audio_file_filter(audio)

            if not is_accepted:
                continue

            properties = {
                'audio_file_path': absolute_file_path,
                'filename': Path(absolute_file_path).stem,
                'original_duration': features['original_duration'],
                'rms': features['rms'],
                'category': category,
                # We fill up these later
                'start_time': np.NaN,
                'end_time': np.NaN,
            }

            dataframe_rows.append(properties)

        return pandas.DataFrame(dataframe_rows)

    def load_raw_audio(self, absolute_file_path: str, sample_rate: int = config.DEFAULT_SAMPLE_RATE, offset=0,
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
        original_duration = len(raw_audio) / float(config.DEFAULT_SAMPLE_RATE)
        if original_duration > config.LIBRARY_AUDIO_FILE_MAX_DURATION_SEC:
            return False, {}

        # Exclude all files quieter than required
        rms = get_rms(raw_audio)
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

            matches = 0
            for keyword in keywords:
                if keyword in filename:
                    matches += 1

            # Basically, if all keywords matched with a filename
            if matches >= len(keywords):
                return category

        return None
