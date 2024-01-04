import os
import librosa
import numpy as np
import config
import pandas

from pathlib import Path
from utils import exc_to_message, get_logger


class AudioLibraryReader:
    def __init__(self):
        self.logger = get_logger(AudioLibraryReader.__name__)

    def read(self, input_dir_path: Path) -> pandas.DataFrame:
        self.logger.info(f'Searching for audio within {input_dir_path}')
        dataframe_rows = []
        for input_file in input_dir_path.glob('**/*.*'):
            if len(dataframe_rows) >= config.LIBRARY_SIZE_LIMIT:
                break

            absolute_file_path = input_file.resolve().as_posix()
            if not self.can_load_audio(absolute_file_path):
                continue

            audio = self.load_raw_audio(absolute_file_path, fast=True)

            properties = {
                'audio_path': absolute_file_path,
                'store_path': input_file.as_posix(),
                'filename': Path(absolute_file_path).stem,
                'start_time': 0.0,
                'end_time': np.NaN,
                'original_duration': len(audio) / float(config.DEFAULT_SAMPLE_RATE)
            }

            dataframe_rows.append(properties)

        return pandas.DataFrame(dataframe_rows)

    def can_load_audio(self, absolute_file_path: str) -> bool:
        if not os.path.isfile(absolute_file_path):
            return False

        try:
            librosa.load(absolute_file_path, mono=True, res_type='kaiser_fast', duration=.01)
            return True
        except BaseException:
            self.logger.warning(exc_to_message())
            return False

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
