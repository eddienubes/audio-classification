import pandas
import utils
import config
from pathlib import Path
from audio_library_reader import AudioLibraryReader

path = Path(config.LIBRARY_DATAFRAME_PATH)

if not path.is_file():
    source_path = Path(config.LIBRARY_SOURCE_PATH)
    reader = AudioLibraryReader()
    audio_files = reader.read(source_path)
    print(f'Writing {len(audio_files)} to pickle')
    audio_files.to_pickle(config.LIBRARY_DATAFRAME_PATH)

audio_files: pandas.DataFrame = pandas.read_pickle(config.LIBRARY_DATAFRAME_PATH)
utils.disable_pandas_print_limit()
print(audio_files.columns)
print(audio_files['category'].value_counts())
print(audio_files.drop('audio_file_path', axis=1).head())
# print(audio_files['category'].value_counts())
