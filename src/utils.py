import sys
import traceback
import logging

import pandas
from rich.logging import RichHandler

import config


def exc_to_message() -> str:
    exc_type, exc_value, trace = sys.exc_info()
    exception_description_lines = traceback.format_exception_only(exc_type, exc_value)
    exception_description = ''.join(exception_description_lines).rstrip()
    return exception_description


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)

    formatter = logging.Formatter('%(message)s')

    handler = RichHandler()
    handler.setFormatter(formatter)
    handler.setLevel(config.LOG_LEVEL)

    logger.addHandler(handler)

    return logger


def disable_pandas_print_limit() -> None:
    pandas.set_option('display.width', None)
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None, 'display.max_columns', None)


def reset_pandas_print_limit() -> None:
    pandas.set_option('^display.', silent=True)
