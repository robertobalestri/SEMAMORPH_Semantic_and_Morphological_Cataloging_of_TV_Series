import logging
from colorlog import ColoredFormatter
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def setup_logging(name: str) -> logging.Logger:
    """
    Set up colored logging for the given logger name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Set up colored logging
    log_format = "%(log_color)s%(levelname)s:%(name)s: %(message)s"
    formatter = ColoredFormatter(log_format, 
        datefmt=None,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)
    logging.getLogger('sqlalchemy.orm').setLevel(logging.DEBUG)

    return logger