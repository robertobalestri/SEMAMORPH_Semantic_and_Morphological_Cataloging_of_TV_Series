import logging
import sys
from colorlog import ColoredFormatter
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def setup_logging(name: str) -> logging.Logger:
    """
    Set up colored logging for the given logger name.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.propagate = False  # Prevent double logging
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create file handler
    file_handler = logging.FileHandler('api.log')
    
    # Create formatters and add it to the handlers
    color_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    console_handler.setFormatter(color_formatter)
    file_handler.setFormatter(file_formatter)
    
    # Set level for handlers and logger
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger