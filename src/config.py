"""
Configuration loader for SEMAMORPH project.

This module provides functionality to load and access configuration settings
from the config.ini file.
"""

import configparser
import os
from pathlib import Path
from typing import Any, Optional


class Config:
    """Configuration manager for SEMAMORPH project."""
    
    def __init__(self, config_file: str = "config.ini"):
        """
        Initialize configuration from file.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            # Create default config if file doesn't exist
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration file."""
        self.config['processing'] = {
            'use_original_plot_as_summary': 'false',
            'max_summary_length': '2000',
            'summarization_model': 'cheap'
        }
        
        self.config['paths'] = {
            'data_dir': 'data',
            'narrative_storage_dir': 'narrative_storage'
        }
        
        self.config['api'] = {
            'host': '0.0.0.0',
            'port': '8000'
        }
        
        self.config['logging'] = {
            'level': 'INFO',
            'log_to_file': 'true',
            'log_file': 'processing.log'
        }
        
        # Save default config
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get_bool(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean value from config."""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def get_str(self, section: str, key: str, fallback: str = "") -> str:
        """Get string value from config."""
        return self.config.get(section, key, fallback=fallback)
    
    def get_int(self, section: str, key: str, fallback: int = 0) -> int:
        """Get integer value from config."""
        return self.config.getint(section, key, fallback=fallback)
    
    def get_float(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Get float value from config."""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def set_value(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value."""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    # Convenience properties for commonly used settings
    @property
    def use_original_plot_as_summary(self) -> bool:
        """Whether to use original plot as summary instead of generating new one."""
        return self.get_bool('processing', 'use_original_plot_as_summary', False)
    
    @property
    def max_summary_length(self) -> int:
        """Maximum length for plot summarization."""
        return self.get_int('processing', 'max_summary_length', 2000)
    
    @property
    def summarization_model(self) -> str:
        """Model to use for summarization."""
        return self.get_str('processing', 'summarization_model', 'cheap')
    
    @property
    def pronoun_replacement_batch_size(self) -> int:
        """Batch size for pronoun replacement processing."""
        return self.get_int('processing', 'pronoun_replacement_batch_size', 18)
    
    @property
    def pronoun_replacement_context_size(self) -> int:
        """Context size for pronoun replacement processing."""
        return self.get_int('processing', 'pronoun_replacement_context_size', 6)
    
    @property
    def text_simplification_batch_size(self) -> int:
        """Batch size for text simplification processing."""
        return self.get_int('processing', 'text_simplification_batch_size', 10)
    
    @property
    def semantic_segmentation_window_size(self) -> int:
        """Window size for semantic segmentation processing."""
        return self.get_int('processing', 'semantic_segmentation_window_size', 20)
    
    @property
    def semantic_correction_batch_size(self) -> int:
        """Batch size for semantic segment correction."""
        return self.get_int('processing', 'semantic_correction_batch_size', 3)
    
    @property
    def narrative_arc_extraction_method(self) -> str:
        """Method to use for narrative arc extraction."""
        return self.get_str('processing', 'narrative_arc_extraction_method', 'original')
    
    @property
    def data_dir(self) -> str:
        """Data directory path."""
        return self.get_str('paths', 'data_dir', 'data')
    
    @property
    def narrative_storage_dir(self) -> str:
        """Narrative storage directory path."""
        return self.get_str('paths', 'narrative_storage_dir', 'narrative_storage')
    
    @property
    def api_host(self) -> str:
        """API server host."""
        return self.get_str('api', 'host', '0.0.0.0')
    
    @property
    def api_port(self) -> int:
        """API server port."""
        return self.get_int('api', 'port', 8000)
    
    @property
    def log_level(self) -> str:
        """Logging level."""
        return self.get_str('logging', 'level', 'INFO')
    
    @property
    def log_to_file(self) -> bool:
        """Whether to log to file."""
        return self.get_bool('logging', 'log_to_file', True)
    
    @property
    def log_file(self) -> str:
        """Log file path."""
        return self.get_str('logging', 'log_file', 'processing.log')


# Global configuration instance
config = Config()
