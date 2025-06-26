#!/usr/bin/env python3
"""Simple configuration management for SEMAMORPH project."""

import sys
import os
sys.path.insert(0, os.getcwd())

from src.config import config

def show_config():
    print("SEMAMORPH Configuration Settings")
    print("=" * 40)
    print(f"Use original plot as summary: {config.use_original_plot_as_summary}")
    print(f"Summarization model: {config.summarization_model}")
    print(f"Max summary length: {config.max_summary_length}")
    print(f"Data directory: {config.data_dir}")
    print()
    print("LLM Batch Sizes:")
    print(f"  Pronoun replacement batch: {config.pronoun_replacement_batch_size}")
    print(f"  Pronoun replacement context: {config.pronoun_replacement_context_size}")
    print(f"  Text simplification batch: {config.text_simplification_batch_size}")
    print(f"  Semantic segmentation window: {config.semantic_segmentation_window_size}")
    print(f"  Semantic correction batch: {config.semantic_correction_batch_size}")

def set_use_original(value):
    config.set_value('processing', 'use_original_plot_as_summary', str(value).lower())
    config.save_config()
    print(f"Set use_original_plot_as_summary to: {value}")

def set_batch_size(setting_name, value):
    """Set a batch size configuration value."""
    try:
        value = int(value)
        if value < 1:
            print(f"Error: Batch size must be at least 1")
            return
        config.set_value('processing', setting_name, str(value))
        config.save_config()
        print(f"Set {setting_name} to: {value}")
    except ValueError:
        print(f"Error: Invalid number: {value}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_config()
    elif len(sys.argv) == 2 and sys.argv[1] == "show":
        show_config()
    elif len(sys.argv) == 3 and sys.argv[1] == "set-original":
        value = sys.argv[2].lower() == 'true'
        set_use_original(value)
    elif len(sys.argv) == 4 and sys.argv[1] == "set-batch":
        setting_name = sys.argv[2]
        value = sys.argv[3]
        
        valid_settings = [
            'pronoun_replacement_batch_size',
            'pronoun_replacement_context_size', 
            'text_simplification_batch_size',
            'semantic_segmentation_window_size',
            'semantic_correction_batch_size'
        ]
        
        if setting_name in valid_settings:
            set_batch_size(setting_name, value)
        else:
            print(f"Error: Invalid setting name. Valid options: {', '.join(valid_settings)}")
    else:
        print("Usage:")
        print("  python config_manager.py                     # Show config")
        print("  python config_manager.py show                # Show config")
        print("  python config_manager.py set-original true   # Use original plot")
        print("  python config_manager.py set-original false  # Generate summary")
        print()
        print("Batch size settings:")
        print("  python config_manager.py set-batch pronoun_replacement_batch_size 30")
        print("  python config_manager.py set-batch text_simplification_batch_size 20")
        print("  python config_manager.py set-batch semantic_segmentation_window_size 40")
        print("  python config_manager.py set-batch semantic_correction_batch_size 5")
