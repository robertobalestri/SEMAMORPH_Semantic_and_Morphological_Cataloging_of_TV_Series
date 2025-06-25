#!/usr/bin/env python3
"""
Configuration Management Script for SEMAMORPH Project

This script allows users to view and modify configuration settings for the SEMAMORPH project.
"""

import argparse
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.getcwd())

try:
    from src.config import config
except ImportError as e:
    print(f"Error importing configuration: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def show_config():
    """Display current configuration settings."""
    print("SEMAMORPH Configuration Settings")
    print("=" * 40)
    print()
    print("[Processing]")
    print(f"  use_original_plot_as_summary: {config.use_original_plot_as_summary}")
    print(f"  max_summary_length: {config.max_summary_length}")
    print(f"  summarization_model: {config.summarization_model}")
    print()
    print("[Paths]")
    print(f"  data_dir: {config.data_dir}")
    print(f"  narrative_storage_dir: {config.narrative_storage_dir}")
    print()
    print("[API]")
    print(f"  host: {config.api_host}")
    print(f"  port: {config.api_port}")
    print()
    print("[Logging]")
    print(f"  level: {config.log_level}")
    print(f"  log_to_file: {config.log_to_file}")
    print(f"  log_file: {config.log_file}")
    sys.stdout.flush()  # Force output


def set_original_plot_as_summary(use_original: bool):
    """Set whether to use original plot as summary."""
    config.set_value('processing', 'use_original_plot_as_summary', str(use_original).lower())
    config.save_config()
    print(f"✓ Set use_original_plot_as_summary to: {use_original}")
    print()
    if use_original:
        print("ℹ  Now when processing episodes, the original plot file will be used")
        print("   as the summarized plot instead of generating a new summary.")
    else:
        print("ℹ  Now when processing episodes, a new summarized plot will be generated")
        print("   using the configured LLM model.")


def set_summarization_model(model: str):
    """Set the summarization model."""
    valid_models = ['cheap', 'intelligent']
    if model not in valid_models:
        print(f"❌ Error: Model must be one of: {', '.join(valid_models)}")
        return
    
    config.set_value('processing', 'summarization_model', model)
    config.save_config()
    print(f"✓ Set summarization_model to: {model}")


def set_max_summary_length(length: int):
    """Set the maximum summary length."""
    if length < 100:
        print("❌ Error: Maximum summary length must be at least 100 characters")
        return
    
    config.set_value('processing', 'max_summary_length', str(length))
    config.save_config()
    print(f"✓ Set max_summary_length to: {length}")


def interactive_config():
    """Interactive configuration mode."""
    print("SEMAMORPH Interactive Configuration")
    print("=" * 40)
    print()
    
    while True:
        print("Current settings:")
        print(f"1. Use original plot as summary: {config.use_original_plot_as_summary}")
        print(f"2. Summarization model: {config.summarization_model}")
        print(f"3. Max summary length: {config.max_summary_length}")
        print()
        print("Options:")
        print("1) Toggle original plot as summary")
        print("2) Change summarization model")
        print("3) Change max summary length")
        print("4) Show all settings")
        print("5) Exit")
        print()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            current = config.use_original_plot_as_summary
            new_value = not current
            set_original_plot_as_summary(new_value)
            
        elif choice == '2':
            print("Available models:")
            print("- cheap: Faster, less accurate")
            print("- intelligent: Slower, more accurate")
            model = input("Enter model (cheap/intelligent): ").strip().lower()
            set_summarization_model(model)
            
        elif choice == '3':
            try:
                length = int(input("Enter max summary length (characters): ").strip())
                set_max_summary_length(length)
            except ValueError:
                print("❌ Error: Please enter a valid number")
                
        elif choice == '4':
            print()
            show_config()
            print()
            
        elif choice == '5':
            print("Configuration saved. Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Manage SEMAMORPH configuration settings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--show', 
        action='store_true',
        help='Show current configuration settings'
    )
    
    parser.add_argument(
        '--use-original-plot', 
        type=str,
        choices=['true', 'false'],
        help='Set whether to use original plot as summary (true/false)'
    )
    
    parser.add_argument(
        '--summarization-model',
        type=str,
        choices=['cheap', 'intelligent'],
        help='Set the model for summarization'
    )
    
    parser.add_argument(
        '--max-summary-length',
        type=int,
        help='Set maximum summary length in characters'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show configuration by default
    if len(sys.argv) == 1:
        show_config()
        return
    
    # Handle arguments
    if args.show:
        show_config()
    
    if args.use_original_plot:
        use_original = args.use_original_plot.lower() == 'true'
        set_original_plot_as_summary(use_original)
    
    if args.summarization_model:
        set_summarization_model(args.summarization_model)
    
    if args.max_summary_length:
        set_max_summary_length(args.max_summary_length)
    
    if args.interactive:
        interactive_config()


if __name__ == "__main__":
    main()
