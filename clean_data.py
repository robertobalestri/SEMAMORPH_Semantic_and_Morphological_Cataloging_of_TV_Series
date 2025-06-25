#!/usr/bin/env python3
"""
Data Cleaning Script for SEMAMORPH Project

This script cleans the data folder by removing only the multiagent suggested episode arcs files.
All other files will be preserved, including:
- *_plot.txt files (original plot files)
- *_full_dialogues.json files (original dialogue files)
- *_plot_simplified.txt (simplified plots)
- *_plot_named.txt (named plots)
- *_plot_entities_*.txt (entity-processed plots)
- *_refined_entities.json (refined entity files)
- *_raw_spacy_entities.json (raw entity files)
- *_plot_semantic_segments.json (semantic segments)
- *_summarized_plot.txt (summarized plots)
- *_extracted_entities.json (extracted entities)

Only the following files will be deleted:
- *_multiagent_suggested_episode_arcs.json (multiagent narrative arcs)

This allows for regenerating narrative arcs while preserving all other processing results.

Usage:
    python clean_data.py [--dry-run] [--series SERIES] [--season SEASON]

Options:
    --dry-run    Show what would be deleted without actually deleting
    --series     Clean only specific series (e.g., GA, FIABA)
    --season     Clean only specific season (e.g., S01, S02)

Examples:
    python clean_data.py --dry-run                    # Preview all deletions
    python clean_data.py                              # Clean all data
    python clean_data.py --series GA                  # Clean only GA series
    python clean_data.py --series GA --season S01     # Clean only GA S01
"""

import os
import argparse
import sys
from pathlib import Path
from typing import List, Set, Tuple


def get_files_to_preserve() -> Set[str]:
    """Get the set of file patterns that should be preserved."""
    # Note: This function is kept for compatibility, but now we preserve everything 
    # except multiagent suggested episode arcs
    return set()  # Empty set since we preserve everything except specific files


def should_preserve_file(filename: str, preserve_patterns: Set[str]) -> bool:
    """Check if a file should be preserved based on its name."""
    # Only delete multiagent suggested episode arcs files
    if filename.endswith('_multiagent_suggested_episode_arcs.json'):
        return False
    
    # Preserve all other files
    return True


def get_files_to_delete(directory: Path, preserve_patterns: Set[str]) -> List[Path]:
    """Get list of files to delete in a directory."""
    files_to_delete = []
    
    if not directory.exists() or not directory.is_dir():
        return files_to_delete
    
    for item in directory.iterdir():
        if item.is_file():
            if not should_preserve_file(item.name, preserve_patterns):
                files_to_delete.append(item)
        elif item.is_dir():
            # Recursively check subdirectories
            files_to_delete.extend(get_files_to_delete(item, preserve_patterns))
    
    return files_to_delete


def get_empty_directories(directory: Path, preserve_patterns: Set[str]) -> List[Path]:
    """Get list of directories that would be empty after deletion."""
    empty_dirs = []
    
    if not directory.exists() or not directory.is_dir():
        return empty_dirs
    
    for item in directory.iterdir():
        if item.is_dir():
            # Check subdirectories first
            empty_dirs.extend(get_empty_directories(item, preserve_patterns))
            
            # Check if this directory would be empty after cleanup
            remaining_items = []
            for subitem in item.iterdir():
                if subitem.is_file():
                    if should_preserve_file(subitem.name, preserve_patterns):
                        remaining_items.append(subitem)
                elif subitem.is_dir():
                    # Directory will remain if it has preserved files or non-empty subdirs
                    if get_files_to_preserve_in_dir(subitem, preserve_patterns) or \
                       not would_be_empty_after_cleanup(subitem, preserve_patterns):
                        remaining_items.append(subitem)
            
            if not remaining_items:
                empty_dirs.append(item)
    
    return empty_dirs


def get_files_to_preserve_in_dir(directory: Path, preserve_patterns: Set[str]) -> List[Path]:
    """Get files that would be preserved in a directory."""
    preserved_files = []
    
    if not directory.exists() or not directory.is_dir():
        return preserved_files
    
    for item in directory.iterdir():
        if item.is_file():
            if should_preserve_file(item.name, preserve_patterns):
                preserved_files.append(item)
        elif item.is_dir():
            preserved_files.extend(get_files_to_preserve_in_dir(item, preserve_patterns))
    
    return preserved_files


def would_be_empty_after_cleanup(directory: Path, preserve_patterns: Set[str]) -> bool:
    """Check if a directory would be empty after cleanup."""
    if not directory.exists() or not directory.is_dir():
        return True
    
    for item in directory.iterdir():
        if item.is_file():
            if should_preserve_file(item.name, preserve_patterns):
                return False
        elif item.is_dir():
            if not would_be_empty_after_cleanup(item, preserve_patterns):
                return False
    
    return True


def clean_data_folder(data_dir: Path, series_filter: str = None, season_filter: str = None, 
                     dry_run: bool = False) -> Tuple[int, int]:
    """
    Clean the data folder while preserving source files.
    
    Returns:
        Tuple of (files_deleted, directories_deleted)
    """
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        return 0, 0
    
    preserve_patterns = get_files_to_preserve()
    files_deleted = 0
    directories_deleted = 0
    
    print(f"Cleaning data folder: {data_dir}")
    print(f"Will delete only: *_multiagent_suggested_episode_arcs.json files")
    print(f"Will preserve: ALL other files")
    if series_filter:
        print(f"Series filter: {series_filter}")
    if season_filter:
        print(f"Season filter: {season_filter}")
    print(f"Dry run: {dry_run}")
    print("-" * 50)
    
    # Iterate through series directories
    for series_dir in data_dir.iterdir():
        if not series_dir.is_dir() or series_dir.name.startswith('.'):
            continue
            
        # Apply series filter
        if series_filter and series_dir.name != series_filter:
            continue
            
        print(f"\nProcessing series: {series_dir.name}")
        
        # Iterate through season directories
        for season_dir in series_dir.iterdir():
            if not season_dir.is_dir() or season_dir.name.startswith('.'):
                continue
                
            # Apply season filter
            if season_filter and season_dir.name != season_filter:
                continue
                
            print(f"  Processing season: {season_dir.name}")
            
            # Get files to delete in this season
            files_to_delete = get_files_to_delete(season_dir, preserve_patterns)
            
            # Delete files
            for file_path in files_to_delete:
                rel_path = file_path.relative_to(data_dir)
                if dry_run:
                    print(f"    [DRY RUN] Would delete file: {rel_path}")
                    files_deleted += 1  # Count files in dry run too
                else:
                    try:
                        file_path.unlink()
                        print(f"    Deleted file: {rel_path}")
                        files_deleted += 1
                    except Exception as e:
                        print(f"    Error deleting file {rel_path}: {e}")
            
            # Clean up empty directories (bottom-up)
            empty_dirs = get_empty_directories(season_dir, preserve_patterns)
            # Sort by depth (deepest first) to delete from bottom up
            empty_dirs.sort(key=lambda p: len(p.parts), reverse=True)
            
            for dir_path in empty_dirs:
                if dir_path.exists() and dir_path != season_dir:  # Don't delete season dir itself
                    rel_path = dir_path.relative_to(data_dir)
                    if dry_run:
                        print(f"    [DRY RUN] Would delete empty directory: {rel_path}")
                        directories_deleted += 1  # Count directories in dry run too
                    else:
                        try:
                            dir_path.rmdir()
                            print(f"    Deleted empty directory: {rel_path}")
                            directories_deleted += 1
                        except Exception as e:
                            print(f"    Error deleting directory {rel_path}: {e}")
    
    return files_deleted, directories_deleted


def main():
    parser = argparse.ArgumentParser(
        description="Clean data folder while preserving source files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--series',
        type=str,
        help='Clean only specific series (e.g., GA, FIABA)'
    )
    
    parser.add_argument(
        '--season',
        type=str,
        help='Clean only specific season (e.g., S01, S02)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory (default: data)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.season and not args.series:
        print("Error: --season requires --series to be specified")
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        sys.exit(1)
    
    # Confirm deletion if not dry run
    if not args.dry_run:
        filter_info = ""
        if args.series:
            filter_info = f" (series: {args.series}"
            if args.season:
                filter_info += f", season: {args.season}"
            filter_info += ")"
        
        response = input(f"\nThis will delete only multiagent suggested episode arcs files in {data_dir}{filter_info}.\n"
                        f"All other files (*_plot.txt, *_full_dialogues.json, processed files, etc.) will be preserved.\n"
                        f"Continue? (y/N): ")
        
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            sys.exit(0)
    
    # Perform cleanup
    files_deleted, dirs_deleted = clean_data_folder(
        data_dir, 
        args.series, 
        args.season, 
        args.dry_run
    )
    
    # Summary
    print("\n" + "=" * 50)
    action = "Would delete" if args.dry_run else "Deleted"
    print(f"Summary: {action} {files_deleted} files and {dirs_deleted} directories")
    
    if args.dry_run:
        print("\nTo actually perform the cleanup, run without --dry-run")


if __name__ == "__main__":
    main()
