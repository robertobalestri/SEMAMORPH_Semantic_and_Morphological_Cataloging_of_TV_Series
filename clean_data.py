#!/usr/bin/env python3
"""
Data Cleaning Script for SEMAMORPH Project

This script cleans the data folder while preserving only the original source files:
- *.srt files (subtitle files)
- entities.json files (original entity files)

All other processed files will be deleted, including:
- *_plot.txt (generated plot files)
- *_plot_scenes.json (plot scenes)
- *_scene_timestamps.json (scene timestamps)
- *_plot_named.txt (named plots)
- *_plot_entities_*.txt (entity-processed plots)
- *_refined_entities.json (refined entity files)
- *_raw_spacy_entities.json (raw entity files)
- *_plot_semantic_segments.json (semantic segments)
- *_multiagent_suggested_episode_arcs.json (narrative arcs)

Additionally, this script will:
- Clean narrative.db database (drop all tables except 'character' and 'character_appellation')
- Remove the chroma_db folder completely

This allows for a fresh reprocessing of the data while maintaining the original source material.

Usage:
    python clean_data.py [--dry-run] [--series SERIES] [--season SEASON] [--no-db] [--no-chroma]

Options:
    --dry-run    Show what would be deleted without actually deleting
    --series     Clean only specific series (e.g., GA, FIABA)
    --season     Clean only specific season (e.g., S01, S02)
    --no-db      Skip database cleaning
    --no-chroma  Skip chroma_db folder removal

Examples:
    python clean_data.py --dry-run                    # Preview all deletions
    python clean_data.py                              # Clean all data, db, and chroma
    python clean_data.py --series GA                  # Clean only GA series
    python clean_data.py --series GA --season S01     # Clean only GA S01
    python clean_data.py --no-db --no-chroma          # Clean only file data
"""

import os
import argparse
import sys
import sqlite3
import shutil
from pathlib import Path
from typing import List, Set, Tuple


def get_files_to_preserve() -> Set[str]:
    """Get the set of file patterns that should be preserved."""
    return {
        '.srt',
        #'entities.json',
        #'timestamps.json',
        #'_plot.txt',
        #'_plot_scenes.json',
    }


def clean_database(db_path: Path, dry_run: bool = False) -> int:
    """
    Clean the narrative.db database by dropping all tables except 'character' and 'character_appellation'.
    
    Returns:
        Number of tables dropped
    """
    if not db_path.exists():
        print(f"Database {db_path} does not exist - skipping database cleanup")
        return 0
    
    tables_dropped = 0
    tables_to_preserve = {}# {} #{'character', 'character_appellation'}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_tables = [row[0] for row in cursor.fetchall()]
        
        # Determine tables to drop
        tables_to_drop = [table for table in all_tables if table not in tables_to_preserve]
        
        print(f"\nDatabase cleanup for: {db_path}")
        print(f"Tables to preserve: {', '.join(tables_to_preserve)}")
        print(f"Tables to drop: {', '.join(tables_to_drop) if tables_to_drop else 'None'}")
        
        if dry_run:
            print(f"[DRY RUN] Would drop {len(tables_to_drop)} tables")
            tables_dropped = len(tables_to_drop)
        else:
            for table in tables_to_drop:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
                    print(f"  Dropped table: {table}")
                    tables_dropped += 1
                except sqlite3.Error as e:
                    print(f"  Error dropping table {table}: {e}")
            
            conn.commit()
            print(f"Successfully dropped {tables_dropped} tables")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error cleaning database: {e}")
    
    return tables_dropped


def remove_chroma_folder(chroma_path: Path, dry_run: bool = False) -> bool:
    """
    Remove the chroma_db folder completely.
    
    Returns:
        True if folder was removed (or would be removed in dry run), False otherwise
    """
    if not chroma_path.exists():
        print(f"Chroma folder {chroma_path} does not exist - skipping")
        return False
    
    print(f"\nChroma folder cleanup: {chroma_path}")
    
    if dry_run:
        print(f"[DRY RUN] Would remove entire chroma_db folder and all contents")
        return True
    else:
        try:
            shutil.rmtree(chroma_path)
            print(f"Successfully removed chroma_db folder")
            return True
        except Exception as e:
            print(f"Error removing chroma_db folder: {e}")
            return False


def should_preserve_file(filename: str, preserve_patterns: Set[str]) -> bool:
    """Check if a file should be preserved based on its name."""
    # Check each preservation pattern
    for pattern in preserve_patterns:
        if filename.endswith(pattern):
            # For file extensions like .srt
            return True
    
    return False


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
                     dry_run: bool = False, clean_db: bool = True, clean_chroma: bool = True) -> Tuple[int, int, int, bool]:
    """
    Clean the data folder while preserving source files.
    
    Returns:
        Tuple of (files_deleted, directories_deleted, tables_dropped, chroma_removed)
    """
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        return 0, 0, 0, False
    
    preserve_patterns = get_files_to_preserve()
    files_deleted = 0
    directories_deleted = 0
    tables_dropped = 0
    chroma_removed = False
    
    print(f"Cleaning data folder: {data_dir}")
    print(f"Preserving files ending with: {', '.join(preserve_patterns)}")
    if series_filter:
        print(f"Series filter: {series_filter}")
    if season_filter:
        print(f"Season filter: {season_filter}")
    print(f"Database cleanup: {'Yes' if clean_db else 'No'}")
    print(f"Chroma cleanup: {'Yes' if clean_chroma else 'No'}")
    print(f"Dry run: {dry_run}")
    print("-" * 50)
    
    # Clean database if requested
    if clean_db:
        narrative_db_path = Path("narrative_storage/narrative.db")
        tables_dropped = clean_database(narrative_db_path, dry_run)
    
    # Remove chroma_db folder if requested
    if clean_chroma:
        chroma_db_path = Path("narrative_storage/chroma_db")
        chroma_removed = remove_chroma_folder(chroma_db_path, dry_run)
    
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
    
    return files_deleted, directories_deleted, tables_dropped, chroma_removed


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
    
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Skip database cleaning'
    )
    
    parser.add_argument(
        '--no-chroma',
        action='store_true',
        help='Skip chroma_db folder removal'
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
        
        cleanup_items = []
        if not args.no_db:
            cleanup_items.append("database tables (except character and character_appellation)")
        if not args.no_chroma:
            cleanup_items.append("chroma_db folder")
        cleanup_items.append("processed files")
        
        cleanup_text = ", ".join(cleanup_items)
        
        response = input(f"\nThis will delete {cleanup_text} in {data_dir}{filter_info}.\n"
                        f"Original *.srt and entities.json files will be preserved.\n"
                        f"Continue? (y/N): ")
        
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            sys.exit(0)
    
    # Perform cleanup
    files_deleted, dirs_deleted, tables_dropped, chroma_removed = clean_data_folder(
        data_dir, 
        args.series, 
        args.season, 
        args.dry_run,
        clean_db=not args.no_db,
        clean_chroma=not args.no_chroma
    )
    
    # Summary
    print("\n" + "=" * 50)
    action = "Would delete" if args.dry_run else "Deleted"
    print(f"Summary:")
    print(f"  {action} {files_deleted} files and {dirs_deleted} directories")
    
    if not args.no_db:
        print(f"  {action} {tables_dropped} database tables")
    
    if not args.no_chroma:
        chroma_action = "Would remove" if args.dry_run else ("Removed" if chroma_removed else "Failed to remove")
        print(f"  {chroma_action} chroma_db folder")
    
    if args.dry_run:
        print("\nTo actually perform the cleanup, run without --dry-run")


if __name__ == "__main__":
    main()
