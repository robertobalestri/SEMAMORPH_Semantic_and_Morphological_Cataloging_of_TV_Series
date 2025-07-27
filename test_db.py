#!/usr/bin/env python3
"""
Test script to debug database initialization
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

def test_database_path():
    """Test database path construction"""
    print("Testing database path construction...")
    
    # Test the path construction logic
    db_name = os.getenv("DATABASE_NAME", "narrative.db")
    print(f"Database name from env: {db_name}")
    
    db_path = os.path.join("narrative_storage", db_name)
    print(f"Full database path: {db_path}")
    
    # Check if directory exists
    db_dir = os.path.dirname(db_path)
    print(f"Database directory: {db_dir}")
    print(f"Directory exists: {os.path.exists(db_dir)}")
    
    # Check if we can write to the directory
    try:
        test_file = os.path.join(db_dir, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✅ Can write to database directory")
    except Exception as e:
        print(f"❌ Cannot write to database directory: {e}")

def test_database_init():
    """Test database initialization"""
    print("\nTesting database initialization...")
    
    try:
        from backend.src.narrative_storage_management.repositories import DatabaseSessionManager
        print("✅ Successfully imported DatabaseSessionManager")
        
        # Try to create the database manager
        db_manager = DatabaseSessionManager()
        print("✅ Successfully created DatabaseSessionManager")
        
        # Check if database file was created
        db_path = os.path.join("narrative_storage", "narrative.db")
        if os.path.exists(db_path):
            print(f"✅ Database file created: {db_path}")
            print(f"   File size: {os.path.getsize(db_path)} bytes")
        else:
            print(f"❌ Database file not found: {db_path}")
            
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_path()
    test_database_init() 