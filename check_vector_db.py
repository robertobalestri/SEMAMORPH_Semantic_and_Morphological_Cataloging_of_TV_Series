#!/usr/bin/env python3
"""
Check Vector Database Contents for Recap Generation.

This script checks what's actually in the vector database to understand
why the recap generation found 0 events.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
backend_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', '.env')
load_dotenv(backend_env_path, override=True)

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from src.narrative_storage_management.vector_store_service import VectorStoreService
from src.utils.logger_utils import setup_logging

logger = setup_logging(__name__)

def check_vector_database():
    """Check what's in the vector database."""
    print("🔍 CHECKING VECTOR DATABASE CONTENTS")
    print("=" * 60)
    
    try:
        vector_store = VectorStoreService()
        
        # Check if vector store is accessible
        print("📊 Vector Database Status:")
        
        # Try to query without filters to see what's there
        print("\n🔍 Searching for any narrative arcs (no filters)...")
        results = vector_store.find_similar_events(
            query="Meredith Grey",
            n_results=10,
            series="GA",
            min_confidence=0.0  # No confidence filter
        )
        
        print(f"✅ Found {len(results)} total events for 'Meredith Grey'")
        
        # Show some results
        if results:
            print("\n📋 Sample results:")
            for i, result in enumerate(results[:5], 1):
                metadata = result.get('metadata', {})
                content = result.get('page_content', '')[:100] + "..."
                
                print(f"   {i}. {metadata.get('series', 'Unknown')} {metadata.get('season', 'Unknown')} {metadata.get('episode', 'Unknown')}")
                print(f"      Arc: {metadata.get('arc_title', 'Unknown')}")
                print(f"      Arc ID: {metadata.get('main_arc_id', 'Unknown')}")
                print(f"      Content: {content}")
                print(f"      Distance: {result.get('cosine_distance', 'Unknown')}")
                print()
        
        # Check for specific series/season/episode patterns
        print("\n🔍 Checking for Grey's Anatomy Season 1 content...")
        ga_results = vector_store.find_similar_events(
            query="surgical intern",
            n_results=20,
            series="GA",
            min_confidence=0.0
        )
        
        print(f"✅ Found {len(ga_results)} events for GA surgical interns")
        
        # Group by episodes to see coverage
        episodes = {}
        for result in ga_results:
            metadata = result.get('metadata', {})
            season = metadata.get('season', 'Unknown')
            episode = metadata.get('episode', 'Unknown')
            key = f"{season}_{episode}"
            episodes[key] = episodes.get(key, 0) + 1
        
        print(f"\n📊 Episode coverage:")
        for ep, count in sorted(episodes.items()):
            print(f"   {ep}: {count} events")
        
        # Check for specific narrative arc IDs from the test
        test_arc_ids = [
            "8478c751-2f62-4f25-9d9b-663d0ba51c5a",  # Meredith & Ellis
            "6ac77639-cc0d-49ed-a35b-81cfbfdb9991",  # Meredith & Derek  
            "e490ee68-f5ff-46f3-9ad8-1c8964ad1e61",  # Intern rivalries
            "ed5bc0bf-5654-47fb-b429-91a09dd7fb15",  # Bailey mentorship
            "3a745869-d4d0-442f-a92d-c2e7adb32fad"   # Power dynamics
        ]
        
        print(f"\n🔍 Checking for specific narrative arc IDs from test...")
        for i, arc_id in enumerate(test_arc_ids, 1):
            print(f"\n   Arc {i}: {arc_id[:8]}...")
            arc_results = vector_store.find_similar_events(
                query="characters dialogue",
                n_results=5,
                series="GA", 
                narrative_arc_ids=[arc_id],
                min_confidence=0.0
            )
            print(f"   Found: {len(arc_results)} events")
            
            if arc_results:
                for result in arc_results[:2]:  # Show first 2
                    metadata = result.get('metadata', {})
                    print(f"      • {metadata.get('season', 'Unknown')} {metadata.get('episode', 'Unknown')}: {metadata.get('arc_title', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector database check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_vector_database()
    
    if success:
        print("\n✅ Vector database check completed")
        print("\n💡 Key insights:")
        print("   • If 0 events found, database may be empty for previous episodes")
        print("   • If events found but 0 in recap test, arc ID filtering may be too strict")
        print("   • Events should have NO duration limits in retrieval")
        print("   • LLM subtitle selection should trim to ~10s dialogue")
    else:
        print("\n❌ Vector database check failed")
    
    exit(0 if success else 1)
