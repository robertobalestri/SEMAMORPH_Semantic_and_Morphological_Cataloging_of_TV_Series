#!/usr/bin/env python3
"""
Simple script to search events in a narrative arc using vector embeddings.

USAGE:
1. Edit the phrase2search variable below
2. Run: python search_events.py
"""

import os
import sys
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# ========================================
# EDIT THIS SECTION
# ========================================

phrase2search = {
    "phrase": "Asshole",
    "narrative_arc_name": "The Interns' Rivalries And Bonds In Surgical Residency"
}

# ========================================
# SCRIPT - DON'T EDIT BELOW
# ========================================

from backend.src.utils.logger_utils import setup_logging
from backend.src.narrative_storage_management.vector_store_service import VectorStoreService
from backend.src.narrative_storage_management.repositories import DatabaseSessionManager, NarrativeArcRepository

logger = setup_logging(__name__)

def main():
    search_phrase = phrase2search["phrase"]
    arc_name = phrase2search["narrative_arc_name"]
    
    print(f"🔍 Searching for: '{search_phrase}'")
    print(f"📚 In arc: '{arc_name}'")
    print("=" * 60)
    
    try:
        # Initialize services
        vector_service = VectorStoreService(collection_name="narrative_arcs")
        db_manager = DatabaseSessionManager()
        
        # Find the arc
        arc_id = None
        arc_title = None
        with db_manager.session_scope() as session:
            arc_repo = NarrativeArcRepository(session)
            arcs = arc_repo.get_all()
            
            # Search for matching arc
            for arc in arcs:
                if arc_name.lower() in arc.title.lower():
                    arc_id = arc.id
                    arc_title = arc.title
                    break
            
            if not arc_id:
                print(f"❌ Arc '{arc_name}' not found!")
                print("Available arcs:")
                for arc in arcs[:10]:
                    print(f"  - {arc.title} ({arc.series})")
                return
        
        print(f"✅ Found arc: {arc_title} (ID: {arc_id})")
        print()
        
        # Search events using vector similarity - Use the collection directly for arc filtering
        results = vector_service.collection.similarity_search_with_score(
            query=search_phrase,
            k=10,
            filter={"$and": [
                {"doc_type": "event"},
                {"main_arc_id": arc_id}
            ]}
        )
        
        if not results:
            print("❌ No events found!")
            return
        
        print(f"📋 Found {len(results)} events:")
        print()
        
        # Display results
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            print(f"#{i} (Similarity: {score:.3f})")
            print(f"   Content: {doc.page_content}")
            print(f"   Episode: {metadata.get('series', 'N/A')} {metadata.get('season', 'N/A')}{metadata.get('episode', 'N/A')}")
            if metadata.get('start_timestamp'):
                print(f"   Time: {metadata.get('start_timestamp')} - {metadata.get('end_timestamp', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
