#!/usr/bin/env python3
"""
Script to search for events related to a specific narrative arc using vector embeddings.

Usage:
    1. Set the phrase2search variable with your search phrase and narrative arc name
    2. Run the script: python search_arc_events.py
    3. View the semantically similar events from the specified narrative arc

The script uses vector embeddings to find events that are semantically similar to your search phrase
within the context of a specific narrative arc.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.src.utils.logger_utils import setup_logging
from backend.src.narrative_storage_management.vector_store_service import VectorStoreService
from backend.src.narrative_storage_management.repositories import DatabaseSessionManager, NarrativeArcRepository
from backend.src.ai_models.ai_models import get_embedding_model

logger = setup_logging(__name__)

# ========================================
# CONFIGURATION - EDIT THIS SECTION
# ========================================

phrase2search = {
    "phrase": "medical emergency surgery patient",
    "narrative_arc_name": "Medical Drama Arc"
}

# Optional: Limit results
MAX_RESULTS = 10
SIMILARITY_THRESHOLD = 0.7  # Only show results above this similarity score

# ========================================
# SEARCH IMPLEMENTATION
# ========================================

class ArcEventSearcher:
    def __init__(self):
        self.vector_service = VectorStoreService(collection_name="narrative_arcs")
        self.db_manager = DatabaseSessionManager()
        self.embedding_model = get_embedding_model()
        
    def find_arc_by_name(self, arc_name: str, series: Optional[str] = None) -> Optional[Dict]:
        """Find narrative arc by name (with optional series filter)."""
        try:
            with self.db_manager.session_scope() as session:
                arc_repo = NarrativeArcRepository(session)
                
                if series:
                    arcs = arc_repo.get_all(series=series)
                else:
                    arcs = arc_repo.get_all()
                
                # Search for arc by name (case-insensitive partial match)
                matching_arcs = []
                for arc in arcs:
                    if arc_name.lower() in arc.title.lower():
                        matching_arcs.append({
                            "id": arc.id,
                            "title": arc.title,
                            "series": arc.series,
                            "description": arc.description,
                            "arc_type": arc.arc_type
                        })
                
                if not matching_arcs:
                    logger.warning(f"No narrative arc found matching '{arc_name}'")
                    return None
                
                if len(matching_arcs) > 1:
                    logger.info(f"Found {len(matching_arcs)} matching arcs:")
                    for i, arc in enumerate(matching_arcs):
                        logger.info(f"  {i+1}. {arc['title']} ({arc['series']}) - {arc['arc_type']}")
                    logger.info("Using the first match. Be more specific if needed.")
                
                return matching_arcs[0]
                
        except Exception as e:
            logger.error(f"Error finding arc by name: {e}")
            return None
    
    def search_events_in_arc(self, search_phrase: str, arc_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for events within a specific narrative arc using vector similarity."""
        try:
            logger.info(f"🔍 Searching for events related to: '{search_phrase}'")
            logger.info(f"📚 Within narrative arc ID: {arc_id}")
            
            # Perform vector similarity search with arc filter
            results = self.vector_service.similarity_search_with_score(
                query=search_phrase,
                k=max_results * 3,  # Get more results to filter by arc
                filter={"$and": [
                    {"doc_type": "event"},
                    {"main_arc_id": arc_id}
                ]}
            )
            
            if not results:
                logger.warning(f"No events found for arc ID {arc_id}")
                return []
            
            # Process and format results
            formatted_results = []
            for doc, score in results:
                if score >= SIMILARITY_THRESHOLD:
                    # Extract metadata
                    metadata = doc.metadata
                    
                    result = {
                        "content": doc.page_content,
                        "similarity_score": round(score, 3),
                        "event_id": metadata.get("event_id", "unknown"),
                        "progression_id": metadata.get("progression_id", "unknown"),
                        "series": metadata.get("series", "unknown"),
                        "season": metadata.get("season", "unknown"),
                        "episode": metadata.get("episode", "unknown"),
                        "ordinal_position": metadata.get("ordinal_position", 0),
                        "start_timestamp": metadata.get("start_timestamp"),
                        "end_timestamp": metadata.get("end_timestamp"),
                        "confidence_score": metadata.get("confidence_score", 0),
                        "extraction_method": metadata.get("extraction_method", "unknown")
                    }
                    formatted_results.append(result)
            
            # Sort by similarity score (highest first)
            formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Limit results
            if len(formatted_results) > max_results:
                formatted_results = formatted_results[:max_results]
            
            logger.info(f"✅ Found {len(formatted_results)} relevant events (similarity >= {SIMILARITY_THRESHOLD})")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching events: {e}")
            return []
    
    def search_all_arc_events(self, search_phrase: str, arc_id: str) -> List[Dict[str, Any]]:
        """Get all events for an arc (for debugging/exploration)."""
        try:
            logger.info(f"📋 Retrieving all events for arc ID: {arc_id}")
            
            # Get all events for the arc
            all_docs = self.vector_service.collection.get(
                where={"$and": [
                    {"doc_type": "event"},
                    {"main_arc_id": arc_id}
                ]},
                include=["documents", "metadatas"]
            )
            
            if not all_docs.get("documents"):
                logger.warning(f"No events found for arc ID {arc_id}")
                return []
            
            # Format results
            results = []
            for i, (content, metadata) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
                result = {
                    "content": content,
                    "event_id": metadata.get("event_id", f"unknown_{i}"),
                    "progression_id": metadata.get("progression_id", "unknown"),
                    "series": metadata.get("series", "unknown"),
                    "season": metadata.get("season", "unknown"),  
                    "episode": metadata.get("episode", "unknown"),
                    "ordinal_position": metadata.get("ordinal_position", 0),
                    "start_timestamp": metadata.get("start_timestamp"),
                    "end_timestamp": metadata.get("end_timestamp"),
                    "confidence_score": metadata.get("confidence_score", 0),
                    "extraction_method": metadata.get("extraction_method", "unknown")
                }
                results.append(result)
            
            # Sort by ordinal position
            results.sort(key=lambda x: (x["season"], x["episode"], x["ordinal_position"]))
            
            logger.info(f"📊 Found {len(results)} total events for this arc")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving all arc events: {e}")
            return []

def display_results(results: List[Dict[str, Any]], search_phrase: str, arc_info: Dict):
    """Display search results in a formatted way."""
    print(f"\n{'='*80}")
    print(f"🔍 SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"Search Phrase: '{search_phrase}'")
    print(f"Narrative Arc: {arc_info['title']} ({arc_info['series']})")
    print(f"Arc Type: {arc_info['arc_type']}")
    print(f"Results Found: {len(results)}")
    print(f"{'='*80}\n")
    
    if not results:
        print("❌ No relevant events found. Try:")
        print("   - Different search terms")
        print("   - Lower similarity threshold")
        print("   - Check if the arc has events in the database")
        return
    
    for i, result in enumerate(results, 1):
        print(f"📋 RESULT #{i}")
        print(f"   📝 Content: {result['content']}")
        print(f"   📍 Location: {result['series']} {result['season']}{result['episode']}")
        
        if "similarity_score" in result:
            print(f"   🎯 Similarity: {result['similarity_score']:.3f}")
        
        if result.get('start_timestamp'):
            print(f"   ⏰ Timestamp: {result['start_timestamp']} - {result.get('end_timestamp', 'N/A')}")
        
        print(f"   🔢 Position: {result['ordinal_position']}")
        print(f"   📊 Confidence: {result['confidence_score']:.2f}")
        print(f"   🔧 Method: {result['extraction_method']}")
        print(f"   🆔 Event ID: {result['event_id']}")
        print()

def main():
    """Main search function."""
    logger.info("🚀 Starting narrative arc event search")
    
    # Extract search parameters
    search_phrase = phrase2search["phrase"]
    arc_name = phrase2search["narrative_arc_name"]
    
    logger.info(f"🔍 Search phrase: '{search_phrase}'")
    logger.info(f"📚 Target arc: '{arc_name}'")
    
    # Initialize searcher
    try:
        searcher = ArcEventSearcher()
    except Exception as e:
        logger.error(f"❌ Failed to initialize searcher: {e}")
        logger.error("Make sure the database and vector store are accessible")
        return
    
    # Find the narrative arc
    arc_info = searcher.find_arc_by_name(arc_name)
    if not arc_info:
        logger.error(f"❌ Could not find narrative arc: '{arc_name}'")
        logger.info("💡 Available arcs in database:")
        
        # Show available arcs for reference
        try:
            with searcher.db_manager.session_scope() as session:
                arc_repo = NarrativeArcRepository(session)
                all_arcs = arc_repo.get_all()
                
                if all_arcs:
                    for arc in all_arcs[:20]:  # Show first 20
                        logger.info(f"   - {arc.title} ({arc.series}) - {arc.arc_type}")
                    if len(all_arcs) > 20:
                        logger.info(f"   ... and {len(all_arcs) - 20} more")
                else:
                    logger.info("   No arcs found in database")
        except Exception as e:
            logger.error(f"Error listing available arcs: {e}")
        
        return
    
    logger.info(f"✅ Found arc: {arc_info['title']} (ID: {arc_info['id']})")
    
    # Perform the search
    results = searcher.search_events_in_arc(
        search_phrase=search_phrase,
        arc_id=arc_info["id"],
        max_results=MAX_RESULTS
    )
    
    # Display results
    display_results(results, search_phrase, arc_info)
    
    # Optional: Show all events for debugging
    if not results:
        user_input = input("\n🤔 No results found. Show all events for this arc? (y/n): ")
        if user_input.lower().startswith('y'):
            logger.info("📋 Retrieving all events for exploration...")
            all_events = searcher.search_all_arc_events(search_phrase, arc_info["id"])
            if all_events:
                print(f"\n{'='*80}")
                print(f"📋 ALL EVENTS IN ARC: {arc_info['title']}")
                print(f"{'='*80}")
                for i, event in enumerate(all_events[:10]):  # Show first 10
                    print(f"{i+1:2d}. [{event['season']}{event['episode']}] {event['content'][:100]}...")
                if len(all_events) > 10:
                    print(f"    ... and {len(all_events) - 10} more events")
            else:
                print("📭 No events found for this arc")

if __name__ == "__main__":
    main()
