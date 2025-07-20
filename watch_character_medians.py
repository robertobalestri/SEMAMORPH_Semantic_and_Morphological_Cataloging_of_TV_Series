#!/usr/bin/env python3
"""
Character Medians Vector Store Monitor

This script monitors and displays character medians stored in the vector store.
It can run in watch mode to continuously monitor for changes.

Usage:
    python watch_character_medians.py                 # Show current state
    python watch_character_medians.py --watch         # Continuous monitoring
    python watch_character_medians.py --series GA     # Filter by series
    python watch_character_medians.py --detailed      # Show detailed information
"""

import sys
import os
import time
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from backend.src.narrative_storage_management.vector_store_service import FaceEmbeddingVectorStore
    from backend.src.utils.logger_utils import setup_logging
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

# Set up logging
logger = setup_logging(__name__)

class CharacterMedianMonitor:
    """Monitor for character medians in the vector store."""
    
    def __init__(self):
        self.vector_store = FaceEmbeddingVectorStore()
        self.last_count = 0
        self.last_update = None
        # Use the correct collection name that face clustering uses
        self.collection_name = "dialogue_faces_facenet512"
        
    def get_character_medians(self, series_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all character medians from the vector store.
        
        Args:
            series_filter: Optional series filter (e.g., 'GA')
            
        Returns:
            List of character median entries
        """
        try:
            # Build filter criteria for ChromaDB
            if series_filter:
                filter_criteria = {
                    "$and": [
                        {"type": "character_median"},
                        {"series": series_filter}
                    ]
                }
            else:
                filter_criteria = {"type": "character_median"}
            
            # Access the correct collection directly using ChromaDB
            import chromadb
            client = chromadb.PersistentClient(path="./narrative_storage/chroma_db")
            collection = client.get_collection(self.collection_name)
            
            # Query for character medians
            results = collection.get(
                where=filter_criteria,
                include=['metadatas', 'documents']
            )
            
            if not results or not results.get('ids'):
                return []
            
            # Format results
            character_medians = []
            for i, char_id in enumerate(results['ids']):
                metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                document = results['documents'][i] if i < len(results['documents']) else ""
                
                character_medians.append({
                    'id': char_id,
                    'metadata': metadata,
                    'document': document,
                    'character_name': metadata.get('character_name', 'Unknown'),
                    'series': metadata.get('series', 'Unknown'),
                    'season': metadata.get('season', 'Unknown'),
                    'episode': metadata.get('episode', 'Unknown'),
                    'episode_code': metadata.get('episode_code', 'Unknown'),
                    'face_count': metadata.get('face_count', 0),
                    'cluster_count': metadata.get('cluster_count', 0),
                    'avg_cluster_confidence': metadata.get('avg_cluster_confidence', 0.0),
                    'created_timestamp': metadata.get('created_timestamp', ''),
                    'median_level': metadata.get('median_level', 'character')
                })
            
            # Sort by series, season, episode, character name
            character_medians.sort(key=lambda x: (
                x['series'], 
                x['season'], 
                x['episode'], 
                x['character_name']
            ))
            
            return character_medians
            
        except Exception as e:
            logger.error(f"❌ Error getting character medians: {e}")
            return []
    
    def get_cluster_medians(self, series_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all cluster medians from the vector store.
        
        Args:
            series_filter: Optional series filter (e.g., 'GA')
            
        Returns:
            List of cluster median entries
        """
        try:
            # Build filter criteria for ChromaDB
            if series_filter:
                filter_criteria = {
                    "$and": [
                        {"type": "cluster_median"},
                        {"series": series_filter}
                    ]
                }
            else:
                filter_criteria = {"type": "cluster_median"}
            
            # Access the correct collection directly using ChromaDB
            import chromadb
            client = chromadb.PersistentClient(path="./narrative_storage/chroma_db")
            collection = client.get_collection(self.collection_name)
            
            # Query for cluster medians
            results = collection.get(
                where=filter_criteria,
                include=['metadatas']
            )
            
            if not results or not results.get('ids'):
                return []
            
            # Format results
            cluster_medians = []
            for i, cluster_id in enumerate(results['ids']):
                metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                
                cluster_medians.append({
                    'id': cluster_id,
                    'metadata': metadata,
                    'character_name': metadata.get('character_name', 'Unknown'),
                    'series': metadata.get('series', 'Unknown'),
                    'season': metadata.get('season', 'Unknown'),
                    'episode': metadata.get('episode', 'Unknown'),
                    'episode_code': metadata.get('episode_code', 'Unknown'),
                    'cluster_id': metadata.get('cluster_id', -1),
                    'face_count': metadata.get('face_count', 0),
                    'cluster_confidence': metadata.get('cluster_confidence', 0.0),
                    'cluster_status': metadata.get('cluster_status', 'Unknown'),
                    'has_been_validated': metadata.get('has_been_validated', False),
                    'created_timestamp': metadata.get('created_timestamp', '')
                })
            
            return cluster_medians
            
        except Exception as e:
            logger.error(f"❌ Error getting cluster medians: {e}")
            return []
    
    def display_summary(self, character_medians: List[Dict], cluster_medians: List[Dict], series_filter: Optional[str] = None):
        """Display a summary of character and cluster medians."""
        print(f"\n{'='*80}")
        print(f"📊 CHARACTER MEDIANS VECTOR STORE SUMMARY")
        print(f"{'='*80}")
        print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if series_filter:
            print(f"🎬 Series Filter: {series_filter}")
        
        print(f"\n📈 OVERVIEW:")
        print(f"   🎭 Character Medians: {len(character_medians)}")
        print(f"   🔗 Cluster Medians: {len(cluster_medians)}")
        
        if not character_medians and not cluster_medians:
            print(f"\n⚠️  No medians found in vector store")
            if series_filter:
                print(f"   Try without --series filter to see all data")
            return
        
        # Group by series and episode
        series_episodes = {}
        for median in character_medians:
            series = median['series']
            episode_code = median['episode_code']
            if series not in series_episodes:
                series_episodes[series] = {}
            if episode_code not in series_episodes[series]:
                series_episodes[series][episode_code] = {'characters': [], 'clusters': 0}
            series_episodes[series][episode_code]['characters'].append(median)
        
        # Add cluster counts
        for median in cluster_medians:
            series = median['series']
            episode_code = median['episode_code']
            if series in series_episodes and episode_code in series_episodes[series]:
                series_episodes[series][episode_code]['clusters'] += 1
        
        # Display by series and episode
        print(f"\n📺 BY SERIES & EPISODE:")
        for series, episodes in sorted(series_episodes.items()):
            print(f"\n   🎬 {series}:")
            for episode_code, data in sorted(episodes.items()):
                characters = data['characters']
                cluster_count = data['clusters']
                char_count = len(characters)
                
                print(f"      📺 {episode_code}: {char_count} characters, {cluster_count} clusters")
                
                # Show character names
                char_names = sorted([c['character_name'] for c in characters])
                if char_names:
                    names_str = ', '.join(char_names)
                    if len(names_str) > 60:
                        names_str = names_str[:57] + "..."
                    print(f"         🎭 Characters: {names_str}")
    
    def display_detailed(self, character_medians: List[Dict], cluster_medians: List[Dict]):
        """Display detailed information about character and cluster medians."""
        print(f"\n{'='*80}")
        print(f"🔍 DETAILED CHARACTER MEDIANS")
        print(f"{'='*80}")
        
        if not character_medians:
            print("⚠️  No character medians found")
            return
        
        for median in character_medians:
            print(f"\n🎭 CHARACTER: {median['character_name']}")
            print(f"   📺 Episode: {median['episode_code']} ({median['series']} {median['season']} {median['episode']})")
            print(f"   🆔 ID: {median['id']}")
            print(f"   👥 Face Count: {median['face_count']}")
            print(f"   🔗 Cluster Count: {median['cluster_count']}")
            print(f"   📊 Avg Confidence: {median['avg_cluster_confidence']:.2f}")
            print(f"   🕒 Created: {median['created_timestamp']}")
            print(f"   📊 Level: {median['median_level']}")
            
            # Show related cluster medians
            related_clusters = [
                c for c in cluster_medians 
                if (c['character_name'] == median['character_name'] and 
                    c['episode_code'] == median['episode_code'])
            ]
            
            if related_clusters:
                print(f"   🔗 Related Clusters ({len(related_clusters)}):")
                for cluster in related_clusters:
                    status = cluster['cluster_status']
                    validated = "✅" if cluster['has_been_validated'] else "❌"
                    print(f"      • Cluster {cluster['cluster_id']}: {cluster['face_count']} faces, "
                          f"conf={cluster['cluster_confidence']:.2f}, status={status} {validated}")
    
    def watch_changes(self, series_filter: Optional[str] = None, interval: int = 5):
        """Watch for changes in character medians."""
        print(f"👀 WATCHING CHARACTER MEDIANS (refresh every {interval}s)")
        print(f"   Press Ctrl+C to stop")
        
        try:
            while True:
                character_medians = self.get_character_medians(series_filter)
                cluster_medians = self.get_cluster_medians(series_filter)
                
                current_count = len(character_medians)
                current_time = datetime.now()
                
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Show changes since last check
                if self.last_count != current_count:
                    change = current_count - self.last_count
                    print(f"🔄 CHANGE DETECTED: {change:+d} character medians")
                    self.last_count = current_count
                    self.last_update = current_time
                
                # Display current state
                self.display_summary(character_medians, cluster_medians, series_filter)
                
                if self.last_update:
                    print(f"\n🕒 Last Change: {self.last_update.strftime('%H:%M:%S')}")
                
                print(f"\n⏱️  Next refresh in {interval}s... (Ctrl+C to stop)")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Monitoring stopped")
    
    def export_to_json(self, filename: str, series_filter: Optional[str] = None):
        """Export character medians to JSON file."""
        try:
            character_medians = self.get_character_medians(series_filter)
            cluster_medians = self.get_cluster_medians(series_filter)
            
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "series_filter": series_filter,
                "character_medians_count": len(character_medians),
                "cluster_medians_count": len(cluster_medians),
                "character_medians": character_medians,
                "cluster_medians": cluster_medians
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Exported to {filename}")
            print(f"   📊 {len(character_medians)} character medians")
            print(f"   🔗 {len(cluster_medians)} cluster medians")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Monitor character medians in the vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python watch_character_medians.py                    # Show current state
  python watch_character_medians.py --watch            # Continuous monitoring  
  python watch_character_medians.py --series GA        # Filter by series
  python watch_character_medians.py --detailed         # Show detailed info
  python watch_character_medians.py --export data.json # Export to JSON
        """
    )
    
    parser.add_argument('--watch', action='store_true', 
                       help='Continuously monitor for changes')
    parser.add_argument('--series', type=str, 
                       help='Filter by series (e.g., GA)')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed information')
    parser.add_argument('--interval', type=int, default=5,
                       help='Refresh interval in seconds for watch mode (default: 5)')
    parser.add_argument('--export', type=str, metavar='FILENAME',
                       help='Export data to JSON file')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = CharacterMedianMonitor()
    
    try:
        if args.export:
            monitor.export_to_json(args.export, args.series)
        elif args.watch:
            monitor.watch_changes(args.series, args.interval)
        else:
            # One-time display
            character_medians = monitor.get_character_medians(args.series)
            cluster_medians = monitor.get_cluster_medians(args.series)
            
            if args.detailed:
                monitor.display_detailed(character_medians, cluster_medians)
            else:
                monitor.display_summary(character_medians, cluster_medians, args.series)
                
    except KeyboardInterrupt:
        print(f"\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 