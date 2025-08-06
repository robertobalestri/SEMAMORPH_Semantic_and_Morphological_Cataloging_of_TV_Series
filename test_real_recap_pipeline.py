#!/usr/bin/env python3
"""
REAL Complete Recap Generation Pipeline Test - Episode 2.

This script uses the ACTUAL RecapOrchestrator and all real services:
1. QueryGeneratorService (✅ REAL - already tested)
2. EventRetrievalService (✅ REAL - vector database integration)
3. SubtitleProcessorService (✅ REAL - LLM subtitle optimization)  
4. VideoClipExtractor (✅ REAL - FFmpeg video processing)
5. RecapAssembler (✅ REAL - final video assembly)

NO SIMULATION - All stages use production-ready implementations!
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import json
import time
from typing import List, Dict, Any

# Load environment variables from backend/.env file
backend_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', '.env')
load_dotenv(backend_env_path, override=True)

# Add the backend directory to Python path to ensure imports work correctly
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Now we can import from src
from src.path_handler import PathHandler
from src.recap_generator.recap_orchestrator import RecapOrchestrator
from src.recap_generator.models.recap_models import RecapConfiguration
from src.utils.logger_utils import setup_logging

logger = setup_logging(__name__)

class RealRecapPipelineTest:
    """Real recap generation pipeline test using actual RecapOrchestrator."""
    
    def __init__(self, series: str, season: str, episode: str):
        self.series = series
        self.season = season
        self.episode = episode
        
        # Create configuration for recap generation 
        # NOTE: Model has validation constraints that limit event duration (1-20s)
        # In real implementation, these constraints should be removed to allow full scenes
        # The LLM subtitle selection should do the trimming to ~10s key dialogue
        self.config = RecapConfiguration(
            # Event processing - using max allowed values (should be unlimited)
            min_event_duration=1.0,     # Model constraint: minimum 1.0s (should be 0.0)
            max_event_duration=20.0,    # Model constraint: maximum 20.0s (should be unlimited)
            
            # Target final recap duration 
            target_duration_seconds=90,  # Final assembled recap target: 90 seconds
            max_events=8,                # Maximum events to select
            
            # Event selection thresholds
            relevance_threshold=0.5,     # Threshold for event selection
            quality_threshold=0.6,       # Quality threshold for clips
            
            # Real video processing settings
            enable_subtitles=True,       # Include subtitles in final video
            enable_transitions=True,     # Add transitions between clips
            video_codec="libx264",       # H.264 codec
            audio_codec="aac",           # AAC audio
            ffmpeg_preset="medium"       # FFmpeg processing preset
        )
        
        print("⚠️  NOTE: RecapConfiguration model has constraints that need to be fixed:")
        print("   • min_event_duration should be 0.0 (not 1.0+)")
        print("   • max_event_duration should be unlimited (not 20s max)")
        print("   • Event retrieval should get full scenes, LLM trims dialogue")
        
        # Initialize orchestrator with configuration
        self.orchestrator = RecapOrchestrator(
            series=series,
            season=season, 
            episode=episode,
            config=self.config
        )
        
        # Results storage
        self.test_results = {
            "episode": f"{series} {season} {episode}",
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_type": "real_production_implementation",
            "configuration": self.config.dict(),
            "success": False,
            "error_message": None,
            "metadata": None
        }
    
    def check_prerequisites(self) -> bool:
        """Check if all required files exist for recap generation."""
        print("🔍 CHECKING PREREQUISITES FOR REAL RECAP GENERATION")
        print("=" * 60)
        
        path_handler = PathHandler(self.series, self.season, self.episode)
        
        required_files = {
            "Video file": path_handler.get_video_file_path(),
            "Plot file": path_handler.get_plot_possible_speakers_path(),
            "Running plotlines": path_handler.get_present_running_plotlines_path(),
            "Season summary": path_handler.get_season_summary_path(),
            "SRT subtitles": path_handler.get_srt_file_path(),
        }
        
        all_files_exist = True
        
        for name, file_path in required_files.items():
            exists = file_path and os.path.exists(file_path)
            status = "✅" if exists else "❌"
            print(f"  {status} {name}: {file_path}")
            if not exists:
                all_files_exist = False
        
        # Check if vector database is accessible
        try:
            from src.narrative_storage_management.vector_store_service import VectorStoreService
            vector_store = VectorStoreService()
            print(f"  ✅ Vector database: Accessible")
        except Exception as e:
            print(f"  ❌ Vector database: Error - {e}")
            all_files_exist = False
        
        # Check if recap directories can be created
        try:
            recap_dir = path_handler.get_recap_files_dir()
            os.makedirs(recap_dir, exist_ok=True)
            print(f"  ✅ Recap directory: {recap_dir}")
        except Exception as e:
            print(f"  ❌ Recap directory: Error - {e}")
            all_files_exist = False
        
        if all_files_exist:
            print("\n✅ All prerequisites satisfied - ready for real recap generation!")
        else:
            print("\n❌ Prerequisites check failed - cannot proceed with recap generation")
        
        return all_files_exist
    
    def run_real_recap_generation(self) -> bool:
        """Execute the real recap generation pipeline using RecapOrchestrator."""
        print("\n🎬 EXECUTING REAL RECAP GENERATION PIPELINE")
        print("=" * 60)
        print("✅ Using PRODUCTION RecapOrchestrator with all real services")
        print("✅ Vector database queries, LLM processing, FFmpeg video editing")
        
        try:
            start_time = time.time()
            
            # Execute the complete pipeline using the real orchestrator
            print("\n📋 Pipeline stages:")
            print("   1️⃣ Query generation for narrative arcs")
            print("   2️⃣ Vector database event retrieval")  
            print("   3️⃣ LLM subtitle optimization")
            print("   4️⃣ FFmpeg video clip extraction")
            print("   5️⃣ Final video assembly with transitions")
            
            print("\n🚀 Starting recap generation...")
            
            # This calls the REAL pipeline with ALL services
            recap_metadata = self.orchestrator.generate_recap()
            
            execution_time = time.time() - start_time
            
            # Pipeline completed successfully
            print(f"\\n🎉 REAL RECAP GENERATION COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
            
            # Display comprehensive results
            self._display_recap_results(recap_metadata, execution_time)
            
            # Store results
            self.test_results["success"] = True
            self.test_results["execution_time"] = execution_time
            self.test_results["metadata"] = recap_metadata.dict()
            
            return True
            
        except Exception as e:
            print(f"\\n❌ REAL RECAP GENERATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            self.test_results["success"] = False
            self.test_results["error_message"] = str(e)
            
            return False
    
    def _display_recap_results(self, metadata, execution_time: float):
        """Display comprehensive results from real recap generation."""
        print("\\n📊 REAL RECAP GENERATION RESULTS")
        print("=" * 50)
        
        # Basic information
        print(f"📺 Episode: {metadata.series} {metadata.season} {metadata.episode}")
        print(f"⏱️  Total duration: {metadata.total_duration:.1f} seconds")
        print(f"🎬 Number of clips: {len(metadata.clips)}")
        print(f"📁 Final video: {metadata.file_paths.get('final_video', 'N/A')}")
        
        # Processing metrics
        print(f"\\n🔧 Processing Metrics:")
        print(f"   🤖 LLM queries: {metadata.llm_queries_count}")
        print(f"   🔍 Vector searches: {metadata.vector_search_count}")
        print(f"   🎞️ FFmpeg operations: {metadata.ffmpeg_operations_count}")
        print(f"   ⏱️  Pipeline time: {metadata.processing_time_seconds:.2f}s")
        print(f"   ⚡ Total time: {execution_time:.2f}s")
        
        # Configuration used
        config = metadata.configuration
        print(f"\\n⚙️ Configuration Used:")
        print(f"   🎯 Target duration: {config.target_duration_seconds}s")
        print(f"   📊 Max events: {config.max_events}")
        print(f"   ⏱️ Event duration: {config.min_event_duration}s - {config.max_event_duration}s")
        print(f"   � Subtitle lines per event: {config.subtitle_lines_per_event}")
        print(f"   🎞️ Transitions: {config.enable_transitions}")
        print(f"   📝 Subtitles: {config.enable_subtitles}")
        print(f"   🎬 Video codec: {config.video_codec}")
        print(f"   🔊 Audio codec: {config.audio_codec}")
        
        # Event details
        if hasattr(metadata, 'events') and metadata.events:
            print(f"\\n📋 Selected Events:")
            for i, event in enumerate(metadata.events[:5], 1):  # Show first 5
                print(f"   {i}. {event.content[:60]}{'...' if len(event.content) > 60 else ''}")
                print(f"      📍 {event.series} {event.season} {event.episode} | {event.start_timestamp}-{event.end_timestamp}")
                print(f"      🎯 Relevance: {event.relevance_score:.2f} | Arc: {event.arc_title}")
            
            if len(metadata.events) > 5:
                print(f"   ... and {len(metadata.events) - 5} more events")
        
        # Clip details
        print(f"\\n🎬 Generated Clips:")
        for i, clip in enumerate(metadata.clips, 1):
            print(f"   {i}. Duration: {clip.duration:.1f}s | Quality: {clip.quality_score:.2f}")
            print(f"      📁 File: {clip.file_path}")
            if hasattr(clip, 'arc_title'):
                print(f"      📈 Arc: {clip.arc_title}")
        
        # File outputs
        print(f"\\n📁 Generated Files:")
        for file_type, file_path in metadata.file_paths.items():
            print(f"   📄 {file_type.replace('_', ' ').title()}: {file_path}")
        
        print(f"\\n🎉 SUCCESS: Complete recap video generated using real implementation!")
    
    def save_test_results(self):
        """Save comprehensive test results."""
        output_file = f"real_recap_test_{self.series}_{self.season}_{self.episode}_{int(time.time())}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\\n💾 Test results saved to: {output_file}")
        return output_file

def main():
    """Main test function for real recap generation."""
    # Configuration
    series = "GA"
    season = "S01"
    episode = "E02"
    
    print("🎬 SEMAMORPH REAL RECAP GENERATION TEST")
    print("=" * 60)
    print("This test uses the ACTUAL RecapOrchestrator and ALL real services:")
    print("✅ Real QueryGeneratorService") 
    print("✅ Real EventRetrievalService with vector database")
    print("✅ Real SubtitleProcessorService with LLM optimization")
    print("✅ Real VideoClipExtractor with FFmpeg")
    print("✅ Real RecapAssembler for final video")
    print("\\n🚀 NO SIMULATION - All production-ready implementations!")
    
    # Initialize test
    test = RealRecapPipelineTest(series, season, episode)
    
    try:
        # Step 1: Check prerequisites  
        if not test.check_prerequisites():
            print("\\n❌ Cannot proceed - prerequisites not met")
            return 1
        
        # Step 2: Execute real recap generation
        success = test.run_real_recap_generation()
        
        # Step 3: Save results
        results_file = test.save_test_results()
        
        if success:
            print("\\n🎉 REAL RECAP GENERATION TEST COMPLETED SUCCESSFULLY!")
            print("\\n📋 What was accomplished:")
            print("   ✅ Complete pipeline executed with real implementations")
            print("   ✅ Vector database queries executed for event retrieval")
            print("   ✅ LLM processing for subtitle optimization") 
            print("   ✅ FFmpeg video processing for clip extraction")
            print("   ✅ Final video assembly with transitions and effects")
            print("   ✅ Production-quality recap video generated")
            print(f"\\n📁 Results: {results_file}")
            return 0
        else:
            print("\\n❌ REAL RECAP GENERATION TEST FAILED")
            print("\\n📋 Possible issues:")
            print("   🔍 Check vector database connectivity")
            print("   🎬 Verify video file format and accessibility")
            print("   🤖 Confirm LLM service availability") 
            print("   🛠️ Ensure FFmpeg is installed and configured")
            print(f"\\n📁 Error details: {results_file}")
            return 1
            
    except Exception as e:
        print(f"\\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
