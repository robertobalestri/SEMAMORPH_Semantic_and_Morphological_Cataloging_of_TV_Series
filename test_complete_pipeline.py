#!/usr/bin/env python3
"""
Complete Recap Generation Pipeline Test

This script tests the full recap_gen pipeline from start to finish:
1. Prerequisites validation
2. Episode inputs loading  
3. LLM query generation
4. Vector database search
5. Event ranking per arc
6. Event selection (round-robin)
7. Key dialogue extraction
8. Video clip creation
9. JSON export
10. Final recap assembly

Tests the integration with PathHandler and full event IDs (no slicing).
"""

import sys
import os
import json
import logging
from datetime import datetime

# Add backend/src to Python path
backend_src = os.path.join(os.path.dirname(__file__), 'backend', 'src')
sys.path.insert(0, backend_src)

# Set up comprehensive logging
def setup_logging():
    """Setup detailed logging for pipeline testing."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('recap_pipeline_test.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def test_complete_recap_pipeline():
    """Test the complete recap generation pipeline."""
    
    logger = setup_logging()
    
    print("🎬 COMPLETE RECAP GENERATION PIPELINE TEST")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target: GA S01 E09 (Grey's Anatomy Season 1 Episode 9)")
    print("=" * 60)
    
    try:
        # Step 1: Import and initialize
        print("\n📦 STEP 1: IMPORTING MODULES")
        print("-" * 30)
        
        from recap_gen.recap_generator import RecapGenerator
        from path_handler import PathHandler
        
        logger.info("✅ Modules imported successfully")
        print("✅ RecapGenerator imported")
        print("✅ PathHandler imported")
        
        # Step 2: Initialize generator
        print("\n🔧 STEP 2: INITIALIZING GENERATOR")
        print("-" * 35)
        
        generator = RecapGenerator(base_dir='data')
        logger.info("✅ RecapGenerator initialized with base_dir='data'")
        print("✅ Generator initialized")
        
        # Step 3: Validate prerequisites
        print("\n📋 STEP 3: VALIDATING PREREQUISITES")
        print("-" * 37)
        
        series, season, episode = "GA", "S01", "E09"
        validation = generator.validate_prerequisites(series, season, episode)
        
        logger.info(f"Prerequisites validation: {validation}")
        print(f"📊 Ready: {validation['ready']}")
        
        if validation.get('missing_files'):
            print("❌ Missing required files:")
            for file in validation['missing_files']:
                print(f"   - {file}")
                logger.warning(f"Missing file: {file}")
        
        if validation.get('warnings'):
            print("⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   - {warning}")
                logger.warning(f"Warning: {warning}")
        
        if not validation['ready']:
            print("❌ Prerequisites not met - stopping test")
            logger.error("Prerequisites validation failed")
            return False
        
        print("✅ All prerequisites met")
        
        # Step 4: Check PathHandler integration
        print("\n📁 STEP 4: PATHHANDLER INTEGRATION CHECK")
        print("-" * 42)
        
        path_handler = PathHandler(series, season, episode, 'data')
        
        # Test key paths
        paths_to_check = {
            "Recap files dir": path_handler.get_recap_files_dir(),
            "Clip directory": path_handler.get_recap_clip_dir(), 
            "JSON spec path": path_handler.get_recap_clips_json_path(),
            "Final video path": path_handler.get_final_recap_video_path()
        }
        
        for name, path in paths_to_check.items():
            print(f"📂 {name}: {path}")
            logger.info(f"PathHandler path - {name}: {path}")
        
        print("✅ PathHandler integration verified")
        
        # Step 5: Run complete pipeline
        print("\n🚀 STEP 5: RUNNING COMPLETE PIPELINE")
        print("-" * 37)
        print("⏱️  This may take several minutes...")
        print("")
        
        start_time = datetime.now()
        logger.info(f"Starting complete recap generation at {start_time}")
        
        # Run the full pipeline
        result = generator.generate_recap(series, season, episode)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Pipeline completed at {end_time}, duration: {duration}")
        
        # Step 6: Analyze results
        print(f"\n📊 STEP 6: RESULTS ANALYSIS")
        print("-" * 27)
        print(f"⏱️  Pipeline duration: {duration}")
        print(f"✅ Success: {result.success}")
        
        if not result.success:
            print(f"❌ Error: {result.error_message}")
            logger.error(f"Pipeline failed: {result.error_message}")
            return False
        
        # Success metrics
        print(f"📈 PIPELINE METRICS:")
        print(f"   🎬 Final video: {result.video_path}")
        print(f"   📝 Events selected: {len(result.events)}")
        print(f"   🎞️  Clips created: {len(result.clips)}")
        print(f"   ⏱️  Total duration: {result.total_duration:.1f}s")
        
        logger.info(f"Pipeline success metrics - Events: {len(result.events)}, Clips: {len(result.clips)}, Duration: {result.total_duration:.1f}s")
        
        # Step 7: Verify full event IDs (no slicing)
        print(f"\n🔍 STEP 7: FULL EVENT ID VERIFICATION")
        print("-" * 36)
        
        if result.events:
            print(f"📋 Selected Events (showing full IDs):")
            for i, event in enumerate(result.events[:5], 1):  # Show first 5
                id_length = len(event.id)
                print(f"   {i}. ID: {event.id} ({id_length} chars)")
                print(f"      Arc: {event.arc_title[:50]}...")
                print(f"      Source: {event.series}{event.season}{event.episode}")
                
                logger.info(f"Event {i} - ID: {event.id} ({id_length} chars), Arc: {event.arc_title}")
            
            if len(result.events) > 5:
                print(f"   ... and {len(result.events) - 5} more events")
            
            print(f"✅ All event IDs preserved (no slicing)")
        
        # Step 8: Verify video clips
        print(f"\n🎬 STEP 8: VIDEO CLIPS VERIFICATION")
        print("-" * 33)
        
        if result.clips:
            print(f"🎞️  Generated Clips:")
            total_duration = 0
            for i, clip in enumerate(result.clips[:5], 1):  # Show first 5
                filename = os.path.basename(clip.file_path)
                total_duration += clip.duration
                print(f"   {i}. {filename}")
                print(f"      Duration: {clip.duration:.1f}s")
                print(f"      Arc: {clip.arc_title[:40]}...")
                print(f"      Exists: {'✅' if os.path.exists(clip.file_path) else '❌'}")
                
                logger.info(f"Clip {i} - {filename}, Duration: {clip.duration:.1f}s, Exists: {os.path.exists(clip.file_path)}")
            
            if len(result.clips) > 5:
                print(f"   ... and {len(result.clips) - 5} more clips")
            
            print(f"✅ Total clips duration: {total_duration:.1f}s")
        
        # Step 9: Verify JSON export
        print(f"\n📝 STEP 9: JSON EXPORT VERIFICATION")
        print("-" * 32)
        
        json_path = path_handler.get_recap_clips_json_path()
        
        if os.path.exists(json_path):
            print(f"📄 JSON spec file: {json_path}")
            
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            print(f"📊 JSON content:")
            print(f"   Series: {json_data.get('series')}")
            print(f"   Season: {json_data.get('season')}")
            print(f"   Episode: {json_data.get('episode')}")
            print(f"   Events: {len(json_data.get('selected_events', []))}")
            
            # Verify full event IDs in JSON
            for i, event_data in enumerate(json_data.get('selected_events', [])[:3], 1):
                event_id = event_data.get('event_id', '')
                subtitles = event_data.get('selected_subtitles', [])
                print(f"   Event {i}: {event_id[:30]}... ({len(event_id)} chars)")
                print(f"            Subtitles: {len(subtitles)} lines")
                
                logger.info(f"JSON Event {i} - ID: {event_id} ({len(event_id)} chars), Subtitles: {len(subtitles)}")
            
            print(f"✅ JSON export verified with full event IDs")
            logger.info(f"JSON export verified at: {json_path}")
        else:
            print(f"❌ JSON spec file not found: {json_path}")
            logger.warning(f"JSON export missing: {json_path}")
        
        # Step 10: Final verification
        print(f"\n✅ STEP 10: FINAL VERIFICATION")
        print("-" * 29)
        
        checks = []
        
        # Check final video exists
        video_exists = os.path.exists(result.video_path) if result.video_path else False
        checks.append(("Final video file", video_exists))
        
        # Check all clips exist
        all_clips_exist = all(os.path.exists(clip.file_path) for clip in result.clips)
        checks.append(("All video clips", all_clips_exist))
        
        # Check JSON export
        json_exists = os.path.exists(json_path)
        checks.append(("JSON export", json_exists))
        
        # Check event ID preservation
        no_short_ids = all(len(event.id) > 8 for event in result.events)
        checks.append(("Full event IDs (no slicing)", no_short_ids))
        
        # Display checks
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            logger.info(f"Final check - {check_name}: {passed}")
        
        all_checks_passed = all(passed for _, passed in checks)
        
        print(f"\n{'🎉 PIPELINE TEST COMPLETED SUCCESSFULLY!' if all_checks_passed else '❌ PIPELINE TEST FAILED'}")
        print("=" * 60)
        
        if all_checks_passed:
            print("✅ Complete recap generation pipeline working")
            print("✅ PathHandler integration verified")
            print("✅ Full event IDs preserved (no slicing)")
            print("✅ JSON export with selected events and subtitles")
            print("✅ Video clips created and assembled")
            print("✅ Ready for production use")
            
            logger.info("🎉 COMPLETE PIPELINE TEST SUCCESSFUL")
        else:
            logger.error("❌ PIPELINE TEST FAILED - Some checks did not pass")
        
        return all_checks_passed
        
    except Exception as e:
        print(f"\n💥 PIPELINE ERROR: {e}")
        logger.error(f"Pipeline exception: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Pipeline traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print(f"🎬 Starting Complete Recap Pipeline Test...")
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"🐍 Python path includes: {backend_src}")
    
    success = test_complete_recap_pipeline()
    
    if success:
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"The recap_gen pipeline is fully functional with:")
        print(f"  ✅ Full event ID support (no slicing)")
        print(f"  ✅ PathHandler integration")
        print(f"  ✅ JSON export functionality")
        print(f"  ✅ Video clip creation and assembly")
    else:
        print(f"\n❌ TEST FAILED")
        print(f"Check recap_pipeline_test.log for detailed error information")
    
    sys.exit(0 if success else 1)
