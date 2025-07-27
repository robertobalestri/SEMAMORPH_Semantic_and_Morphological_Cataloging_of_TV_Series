#!/usr/bin/env python3
"""
Test script for WhisperX transcription with audio_test.wav
"""

import whisperx
import time
import os
from pathlib import Path

def test_whisperx_transcription():
    """Test WhisperX transcription with audio_test.wav"""
    
    # Check if audio file exists
    audio_file = "audio_test.wav"
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found!")
        return
    
    print(f"Testing WhisperX transcription with {audio_file}")
    print(f"File size: {os.path.getsize(audio_file) / (1024*1024):.2f} MB")
    
    try:
        # Load model
        print("Loading WhisperX model...")
        start_time = time.time()
        model = whisperx.load_model("base", device="cpu", compute_type="float32")
        model_load_time = time.time() - start_time
        print(f"Model loaded in {model_load_time:.2f} seconds")
        
        # Transcribe audio
        print("Transcribing audio...")
        start_time = time.time()
        result = model.transcribe(audio_file)
        transcription_time = time.time() - start_time
        print(f"Transcription completed in {transcription_time:.2f} seconds")
        
        # Print results
        print("\n=== TRANSCRIPTION RESULTS ===")
        print(f"Language detected: {result.get('language', 'Unknown')}")
        print(f"Language probability: {result.get('language_prob', 'Unknown')}")
        
        # Print segments
        if 'segments' in result:
            print(f"\nNumber of segments: {len(result['segments'])}")
            print("\nSegments:")
            for i, segment in enumerate(result['segments']):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '').strip()
                print(f"[{start:.2f}s - {end:.2f}s]: {text}")
        
        # Print full text
        if 'text' in result:
            print(f"\nFull transcription:")
            print(result['text'])
        
        # Save results to file
        output_file = "whisperx_test_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"WhisperX Test Results for {audio_file}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Language: {result.get('language', 'Unknown')}\n")
            f.write(f"Language probability: {result.get('language_prob', 'Unknown')}\n")
            f.write(f"Model load time: {model_load_time:.2f} seconds\n")
            f.write(f"Transcription time: {transcription_time:.2f} seconds\n\n")
            
            if 'segments' in result:
                f.write(f"Number of segments: {len(result['segments'])}\n\n")
                for i, segment in enumerate(result['segments']):
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    f.write(f"[{start:.2f}s - {end:.2f}s]: {text}\n")
            
            if 'text' in result:
                f.write(f"\nFull transcription:\n{result['text']}\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()

def test_whisperx_with_alignment():
    """Test WhisperX with word-level alignment"""
    
    audio_file = "audio_test.wav"
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found!")
        return
    
    print(f"\nTesting WhisperX with word-level alignment for {audio_file}")
    
    try:
        # Load model
        print("Loading WhisperX model...")
        model = whisperx.load_model("base", device="cpu", compute_type="float32")
        
        # Transcribe with alignment
        print("Transcribing with word-level alignment...")
        start_time = time.time()
        result = model.transcribe(audio_file)
        
        # Load alignment model
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
        
        # Align whisper output
        result = whisperx.align(result["segments"], model_a, metadata, audio_file, "cpu")
        alignment_time = time.time() - start_time
        print(f"Alignment completed in {alignment_time:.2f} seconds")
        
        # Print aligned results
        print("\n=== ALIGNED TRANSCRIPTION RESULTS ===")
        if 'segments' in result:
            print(f"Number of aligned segments: {len(result['segments'])}")
            for i, segment in enumerate(result['segments']):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '').strip()
                print(f"[{start:.2f}s - {end:.2f}s]: {text}")
                
                # Print word-level timing if available
                if 'words' in segment:
                    print("  Words:")
                    for word in segment['words']:
                        word_start = word.get('start', 0)
                        word_end = word.get('end', 0)
                        word_text = word.get('word', '')
                        print(f"    [{word_start:.2f}s - {word_end:.2f}s]: {word_text}")
        
    except Exception as e:
        print(f"Error during alignment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting WhisperX tests...")
    
    # Test basic transcription
    test_whisperx_transcription()
    
    # Test with alignment
    test_whisperx_with_alignment()
    
    print("\nWhisperX tests completed!")