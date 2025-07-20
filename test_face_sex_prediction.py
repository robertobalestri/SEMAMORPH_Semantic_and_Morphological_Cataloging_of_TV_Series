#!/usr/bin/env python3
"""
Simple script to test DeepFace sex/gender prediction on a single face image.
Usage: python test_face_sex_prediction.py [image_path]
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("❌ DeepFace not available. Please install: pip install deepface")
    sys.exit(1)

def analyze_face_sex(image_path: str):
    """Analyze sex/gender of a face image using DeepFace."""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔬 Analyzing sex/gender for: {image_path}")
    print("-" * 60)
    
    try:
        # Analyze gender using DeepFace (same as sex validator)
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['gender'],
            enforce_detection=False  # Don't fail if face not detected perfectly
        )
        
        # Handle both list and dict returns from DeepFace
        if isinstance(result, list):
            result = result[0] if result else {}
        
        gender_data = result.get('gender', {})
        
        if not gender_data:
            print("❌ No gender data returned from DeepFace")
            return
        
        # Extract confidence scores
        male_confidence = gender_data.get('Man', 0)
        female_confidence = gender_data.get('Woman', 0)
        
        # Make prediction
        predicted_sex = 'M' if male_confidence > female_confidence else 'F'
        confidence_difference = abs(male_confidence - female_confidence)
        
        # Display results
        print(f"📊 DeepFace Gender Analysis Results:")
        print(f"   Male confidence:   {male_confidence:.1f}%")
        print(f"   Female confidence: {female_confidence:.1f}%")
        print(f"   Predicted sex:     {predicted_sex} ({'Male' if predicted_sex == 'M' else 'Female'})")
        print(f"   Confidence diff:   {confidence_difference:.1f}%")
        
        # Check reliability (using same threshold as config)
        reliable_threshold = 75.0  # Default from config
        reliable_prediction = confidence_difference >= reliable_threshold
        
        print(f"   Reliable prediction: {'✅ Yes' if reliable_prediction else '❌ No'} (threshold: {reliable_threshold}%)")
        
        if not reliable_prediction:
            print(f"   ⚠️  Low confidence difference - prediction may not be reliable")
        
        # Additional analysis info if available
        if 'region' in result:
            region = result['region']
            print(f"\n📐 Face Detection:")
            print(f"   Face region: x={region.get('x', 'N/A')}, y={region.get('y', 'N/A')}, w={region.get('w', 'N/A')}, h={region.get('h', 'N/A')}")
            if 'w' in region and 'h' in region:
                face_area = region['w'] * region['h']
                print(f"   Face size: {face_area} pixels ({region['w']}×{region['h']})")
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        print(f"   Make sure the image contains a clear face")
        return

def main():
    """Main function."""
    
    # Default image path if none provided
    default_image = "data/GA/S01/E01/dialogue_faces/dialogue_0363_face_00.png"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image
        print(f"📁 No image path provided, using default: {image_path}")
    
    # Convert to absolute path
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.getcwd(), image_path)
    
    print(f"🎯 Testing DeepFace sex prediction")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🖼️  Image path: {image_path}")
    print()
    
    analyze_face_sex(image_path)

if __name__ == "__main__":
    main() 