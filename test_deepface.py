#!/usr/bin/env python3
"""
Test script for DeepFace with RetinaFace detection and DeepFace512 recognition
"""

from deepface import DeepFace
import time
import os
import numpy as np
from pathlib import Path

def test_deepface_retinaface_detection():
    """Test DeepFace face detection using RetinaFace"""
    
    # Use path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "img.png")
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return
    
    print(f"Testing DeepFace face detection with RetinaFace on {image_path}")
    print(f"File size: {os.path.getsize(image_path) / (1024*1024):.2f} MB")
    
    try:
        # Extract faces using RetinaFace detector
        print("Extracting faces using RetinaFace...")
        start_time = time.time()
        
        face_objs = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
            expand_percentage=0
        )
        
        detection_time = time.time() - start_time
        print(f"Face detection completed in {detection_time:.2f} seconds")
        
        # Print results
        print(f"\n=== FACE DETECTION RESULTS ===")
        print(f"Number of faces detected: {len(face_objs)}")
        
        for i, face_obj in enumerate(face_objs):
            print(f"\nFace {i+1}:")
            print(f"  Confidence: {face_obj.get('confidence', 'N/A'):.4f}")
            
            facial_area = face_obj.get('facial_area', {})
            print(f"  Position: x={facial_area.get('x', 'N/A')}, y={facial_area.get('y', 'N/A')}")
            print(f"  Size: w={facial_area.get('w', 'N/A')}, h={facial_area.get('h', 'N/A')}")
            
            # Print landmarks if available
            if 'left_eye' in facial_area and facial_area['left_eye']:
                print(f"  Left eye: {facial_area['left_eye']}")
            if 'right_eye' in facial_area and facial_area['right_eye']:
                print(f"  Right eye: {facial_area['right_eye']}")
            if 'nose' in facial_area and facial_area['nose']:
                print(f"  Nose: {facial_area['nose']}")
            if 'mouth_left' in facial_area and facial_area['mouth_left']:
                print(f"  Mouth left: {facial_area['mouth_left']}")
            if 'mouth_right' in facial_area and facial_area['mouth_right']:
                print(f"  Mouth right: {facial_area['mouth_right']}")
            
            # Print face image info
            face_img = face_obj.get('face')
            if face_img is not None:
                print(f"  Face image shape: {face_img.shape}")
                print(f"  Face image dtype: {face_img.dtype}")
                print(f"  Face image range: {face_img.min():.3f} to {face_img.max():.3f}")
        
        # Save results to file
        output_file = "deepface_detection_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"DeepFace RetinaFace Detection Results for {image_path}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Detection time: {detection_time:.2f} seconds\n")
            f.write(f"Number of faces detected: {len(face_objs)}\n\n")
            
            for i, face_obj in enumerate(face_objs):
                f.write(f"Face {i+1}:\n")
                f.write(f"  Confidence: {face_obj.get('confidence', 'N/A'):.4f}\n")
                
                facial_area = face_obj.get('facial_area', {})
                f.write(f"  Position: x={facial_area.get('x', 'N/A')}, y={facial_area.get('y', 'N/A')}\n")
                f.write(f"  Size: w={facial_area.get('w', 'N/A')}, h={facial_area.get('h', 'N/A')}\n")
                
                if 'left_eye' in facial_area and facial_area['left_eye']:
                    f.write(f"  Left eye: {facial_area['left_eye']}\n")
                if 'right_eye' in facial_area and facial_area['right_eye']:
                    f.write(f"  Right eye: {facial_area['right_eye']}\n")
                if 'nose' in facial_area and facial_area['nose']:
                    f.write(f"  Nose: {facial_area['nose']}\n")
                if 'mouth_left' in facial_area and facial_area['mouth_left']:
                    f.write(f"  Mouth left: {facial_area['mouth_left']}\n")
                if 'mouth_right' in facial_area and facial_area['mouth_right']:
                    f.write(f"  Mouth right: {facial_area['mouth_right']}\n")
                f.write("\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error during face detection: {e}")
        import traceback
        traceback.print_exc()

def test_deepface512_recognition():
    """Test DeepFace512 face recognition"""
    
    # Use path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "img.png")
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return
    
    print(f"\nTesting DeepFace512 face recognition on {image_path}")
    
    try:
        # Generate embeddings using DeepFace512
        print("Generating embeddings using DeepFace512...")
        start_time = time.time()
        
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
            expand_percentage=0,
            normalization="base"
        )
        
        recognition_time = time.time() - start_time
        print(f"Face recognition completed in {recognition_time:.2f} seconds")
        
        # Print results
        print(f"\n=== DEEPFACE512 RECOGNITION RESULTS ===")
        print(f"Number of faces processed: {len(embedding_objs)}")
        
        for i, embedding_obj in enumerate(embedding_objs):
            print(f"\nFace {i+1}:")
            
            # Print embedding info
            embedding = embedding_obj.get('embedding', [])
            print(f"  Embedding dimensions: {len(embedding)}")
            print(f"  Embedding type: {type(embedding)}")
            if embedding:
                embedding_array = np.array(embedding)
                print(f"  Embedding shape: {embedding_array.shape}")
                print(f"  Embedding range: {embedding_array.min():.6f} to {embedding_array.max():.6f}")
                print(f"  Embedding mean: {embedding_array.mean():.6f}")
                print(f"  Embedding std: {embedding_array.std():.6f}")
            
            # Print facial area info
            facial_area = embedding_obj.get('facial_area', {})
            print(f"  Facial area: x={facial_area.get('x', 'N/A')}, y={facial_area.get('y', 'N/A')}")
            print(f"  Facial area size: w={facial_area.get('w', 'N/A')}, h={facial_area.get('h', 'N/A')}")
            
            # Print confidence
            face_confidence = embedding_obj.get('face_confidence', 'N/A')
            print(f"  Face confidence: {face_confidence}")
        
        # Save results to file
        output_file = "deepface512_recognition_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"DeepFace512 Recognition Results for {image_path}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Recognition time: {recognition_time:.2f} seconds\n")
            f.write(f"Number of faces processed: {len(embedding_objs)}\n\n")
            
            for i, embedding_obj in enumerate(embedding_objs):
                f.write(f"Face {i+1}:\n")
                
                embedding = embedding_obj.get('embedding', [])
                f.write(f"  Embedding dimensions: {len(embedding)}\n")
                if embedding:
                    embedding_array = np.array(embedding)
                    f.write(f"  Embedding shape: {embedding_array.shape}\n")
                    f.write(f"  Embedding range: {embedding_array.min():.6f} to {embedding_array.max():.6f}\n")
                    f.write(f"  Embedding mean: {embedding_array.mean():.6f}\n")
                    f.write(f"  Embedding std: {embedding_array.std():.6f}\n")
                
                facial_area = embedding_obj.get('facial_area', {})
                f.write(f"  Facial area: x={facial_area.get('x', 'N/A')}, y={facial_area.get('y', 'N/A')}\n")
                f.write(f"  Facial area size: w={facial_area.get('w', 'N/A')}, h={facial_area.get('h', 'N/A')}\n")
                
                face_confidence = embedding_obj.get('face_confidence', 'N/A')
                f.write(f"  Face confidence: {face_confidence}\n\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error during face recognition: {e}")
        import traceback
        traceback.print_exc()

def test_deepface_analysis():
    """Test DeepFace facial attribute analysis with RetinaFace"""
    
    # Use path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "img.png")
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return
    
    print(f"\nTesting DeepFace facial attribute analysis with RetinaFace on {image_path}")
    
    try:
        # Analyze facial attributes
        print("Analyzing facial attributes...")
        start_time = time.time()
        
        analysis_results = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'race', 'emotion'],
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
            expand_percentage=0
        )
        
        analysis_time = time.time() - start_time
        print(f"Facial analysis completed in {analysis_time:.2f} seconds")
        
        # Print results
        print(f"\n=== FACIAL ATTRIBUTE ANALYSIS RESULTS ===")
        print(f"Number of faces analyzed: {len(analysis_results)}")
        
        for i, result in enumerate(analysis_results):
            print(f"\nFace {i+1}:")
            
            # Print region info
            region = result.get('region', {})
            print(f"  Region: x={region.get('x', 'N/A')}, y={region.get('y', 'N/A')}")
            print(f"  Region size: w={region.get('w', 'N/A')}, h={region.get('h', 'N/A')}")
            
            # Print age
            age = result.get('age', 'N/A')
            print(f"  Age: {age}")
            
            # Print gender
            gender = result.get('gender', {})
            dominant_gender = result.get('dominant_gender', 'N/A')
            print(f"  Dominant gender: {dominant_gender}")
            if gender:
                for g, score in gender.items():
                    print(f"    {g}: {score:.4f}")
            
            # Print emotion
            emotion = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'N/A')
            print(f"  Dominant emotion: {dominant_emotion}")
            if emotion:
                for e, score in emotion.items():
                    print(f"    {e}: {score:.4f}")
            
            # Print race
            race = result.get('race', {})
            dominant_race = result.get('dominant_race', 'N/A')
            print(f"  Dominant race: {dominant_race}")
            if race:
                for r, score in race.items():
                    print(f"    {r}: {score:.4f}")
            
            # Print face confidence
            face_confidence = result.get('face_confidence', 'N/A')
            print(f"  Face confidence: {face_confidence}")
        
        # Save results to file
        output_file = "deepface_analysis_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"DeepFace Facial Attribute Analysis Results for {image_path}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis time: {analysis_time:.2f} seconds\n")
            f.write(f"Number of faces analyzed: {len(analysis_results)}\n\n")
            
            for i, result in enumerate(analysis_results):
                f.write(f"Face {i+1}:\n")
                
                region = result.get('region', {})
                f.write(f"  Region: x={region.get('x', 'N/A')}, y={region.get('y', 'N/A')}\n")
                f.write(f"  Region size: w={region.get('w', 'N/A')}, h={region.get('h', 'N/A')}\n")
                
                age = result.get('age', 'N/A')
                f.write(f"  Age: {age}\n")
                
                gender = result.get('gender', {})
                dominant_gender = result.get('dominant_gender', 'N/A')
                f.write(f"  Dominant gender: {dominant_gender}\n")
                if gender:
                    for g, score in gender.items():
                        f.write(f"    {g}: {score:.4f}\n")
                
                emotion = result.get('emotion', {})
                dominant_emotion = result.get('dominant_emotion', 'N/A')
                f.write(f"  Dominant emotion: {dominant_emotion}\n")
                if emotion:
                    for e, score in emotion.items():
                        f.write(f"    {e}: {score:.4f}\n")
                
                race = result.get('race', {})
                dominant_race = result.get('dominant_race', 'N/A')
                f.write(f"  Dominant race: {dominant_race}\n")
                if race:
                    for r, score in race.items():
                        f.write(f"    {r}: {score:.4f}\n")
                
                face_confidence = result.get('face_confidence', 'N/A')
                f.write(f"  Face confidence: {face_confidence}\n\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error during facial analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting DeepFace tests with RetinaFace and DeepFace512...")
    
    # Test face detection with RetinaFace
    test_deepface_retinaface_detection()
    
    # Test face recognition with DeepFace512
    test_deepface512_recognition()
    
    # Test facial attribute analysis
    test_deepface_analysis()
    
    print("\nDeepFace tests completed!")

