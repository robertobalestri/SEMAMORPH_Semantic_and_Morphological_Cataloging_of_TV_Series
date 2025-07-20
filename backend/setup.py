#!/usr/bin/env python3
"""
SEMAMORPH Backend Setup Script
Installs all dependencies and sets up the environment for speaker identification.
"""
import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"🔄 {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False

def check_system_dependencies():
    """Check for required system dependencies."""
    print("🔍 Checking system dependencies...")
    
    # Check for cmake
    if not run_command("cmake --version", "Checking CMake"):
        print("⚠️  CMake not found. Please install:")
        if platform.system() == "Linux":
            print("   sudo apt-get install cmake")
        elif platform.system() == "Darwin":
            print("   brew install cmake")
        elif platform.system() == "Windows":
            print("   Download from https://cmake.org/download/")
        return False
    
    # Check for ffmpeg
    if not run_command("ffmpeg -version", "Checking FFmpeg"):
        print("⚠️  FFmpeg not found. Please install:")
        if platform.system() == "Linux":
            print("   sudo apt-get install ffmpeg")
        elif platform.system() == "Darwin":
            print("   brew install ffmpeg")
        elif platform.system() == "Windows":
            print("   Download from https://ffmpeg.org/download.html")
        return False
    
    print("✅ System dependencies OK")
    return True

def install_python_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Install core requirements
    if not run_command("pip install -r requirements.txt", "Installing core requirements"):
        return False
    
    # Install speaker identification requirements
    if os.path.exists("requirements-speaker-id.txt"):
        if not run_command("pip install -r requirements-speaker-id.txt", "Installing speaker ID requirements"):
            return False
    
    print("✅ Python dependencies installed")
    return True

def download_spacy_models():
    """Download required spaCy models."""
    print("🧠 Downloading spaCy models...")
    
    models = [
        "en_core_web_trf"
    ]
    
    for model in models:
        if not run_command(f"python -m spacy download {model}", f"Downloading {model}"):
            print(f"⚠️  Failed to download {model}, but continuing...")
    
    print("✅ spaCy models downloaded")
    return True

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        "data",
        "narrative_storage",
        "narrative_storage/chroma_db",
        "agent_logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")
    
    return True

def verify_installation():
    """Verify the installation by testing imports."""
    print("🧪 Verifying installation...")
    
    test_imports = [
        "fastapi",
        "sqlmodel", 
        "chromadb",
        "cv2",
        "deepface",
        "face_recognition",
        "spacy",
        "langchain",
        "openai",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "sklearn",
        "hdbscan"
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Some modules may need manual installation or system dependencies.")
        return False
    
    print("✅ All imports successful!")
    return True

def main():
    """Main setup function."""
    print("🚀 SEMAMORPH Backend Setup")
    print("=" * 50)
    
    # Change to backend directory if not already there
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    steps = [
        ("Checking system dependencies", check_system_dependencies),
        ("Installing Python dependencies", install_python_dependencies),
        ("Downloading spaCy models", download_spacy_models),
        ("Setting up directories", setup_directories),
        ("Verifying installation", verify_installation)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            failed_steps.append(step_name)
            print(f"❌ {step_name} failed!")
        else:
            print(f"✅ {step_name} completed!")
    
    print("\n" + "=" * 50)
    if failed_steps:
        print(f"❌ Setup completed with {len(failed_steps)} issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease address the issues above before proceeding.")
        return 1
    else:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Set up your .env file with API keys")
        print("2. Configure config.ini with your settings")
        print("3. Start the API server: uvicorn api.api_main_updated:app --reload")
        return 0

if __name__ == "__main__":
    sys.exit(main())
