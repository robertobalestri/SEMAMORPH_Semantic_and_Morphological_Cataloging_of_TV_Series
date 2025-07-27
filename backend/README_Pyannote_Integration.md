# Pyannote Audio Integration for Speaker Identification

This document describes the implementation of Pyannote 3.1 audio-based speaker diarization integration into the existing speaker identification pipeline.

## Overview

The Pyannote Audio Integration provides a **modular speaker identification pipeline** that supports three modes:

- **`audio_only`**: Pure audio-based speaker identification using Pyannote
- **`face_only`**: Pure face-based speaker identification using existing clustering
- **`complete`**: Hybrid approach combining audio + face + character mapping

## Architecture

### Base Pipeline Structure

```
BaseSpeakerIdentificationPipeline
├── AudioOnlyPipeline
├── FaceOnlyPipeline
└── CompletePipeline
```

### Shared Components

All pipeline modes use shared components to avoid code duplication:

- **AudioProcessor**: Pyannote integration and audio extraction
- **FaceProcessor**: Face extraction and clustering
- **TimelineAligner**: Aligns audio segments with dialogue lines
- **CharacterMapper**: Maps speaker IDs to character names
- **ConfidenceScorer**: Calculates confidence scores for assignments

## Installation

### 1. Install Pyannote Dependencies

```bash
# Install Pyannote and audio processing dependencies
pip install -r requirements-pyannote.txt

# Or install manually
pip install pyannote.audio>=3.1.0
pip install ffmpeg-python>=0.2.0
pip install transformers>=4.30.0
pip install torch>=2.0.0
pip install torchaudio>=2.0.0
```

### 2. HuggingFace Authentication

You need a HuggingFace account and authentication token:

1. Create account at [HuggingFace](https://huggingface.co/)
2. Accept the Pyannote model license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Generate an access token in your HuggingFace settings
4. Add the token to your `config.ini` in the project root:

```ini
[audio]
auth_token = YOUR_HF_TOKEN_HERE
```

### 3. System Dependencies

Ensure you have FFmpeg installed:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Configuration

### Basic Configuration

The configuration has been updated in the main `config.ini` file. The new sections include:

```ini
[speaker_identification]
# Pipeline mode: audio_only, face_only, complete
mode = complete
audio_enabled = true
face_enabled = true
character_mapping_enabled = true

[audio]
# Pyannote settings
model = pyannote/speaker-diarization-3.1
confidence_threshold = 0.8
min_speaker_duration = 0.5
auth_token = YOUR_HF_TOKEN_HERE

[face]
# Face processing settings
similarity_threshold = 0.8
min_occurrences = 1.5
embedding_model = Facenet512
face_detector = retinaface
```

### Configuration Modes

#### Audio-Only Mode
```ini
[speaker_identification]
mode = audio_only
audio_enabled = true
face_enabled = false
character_mapping_enabled = false
```

#### Face-Only Mode
```ini
[speaker_identification]
mode = face_only
audio_enabled = false
face_enabled = true
character_mapping_enabled = false
```

#### Complete Mode
```ini
[speaker_identification]
mode = complete
audio_enabled = true
face_enabled = true
character_mapping_enabled = true
```

## Usage

### Basic Usage

```python
from src.subtitle_speaker_identification.pipeline_factory import run_speaker_identification_pipeline
from src.subtitle_speaker_identification.base_pipeline import DialogueLine

# Create dialogue lines
dialogue_lines = [
    DialogueLine(start_time=0, end_time=2, text="Hello"),
    DialogueLine(start_time=2, end_time=4, text="Hi there"),
]

# Run pipeline
result = run_speaker_identification_pipeline(
    series="TestSeries",
    season="S01",
    episode="E01",
    video_path="path/to/video.mp4",
    dialogue_lines=dialogue_lines,
    episode_entities=[]
)

print(f"Pipeline mode: {result['pipeline_mode']}")
print(f"Success: {result['success']}")
print(f"Statistics: {result['statistics']}")
```

### Advanced Usage

```python
from src.subtitle_speaker_identification.pipeline_factory import SpeakerIdentificationPipelineFactory
from src.subtitle_speaker_identification.base_pipeline import SpeakerIdentificationConfig
from src.path_handler import PathHandler
from src.config import config

# Create pipeline manually
path_handler = PathHandler("TestSeries", "S01", "E01")
pipeline_config = SpeakerIdentificationConfig(config)

# Validate configuration
validation = SpeakerIdentificationPipelineFactory.validate_configuration(pipeline_config)
if not validation['valid']:
    print("Configuration issues:", validation['issues'])

# Create specific pipeline
pipeline = SpeakerIdentificationPipelineFactory.create_pipeline(path_handler, pipeline_config)

# Run pipeline
result_dialogues = pipeline.run_pipeline(
    video_path="path/to/video.mp4",
    dialogue_lines=dialogue_lines,
    episode_entities=[]
)
```

## Pipeline Modes

### Audio-Only Pipeline

**Flow**: `Video → Audio Extraction → Pyannote Diarization → Speaker Segments → Timeline Alignment → Speaker Assignment`

**Use Cases**:
- When face data is unavailable or unreliable
- For audio-focused analysis
- When processing time is critical

**Advantages**:
- Fast processing
- Good speaker separation
- Works with audio-only content

### Face-Only Pipeline

**Flow**: `Video → Face Extraction → Face Embeddings → Face Clustering → Speaker Assignment`

**Use Cases**:
- When audio quality is poor
- For visual character analysis
- When audio processing fails

**Advantages**:
- Visual character identification
- Works with silent content
- Leverages existing face clustering

### Complete Pipeline

**Flow**: `Audio Processing + Face Processing → Modality Combination → Character Mapping → Final Assignment`

**Use Cases**:
- Maximum accuracy requirements
- Complete character analysis
- Research applications

**Advantages**:
- Highest accuracy
- Robust fallback mechanisms
- Comprehensive character mapping

## Confidence Logic

### Resolution Methods

- `audio_only`: Pure audio-based assignment
- `face_only`: Pure face-based assignment
- `audio_preferred`: Audio used when face uncertain
- `face_preferred`: Face used when audio uncertain
- `audio_face_agreement`: Both modalities agree
- `audio_face_disagreement`: Modalities disagree
- `audio_fallback`: Audio used as fallback
- `complete_pipeline`: Final complete pipeline result

### Confidence Scoring

```python
# Audio confidence
audio_confident = audio_confidence > audio_threshold

# Face confidence  
face_confident = face_confidence > face_threshold

# Hybrid confidence
hybrid_confident = audio_confident or face_confident
```

## Testing

### Run Tests

```bash
# Run the test suite
python test_pyannote_integration.py
```

### Test Coverage

The test suite validates:

- ✅ Configuration loading and validation
- ✅ Pipeline factory creation
- ✅ Dialogue line serialization
- ✅ Shared components initialization
- ✅ Statistics calculation

### Manual Testing

```python
# Test with sample video
from src.subtitle_speaker_identification.pipeline_factory import run_speaker_identification_pipeline

# Create test dialogue lines
dialogue_lines = [
    DialogueLine(start_time=0, end_time=2, text="Test dialogue 1"),
    DialogueLine(start_time=2, end_time=4, text="Test dialogue 2"),
]

# Run test
result = run_speaker_identification_pipeline(
    series="TestSeries",
    season="S01", 
    episode="E01",
    video_path="test_video.mp4",
    dialogue_lines=dialogue_lines,
    episode_entities=[]
)

print(f"Success: {result['success']}")
print(f"Statistics: {result['statistics']}")
```

## Performance

### Expected Performance

| Mode | Processing Time | Memory Usage | Accuracy Target |
|------|----------------|--------------|-----------------|
| `audio_only` | ~15 minutes | ~2GB | >80% |
| `face_only` | ~30 minutes | ~4GB | >70% |
| `complete` | ~45 minutes | ~6GB | >90% |

### Optimization Tips

1. **Audio Processing**:
   - Use 16kHz mono audio for faster processing
   - Normalize audio levels before processing
   - Remove silence at start/end

2. **Face Processing**:
   - Use appropriate similarity thresholds
   - Enable spatial outlier removal
   - Use cross-episode consistency

3. **Memory Management**:
   - Process in batches for large videos
   - Clear intermediate results
   - Use checkpointing for long processes

## Troubleshooting

### Common Issues

#### Pyannote Import Error
```
ImportError: No module named 'pyannote.audio'
```
**Solution**: Install Pyannote with `pip install pyannote.audio>=3.1.0`

#### HuggingFace Authentication Error
```
Authentication failed for pyannote/speaker-diarization-3.1
```
**Solution**: 
1. Accept model license at HuggingFace
2. Generate access token
3. Add token to config.ini

#### FFmpeg Not Found
```
FFmpeg not available
```
**Solution**: Install FFmpeg system dependency

#### Memory Issues
```
Out of memory error
```
**Solution**:
1. Reduce batch sizes
2. Use audio-only mode
3. Process shorter video segments

### Debug Mode

Enable debug logging:

```ini
[logging]
level = DEBUG
log_to_file = true
```

### Checkpoint Recovery

Pipelines automatically save checkpoints. To force regeneration:

```python
result = run_speaker_identification_pipeline(
    # ... other params ...
    force_regenerate=True
)
```

## API Reference

### Core Classes

#### `DialogueLine`
Represents a single dialogue line with speaker identification data.

```python
dialogue = DialogueLine(
    start_time=0.0,
    end_time=2.5,
    text="Hello",
    speaker="Speaker_1",
    is_llm_confident=True,
    resolution_method="audio_only",
    audio_confidence=0.85,
    face_confidence=0.0
)
```

#### `SpeakerIdentificationConfig`
Configuration wrapper for speaker identification settings.

```python
config = SpeakerIdentificationConfig(app_config)
mode = config.get_pipeline_mode()
audio_enabled = config.is_audio_enabled()
```

#### `BaseSpeakerIdentificationPipeline`
Base class for all pipeline modes.

```python
pipeline = BaseSpeakerIdentificationPipeline(path_handler, config)
result = pipeline.run_pipeline(video_path, dialogue_lines, episode_entities)
```

### Pipeline Classes

#### `AudioOnlyPipeline`
Audio-only speaker identification using Pyannote.

#### `FaceOnlyPipeline`
Face-only speaker identification using clustering.

#### `CompletePipeline`
Complete hybrid pipeline combining all modalities.

### Factory Functions

#### `SpeakerIdentificationPipelineFactory.create_pipeline()`
Creates the appropriate pipeline based on configuration.

#### `run_speaker_identification_pipeline()`
High-level function to run the complete pipeline.

## Contributing

### Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-pyannote.txt
```

2. Configure settings:
```bash
# Edit config.ini in the project root with your settings
# Add your HuggingFace auth token to the [audio] section
```

3. Run tests:
```bash
python test_pyannote_integration.py
```

### Code Style

- Follow existing code style
- Add type hints
- Include docstrings
- Write tests for new features

### Adding New Components

1. Extend `BaseSpeakerIdentificationPipeline`
2. Implement required methods
3. Add to `SpeakerIdentificationPipelineFactory`
4. Update configuration validation
5. Add tests

## License

This implementation follows the same license as the main SEMAMORPH project.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the test suite
3. Check configuration settings
4. Enable debug logging
5. Create an issue with detailed information 