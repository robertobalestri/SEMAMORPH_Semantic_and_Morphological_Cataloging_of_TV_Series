# GitHub Copilot Instructions for SEMAMORPH

## Project Overview

SEMAMORPH is a semantic and morphological cataloging system for TV series that processes episode transcripts through an AI pipeline to extract narrative arcs, characters, and relationships. The system combines NLP, vector databases, and LLM services to analyze television content.

## Core Commands

### Backend Development
```bash
# Setup
pip install -r requirements.txt
python config_manager.py  # View/configure settings

# Run API server
uvicorn api.api_main:app --reload

# Process episodes
python main.py --series GA --season S01 --episode E01
python main.py --series GA --season S01 --start-episode 1 --end-episode 5

# Configuration management
python config_manager.py set-original true    # Use original plots vs LLM summaries
python config_manager.py set-batch pronoun_replacement_batch_size 30

# Data cleanup (preserves .srt and entities.json)
python clean_data.py --dry-run               # Preview cleanup
python clean_data.py --series GA --season S01  # Clean specific data
```

### Frontend Development
```bash
cd frontend/
npm install
npm run dev     # Development server
npm run build   # Production build
npm run lint    # ESLint
```

### Database Operations
```bash
# SQLite direct access
sqlite3 narrative_storage/narrative.db ".tables"
sqlite3 narrative_storage/narrative.db "SELECT COUNT(*) FROM character;"
```

## Architecture

### Major Components
- **`main.py`**: Main processing pipeline orchestrator
- **`api/api_main.py`**: FastAPI backend with CRUD operations for arcs, characters, progressions
- **`frontend/`**: React + TypeScript UI with Chakra UI, Plotly visualizations
- **`src/plot_processing/`**: NLP pipeline for text processing and entity extraction
- **`src/narrative_storage_management/`**: Database services, repositories, and models
- **`src/ai_models/`**: LLM service abstractions (OpenAI integration)
- **`src/langgraph_narrative_arcs_extraction/`**: LangGraph-based narrative arc extraction

### Processing Pipeline
1. **SRT → Plot**: Convert subtitles to narrative text using LLM
2. **Pronoun Resolution**: Replace pronouns with character names
3. **Entity Extraction**: spaCy + LLM-based NER for characters/locations
4. **Entity Refinement**: LLM deduplication and conflict resolution
5. **Semantic Segmentation**: Split episodes into narrative segments
6. **Arc Extraction**: LangGraph multi-agent extraction of narrative arcs
7. **Database Storage**: SQLModel/SQLAlchemy persistence with vector embeddings

### Data Stores
- **SQLite**: Primary database (`narrative_storage/narrative.db`) with tables for `narrativearc`, `character`, `arcprogression`
- **ChromaDB**: Vector embeddings for semantic similarity (`narrative_storage/chroma_db/`)
- **File System**: Organized as `data/{SERIES}/{SEASON}/{EPISODE}/` with .srt sources and processed outputs

### External APIs
- **OpenAI**: GPT models for text processing, entity extraction, narrative analysis
- **spaCy**: en_core_web_lg and en_core_web_trf models for NLP
- **AgentOps**: LLM call monitoring and analytics

## Code Style & Patterns

### Python Standards
- **Type Hints**: Mandatory for all function signatures and class attributes
- **Pydantic Models**: Use for API request/response schemas and data validation
- **Enums**: For constants and status values (`ProcessingStatus`, `LLMType`)
- **Logging**: Structured logging with emoji prefixes (🔍, ✅, ❌, ⚠️) for pipeline stages
- **Error Handling**: Comprehensive try/catch with detailed logging
- **Single Responsibility**: Keep functions focused and small

### Architecture Patterns
- **Repository Pattern**: Database access through repository classes (`NarrativeArcRepository`)
- **Service Layer**: Business logic in service classes (`CharacterService`, `NarrativeArcService`)
- **Dependency Injection**: Services receive repositories as constructor parameters
- **Path Management**: Centralized file path handling via `PathHandler` class
- **Configuration**: Centralized config management in `src/config.py`

### Naming Conventions
- **Files**: snake_case for Python files, PascalCase for React components
- **Classes**: PascalCase (`EntityLink`, `NarrativeArc`)
- **Functions/Variables**: snake_case (`extract_entities`, `episode_plot_path`)
- **Constants**: UPPER_SNAKE_CASE
- **Database**: snake_case table/column names

### Import Organization
```python
# Standard library
import os
import json
from typing import List, Optional

# Third-party
from fastapi import FastAPI
from pydantic import BaseModel
from sqlmodel import Session

# Local imports - use absolute paths from src/
from src.utils.logger_utils import setup_logging
from src.narrative_storage_management.repositories import NarrativeArcRepository
```

### Error Handling
```python
try:
    # Operation
    result = some_operation()
    logger.info(f"✅ Operation completed: {result}")
except SpecificException as e:
    logger.error(f"❌ Specific error: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"❌ Unexpected error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Logging Patterns
```python
logger = setup_logging(__name__)

# Pipeline stages with emojis
logger.info("🔍 STEP 1: EXTRACTING ENTITIES WITH SPACY")
logger.info("🔄 Improved merging: Checking X entities")
logger.info("✅ Successfully processed entity")
logger.warning("⚠️ No entities found")
logger.error("❌ Processing failed")
```

## Established Coding Rules

### From .cursorrules
- Use type hints and Pydantic models for type safety
- Implement comprehensive testing (note: test suite needs development)
- Use logging extensively for tracking operations and errors
- Keep functions small and focused (single responsibility)
- Prioritize code readability
- Use type guards for runtime validation
- Prefer classes for data structures

### Data Validation
- Always use Pydantic models for API inputs/outputs
- Validate file existence before processing
- Handle empty/null data gracefully
- Use type guards for runtime type checking

### Database Operations
- Use SQLModel for type-safe database models
- Implement proper session management with context managers
- Handle database errors with specific exception types
- Use repositories for data access abstraction

### LLM Integration
- Centralize LLM calls through service classes
- Implement proper error handling for API failures
- Use structured prompts with clear instructions
- Parse LLM responses with robust JSON handling
- Log LLM interactions for debugging

## Project-Specific Notes

### Entity Processing
- Entities have `entity_name` (normalized) and `best_appellation` (display name)
- Support multiple appellations per character for nickname handling
- Implement smart deduplication to avoid merging distinct characters
- Use LLM verification for ambiguous entity merges

### File Management
- Preserve original .srt files and entities.json - never delete
- Use episode-specific processing to avoid cross-contamination
- Implement caching - skip reprocessing if output files exist
- Use `PathHandler` for consistent file path management

### Vector Store Operations
- Store narrative arc embeddings in ChromaDB for similarity search
- Use cosine similarity for arc comparison and clustering
- Implement HDBSCAN clustering for finding similar narrative patterns
- Update embeddings when arc content changes

### Configuration System
- Use `config_manager.py` for runtime configuration changes
- Support batch size tuning for LLM operations
- Toggle between original plots and LLM-generated summaries
- Centralize all processing parameters in `src/config.py`

### No Testing Framework
The project currently lacks a formal test suite. When implementing tests:
- Use pytest as the testing framework
- Create test fixtures for database operations
- Mock LLM calls for consistent testing
- Test entity extraction and merging logic thoroughly
- Implement integration tests for the full processing pipeline
