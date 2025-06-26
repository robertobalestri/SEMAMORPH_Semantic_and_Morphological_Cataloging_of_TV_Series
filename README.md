# SEMAMORPH: Semantic and Morphological Cataloging of TV Series

This project provides tools for semantic and morphological analysis of TV series content, including narrative arc extraction, character analysis, and entity linking.

## Setup

### Backend Setup
```bash
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Running the Application

1. **Start the API server:**
   ```bash
   uvicorn api.api_main:app --reload
   ```

2. **Start the frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Process episodes:**
   Run `main.py` to process individual episodes, or use the web interface's Processing tab.

## Configuration

The project includes a configuration system that allows you to customize processing behavior:

### Viewing Configuration
```bash
python3 config_manager.py
```

### Key Configuration Options

**Use Original Plot as Summary:**
```bash
# Use original plot files as summarized plots (faster processing)
python3 config_manager.py set-original true

# Generate new summarized plots using LLM (default)
python3 config_manager.py set-original false
```

When `use_original_plot_as_summary` is set to `true`, the system will copy the original plot file as the summarized plot instead of generating a new summary using LLM. This significantly speeds up processing while maintaining compatibility with the narrative arc extraction system.

## Data Management

### Processing Data
Use the web interface's "Processing" tab to process episodes and generate narrative arcs, or run:
```bash
python main.py --series GA --season S01 --episode E01
```

### Cleaning Processed Data
The project includes tools to clean processed data while preserving original source files:

#### Interactive Cleanup (Recommended)
```bash
./clean_data_interactive.sh
```

#### Command Line Cleanup
```bash
# Preview what would be deleted
python3 clean_data.py --dry-run

# Clean all processed data
python3 clean_data.py

# Clean specific series
python3 clean_data.py --series GA

# Clean specific season
python3 clean_data.py --series GA --season S01
```

**Note:** The cleaning scripts preserve:
- `*_plot.txt` (original plot files)
- `*_full_dialogues.json` (original dialogue files)

All other processed files (entities, semantic segments, narrative arcs, season plots, etc.) will be deleted.

## Processing Architecture

The system processes TV series episodes independently, without relying on season-level summaries. Each episode is processed through the following pipeline:

1. **Text Simplification** - Simplifies and clarifies the original plot text
2. **Pronoun Resolution** - Replaces pronouns with character names for clarity
3. **Entity Extraction** - Identifies and normalizes character and location entities  
4. **Semantic Segmentation** - Splits episodes into meaningful narrative segments
5. **Narrative Arc Extraction** - Identifies story arcs within each episode

## Project Structure

- `data/` - TV series data organized by series/season/episode
- `frontend/` - React-based web interface
- `api/` - FastAPI backend
- `src/` - Core processing modules
- `main.py` - Episode processing script
- `clean_data.py` - Data cleaning utility