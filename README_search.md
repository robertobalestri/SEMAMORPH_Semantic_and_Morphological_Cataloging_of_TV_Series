# Narrative Arc Event Search Tool

This tool allows you to search for events within specific narrative arcs using vector embeddings for semantic similarity.

## Files

- **`search_arc_events.py`** - Main search script
- **`search_examples.py`** - Configuration examples  
- **`test_search_setup.py`** - Test script to verify setup
- **`README_search.md`** - This documentation

## Quick Start

### 1. Test Your Setup
```bash
python test_search_setup.py
```
This will verify that your database and vector store connections are working.

### 2. Configure Your Search
Edit `search_arc_events.py` and modify the `phrase2search` variable:

```python
phrase2search = {
    "phrase": "medical emergency surgery patient",
    "narrative_arc_name": "Medical Drama Arc"
}
```

### 3. Run the Search
```bash
python search_arc_events.py
```

## Configuration Options

### In search_arc_events.py:

```python
# Main search configuration
phrase2search = {
    "phrase": "your search terms here",
    "narrative_arc_name": "Arc Name (partial match OK)"
}

# Optional settings
MAX_RESULTS = 10                # Maximum number of results to return
SIMILARITY_THRESHOLD = 0.7      # Only show results above this similarity score (0.0 - 1.0)
```

### Search Phrase Tips:
- Use specific keywords related to your target content
- Include synonyms and related terms  
- Use multiple words for better semantic matching
- Examples: 
  - `"medical emergency surgery patient hospital"`
  - `"romantic tension kiss love relationship"`
  - `"investigation evidence crime detective"`

### Narrative Arc Name:
- Uses partial matching (case-insensitive)
- Examples:
  - `"Medical"` will match "Medical Drama Arc"
  - `"Romance"` will match "Romance Storyline"
  - `"Mystery"` will match "Murder Mystery Arc"

## Example Searches

See `search_examples.py` for pre-configured search examples:

```python
# Medical Emergency Search
medical_search = {
    "phrase": "medical emergency surgery patient hospital treatment",
    "narrative_arc_name": "Medical Drama Arc"
}

# Character Relationship Search  
relationship_search = {
    "phrase": "romantic tension kiss love relationship",
    "narrative_arc_name": "Romance Arc"
}

# Crime Investigation Search
crime_search = {
    "phrase": "investigation evidence murder crime detective",
    "narrative_arc_name": "Mystery Arc"
}
```

## Output Format

The script will display:

```
🔍 SEARCH RESULTS
================================================================================
Search Phrase: 'medical emergency surgery patient'
Narrative Arc: Medical Drama Arc (GA)
Arc Type: Soap Arc
Results Found: 5
================================================================================

📋 RESULT #1
   📝 Content: Dr. Smith performs emergency surgery on a patient...
   📍 Location: GA S01E03
   🎯 Similarity: 0.892
   ⏰ Timestamp: 00:15:30,000 - 00:17:45,000
   🔢 Position: 3
   📊 Confidence: 0.85
   🔧 Method: dialogue_matching
   🆔 Event ID: event_12345
```

## Troubleshooting

### No Results Found?
1. **Lower the similarity threshold**: Change `SIMILARITY_THRESHOLD = 0.5`
2. **Try different search terms**: Use broader or more specific keywords
3. **Check arc name**: Run with wrong arc name to see available arcs
4. **Increase max results**: Change `MAX_RESULTS = 20`

### Database Connection Issues?
1. Make sure your `.env` file is configured correctly
2. Check that the database is accessible
3. Run `test_search_setup.py` to diagnose issues

### Vector Store Issues?
1. Ensure the `narrative_storage/chroma_db/` directory exists
2. Check that events have been processed and stored in the vector database
3. Verify the vector store collection name is correct

### No Events for an Arc?
- The arc might not have events processed yet
- Events are only created when episodes are processed with narrative arc extraction
- Use the "show all events" option when no results are found for debugging

## Advanced Usage

### Custom Filters
You can modify the search to add additional filters in the `search_events_in_arc` method:

```python
# Search only in specific episodes
filter={
    "$and": [
        {"doc_type": "event"},
        {"main_arc_id": arc_id},
        {"episode": "E01"}  # Only episode 1
    ]
}

# Search only in specific seasons  
filter={
    "$and": [
        {"doc_type": "event"},
        {"main_arc_id": arc_id},
        {"season": "S01"}  # Only season 1
    ]
}
```

### Multiple Arc Search
To search across multiple arcs, you can modify the script or run it multiple times with different arc names.

### Batch Processing
For multiple searches, you can create a list of `phrase2search` configurations and loop through them.

## Technical Details

- **Vector Database**: Uses ChromaDB for storing and searching embeddings
- **Embedding Model**: Uses the same model as configured in your main application
- **Similarity Metric**: Cosine similarity between query and event embeddings
- **Search Scope**: Limited to events within the specified narrative arc
- **Performance**: Scales well with large numbers of events due to vector indexing

## Integration

This script uses the same infrastructure as your main SEMAMORPH application:
- Database models and repositories
- Vector store service
- Embedding models
- Logging utilities

It's designed to work seamlessly with your existing processed data.
