# Product Requirements Document: Generate Recap Feature

## Executive Summary

The "Generate Recap" feature will automatically create concise, contextually relevant video summaries for TV series episodes. By analyzing ongoing narrative arcs and extracting meaningful past events from the vector database, this feature will generate approximately one-minute recap videos that help viewers understand current episode storylines.

---

## 1. Problem Statement

### Current Pain Points
- **Context Loss**: Viewers who missed previous episodes or are returning to a series after time away lack essential narrative context
- **Manual Inefficiency**: Creating episode recaps currently requires manual video editing and narrative selection
- **Inconsistent Quality**: Human-created recaps vary in quality and relevance to current episode content
- **Scalability Challenges**: Manual recap creation doesn't scale across multiple series and episodes

### User Needs
- **Quick Context**: Viewers need rapid understanding of relevant backstory before watching current episodes
- **Selective Content**: Only events directly relevant to current episode narratives should be included
- **Consistent Experience**: Recap quality and format should be predictable and professional
- **Automated Generation**: The system should work without manual intervention for processed episodes

---

## 2. Goals & Success Criteria

### Primary Goals
1. **Narrative Relevance**: Generate recaps containing only events that provide explanatory value for the current episode
2. **Optimal Duration**: Consistently produce ~1-minute recap videos that maintain viewer engagement
3. **Audio-Visual Quality**: Create professional-quality video clips with clear dialogue and smooth transitions
4. **Automated Workflow**: Enable hands-off recap generation for any processed episode after the first

### Success Metrics
- **Content Accuracy**: included events should be directly relevant to current episode ongoing arcs
- **Duration Compliance**: generated recaps should be within 50-70 seconds
- **Processing Speed**: Complete recap generation within 5 minutes per episode
- **Audio Quality**: All dialogue in recap clips should be intelligible and properly synchronized
- **System Reliability**: 0% failure rate in recap generation attempts

---

## 3. User Flow

### Trigger Conditions
- User selects an already-processed episode (any episode after Episode 1)
- Episode must have completed the full SEMAMORPH processing pipeline
- Required input files must be present in episode directory

### User Interaction Flow
1. **Episode Selection**: User navigates to episode details in the interface
2. **Recap Tab**: User clicks on new "Generate Recap" tab
3. **Initiation**: User clicks "Generate Recap" button
4. **Processing Indicator**: System displays progress indicator with processing stages
5. **Completion**: User receives notification when recap is ready
6. **Playback**: User can immediately play the generated recap video
7. **Storage**: Recap video and clip specifications JSON are automatically saved in episode's `recap_files/` folder for future access

### Error Handling
- **Missing Dependencies**: Clear error messages for missing input files
- **Processing Failures**: Descriptive error reporting with suggested remediation steps
- **Quality Issues**: Automatic retry mechanisms for failed video generation steps

---

## 4. Inputs & Data Sources

### Required Input Files
| File Type | Location | Purpose |
|-----------|----------|---------|
| `{EPISODE}_plot_possible_speakers.txt` | Episode root directory | Current episode narrative content |
| `{EPISODE}_present_running_plotlines.json` | `recap_files/` folder | Ongoing narrative arcs for current episode |
| `{SERIES}{SEASON}_plot.txt` | Season directory | Broader narrative context (optional) |
| `{EPISODE}_possible_speakers.srt` | Episode root directory | Enhanced subtitle file with speaker identification for dialogue extraction |
| `{EPISODE}.mp4` | Episode root directory | Source video file for clip extraction |
| Vector Database | ChromaDB collection | Historical events with timestamp metadata |

### Data Quality Requirements
- **Subtitle Sync**: Enhanced SRT files with speaker identification must be accurately synchronized with video content
- **Speaker Identification**: SRT files should include speaker attribution for better dialogue context
- **Timestamp Precision**: Vector DB entries must include precise start/end timestamps
- **Arc Completeness**: Running plotlines JSON must contain all active narrative threads
- **Video Quality**: Source MP4 files must be in standard format compatible with FFmpeg

---

## 5. Definition of "Interesting Event"

An event qualifies as "interesting" for recap inclusion if it meets these criteria:

### Primary Criteria
- **Narrative Relevance**: Directly relates to one or more ongoing plotlines in the current episode
- **Explanatory Value**: Provides essential context that enhances understanding of current events
- **Character Development**: Shows significant character growth, relationships, or conflicts referenced in current episode
- **Plot Advancement**: Represents turning points or key decisions that impact current storylines

### Secondary Criteria
- **Emotional Resonance**: Events with strong emotional impact that inform current character states
- **World-Building**: Establishes settings, rules, or relationships crucial to current episode understanding
- **Foreshadowing**: Past events that set up current episode developments

### Exclusion Criteria
- **Redundant Information**: Events that repeat information available in current episode
- **Tangential Content**: Events only loosely connected to current narratives
- **Technical Scenes**: Pure exposition or procedural content without character/plot significance
- **Minor Interactions**: Brief conversations or actions without lasting narrative impact

---

## 6. System Behavior

### A. Query Generation (LLM #1)
**Purpose**: Create semantically rich search queries for vector database retrieval

**Input Processing**:
- Analyze current episode plot from `_plot_possible_speakers.txt`
- Parse ongoing arcs from `_present_running_plotlines.json`
- Extract key characters, conflicts, and themes
- Incorporate season summary context when available

**Query Formulation**:
- Generate 3-5 targeted queries per ongoing arc
- Include character combinations and relationship dynamics
- Focus on conflict resolutions and emotional beats
- Prioritize events with temporal relevance to current storylines

**Output**: Structured query list with confidence weights and arc associations

### B. Event Retrieval and Ranking (LLM #2)
**Purpose**: Filter and prioritize retrieved events for maximum narrative impact

**Retrieval Process**:
- Execute generated queries against vector database
- Collect candidate events from multiple seasons/episodes
- Ensure minimum 30-second total content per ongoing arc
- Maintain diversity in event types and timeframes

**Ranking Algorithm**:
1. **Arc Relevance Score** (40%): Direct connection to current ongoing plotlines
2. **Narrative Impact Score** (30%): Significance of event to overall story development
3. **Character Importance Score** (20%): Involvement of main characters in current episode
4. **Temporal Proximity Score** (10%): Recency of event relative to current episode

**Quality Filters**:
- Remove events with poor timestamp metadata
- Exclude clips shorter than 5 seconds or longer than 30 seconds
- Filter out duplicate or highly similar events
- Ensure balanced representation across all ongoing arcs

### C. Subtitle Pruning (LLM #3)
**Purpose**: Select optimal consecutive subtitle sequences that capture event essence

**Analysis Process**:
- Parse SRT files for each selected event's full timestamp range
- Identify all subtitle entries within the event timespan
- Analyze dialogue flow and narrative progression within the event
- Evaluate which consecutive subtitle sequences best convey the event's core meaning

**Pruning Strategy**:
- Select 3-7 consecutive subtitles that most directly communicate the event's essence
- Prioritize dialogue that establishes context, conflict, or resolution
- Choose sequences that maintain natural conversation flow
- Ensure selected subtitles capture key character emotions and decisions
- Avoid breaking mid-conversation or mid-thought sequences

**Selection Criteria**:
- **Narrative Completeness**: Chosen subtitles must tell a coherent mini-story
- **Emotional Impact**: Include moments of highest dramatic tension or character revelation
- **Context Clarity**: Selected sequence should be understandable without additional context
- **Brevity**: Minimize duration while preserving essential meaning
- **Natural Boundaries**: Start and end at natural conversation or action breaks

**Output Optimization**:
- Generate JSON file with selected clip specifications stored in `recap_files/` folder
- Record precise start and end timestamps from selected subtitle sequences
- Store video source file references and clip metadata
- Maintain mapping between events, timestamps, and narrative arcs

### D. Video Clip Extraction
**Purpose**: Generate high-quality video segments using FFmpeg based on JSON specifications

**Technical Specifications**:
- Input: Original MP4 files with timestamp ranges from JSON clip specifications
- Output: Individual MP4 clips with embedded subtitles
- Quality: Maintain original resolution and frame rate
- Audio: Preserve stereo audio tracks with normalization

**Processing Pipeline**:
```bash
ffmpeg -ss [START_TIME] -i [INPUT_VIDEO] -t [DURATION] \
       -vf "subtitles=[SUBTITLE_FILE]:force_style='FontSize=16'" \
       -c:v libx264 -c:a aac \
       -avoid_negative_ts make_zero \
       [OUTPUT_CLIP]
```

**JSON-Driven Processing**:
- Read clip specifications from `recap_files/{EPISODE}_recap_clips.json`
- Extract start/end timestamps for each selected event
- Generate individual video clips based on timestamp ranges
- Apply subtitle overlay only for the selected subtitle sequences

**Quality Assurance**:
- Validate clip duration accuracy (±0.1 seconds)
- Ensure audio-video synchronization
- Verify subtitle rendering quality for selected sequences only
- Check for corruption or encoding errors

### E. Recap Assembly
**Purpose**: Merge individual clips into cohesive recap video based on JSON specifications

**Assembly Strategy**:
- Read clip order and metadata from `recap_files/{EPISODE}_recap_clips.json`
- Order clips by narrative importance and temporal logic as specified in JSON
- Add 0.5-second fade transitions between clips
- Include brief arc-identifying title cards (2-3 seconds each)
- Ensure total duration stays within 50-70 second target

**JSON-Based Processing**:
- Use clip specifications to determine final assembly order
- Apply transitions and effects based on metadata in JSON file
- Generate final recap video with all clips and transitions
- Store final recap as `{EPISODE}_recap_final.mp4` in `recap_files/` folder

**Final Processing**:
- Normalize audio levels across all clips
- Add opening/closing title sequences (3-5 seconds total)
- Apply consistent color grading and visual style
- Generate final MP4 with optimized compression

**Quality Control**:
- Automated duration verification against JSON specifications
- Audio continuity checking
- Visual transition smoothness validation
- Final output format compliance testing

---

## 7. Technical Architecture

### Core Directory Structure
```
backend/src/recap_generator/
├── __init__.py
├── recap_orchestrator.py          # Main coordination service
├── query_generator.py             # LLM #1 - Query generation
├── event_retrieval_service.py     # LLM #2 - Event filtering/ranking
├── subtitle_processor.py          # LLM #3 - Dialogue extraction
├── video_clip_extractor.py        # FFmpeg wrapper
├── recap_assembler.py             # Final video composition
├── models/
│   ├── recap_models.py            # Pydantic data models
│   └── event_models.py            # Event and ranking structures
├── utils/
│   ├── ffmpeg_utils.py            # Video processing utilities
│   ├── subtitle_utils.py          # SRT parsing and manipulation
│   └── validation_utils.py        # Input/output validation
└── exceptions/
    └── recap_exceptions.py        # Custom exception classes
```

### Integration Dependencies
- **Vector Database**: ChromaDB collection with timestamped narrative events
- **LLM Services**: Three distinct LLM instances with specialized prompts
- **FFmpeg**: System-level installation with codec support
- **Subtitle Parser**: SRT format processing with timestamp conversion
- **File System**: Structured access to episode directories and assets

### Data Flow Architecture
1. **Input Validation**: Verify all required files exist and are properly formatted
2. **Query Generation**: Generate semantic queries based on current episode context
3. **Event Retrieval**: Search vector DB and rank results by relevance
4. **Content Processing**: Extract and refine subtitle content for selected events
5. **Video Processing**: Generate individual clips using FFmpeg
6. **Assembly**: Combine clips into final recap with transitions and titles
7. **Storage**: Save final recap to episode's `recap_files/` directory

### Performance Considerations
- **Concurrent Processing**: Parallel video clip extraction for improved speed
- **Caching Strategy**: Cache generated queries and rankings for similar episodes
- **Resource Management**: Memory-efficient handling of large video files
- **Error Recovery**: Robust retry mechanisms for transient failures

---

## 8. Interface Requirements

### New UI Tab: "Generate Recap"
**Location**: Episode details interface, alongside existing episode information tabs

### User Interface Elements
1. **Recap Status Indicator**
   - Shows whether recap already exists for selected episode
   - Displays last generation timestamp if available
   - Indicates file size and duration of existing recap

2. **Generation Controls**
   - Primary "Generate Recap" button (disabled for Episode 1)
   - Progress indicator with stage-specific status updates
   - Cancel operation button during processing
   - Regenerate option for existing recaps

3. **Configuration Options** (Advanced)
   - Target duration slider (30-90 seconds)

4. **Preview and Playback**
   - Embedded video player for immediate recap viewing
   - Download link for recap file
   - Share functionality for generated recaps

### Backend API Endpoints
```typescript
POST /api/episodes/{seriesId}/{seasonId}/{episodeId}/recap/generate
GET  /api/episodes/{seriesId}/{seasonId}/{episodeId}/recap/status
GET  /api/episodes/{seriesId}/{seasonId}/{episodeId}/recap/download
DELETE /api/episodes/{seriesId}/{seasonId}/{episodeId}/recap
```

---

## 9. Constraints & Limitations

### Technical Constraints
- **Episode Prerequisites**: Only works for episodes after the first in any season
- **File Dependencies**: Requires complete SEMAMORPH processing pipeline completion
- **Duration Limits**: Recap length constrained to 50-70 seconds for optimal engagement

### Processing Constraints
- **Synchronous Operation**: Recap generation is not asynchronous (blocking operation)
- **Resource Intensive**: Requires significant CPU/memory during video processing
- **Storage Requirements**: Each recap requires ~10-50MB of additional storage
- **Network Dependencies**: Requires stable connection to vector database and LLM services

### Content Constraints
- **Subtitle Dependency**: Quality heavily dependent on accurate subtitle timing
- **Vector Database Coverage**: Limited by completeness of historical event indexing
- **Arc Consistency**: Requires well-defined ongoing plotlines in source JSON
- **Language Support**: Currently optimized for English-language content

### User Experience Constraints
- **Processing Time**: 3-5 minute generation time may impact user workflow
- **Preview Limitations**: No preview capability during generation process
- **Regeneration Cost**: Each regeneration requires full processing cycle
- **Error Recovery**: Limited automatic retry for failed operations

---

## 10. Open Questions & Considerations

### Content Strategy Questions
1. **Arc Weighting**: How should the system handle episodes with significantly different numbers of ongoing arcs (e.g., 2 arcs vs. 8 arcs)?

2. **Timeline Sensitivity**: Should recent events be weighted more heavily than older events, even if older events are more narratively significant?

3. **Cross-Season Dependencies**: How should the system handle ongoing arcs that span multiple seasons, particularly when season boundaries create natural break points?

4. **Character Focus Balance**: What's the optimal balance between main character focus and ensemble storylines in recap content?

### Technical Implementation Questions
5. **Redundancy Handling**: What's the best approach when multiple ongoing arcs reference the same past event? Should it be included once or multiple times with different context?

6. **Quality Thresholds**: At what point should the system refuse to generate a recap due to insufficient or poor-quality source material?

7. **Scalability Planning**: How will performance scale as the vector database grows to contain thousands of episodes and events?

8. **Format Flexibility**: Should the system support alternative recap formats (e.g., 30-second versions, audio-only versions, extended 2-minute versions)?

### Business Logic Questions
9. **Season Summary Integration**: Are season summaries consistently available and structured well enough to provide meaningful context, or should they be optional?

10. **User Customization**: Should users be able to manually adjust arc priorities or exclude specific types of content from recaps?

11. **Quality Metrics**: How can the system automatically assess the quality of generated recaps to identify when manual review might be needed?

12. **Content Rights**: Are there any copyright or licensing considerations for automatically generated recap content that combines clips from multiple episodes?

### Future Enhancement Questions
13. **Machine Learning Integration**: Could the system learn from user feedback to improve event selection and ranking over time?

14. **Multi-Language Support**: What modifications would be needed to support non-English content or multiple subtitle languages?

15. **Integration Ecosystem**: How might this feature integrate with existing streaming platforms or content management systems?

---

## Appendix: Technical Implementation Notes

### FFmpeg Command Templates
```bash
# Clip Extraction
ffmpeg -ss ${start_time} -i ${input_video} -t ${duration} \
       -c:v libx264 -c:a aac -preset fast \
       -avoid_negative_ts make_zero ${output_clip}

# Subtitle Overlay
ffmpeg -i ${video_clip} -vf "subtitles=${subtitle_file}" \
       -c:a copy ${output_with_subs}

# Final Assembly
ffmpeg -f concat -i ${clip_list} -c copy ${final_recap}
```

### Data Model Examples
```python
@dataclass
class RecapEvent:
    event_id: str
    start_timestamp: float
    end_timestamp: float
    relevance_score: float
    arc_association: str
    content_summary: str
    selected_subtitle_range: Dict[str, float]  # start/end times for selected subtitles

@dataclass
class RecapConfiguration:
    target_duration: int = 60  # seconds
    max_events_per_arc: int = 3
    minimum_clip_duration: float = 5.0
    maximum_clip_duration: float = 30.0
    transition_duration: float = 0.5
```

### JSON File Structure for Recap Clips
**File Location**: `recap_files/{EPISODE}_recap_clips.json`

```json
{
  "episode_id": "GAS01E02",
  "generation_timestamp": "2025-08-05T14:30:00Z",
  "total_estimated_duration": 58.5,
  "clips": [
    {
      "clip_id": "clip_001",
      "event_id": "event_abc123",
      "arc_title": "Meredith Grey And Derek Shepherd: Secret Romance",
      "source_video": "GAS01E02.mp4",
      "video_start_time": 239.20,
      "video_end_time": 304.56,
      "selected_subtitles": {
        "start_time": 250.15,
        "end_time": 295.30,
        "subtitle_indices": [98, 99, 100, 101, 102, 103]
      },
      "relevance_score": 0.92,
      "narrative_summary": "Derek and Meredith discuss ferry boats and professional boundaries",
      "assembly_order": 1
    }
  ],
  "arc_distribution": {
    "Meredith Grey And Derek Shepherd: Secret Romance": 2,
    "The Interns' Rivalries And Bonds": 1,
    "Miranda Bailey'S Mentorship": 1
  },
  "generation_metadata": {
    "llm_models_used": ["gpt-4", "gpt-4", "gpt-3.5-turbo"],
    "processing_time_seconds": 245,
    "vector_db_queries": 8,
    "total_events_analyzed": 127
  }
}
```

---

*This PRD represents the comprehensive requirements for implementing the "Generate Recap" feature within the SEMAMORPH system. It should be reviewed and approved by development, product, and stakeholder teams before implementation begins.*
