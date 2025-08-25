# Recap Generation Workflow

This document provides a detailed breakdown of the "Previously On" recap generation pipeline. The system is designed to automatically create recap videos by identifying relevant past events, extracting key dialogue, and assembling video clips.

The entire workflow is orchestrated by the `RecapGenerator` class (`recap_generator.py`).

## High-Level Overview

The pipeline can be summarized in the following steps:

1.  **Initialization & Validation**: Set up the generator and validate that all required source files for the target episode are present.
2.  **Input Loading**: Load the necessary data for the target episode, including plot, narrative arcs, and subtitle files.
3.  **LLM #1: Query Generation**: For each narrative arc in the current episode, generate search queries to find relevant historical events.
4.  **Vector DB Search**: Use the generated queries to search a vector database of past events.
5.  **LLM #2: Event Ranking**: For each narrative arc, use an LLM to rank the retrieved events by importance and relevance to the current episode.
6.  **Final Event Selection**: Select a final list of events for the recap, ensuring diversity across narrative arcs using a round-robin strategy.
7.  **LLM #3: Dialogue Pruning**: For each selected event, use an LLM to identify the most impactful and consecutive lines of dialogue from the subtitles. This step includes a fallback mechanism to ensure quality.
8.  **Video Clip Extraction**: Use FFmpeg to extract the video segments corresponding to the selected dialogue.
9.  **Final Assembly**: Concatenate the individual clips into a single recap video file.
10. **Metadata Export**: Save a detailed JSON file that specifies which events and dialogue were used in the final recap for transparency and debugging.

---

## Detailed Step-by-Step Workflow

### 1. Initialization and Prerequisites (`recap_generator.py`)

-   The `RecapGenerator` is initialized with a `base_dir` pointing to the root of the data directory (e.g., `data/`).
-   Before running the main pipeline, the `validate_prerequisites` method is called.
-   It uses `PathHandler` to check for the existence of critical input files for the target episode:
    -   The episode's plot summary (`plot_with_possible_speakers.txt`).
    -   The active narrative arcs for the episode (`present_running_plotlines.json`).
    -   The source video file for the episode (e.g., `E09.mp4`).

### 2. Loading Episode Inputs (`utils.py`)

-   The `load_episode_inputs` function gathers all necessary data:
    -   **Episode Plot**: The main plot of the current episode.
    -   **Season Summary**: A summary of the entire season for broader context.
    -   **Narrative Arcs**: The list of running plotlines active in the current episode.
    -   **Subtitle Data**: It loads all subtitle files for the entire season to have them ready for the dialogue extraction phase. It prioritizes the speaker-identified subtitle files (`possible_speakers.srt`).

### 3. LLM #1: Generate Arc Queries (`llm_services.py`)

-   The `generate_arc_queries` function is the first LLM call in the pipeline.
-   For each narrative arc, it creates a prompt that includes the season summary, the current episode's plot, and the arc's title and description.
-   It asks the LLM to generate 2-3 specific, natural language search queries to find historical events that provide context for that arc.

### 4. Vector Database Search (`utils.py`)

-   The `search_vector_database` function takes the queries from the previous step.
-   It connects to the `VectorStoreService` (ChromaDB) to find similar events.
-   **Crucially, it builds an exclusion list to prevent searching the current or any future episodes**, ensuring the recap only contains past events.
-   The search results are grouped by their respective narrative arcs.

### 5. LLM #2: Rank Events Per Arc (`llm_services.py`)

-   The `rank_events_per_arc` function performs the second LLM call.
-   For each narrative arc, it presents the LLM with the list of events found in the vector database.
-   It prompts the LLM to select the **top 3 most essential events** that provide crucial background for the current episode.

### 6. Final Event Selection (`utils.py`)

-   The `select_events_round_robin` function creates the final list of events for the recap (defaulting to a maximum of 8).
-   It uses a **round-robin** selection strategy:
    1.  It first picks the top-ranked event from *every* narrative arc to guarantee coverage.
    2.  It then performs additional rounds, picking the next-best event from each arc until the maximum number of events is reached.

### 7. LLM #3: Extract Key Dialogue (`llm_services.py`)

-   The `extract_key_dialogue` function is the third and most complex LLM call.
-   For each event in the final selection, it retrieves the corresponding subtitles.
-   It prompts the LLM to select the **best consecutive sequence of dialogue** that is meaningful and fits within a 10-second window.
-   **Fallback Mechanism**: If the LLM determines there is no good dialogue in the originally selected event (or if the process fails), it has a fallback. It will attempt the same dialogue extraction process on the *other ranked events* from the same narrative arc. This significantly increases the quality and success rate of the generated clips.

### 8. Video Clip Extraction (`video_processor.py`)

-   The `extract_video_clips` function orchestrates the use of FFmpeg.
-   For each event, it uses the precise start and end times of the **dialogue selected by LLM #3** (not the original event's full timespan).
-   It calls a helper function (`_extract_clip_ffmpeg`) to run an FFmpeg command that cuts the clip from the source video file.
-   The extracted clips are saved in a dedicated directory managed by `PathHandler`.

### 9. Final Recap Assembly (`video_processor.py`)

-   The `assemble_final_recap` function completes the video generation.
-   It creates a text file listing all the individual clip paths.
-   It then uses FFmpeg's `concat` demuxer to stitch all the clips together into a single, final video file (e.g., `recap_S01_E09.mp4`).

### 10. JSON Export (`recap_generator.py`)

-   As a final step, the `generate_recap` method creates a detailed JSON file (`recap_clips.json`).
-   This file serves as a "spec" for the generated recap and contains:
    -   The list of all ranked events considered for each arc.
    -   The final list of selected events.
    -   The exact dialogue lines that were chosen for each clip.
    -   Debug information from the dialogue extraction step, including which event was used (original or fallback) and the LLM's raw response.
