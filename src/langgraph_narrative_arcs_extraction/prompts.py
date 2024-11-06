# prompts.pys

from langchain.prompts import ChatPromptTemplate
from textwrap import dedent

# ==============================
# Common Guidelines
# ==============================

NARRATIVE_ARC_GUIDELINES = dedent("""
**Guidelines for Narrative Arc Extraction**

1. **Arc Types:**
    - **Soap Arc**: Focuses on personal relationships (romance, family, friendships). Any love story qualifies.
    - **Genre-Specific Arc**: Pertains to the show's core genre elements (medical, political, professional conflicts) and spans multiple episodes.
    - **Anthology Arc**: Self-contained, single-episode stories driven by the show's genre.

2. **Title Creation:**
    - **Be Specific**: Titles must be clear and focused, including main characters if necessary.
    - **Avoid Vague Titles**: E.g., "Character X's Struggles".
    - **Include Key Details**: Reference main characters and the central conflict/theme.
    - **Anthology Format**: "[Genre] Case: [Specific Case Name]" (e.g., "Procedural Case: The Missing Heir").

3. **Arc Description:**
    - **Long-Term Focus**: Do not describe the progression of the arc in the episode, but give an overview of the arc season-wide (unless it's an anthology arc).
    - **Overview**: Summarize what the arc is about (e.g., "Jane and Mark's romantic relationship").
    - **Avoid useless phrases**: E.g., "in this arc".

4. **Character Lists:**
    - **Main Characters**: Protagonists driving the arc (typically two for relationship arcs).
    - **Interfering Characters**: Characters that influence the arc within the episode but are not central.

5. **Arc Distinctness:**
    - **Clarity**: Each arc should be distinct without overlapping with others.
    - **Main Type**: Even if an arc is a little mix of soap and genre-specific type, choose the most appropriate type.

6. **Key Progression Points:**
    - **Major Events**: Identify turning points that advance the storyline or alter character dynamics during the specific episode.

**Example Breakdown:**
- **Arc Type**: Soap Arc
- **Title**: “Jane's Affair and Separation from Mark”
- **Description**: Jane's extramarital affair surfaces, causing tension with Mark and their children, leading to separation.
- **Main Characters**: Jane, Mark
- **Interfering Characters**: Karla, Bob, Julie
- **Progression**: The affair of Jane with her boss is revealed to Mark. Mark decides to separate from Jane. Karla, the judge, confirms the separation. The children, Bob and Julie, become distant.
""")

# ==============================
# Common Output Formats
# ==============================

# JSON format for multiple arcs with detailed information
DETAILED_OUTPUT_JSON_FORMAT = dedent("""
[
    {
        "title": "Specific Arc title",
        "arc_type": "Soap Arc/Genre-Specific Arc/Anthology Arc",
        "description": "Brief season-wide description of the arc",
        "single_episode_progression_string": "Arc progression in this episode with key plot points.",
        "main_characters": "Character 1; Character 2; ...",
        "interfering_episode_characters": "Character 1; Character 2; ..."
    },
    ... more arcs ...
]
""")

# JSON format for multiple arcs with only title and description
BRIEF_OUTPUT_JSON_FORMAT = dedent("""
[
    {
        "title": "Specific Arc title",
        "description": "Brief season-wide description of the arc"
    },
    ... more arcs ...
]
""")

EXTRACTOR_OUTPUT_JSON_FORMAT = dedent("""
[
    {
        "title": "Specific Arc title",
        "description": "Brief season-wide description of the arc",
        "arc_type": "Soap Arc/Genre-Specific Arc"
    },
    ... more arcs ...
]
""")

PRESENT_SEASON_ARCS_OUTPUT_JSON_FORMAT = dedent("""
[
    {
        "title": "Specific Arc title",
        "description": "Brief season-wide description of the arc",
        "explanation": "Explanation of why the arc is present in the episode"
    },
    ... more arcs ...
]
""")

# ==============================
# Prompt Templates
# ==============================

# 1. Present Season Arcs Identifier
PRESENT_SEASON_ARCS_IDENTIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Season Arc Continuity Expert. Identify existing season arcs present in the episode based on the plot below.

**Episode Plot:**
{summarized_episode_plot}

**Existing Season Arcs:**
{existing_season_arcs_summaries}

If an already found seasonal arc is present in the episode, provide its title and description and a brief explanation referencing specific events or character actions from the episode.

**Return the result as a JSON array:**
{output_json_format}
"""
)

# 2. Arc Verifier
ARC_VERIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are an Expert Narratology Scholar. Verify and finalize narrative arcs based on the episode and season plots.

**Episode Plot:**
{episode_plot}

**Season Plot:**
{season_plot}

**Arcs to Verify:**
{arcs_to_verify}

**Present Season Arcs:**
{present_season_arcs_summaries}

**Guidelines:**
{guidelines}

**Instructions:**
1. If possible, match arcs to existing season arcs and update titles accordingly.
2. Following guidelines, focus on verifying titles, descriptions and progression based on the episode plot.

**Return the verified arcs as a JSON array:**
{output_json_format}
"""
)

# 3. Arc Deduplicator
ARC_DEDUPLICATOR_PROMPT = ChatPromptTemplate.from_template(
    """You are a Narrative Arc Deduplication Expert. Merge similar or overlapping arcs from different extraction methods.

**Episode Plot:**
{episode_plot}

**Arcs to Deduplicate:**
{arcs_to_deduplicate}

**Present Season Arcs (if any):**
{present_season_arcs_summaries}

**Guidelines:**
{guidelines}

**Instructions:**
1. Identify arcs describing the exact same narrative from different angles.
2. Only merge arcs if they refer to the same specific storyline with the same characters and events. 
3. Do not merge arcs that involve different relationships, storylines or character development arcs for the same character.
4. If merging, combine the descriptions to capture the full narrative.
5. Choose the most appropriate arc type.

**Additional Instructions:**
- Do not delete or merge anthology arcs, as they are self-contained and should remain distinct.

**Return the deduplicated arcs as a JSON array:**
{output_json_format}
"""
)

# 4. Arc Progression Verifier
ARC_PROGRESSION_VERIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Narrative Arc Progression Specialist. Refine the arc's description and progression based on the episode plot.

**Episode Plot:**
{episode_plot}

**Arc to Verify:**
{arc_to_verify}

**Guidelines:**
1. **Description**: Provide an overview of the entire arc across episodes.
2. **Progression**: Detail developments within this episode only, avoiding phrases like "in this episode" or "the episode shows".

**Return the verified arc as a JSON object:**
{output_json_format}
"""
)

# 5. Character Verifier
CHARACTER_VERIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Character Role Analysis Expert. Categorize characters in the arc as main protagonists or interfering characters.

**Episode Plot:**
{episode_plot}

**Arc to Verify:**
{arc_to_verify}

**Guidelines:**
1. **Main Characters**: Drive the arc's development.
2. **Interfering Characters**: Affect the arc during the specific episode but are not central.
3. Ensure relationship arcs include both protagonists.
4. Avoid background characters with minimal impact.
5. Check the arc title for main characters or groups involved.

**Return the verified arc as a JSON object:**
{output_json_format}
"""
)

# 6. Anthology Arc Extractor
ANTHOLOGY_ARC_EXTRACTOR_PROMPT = ChatPromptTemplate.from_template(
    """You are an Anthology Arc Specialist. Extract standalone anthology arcs from the episode plot.

**Only extract Anthology Arcs:**
- Self-contained cases or challenges within the episode
- Case-of-the-week elements
- Guest character storylines

**Episode Plot:**
{episode_plot}

**Guidelines:**
1. Focus on complete, self-contained stories within the episode.
2. Use the format "[Genre] Case: [Specific Case Name]", like "Procedural Case: The Missing Heir", or "Medical Case: The Heart Attack of Bob Bale".
3. Provide clear descriptions of each case's significance.

You must be sure that the arc is self-contained and not a part of a larger arc in the season.
So here is the season plot:
{season_plot}

**Return only titles and descriptions as a JSON array:**
{output_json_format}
"""
)

# 7. Arc Enhancer
ARC_ENHANCER_PROMPT = ChatPromptTemplate.from_template(
    """You are an Arc Detail Enhancement Specialist. Enrich the arc with characters and progression details.

**Episode Plot:**
{episode_plot}

**Arc to Enhance:**
{arc_to_enhance}

**Guidelines:**
{guidelines}

**Enhancements Needed:**
1. **Main Characters**: List protagonists driving the arc.
2. **Interfering Episode Characters**: List characters affecting the arc in this episode.
3. **Single Episode Progression**: Describe how the arc develops in this episode.

**Avoid temporal phrases. Describe actions and events directly.**

**Return the enhanced details as a JSON object:**
{output_json_format}
"""
)

# 8. Soap and Genre Arc Extractor
SOAP_AND_GENRE_ARC_EXTRACTOR_PROMPT = ChatPromptTemplate.from_template(
    """You are a Narrative Arc Specialist for Soap and Genre arcs. Extract all relevant soap and genre-specific arcs from the episode plot.

**Types of Arcs:**
- **Soap Arc**: Personal relationships, romances, family dynamics, friendships, personal growth, conflicts.
- **Genre-Specific Arc**: Professional conflicts, workplace dynamics, missions/objectives, power struggles, institutional challenges, skill development, political maneuvers.

**Exclude**: Anthology Arcs already identified.

**Episode Plot:**
{episode_plot}

**Existing Season Arcs:**
{present_season_arcs_summaries}

**Identified Anthology Arcs:**
{anthology_arcs}

**Guidelines:**
1. Extract all possible Soap and Genre arcs.
2. Use specific titles including main characters or themes.
3. Avoid duplicating anthology arcs.
4. Ensure arcs are season-wide, not episode-specific.
5. Be exhaustive and thorough.

**Return the arcs as a JSON array:**
{output_json_format}
"""
)

# 9. Seasonal Arc Optimizer
SEASONAL_ARC_OPTIMIZER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Seasonal Arc Optimization Expert. Optimize arc titles/descriptions and merge overlapping arcs based on the season plot.

**Season Plot:**
{season_plot}

**Present Season Arcs (if any):**
{present_season_arcs}

**New Arcs to Optimize:**
{new_arcs}

**Guidelines:**
1. **Title Optimization:**
    - Reflect the specific arc scope across the season.
    - Be precise, including key characters, relationships, and themes.
    - Avoid vague or overly broad titles.
2. **Description Enhancement:**
    - Cover the specific arc development across episodes.
    - Focus on the core narrative thread for each distinct arc.
3. **Arc Merging:**
    - Only combine arcs if they involve the exact same characters and storyline.
    - Do not merge separate character development arcs or different relationship arcs for the same character.
    - Ensure each arc represents a specific, focused narrative thread.
    - Do not merge anthology arcs, as they are self-contained.

**Avoid overmerging. Keep arcs distinct and specific. Choose the most appropriate arc type for each arc.**

**Return the optimized arcs as a JSON array:**
{output_json_format}
"""
)
