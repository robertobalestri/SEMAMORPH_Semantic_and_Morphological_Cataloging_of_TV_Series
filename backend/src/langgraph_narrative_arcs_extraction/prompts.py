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
                                  
4. **Progression**: The progression should be specific to the arc in the episode, without speculations about what do the progressions mean for a character or other reasoning or research of meaning. Progressions should be precise key plot points separated by a dot.

5. **Character Lists:**
    - **Main Characters**: Protagonists driving the arc (typically two for relationship arcs). An arc should have at least one main character.
    - **Interfering Characters**: Characters that influence the arc within the episode (main characters or others)

6. **Arc Distinctness:**
    - **Clarity**: Each arc should be distinct without overlapping with others.
    - **Main Type**: Even if an arc is a little mix of soap and genre-specific type, choose the most appropriate type.
    - **Non-overlapping**: Ensure arcs do not share key plot points or character developments.
    
7. **Key Progression Points:**
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
    """You are a Season Arc Continuity Expert. Determine if the given season arc is present in the episode based on the plot below.

**Episode Plot:**
{summarized_episode_plot}

**Season Arc to Check:**
Title: {arc_title}
Description: {arc_description}

Determine if this specific season arc continues or develops in this episode. If it does, provide a brief explanation referencing specific events or character actions from the episode.

**Return the result as a JSON object:**
{{
    "is_present": true/false,
    "title": "Arc title",
    "description": "Arc description",
    "explanation": "Explanation of why the arc is present in the episode (if is_present is true)"
}}
"""
)

# 2. Arc Verifier
ARC_VERIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are an Expert Narratology Scholar. Verify and finalize narrative arcs based on the episode plot.

**Episode Plot:**
{episode_plot}

**Arcs to Verify:**
{arcs_to_verify}

**Present Season Arcs:**
{present_season_arcs_summaries}

**Guidelines:**
{guidelines}

**Instructions:**
1. If possible, match arcs to existing season arcs and update titles accordingly.
2. Following guidelines, focus on verifying titles, descriptions and progression based on the episode plot.
3. Ensure all character lists and progression details are maintained.
4. Keep the original fields if they are valid, only modify if necessary.

**Return the verified arcs as a JSON array with each arc containing:**
{output_json_format}
"""
)

# 3. Arc Deduplicator
ARC_DEDUPLICATOR_PROMPT = ChatPromptTemplate.from_template(
    """You are a Narrative Arc Synthesizer and Consolidator. Your primary goal is to merge overlapping and fragmented arcs into a smaller set of coherent, distinct, major storylines. Be aggressive and decisive in your consolidation.

**Episode Plot:**
{episode_plot}

**Anthology Arcs (Context):**
{anthology_arcs}

**Arcs to Review:**
{arcs_to_deduplicate}

**Guidelines:**
{guidelines}

**Consolidation Rules:**
1.  **Merge Thematic Facets into a Single Storyline**: If multiple arcs describe different aspects of the same core story (e.g., one arc about internal team conflict, another about external pressure on the team), **merge them**. Create a single, comprehensive arc that covers all these facets.

2.  **Combine Individual Journeys into Group Arcs**: If an individual character's arc is primarily a sub-plot of a larger group's story, **merge it into the group arc**. The description of the consolidated arc should then highlight the key individual journeys within it.
    -   **Exception**: Keep an individual arc separate *only if* it has a major, distinct storyline that exists largely independent of the group's main plot (e.g., a character's secret personal quest, a family drama separate from their professional life).

3.  **Synthesize Relationship Dynamics**: If an arc about a specific relationship (e.g., "Character A and B's Rivalry") is a central dynamic of a larger group or institutional arc, **merge it**. The new description should feature that relationship as a key driver of the consolidated storyline.

4.  **Maintain Title but update Description if needed**: When merging, synthesize a new, more comprehensive description that accurately covers the full scope of the new, consolidated storyline. Just remember that what happens in the specific episode is already covered by the "progression". So you only need to update the description if really necessary.

5.  **Anthology Arcs are Off-Limits**: The anthology arcs provided are for context only. DO NOT modify, merge, or include them in the final output.

**Your Task:**
Review the "Arcs to Review" and apply the consolidation rules to produce a refined, deduplicated list of major, non-anthology storylines.

**Return ONLY the consolidated and refined non-anthology arcs as a JSON array:**
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
2. **Progression**: 
   - Focus ONLY on key plot points specific to this arc in this episode
   - Write brief, focused points separated by dots
   - Include ONLY events directly related to this arc's development
   - Avoid any analysis, speculation, or character motivations
   - Exclude events from other arcs or general episode events
   - Use simple present tense, active voice
   - Omit phrases like "in this episode" or "we see that"

**Example Good Progression:**
"Jane discovers Mark's affair with his secretary. Mark moves out of the house. Their children choose to stay with Jane."

**Example Bad Progression:**
"In this episode, we see Jane struggling with her emotions when she finds out about Mark's affair, which leads to a confrontation where Mark decides to leave, showing how their relationship has deteriorated, and interestingly their children, who are also affected by this situation, decide to stay with their mother."

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
1. **Main Characters**: Drive the arc's development. They are the absolute protagonists of the arc.
2. **Interfering Characters**: Affect the arc during the specific episode. They can be both main characters or others.
3. Ensure relationship arcs include both protagonists as main characters.
4. Avoid background characters with minimal impact.

**Return the verified arc as a JSON object:**
{output_json_format}
"""
)

# 6. Anthology Arc Extractor
ANTHOLOGY_ARC_EXTRACTOR_PROMPT = ChatPromptTemplate.from_template(
    """You are an Anthology Arc Specialist. Extract standalone anthology arcs from the episode plot.

**Only extract Anthology Arcs:**
- Self-contained cases or challenges within the episode (for example episode murders in CSI, episode medical cases in Dr. House etc. etc.)
- Case-of-the-week elements

**Episode Plot:**
{episode_plot}

**Guidelines:**
1. Focus on complete, self-contained stories within the episode.
2. Use the format "[Genre] Case: [Specific Case Name]".
3. Provide clear descriptions of each case's significance.
4. DO NOT create arcs that are not about a genre-specific case of the episode.

Each arc should be self-contained within the episode and focus on the specific case or procedure presented.

**Example of titles:**
- Medical Case: The Heart Attack of Bob Bale
- Procedural Case: The Missing Heir
- Murder Case: Mary Bale stabbed with a kitchen knife

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
2. **Interfering Episode Characters**: List characters affecting the arc in this specific episode.
3. **Single Episode Progression**: 
   - Write ONLY key plot points specific to this arc
   - Use brief, focused statements separated by dots
   - Include ONLY events that directly advance this arc
   - Exclude background events or other arcs' developments
   - Use simple present tense
   - Focus on actions and concrete events
   - Avoid analysis or speculation

**Example Good Progression:**
"Detective Smith finds the murder weapon. The forensic team links it to the suspect. Smith arrests the suspect."

**Example Bad Progression:**
"Detective Smith continues his investigation and makes progress when he discovers the important murder weapon, which leads to a breakthrough in the case as the forensic team does their analysis, ultimately resulting in the arrest of the suspect who had been under surveillance."

**Return the enhanced details as a JSON object:**
{output_json_format}
"""
)

# 8. Soap and Genre Arc Extractor
SOAP_AND_GENRE_ARC_EXTRACTOR_PROMPT = ChatPromptTemplate.from_template(
    """You are a Narrative Arc Specialist for Soap and Genre arcs. Extract all relevant soap and genre-specific arcs from the episode plot.

**Types of Arcs:**
- **Soap Arc**: Personal relationships, romances, family dynamics, friendships, personal growth, conflicts, character development
- **Genre-Specific Arc**: Professional conflicts, workplace dynamics, missions/objectives, power struggles, institutional challenges, skill development, political maneuvers, battles, etc.

**Examples of Genre-Specific Arcs:**
- For a zombie series, a genre-specific arc could be "Building of the new village for the survivors".
- For a drama series on bank employees, a genre-specific arc could be "The new director's conflict with employees".
- For Breaking Bad, a genre-specific arc could be "Heisenberg's work for Gus" or "Saul Goodman's criminal activities".

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
6. Be sure to include every already existing season arc in the output (this is very important)
7. For already existing season arcs, mantain their title and adapt the description to your new knowledge.

**Return the arcs as a JSON array:**
{output_json_format}
"""
)

# 9. Seasonal Arc Optimizer
SEASONAL_ARC_OPTIMIZER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Seasonal Arc Optimization Expert. Optimize arc titles/descriptions and merge overlapping arcs based on the existing seasonal context.

**Present Season Arcs (if any):**
{present_season_arcs}

**New Arcs to Optimize:**
{new_arcs}

**Guidelines:**
1. **Title Optimization:**
    - Reflect the specific arc scope based on existing seasonal context.
    - Be precise, including key characters, relationships, and themes.
    - Avoid vague or overly broad titles.
    - IF AN ARC IS IN PRESENT SEASON ARCS, DO NOT CHANGE THE TITLE!
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
