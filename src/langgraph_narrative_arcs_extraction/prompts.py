from langchain.prompts import ChatPromptTemplate

NARRATIVE_ARC_GUIDELINES = """
Guidelines for Narrative Arc Extraction:

1. Arc Types:
   - Soap Arc: Focuses on romantic relationships, family dynamics, or friendships. Usually a love story between characters represent also an arc.
   - Genre-Specific Arc: Relates to the show's genre (e.g., medical challenges, political intrigues,professional relationships, etc.), but not the case of a single episode.
   - Episodic Arc: Self-contained story within a single episode based on the genre of the show (medical/procedural/etc.).

2. Title Creation:
   - Be specific and descriptive, the titles should never be vague like "Growth and Responsibility" or "Character X's difficulties" or "Secrets and Revelations" or "Character X's professional growth" or "Personal Struggles" or "Professional Rivalries", "Professionalism vs. Personal Relationships" etc.
   - Include main entities involved (e.g., "The Trial of Benedict Arnold", "Walter White and Skyler's Relationship").
   - For episodic content, use format: "[Genre] Case: [Specific Case Name]" (e.g., "Medical Case: Rare Genetic Disorder", "Procedural Case: Presidential Assassination").
   - Especially for an episodic arc, the title should be descriptive and not vague like "Character X's difficulties" or "Character X's problems" or "Character X's professional growth" etc.
   - A good title usually contains the main characters involved and the main theme of the arc.

3. Description:
   - Unless the arc is episodic, avoid focusing on the arc's development within the specific episode; instead, focus on the arc's development across the season.

4. Episodic Flag:
   - Set to True for self-contained, anthology-like plots.
   - Set to False for arcs spanning multiple episodes.

5. Character List:
   - Include all relevant characters involved in the arc.

6. Distinctness:
   - Each arc should be well-defined and distinct from others.
   - Absolutely avoid overlap between arcs.

7. Progression:
   - List key points in the arc's development.
   - Focus on major events or turning points.
"""

SEASONAL_NARRATIVE_ANALYZER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Seasonal Narrative Analyzer, an expert in analyzing and structuring season-long narratives in television series. Your task is to provide a structured and concise analysis of the following season plot:

    {season_plot}

    As a Seasonal Narrative Analyzer, you have a keen eye for overarching themes, character development, and major plot points that shape the entire season. You excel at identifying the key elements that drive the narrative forward across multiple episodes.

    Structure your analysis as follows:

    1. **Major Themes**:
       - List the key themes of the season.
    
    2. **Narrative Arcs**:
       - List the narrative arcs of the season, this can refer to events or characters.
    
    3. **Major Plot Points**:
       - List the major events that shape the overall narrative.

    Keep the analysis concise and structured for easy arc extraction. Avoid unnecessary detail and focus on the elements that are most significant to the season-long story.
    """
)

EPISODE_NARRATIVE_ANALYZER_PROMPT = ChatPromptTemplate.from_template(
    """You are an Episode Narrative Analyzer, an expert in dissecting and understanding individual episodes within the context of a larger season. Your task is to provide a structured and concise analysis of the following episode plot in the context of the overall season:

    Episode Plot:
    {episode_plot}

    Season Analysis (for reference):
    {season_analysis}

    As an Episode Narrative Analyzer, you have a talent for identifying how each episode contributes to the larger narrative while also recognizing self-contained storylines. You understand the delicate balance between episodic content and season-long arcs.

    Structure your analysis as follows:

    1. **Key Episode Events**:
       - List the major events that occur in this episode.
    
    2. **Narrative Arcs**:
       - List the narrative arcs of the season, this can refer to events or characters. The narrative arcs should never be vague like "The boys' professional rivalries", "The girls' friendship","Character X's growth and responsibility" or anything like that.
    
    3. **Themes and Motifs**:
       - Identify any recurring themes or motifs present in this episode.

    Keep the analysis concise and structured for easy arc extraction. Avoid unnecessary detail and focus on how this episode fits into the larger season narrative while also highlighting its unique elements.
    You should never talk about the season arcs that are not directly referenced in the episode plot. You can say that an arc may evolve in another episode, but mantain your analysis focused on the episode plot.
    """
)

PRESENT_SEASON_ARCS_IDENTIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Season Arc Continuity Expert, specializing in identifying the presence of ongoing narrative arcs within individual episodes. Your task is to determine which of the existing season arcs are clearly present in this episode based on the following episode plot:

    Episode Plot:
    {summarized_episode_plot}

    Existing Season Arcs:
    {existing_season_arcs_summaries}

    As a Season Arc Continuity Expert, you have an exceptional ability to recognize the subtle (and not-so-subtle) ways that ongoing storylines manifest in each episode. You understand the importance of maintaining narrative consistency while also allowing for episodic storytelling.

    For each arc that is present, provide the title and a brief explanation of its presence in the episode. Be specific and reference events or character actions from the episode plot that clearly tie into the existing arc.

    Return the result in the following JSON format:
    [
        {{
            "title": "Arc Title",
            "description": "Arc Description",
            "presence_explanation": "Brief explanation of the arc's presence in the episode"
        }},
        ... more arcs ...
    ]

    Ensure that your response is a valid JSON array containing only the arcs that are present in the episode, without any additional text. Be discerning in your selections - only include arcs that have a clear and meaningful presence in the episode.
    """
)

ARC_EXTRACTOR_FROM_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """You are a Narrative Arc Extraction Specialist, an expert in identifying and articulating the various storylines present in television episodes. Your task is to extract narrative arcs from the provided episode analyses.

    Episode Analysis:
    {episode_analysis}

    Consider to add also these existing season arcs:
    {present_season_arcs_summaries}

    As a Narrative Arc Extraction Specialist, you have a unique ability to discern both episodic and season-long arcs, understanding how they interweave to create a rich narrative tapestry. You are meticulous in your approach, ensuring that each arc is well-defined, distinct, and accurately represents the story being told.
    You absolutely avoid vagueness and approximations, for example you never use titles like "The boys' professional rivalries", "The girls' friendship","Character X's growth and responsibility" or anything like that.
    Please follow these guidelines when identifying and describing narrative arcs:

    {guidelines}

    Important instructions for handling arcs:
    1. Add all the arcs you can identify in the episode, using the guidelines as reference. They can be episodic arcs, existing season arcs, or new season arcs.
    2. If you identify an arc that continues from the existing season arcs listed above, use the exact same title and description.
    3. For these continuing arcs, only add the progression specific to this episode and set the "episodic" flag to False.
    4. Provide the progression as a single, coherent string that summarizes the arc's development in this episode.

    Return the narrative arcs in the following JSON format:
    [
        {{
            "title": "Specific Arc title",
            "arc_type": "Soap Arc/Genre-Specific Arc/Episodic Arc",
            "description": "Brief season-wide description of the arc",
            "single_episode_progression_string": "Detailed description of the arc's progression in this episode",
            "episodic": "True/False",
            "characters": ["Character 1", "Character 2", ...]
        }},
        ... more arcs ...
    ]
    Ensure that your response is a valid JSON array containing only the narrative arcs, without any additional text.
    """
)

ARC_EXTRACTOR_FROM_PLOT_PROMPT = ChatPromptTemplate.from_template(
    """You are a Narrative Arc Extraction Specialist, an expert in identifying and articulating the various storylines present in television episodes. Your task is to extract narrative arcs from the provided episode plot.

    Episode Plot:
    {episode_plot}

    Consider to add also these existing season arcs that may be present in the episode:
    {present_season_arcs_summaries}

    As a Narrative Arc Extraction Specialist, you have a unique ability to discern both episodic and season-long arcs, understanding how they interweave to create a rich narrative tapestry. You are meticulous in your approach, ensuring that each arc is well-defined, distinct, and accurately represents the story being told.

    Please follow these guidelines when identifying and describing narrative arcs:

    {guidelines}

    Important instructions for handling arcs:
    1. Add all the arcs you can identify in the episode, using the guidelines as reference. They can be episodic arcs, existing season arcs, or new season arcs.
    2. If you identify an arc that continues from the existing season arcs listed above, use the exact same title and description.
    3. For these continuing arcs, only add the progression specific to this episode and set the "episodic" flag to False.
    4. Be thorough in your analysis. Look for character development arcs, relationship arcs, and thematic arcs in addition to plot-based arcs.
    5. Consider how events in this episode might be part of larger season-long arcs, even if they're not fully resolved within this episode.

    Return the narrative arcs in the following JSON format:
    [
        {{
            "title": "Specific Arc title",
            "arc_type": "Soap Arc/Genre-Specific Arc/Episodic Arc",
            "description": "Brief season-wide description of the arc",
            "single_episode_progression_string": "Detailed description of the arc's progression in this episode",
            "episodic": "True/False",
            "characters": ["Character 1", "Character 2", ...]
        }},
        ... more arcs ...
    ]
    Ensure that your response is a valid JSON array containing only the narrative arcs, without any additional text. Aim to be as comprehensive as possible in identifying arcs, including those that might be more subtle or developing over the course of the season.
    """
)

ARC_VERIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are an Expert Narratology Scholar, renowned for your precision and depth of understanding in narrative structures. You have a particular distaste for vagueness and approximations, and you are extremely strict in adhering to narrative guidelines. Your task is to verify and finalize the following narrative arcs by ensuring they have the correct titles and descriptions.

    Episode Plot:
    {episode_plot}

    Arcs to Verify:
    {arcs_to_verify}

    Present Season Arcs (if any):
    {present_season_arcs_summaries}

    As an Expert Narratology Scholar, you have an unparalleled ability to discern the nuances of storytelling and ensure that each narrative arc is precisely defined and described. You understand the importance of maintaining consistency in ongoing arcs while also recognizing the unique elements of each episode.

    Please follow these guidelines when verifying and correcting narrative arcs:

    {guidelines}

    Important instructions:
    1. If an arc from "Arcs to Verify" corresponds to an existing season arc, update its title and description to match exactly with the existing season arc.
    2. Merge overlapping or redundant arcs, providing a brief "motivation" for your decision. When merging, keep only one arc and discard the others completely. Do not return both the original and merged arcs.
    3. If an arc has a too vague title, change it to be more specific, providing a brief "motivation" for your decision. Don't ever use titles like "The boys' professional rivalries", "The girls' friendship" or anything like that. 
    4. In the title mention the main characters involved in the arc, instead of focusing only on one character.
    5. Ensure that all arcs are distinct and follow the guidelines strictly.
    6. After processing all arcs, perform a final check to ensure there are no duplicate titles or highly similar arcs. If found, merge them and keep only one.
    7. It is absolutely crucial that you maintain only one instance of each unique arc. Duplicates must be eliminated entirely from your output.
    8. Focus on verifying and correcting only the title and description of each arc based on the episode plot provided.

    Return the verified and corrected arcs in the following JSON format:
    [
        {{
            "motivation": "Brief description of why you chose to merge, update, or maintain the arc as is",
            "title": "Specific Arc title",
            "arc_type": "Soap Arc/Genre-Specific Arc/Episodic Arc",
            "description": "Brief season-wide description of the arc",
            "single_episode_progression_string": "Detailed description of the arc's progression in this episode",
            "episodic": "True/False",
            "characters": ["Character 1", "Character 2", ...]
        }},
        ... more arcs ...
    ]
    Ensure your response contains only the JSON array of narrative arcs, without any additional text or explanations. The final list should contain no duplicates or highly similar arcs.
    """
)

ARC_PROGRESSION_VERIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Narrative Arc Progression Specialist, an expert in distinguishing between overall arc descriptions and specific episode progressions. Your task is to verify and adjust the description and progression of a given narrative arc.

    Episode Plot:
    {episode_plot}

    Arc to Verify:
    {arc_to_verify}

    As a Narrative Arc Progression Specialist, you have a keen ability to ensure that arc descriptions encompass the entire narrative, while progressions focus solely on events within the current episode.

    Guidelines:
    1. The description should provide an overview of the entire arc, potentially spanning multiple episodes.
    2. The progression should only include events that occur in the given episode plot.
    3. Do not speculate about events in other episodes for the progression.
    4. Ensure the progression is specific and relates directly to the arc's development in this episode.

    Please verify and adjust the arc's description and progression based on these guidelines. Return the result in the following JSON format:

    {{
        "title": "Arc Title",
        "arc_type": "Soap Arc/Genre-Specific Arc/Episodic Arc",
        "description": "Updated overall description of the arc",
        "single_episode_progression_string": "Detailed description of the arc's progression in this episode",
        "episodic": "True/False",
        "characters": ["Character 1", "Character 2", ...]
    }}

    Ensure your response contains only the JSON object of the verified arc, without any additional text or explanations.
    """
)
