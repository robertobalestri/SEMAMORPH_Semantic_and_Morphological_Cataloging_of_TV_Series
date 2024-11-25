# llm_service.py

from textwrap import dedent
from typing import Optional, Dict
from src.narrative_storage_management.narrative_models import NarrativeArc
from src.ai_models.ai_models import get_llm, LLMType
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_utils import clean_llm_json_response
from src.utils.logger_utils import setup_logging
import logging

logger = setup_logging(__name__)

class LLMService:
    """Service to interact with the LLM."""

    def __init__(self):
        self.llm = get_llm(LLMType.INTELLIGENT)

    def merge_identical_arcs(self, new_arc: NarrativeArc, existing_arc: NarrativeArc) -> dict:
        """Merge two arcs that have identical titles."""
        base_prompt = dedent("""You are an expert in analyzing narrative arcs in TV series.

                Given two versions of the same narrative arc (they have identical titles):

                Version A:
                description: {description_a}
                main_characters: {main_characters_a}

                Version B:
                description: {description_b}
                main_characters: {main_characters_b}

                Create a merged version that combines all the information. These are definitely the same arc, so focus on:
                1. Combining the descriptions comprehensively
                2. Maintaining narrative consistency
                3. Preserving all important details

                Provide your response in JSON format as:
                {{
                    "merged_description": "The combined description that incorporates details from both versions"
                }}
                """)

        prompt = ChatPromptTemplate.from_template(base_prompt)

        response = self.llm.invoke(prompt.format_messages(
            description_a=existing_arc.description,
            main_characters_a=', '.join([f"{char.best_appellation}" for char in existing_arc.main_characters]),
            description_b=new_arc.description,
            main_characters_b=', '.join([f"{char.best_appellation}" for char in new_arc.main_characters])
        ))

        try:
            response_json = clean_llm_json_response(response.content)
            if isinstance(response_json, list):
                response_json = response_json[0]
            
            if "merged_description" not in response_json:
                raise ValueError("Missing 'merged_description' in LLM response")
            
            logger.info(f"Merged identical arcs with title '{existing_arc.title}'")
            return {
                "merged_title": existing_arc.title,  # Keep original title
                "merged_description": response_json["merged_description"]
            }

        except Exception as e:
            logger.error(f"Error parsing LLM response for identical arc merging: {e}")
            return {
                "merged_title": existing_arc.title,
                "merged_description": f"{existing_arc.description}\n\nAdditional context: {new_arc.description}"
            }

    def decide_arc_merging(self, new_arc: NarrativeArc, existing_arc: NarrativeArc) -> dict:
        """Decide if two similar (but not identical) arcs should be merged."""
        base_prompt = dedent("""You are an expert in analyzing narrative arcs in TV series.

                Given these two narrative arcs:

                Arc A:
                title: {title_a}
                description: {description_a}
                main_characters: {main_characters_a}
                Arc Type: {arc_type_a}

                Arc B:
                title: {title_b}
                description: {description_b}
                main_characters: {main_characters_b}
                Arc Type: {arc_type_b}

                Analyze if they represent the same narrative arc. Consider:
                1. The core narrative development being tracked
                2. Whether one is a subset/continuation of the other
                3. The main characters involved
                4. The specific events or developments described
                             
                If they are similar arcs, provide a merged version that combines all the information, even if they are not identical.
                You should not be too strict.

                Provide your response in JSON format as:
                {{
                    "same_arc": true or false,
                    "justification": "Your detailed explanation here",
                    "merged_title": "ONLY IF same_arc is true: Best title for the merged arc",
                    "merged_description": "ONLY IF same_arc is true: Combined description"
                }}
                """)

        prompt = ChatPromptTemplate.from_template(base_prompt)

        response = self.llm.invoke(prompt.format_messages(
            title_a=existing_arc.title,
            description_a=existing_arc.description,
            main_characters_a=', '.join([f"{char.best_appellation}" for char in existing_arc.main_characters]),
            arc_type_a=existing_arc.arc_type,
            title_b=new_arc.title,
            description_b=new_arc.description,
            main_characters_b=', '.join([f"{char.best_appellation}" for char in new_arc.main_characters]),
            arc_type_b=new_arc.arc_type,
        ))

        try:
            response_json = clean_llm_json_response(response.content)
            if isinstance(response_json, list):
                response_json = response_json[0]
            
            if "same_arc" not in response_json or "justification" not in response_json:
                raise ValueError("Missing required fields in LLM response")
            
            if new_arc.title != existing_arc.title:
                logger.warning(f"Decided {response_json['same_arc']} for 'arc merging' for '{existing_arc.title}' and '{new_arc.title}' and they have DIFFERENT titles")
            else:
                logger.warning(f"Decided {response_json['same_arc']} for 'arc merging' for '{existing_arc.title}' and '{new_arc.title}' and they have the SAME title")

            return response_json

        except Exception as e:
            logger.error(f"Error parsing LLM response for arc merging decision: {e}")
            return {
                "same_arc": False,
                "justification": "LLM error",
                "merged_title": existing_arc.title,
                "merged_description": f"{existing_arc.description}\n\nAdditional context: {new_arc.description}"
            }

    def generate_progression_content(self, arc_title: str, arc_description: str, episode_plot_path: str) -> str:
        """Generate progression content for an arc in a specific episode."""
        try:
            # Early validation of required parameters
            if not arc_title or not arc_description:
                logger.error("Missing required arc information for generation")
                logger.error(f"Title: {arc_title}")
                logger.error(f"Description: {arc_description}")
                return ""

            logger.info(f"Generating progression for arc: {arc_title}")
            logger.info(f"Episode plot path: {episode_plot_path}")
            
            # Read the episode plot
            with open(episode_plot_path, 'r') as f:
                episode_plot = f.read()
                logger.info(f"Episode plot length: {len(episode_plot)} characters")

            base_prompt = dedent(
                """You are an expert in analyzing narrative arcs in TV series. Given a narrative arc and an episode plot, identify the key events that advance the arc in the episode.

                Narrative Arc
                Title: {arc_title}
                Description: {arc_description}

                Episode Plot
                {episode_plot}

                Progression Guidelines:

                    Focus solely on events specific to this arc in this episode.
                    Write concise points, each separated by a dot.
                    Use active voice and simple present tense.
                    Include only events directly relevant to the arc.
                    Avoid analysis, speculation, or references to other arcs or general episode events.

                **Example Good Progression:**
                "Jane discovers Mark's affair with his secretary. Mark moves out of the house. Their children choose to stay with Jane."

                **Example Bad Progression:**
                "In this episode, we see Jane struggling with her emotions when she finds out about Mark's affair, which leads to a confrontation where Mark decides to leave, showing how their relationship has deteriorated, and interestingly their children, who are also affected by this situation, decide to stay with their mother."
                                 
                If the arc does not have any significant development in this episode, respond with "NO_PROGRESSION".
                                 
                """)

            prompt = ChatPromptTemplate.from_template(base_prompt)

            logger.info("Sending request to LLM with parameters:")
            logger.info(f"Arc title: {arc_title}")
            logger.info(f"Arc description: {arc_description}")
            logger.info(f"Episode plot excerpt: {episode_plot[:200]}...")
            
            response = self.llm.invoke(prompt.format_messages(
                arc_title=arc_title,
                arc_description=arc_description,
                episode_plot=episode_plot
            ))

            content = response.content.strip()
            logger.info(f"Raw LLM response content: {content}")
            
            if content == "NO_PROGRESSION":
                logger.info("LLM determined NO_PROGRESSION for this episode")
                return content
            
            if not content:
                logger.warning("LLM returned empty content")
                return ""
            
            logger.info(f"Generated progression content: {content}")
            return content

        except FileNotFoundError as e:
            logger.error(f"Episode plot file not found: {episode_plot_path}")
            return ""
        except Exception as e:
            logger.error(f"Error generating progression content: {str(e)}")
            logger.exception(e)
            return ""
