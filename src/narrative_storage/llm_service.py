# llm_service.py

from textwrap import dedent
from typing import Optional, Dict
from src.narrative_storage.narrative_models import NarrativeArc
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
