# llm_service.py

from textwrap import dedent
from typing import Optional, Dict, List
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

    def generate_progression_content(self, arc_title: str, arc_description: str, episode_plot_path: str, other_arcs_context: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate progression content for an arc in a specific episode and validate it against the plot."""
        try:
            # Early validation of required parameters
            if not arc_title or not arc_description:
                logger.error("Missing required arc information for generation")
                logger.error(f"Title: {arc_title}")
                logger.error(f"Description: {arc_description}")
                return "NO_PROGRESSION"

            logger.info(f"Generating progression for arc: {arc_title}")
            logger.info(f"Episode plot path: {episode_plot_path}")
            
            # Read the episode plot
            with open(episode_plot_path, 'r') as f:
                episode_plot = f.read()
                logger.info(f"Episode plot length: {len(episode_plot)} characters")

            # Build context about other arcs in the episode
            other_arcs_prompt = ""
            if other_arcs_context:
                other_arcs_prompt = "\nMajor plot points being covered by other arcs:\n"
                for arc in other_arcs_context:
                    other_arcs_prompt += f"• {arc['title']}: {arc['description']}\n"

            # Initial progression generation prompt
            base_prompt = dedent(
                """You are an expert in analyzing narrative arcs in TV series. Extract ONLY the most crucial plot points that DIRECTLY advance the specific narrative described in this arc.

                Narrative Arc to Track
                Title: {arc_title}
                Description: {arc_description}

                Episode Plot
                {episode_plot}
                
                Progression Guidelines:
                    - Write a MAXIMUM of 3-5 brief sentences
                    - Each sentence must:
                    • DIRECTLY advance the specific narrative described in the arc
                    • Contain only verified plot events
                    • Focus on actions and outcomes
                    - Exclude:
                    • any event that is not explicitly about part of this narrative arc
                    • speculation or interpretation
                    • character emotions or reactions
                    • context or background information
                    - judgement or opinion (such as "X do this demonstrating his resilience")

                If the arc has no significant developments in this episode, respond with "NO_PROGRESSION".

                Your response should contain exclusively the progression content made of sentences separated by dots. No quotes, no numbered lists, no other formatting.
                VERY IMPORTANT: Return the progression ONLY if it is present in the episode plot. If it is not, return "NO_PROGRESSION".
                """
            )

            prompt = ChatPromptTemplate.from_template(base_prompt)

            # First LLM call to generate progression
            logger.info("Sending first request to LLM with parameters:")
            logger.info(f"Arc title: {arc_title}")
            logger.info(f"Arc description: {arc_description}")
            logger.info(f"Episode plot excerpt: {episode_plot[:200]}...")
            if other_arcs_context:
                logger.info(f"Number of other arcs in context: {len(other_arcs_context)}")

            response = self.llm.invoke(prompt.format_messages(
                arc_title=arc_title,
                arc_description=arc_description,
                episode_plot=episode_plot,
                other_arcs_context=other_arcs_prompt
            ))

            progression_content = response.content.strip()
            logger.info(f"Raw LLM response content: {progression_content}")

            # If "NO_PROGRESSION", skip second call
            if progression_content == "NO_PROGRESSION":
                logger.info("LLM determined NO_PROGRESSION for this episode")
                return progression_content

            if not progression_content:
                logger.warning("LLM returned empty content")
                return "NO_PROGRESSION"

            logger.info(f"Generated progression content: {progression_content}")

            # Validation prompt to confirm arc presence and avoid false positives
            validation_prompt = dedent(
                """You are an expert in verifying the presence of narrative arcs in TV series. Analyze the following arc title and description and confirm whether they are explicitly present and relevant to the given episode plot.

                Episode Plot:
                {episode_plot}

                Arc Title: {arc_title}
                Arc Description: {arc_description}

                Validation Guidelines:
                    - Ensure the episode plot explicitly supports the arc title and description.
                    - Verify the arc's relevance to the episode plot.
                    - Pay attention to false positives and learn how to avoid them. For example, if an arc mentions "Battle of Winterfell" and in the plot it is mentioned "Battle of Riverrun", you might think that is the same thing, but it is not. Same thing, if we are talking about a "Brain surgery on patient X", and in the plot it is mentioned "Brain surgery on patient Y", it is not the same thing and the specific arc is not present in the episode plot.
                    - If the arc and description are clearly present and relevant, provide the response "TRUE."
                    - If the arc and description are NOT clearly present or relevant, provide the response "FALSE."

                Return your answer as a JSON object with the following structure:
                {{
                    "Chain of Thought": "Detailed reasoning explaining whether the arc title and description are present and relevant in the plot, referencing specific parts of the plot to justify your decision.",
                    "Response": "TRUE" or "FALSE"
                }}
                """
            )

            prompt = ChatPromptTemplate.from_template(validation_prompt)

            logger.info("Sending validation request to LLM to verify arc presence and relevance:")
            validation_response = self.llm.invoke(prompt.format_messages(
                episode_plot=episode_plot,
                arc_title=arc_title,
                arc_description=arc_description
            ))

            validation_result = clean_llm_json_response(validation_response.content.strip())
            if isinstance(validation_result, list):
                validation_result = validation_result[0]

            logger.info(f"Validation LLM response: {validation_result}")
            chain_of_thought = validation_result.get("Chain of Thought", "")
            response = validation_result.get("Response", "")

            logger.info(f"Chain of Thought: {chain_of_thought}")
            logger.info(f"Final Validation Response: {response}")

            if response == "TRUE":
                logger.info("Arc presence validated successfully")
                return progression_content
            else:
                logger.warning("Arc presence validation failed")
                return "NO_PROGRESSION"

        except FileNotFoundError as e:
            logger.error(f"Episode plot file not found: {episode_plot_path}")
            return "NO_PROGRESSION"
        except Exception as e:
            logger.error(f"Error generating progression content: {str(e)}")
            logger.exception(e)
            return "NO_PROGRESSION"

