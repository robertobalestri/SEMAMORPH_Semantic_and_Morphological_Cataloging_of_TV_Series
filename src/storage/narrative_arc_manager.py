# narrative_arc_manager.py

from typing import List, Optional, Dict, Any
from sqlmodel import Session, select
from src.storage.narrative_arc_models import NarrativeArc, ArcProgression
import src.storage.database as db
import src.storage.vectorstore_collection as vectorstore_collection
from src.ai_models.ai_models import get_llm, LLMType
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_utils import clean_llm_json_response
from src.utils.logger_utils import setup_logging
import uuid
import json

logger = setup_logging(__name__)




class NarrativeArcManager:
    def __init__(self):
        self.db_manager = db.DatabaseManager()
        self.vectorstore_collection = vectorstore_collection.get_vectorstore_collection(vectorstore_collection.CollectionType.NARRATIVE_ARCS)
        self.llm = get_llm(LLMType.INTELLIGENT)

    def add_or_update_narrative_arcs(
        self,
        arcs: List[NarrativeArc],
        series: str,
        season: str,
        episode: str,
        session: Session
    ) -> List[NarrativeArc]:
        if not arcs:
            logger.warning(f"No narrative arcs provided for {series} {season} {episode}.")
            return []

        cosine_distance_threshold = 0.3
        updated_arcs = []

        for arc in arcs:
            logger.info(f"Processing arc: '{arc.title}'")
            
            # Create query from arc information
            query = f"{arc.title}\n\n{arc.description}\n\n{arc.main_characters}"
            similar_arcs = self.vectorstore_collection.find_similar_arcs(query, n_results=5, series=series)

            merged = False
            for similar_arc in similar_arcs:
                if similar_arc['cosine_distance'] < cosine_distance_threshold:
                    id2search = similar_arc['metadata']['id'] if similar_arc['metadata']['doc_type'] == 'main' else similar_arc['metadata']['main_arc_id']

                    existing_arc = self.db_manager.get_narrative_arc_by_id(id2search, session=session)
                    if existing_arc:
                        logger.info(f"Found similar arc '{existing_arc.title}' with distance {similar_arc['cosine_distance']}. Checking if they should be merged...")
                        merged_arc = self.decide_arc_merging(arc, existing_arc)
                        if merged_arc:
                            logger.info(f"Merging arc '{arc.title}' with existing arc '{existing_arc.title}'")
                            # Update the existing arc with new information
                            self._update_existing_arc(existing_arc, arc, series, season, episode, session=session)
                            if existing_arc not in updated_arcs:
                                updated_arcs.append(existing_arc)
                            merged = True
                            break

            if not merged:
                logger.info(f"No similar arcs found. Adding '{arc.title}' as a new arc.")
                self.add_new_arc(arc, series, season, episode, session=session)
                updated_arcs.append(arc)

        logger.info(f"Added or updated {len(updated_arcs)} narrative arcs for {series} {season} {episode}.")
        return updated_arcs

    def decide_arc_merging(self, new_arc: NarrativeArc, existing_arc: NarrativeArc) -> Optional[NarrativeArc]:


        base_prompt = """
                You are an expert in analyzing narrative arcs in TV series.

                Given the following two narrative arcs:

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

                """
        
        if new_arc.title == existing_arc.title:
            base_prompt += """\n\nSince they share the same title, merge them and adapt the description to be comprehensive.
            Provide your response in JSON format as:
                {{
                    "merged_title": "Your merged title",
                    "merged_description": "Your merged description",
                }}
            """
        else:
            base_prompt += """\n\nDo these arcs represent the same narrative arc continuing across episodes, or are they distinct arcs?

            If they are the same arc, suggest a better title and description for the arc combining the key information,
            but remember that if the arc_type is not episodic, the arc description should be season-wide and not episode-specific.
            
            Provide your response in JSON format as:
                {{
                    "same_arc": true or false,
                    "justification": "Your explanation here",
                    "merged_title": "Your merged title if applicable, if not, use the existing arc title",
                    "merged_description": "Your merged description if applicable, if not, use the existing arc description",
                }}
            """

        prompt = ChatPromptTemplate.from_template(base_prompt)

        response = self.llm.invoke(prompt.format_messages(
            title_a=existing_arc.title,
            description_a=existing_arc.description,
            main_characters_a=existing_arc.main_characters,
            arc_type_a=existing_arc.arc_type,
            title_b=new_arc.title,
            description_b=new_arc.description,
            main_characters_b=new_arc.main_characters,
            arc_type_b=new_arc.arc_type,
        ))

        original_arc_title = existing_arc.title

        logger.info(f"LLM response for arc merging: {response.content}")    
        try:
            response_json = clean_llm_json_response(response.content)
            if isinstance(response_json, list):
                response_json = response_json[0]

            if response_json.get("same_arc", None) is None:
                response_json["same_arc"] = True
                response_json["justification"] = "Same titles"

            if isinstance(response_json, dict) and response_json.get("same_arc", False):
                existing_arc.title = response_json.get("merged_title", original_arc_title)
                existing_arc.description = response_json.get("merged_description", existing_arc.description)

                logger.info(f"Merged arc '{original_arc_title}' with '{new_arc.title}' to '{existing_arc.title}'")

                return existing_arc
            else:
                if existing_arc.title != original_arc_title:
                    logger.info(f"Decided not to merge arc '{new_arc.title}' with '{existing_arc.title}'")
                else:
                    logger.error(f"Decided not to merge arc '{new_arc.title}' with '{existing_arc.title}' even if they share the same title. The LLM response was: {json.dumps(response_json, indent=4)}")

        except Exception as e:
            logger.error(f"Error parsing LLM response for arc merging: {e}")

        return None

    def add_new_arc(self, arc: NarrativeArc, series: str, season: str, episode: str, session: Session):
        if not arc.progressions:
            logger.warning(f"No progressions found for arc '{arc.title}'. Skipping addition.")
            return

        # Assign series information if not already set
        arc.series = series

        # Ensure all progressions have necessary fields and are associated with the arc
        for progression in arc.progressions:
            if progression.id is None:
                progression.id = str(uuid.uuid4())
            progression.main_arc_id = arc.id
            progression.narrative_arc = arc  # This links the progression to the arc

            # Assign a valid ordinal_position
            if progression.ordinal_position is None:
                progression.ordinal_position = self.get_next_ordinal_position(arc.id, session)

        # Add or update the arc, which will also handle progressions due to cascading
        self.db_manager.add_or_update_narrative_arc(arc, session=session)

        # Update embeddings and other related tasks
        self._update_embeddings(arc, arc.progressions, session=session)

    def _update_embeddings(self, arc: NarrativeArc, progressions_to_update: List[ArcProgression], session: Session):
        """
        Update embeddings in the vector store for an arc and its specified progressions.
        Maintains the existing arc's ID in the vector store.
        """
        # Create main arc document
        main_doc = Document(
            page_content=f"{arc.title}\n\n{arc.description}",
            metadata={
                "title": arc.title,
                "arc_type": arc.arc_type,
                "description": arc.description,
                "episodic": arc.episodic,
                "main_characters": arc.main_characters,  # Changed from characters
                "series": arc.series,
                "doc_type": "main",
                "id": arc.id,
            }
        )

        docs = [main_doc]
        ids = [arc.id]

        for progression in progressions_to_update:
            # Ensure ordinal_position is set
            if progression.ordinal_position is None:
                progression.ordinal_position = self.get_next_ordinal_position(arc.id, session)

            prog_doc = Document(
                page_content=progression.content,
                metadata={
                    "title": arc.title,
                    "arc_type": arc.arc_type,
                    "episodic": arc.episodic,
                    "main_characters": arc.main_characters,  # Changed from characters
                    "interfering_episode_characters": progression.interfering_episode_characters,  # Added
                    "series": arc.series,
                    "doc_type": "progression",
                    "id": progression.id,
                    "main_arc_id": arc.id,
                    "season": progression.season,
                    "episode": progression.episode,
                    "ordinal_position": progression.ordinal_position
                }
            )
            docs.append(prog_doc)
            ids.append(progression.id)

        try:
            self.vectorstore_collection.add_documents(docs, ids=ids)
            logger.info(f"Updated embeddings for arc '{arc.title}' and {len(progressions_to_update)} progressions.")
        except ValueError as e:
            logger.error(f"Error updating embeddings: {e}")
            raise

    def progression_exists(self, arc_id: str, series: str, season: str, episode: str, session: Session) -> bool:
        existing_progression = session.exec(select(ArcProgression).where(
            ArcProgression.main_arc_id == arc_id,
            ArcProgression.series == series,
            ArcProgression.season == season,
            ArcProgression.episode == episode
        )).first()
        return existing_progression is not None

    def _update_existing_arc(self, existing_arc: NarrativeArc, new_arc: NarrativeArc, series: str, season: str, episode: str, session: Session):
        """
        Internal method to update an existing arc with new information.
        """
        # Update arc fields
        existing_arc.title = new_arc.title
        existing_arc.description = new_arc.description
        existing_arc.arc_type = new_arc.arc_type
        existing_arc.episodic = new_arc.episodic
        existing_arc.series = series

        # Update main characters
        existing_main_characters = set(existing_arc.main_characters.split(", ") if existing_arc.main_characters else [])
        new_main_characters = set(new_arc.main_characters.split(", ") if new_arc.main_characters else [])
        existing_arc.main_characters = ", ".join(existing_main_characters.union(new_main_characters))

        # Handle progressions
        progressions_to_update = []
        for new_progression in new_arc.progressions:
            existing_progression = self.db_manager.get_single_arc_episode_progression(
                arc_id=existing_arc.id,
                series=series,
                season=season,
                episode=episode,
                session=session
            )
            
            if existing_progression:
                # Update existing progression
                existing_progression.content = new_progression.content
                existing_progression.ordinal_position = new_progression.ordinal_position or self.get_next_ordinal_position(existing_arc.id, session)
                existing_progression.interfering_episode_characters = new_progression.interfering_episode_characters  # Added
                progressions_to_update.append(existing_progression)
            else:
                # Create new progression
                new_progression = ArcProgression(
                    id=str(uuid.uuid4()),
                    main_arc_id=existing_arc.id,
                    content=new_progression.content,
                    series=series,
                    season=season,
                    episode=episode,
                    ordinal_position=new_progression.ordinal_position or self.get_next_ordinal_position(existing_arc.id, session),
                    interfering_episode_characters=new_progression.interfering_episode_characters  # Added
                )
                session.add(new_progression)
                existing_arc.progressions.append(new_progression)
                progressions_to_update.append(new_progression)

        # Embeddings update handled within the session
        self._update_embeddings(existing_arc, progressions_to_update, session)


    def process_suggested_arcs(self, suggested_arcs_path: str, series: str, season: str, episode: str) -> List[NarrativeArc]:
        """
        Process the suggested arcs from the JSON file and update the database and vectorstore.
        """
        logger.info(f"Processing suggested arcs from {suggested_arcs_path}")

        with open(suggested_arcs_path, 'r') as f:
            suggested_arcs_data = json.load(f)

        with self.db_manager.session_scope() as session:
            final_arcs = []
            for arc_data in suggested_arcs_data:
                # Create a new NarrativeArc instance
                narrative_arc = NarrativeArc(
                    id=str(uuid.uuid4()),
                    title=arc_data['title'],
                    arc_type=arc_data['arc_type'],
                    description=arc_data['description'],
                    episodic=arc_data['episodic'],
                    main_characters=arc_data['main_characters'],  # Changed from characters
                    series=series,
                    progressions=[]
                )

                # Create and associate ArcProgression instances
                progression_data = arc_data.get('single_episode_progression_string')
                if progression_data:
                    ordinal_position = self.get_next_ordinal_position(narrative_arc.id, session)

                    progression = ArcProgression(
                        id=str(uuid.uuid4()),
                        content=progression_data,
                        series=series,
                        season=season,
                        episode=episode,
                        ordinal_position=ordinal_position,
                        interfering_episode_characters=arc_data.get('interfering_episode_characters', '')  # Added
                    )
                    narrative_arc.progressions.append(progression)
                    logger.info(f"Created progression for arc '{narrative_arc.title}' with ordinal position {ordinal_position}.")

                final_arcs.append(narrative_arc)

            updated_arcs = self.add_or_update_narrative_arcs(
                final_arcs,
                series,
                season,
                episode,
                session
            )

        logger.info(f"Processed {len(updated_arcs)} arcs for {series} {season} {episode}")
        return updated_arcs
    
    def get_next_ordinal_position(self, arc_id: str, session: Session) -> int:
        """
        Calculate the next ordinal position for a new progression within a given arc.

        Args:
            arc_id (str): The ID of the NarrativeArc.
            session (Session): The current database session.

        Returns:
            int: The next ordinal position.
        """
        existing_progressions = self.db_manager.get_arc_progressions(main_arc_id=arc_id, session=session) or []
        max_ordinal = max([p.ordinal_position or 0 for p in existing_progressions], default=0)
        return max_ordinal + 1
    


