# narrative_arc_manager.py

from typing import List, Optional, Dict, Any
from sqlmodel import Session, select
from src.storage.narrative_models import NarrativeArc, ArcProgression, Character
import src.storage.database as db
import src.storage.vectorstore_collection as vectorstore_collection
from src.ai_models.ai_models import get_llm, LLMType
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_utils import clean_llm_json_response
from src.utils.logger_utils import setup_logging
from src.plot_processing.plot_ner_entity_extraction import (
    EntityLink
)
import uuid
import json
from src.storage.character_storage import CharacterStorage
import os

logger = setup_logging(__name__)




class NarrativeArcManager:
    def __init__(self):
        self.db_manager = db.DatabaseManager()
        self.vectorstore_collection = vectorstore_collection.get_vectorstore_collection(vectorstore_collection.CollectionType.NARRATIVE_ARCS)
        self.llm = get_llm(LLMType.INTELLIGENT)
        self.character_storage = CharacterStorage()

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
                if arc not in updated_arcs:  # Add this check to prevent duplicates
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
        """
        try:
            # Convert empty lists to strings for metadata
            main_characters = arc.main_characters if arc.main_characters else ""
            if isinstance(main_characters, list):
                main_characters = ", ".join(main_characters)

            # Create main arc document
            main_doc = Document(
                page_content=f"{arc.title}\n\n{arc.description}",
                metadata={
                    "title": arc.title,
                    "arc_type": arc.arc_type,
                    "description": arc.description,
                    "episodic": arc.episodic,
                    "main_characters": main_characters,
                    "series": arc.series,
                    "doc_type": "main",
                    "id": arc.id,
                }
            )

            docs = [main_doc]
            ids = [arc.id]

            # Create a set to track used IDs
            used_ids = {arc.id}

            for progression in progressions_to_update:
                # Generate a new unique ID if the current one is already used
                prog_id = progression.id
                while prog_id in used_ids:
                    prog_id = str(uuid.uuid4())
                used_ids.add(prog_id)

                # Convert empty lists to strings for metadata
                interfering_chars = progression.interfering_episode_characters if progression.interfering_episode_characters else ""
                if isinstance(interfering_chars, list):
                    interfering_chars = ", ".join(interfering_chars)

                prog_doc = Document(
                    page_content=progression.content,
                    metadata={
                        "title": arc.title,
                        "arc_type": arc.arc_type,
                        "episodic": arc.episodic,
                        "main_characters": main_characters,
                        "interfering_episode_characters": interfering_chars,
                        "series": arc.series,
                        "doc_type": "progression",
                        "id": prog_id,
                        "main_arc_id": arc.id,
                        "season": progression.season,
                        "episode": progression.episode,
                        "ordinal_position": progression.ordinal_position
                    }
                )
                docs.append(prog_doc)
                ids.append(prog_id)

            # Delete existing documents
            try:
                # Delete main arc document
                self.vectorstore_collection.delete(
                    filter={"$and": [{"id": arc.id}, {"doc_type": "main"}]}
                )
                # Delete progression documents
                self.vectorstore_collection.delete(
                    filter={"$and": [{"main_arc_id": arc.id}, {"doc_type": "progression"}]}
                )
            except Exception as e:
                logger.warning(f"Could not delete existing documents: {e}. Continuing with upsert.")

            # Add new documents
            self.vectorstore_collection.add_documents(docs, ids=ids)
            logger.info(f"Updated embeddings for arc '{arc.title}' and {len(progressions_to_update)} progressions.")
        except Exception as e:
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

        # Update main characters (now using List[str])
        existing_arc.main_characters = list(set(existing_arc.main_characters + new_arc.main_characters))

        # Get existing progression for this episode if any
        existing_progression = self.db_manager.get_single_arc_episode_progression(
            arc_id=existing_arc.id,
            series=series,
            season=season,
            episode=episode,
            session=session
        )

        # Get all progressions for this arc to determine the correct ordinal position
        all_progressions = self.db_manager.get_arc_progressions(main_arc_id=existing_arc.id, session=session)
        next_ordinal = max([p.ordinal_position for p in all_progressions], default=0) + 1

        # Handle progressions
        for new_progression in new_arc.progressions:
            if existing_progression:
                # Update existing progression content but keep its original ordinal position
                existing_progression.content = new_progression.content
                existing_progression.interfering_episode_characters = new_progression.interfering_episode_characters
                logger.info(f"Updated existing progression for arc {existing_arc.title} with ordinal position {existing_progression.ordinal_position}")
            else:
                # Create new progression with next available ordinal position
                new_progression.id = str(uuid.uuid4())
                new_progression.main_arc_id = existing_arc.id
                new_progression.narrative_arc = existing_arc
                new_progression.ordinal_position = next_ordinal
                session.add(new_progression)
                existing_arc.progressions.append(new_progression)
                logger.info(f"Added new progression for arc {existing_arc.title} with ordinal position {next_ordinal}")

        # Update embeddings
        self._update_embeddings(existing_arc, existing_arc.progressions, session)

    def link_characters_to_arc(self, characters: List[Character], arc: NarrativeArc):
        """Link characters to a narrative arc."""
        # Filter out None values
        valid_characters = [c for c in characters if c is not None]
        
        # Create sets for existing and new characters
        existing_ids = {c.id for c in arc.characters}
        new_characters = [c for c in valid_characters if c.id not in existing_ids]
        
        if new_characters:
            # Add new characters to the relationship
            arc.characters.extend(new_characters)
            
            # Update main_characters list with all characters' best appellations
            all_appellations = [c.best_appellation for c in arc.characters]
            arc.main_characters = list(set(all_appellations))
            
            # Ensure the relationship is bidirectional
            for character in new_characters:
                if arc not in character.narrative_arcs:
                    character.narrative_arcs.append(arc)
            
            logger.info(f"Linked {len(new_characters)} characters to arc '{arc.title}'")

    def link_characters_to_progression(self, characters: List[Character], progression: ArcProgression):
        """Link characters to an arc progression."""
        # Filter out None values
        valid_characters = [c for c in characters if c is not None]
        
        # Create sets for existing and new characters
        existing_ids = {c.id for c in progression.characters}
        new_characters = [c for c in valid_characters if c.id not in existing_ids]
        
        if new_characters:
            # Add new characters to the relationship
            progression.characters.extend(new_characters)
            
            # Update interfering_episode_characters list with all characters' best appellations
            all_appellations = [c.best_appellation for c in progression.characters]
            progression.interfering_episode_characters = list(set(all_appellations))
            
            # Ensure the relationship is bidirectional
            for character in new_characters:
                if progression not in character.progressions:
                    character.progressions.append(progression)
            
            logger.info(f"Linked {len(new_characters)} characters to progression {progression.id}")

    def process_suggested_arcs(self, suggested_arcs_path: str, series: str, season: str, episode: str) -> List[NarrativeArc]:
        """Process the suggested arcs from the JSON file and update the database and vectorstore."""
        with open(suggested_arcs_path, 'r') as f:
            suggested_arcs_data = json.load(f)

        with self.db_manager.session_scope() as session:
            final_arcs = []
            for arc_data in suggested_arcs_data:
                narrative_arc = NarrativeArc(
                    id=str(uuid.uuid4()),
                    title=arc_data['title'],
                    arc_type=arc_data['arc_type'],
                    description=arc_data['description'],
                    episodic=arc_data['episodic'],
                    series=series,
                    progressions=[],
                    main_characters=[]
                )

                # Get main characters from database
                if arc_data.get('main_characters'):
                    character_names = [name.strip() for name in arc_data['main_characters'].split(';') if name.strip()]
                    arc_characters = self.character_storage.get_characters_by_appellations(character_names, series, session)
                    if arc_characters:
                        self.link_characters_to_arc(arc_characters, narrative_arc)

                # Handle progression
                progression_data = arc_data.get('single_episode_progression_string')
                if progression_data:
                    # First, check if this arc already exists and get its ID
                    existing_arc = self.db_manager.get_narrative_arc_by_title(narrative_arc.title, series, session)
                    arc_id = existing_arc.id if existing_arc else narrative_arc.id
                    
                    # Get the next ordinal position based on existing progressions
                    next_position = self.get_next_ordinal_position(arc_id, session)
                    logger.debug(f"Next ordinal position for arc {arc_id}: {next_position}")

                    progression = ArcProgression(
                        id=str(uuid.uuid4()),
                        main_arc_id=narrative_arc.id,
                        content=progression_data,
                        series=series,
                        season=season,
                        episode=episode,
                        ordinal_position=next_position,  # Use the calculated position
                        interfering_episode_characters=[]
                    )
                    progression.narrative_arc = narrative_arc

                    # Get interfering characters from database
                    if arc_data.get('interfering_episode_characters'):
                        interfering_names = [name.strip() for name in arc_data['interfering_episode_characters'].split(';') if name.strip()]
                        progression_characters = self.character_storage.get_characters_by_appellations(interfering_names, series, session)
                        if progression_characters:
                            self.link_characters_to_progression(progression_characters, progression)

                    narrative_arc.progressions.append(progression)

                final_arcs.append(narrative_arc)

            updated_arcs = self.add_or_update_narrative_arcs(
                final_arcs,
                series,
                season,
                episode,
                session
            )

            return updated_arcs
    
    def get_next_ordinal_position(self, arc_id: str, session: Session) -> int:
        """
        Calculate the next ordinal position for a new progression within a given arc.
        The ordinal position represents the sequence of progressions within the arc.

        Args:
            arc_id (str): The ID of the NarrativeArc.
            session (Session): The current database session.

        Returns:
            int: The next available ordinal position (1-based).
        """
        # Get all progressions for this arc
        existing_progressions = self.db_manager.get_arc_progressions(main_arc_id=arc_id, session=session)
        
        if not existing_progressions:
            return 1

        # Get all used ordinal positions
        used_positions = {p.ordinal_position for p in existing_progressions if p.ordinal_position is not None}
        
        # Find the first available position
        position = 1
        while position in used_positions:
            position += 1
        
        # Log for debugging
        logger.debug(f"Next available ordinal position for arc {arc_id}: {position}")
        
        return position
    






























