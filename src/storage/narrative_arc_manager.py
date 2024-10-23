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
            query = f"{arc.title}\n\n{arc.description}\n\n{arc.characters}"
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
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in analyzing narrative arcs in TV series.

            Given the following two narrative arcs:

            Arc A:
            title: {title_a}
            description: {description_a}
            characters: {characters_a}
            Arc Type: {arc_type_a}

            Arc B:
            title: {title_b}
            description: {description_b}
            characters: {characters_b}
            Arc Type: {arc_type_b}

            Do these arcs represent the same narrative arc continuing across episodes, or are they distinct arcs?

            If they are the same arc, suggest a better title and description for the arc combining the key information,
            but remember that the arc is season-wide and not episode-specific.

            Provide your response in JSON format as:
            {{
                "same_arc": true or false,
                "justification": "Your explanation here",
                "merged_title": "Your merged title if applicable, if not, use the existing arc title",
                "merged_description": "Your merged description if applicable, if not, use the existing arc description"
            }}
            """
        )

        response = self.llm.invoke(prompt.format_messages(
            title_a=existing_arc.title,
            description_a=existing_arc.description,
            characters_a=existing_arc.characters,
            arc_type_a=existing_arc.arc_type,
            title_b=new_arc.title,
            description_b=new_arc.description,
            characters_b=new_arc.characters,
            arc_type_b=new_arc.arc_type,
        ))

        original_arc_title = existing_arc.title

        logger.info(f"LLM response for arc merging: {response.content}")    
        try:
            response_json = clean_llm_json_response(response.content)
            if isinstance(response_json, list):
                response_json = response_json[0]
            if isinstance(response_json, dict) and response_json.get("same_arc", False):
                existing_arc.title = response_json.get("merged_title", original_arc_title)
                existing_arc.description = response_json.get("merged_description", existing_arc.description)

                logger.info(f"Merged arc '{original_arc_title}' with '{new_arc.title}' to '{existing_arc.title}'")

                return existing_arc
        except Exception as e:
            logger.error(f"Error parsing LLM response for arc merging: {e}")

        return None

    def add_new_arc(self, arc: NarrativeArc, series: str, season: str, episode: str, session: Session):
        if not arc.progressions:
            logger.warning(f"No progressions found for arc '{arc.title}'. Skipping addition.")
            return

        
        progressions_to_update = []
        existing_progressions = self.db_manager.get_arc_progressions(arc.id, session) or []
        max_ordinal = max([p.ordinal_position or 0 for p in existing_progressions], default=0)

        for progression in arc.progressions:
            if not self.progression_exists(arc.id, series, season, episode, session):
                progression_copy = progression
                if progression_copy.id is None:
                    progression_copy.id = str(uuid.uuid4())
                progression_copy.main_arc_id = arc.id
                progression_copy.narrative_arc = arc  # Associate progression with the arc
                max_ordinal += 1
                progression_copy.ordinal_position = max_ordinal
                session.add(progression_copy)  # Add progression to the session
                #pop out the original progression from the arc referencing it with the same id
                arc.progressions = [p for p in arc.progressions if p.id != progression.id]
                arc.progressions.append(progression_copy)
                progressions_to_update.append(progression_copy)
                logger.info(f"Added progression for arc '{arc.title}' in episode {series} S{season}E{episode} with ordinal position {progression.ordinal_position}.")

        # Since we're adding a new arc, update embeddings for the arc and its progressions
        self._update_embeddings(arc, progressions_to_update, session=session)
        self.db_manager.add_or_update_narrative_arc(arc, session=session)

    def _update_embeddings(self, arc: NarrativeArc, progressions_to_update: List[ArcProgression], session: Session):
        """
        Update embeddings in the vector store for an arc and its specified progressions.
        Maintains the existing arc's ID in the vector store.
        """
        # Update main arc document with existing ID
        main_doc = Document(
            page_content=f"{arc.title}\n\n{arc.description}",
            metadata={
                "title": arc.title,
                "arc_type": arc.arc_type,
                "description": arc.description,
                "episodic": arc.episodic,
                "characters": arc.characters,
                "series": arc.series,
                "doc_type": "main",
                "id": arc.id,  # Using existing arc's ID
            }
        )

        docs = [main_doc]
        ids = [arc.id]

        for progression in progressions_to_update:
            prog_doc = Document(
                page_content=progression.content,
                metadata={
                    "title": arc.title,
                    "arc_type": arc.arc_type,
                    "episodic": arc.episodic,
                    "characters": arc.characters,
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

        # Update/add documents in vector store
        self.vectorstore_collection.add_documents(docs, ids=ids)
        logger.info(f"Updated embeddings for arc '{arc.title}' and {len(progressions_to_update)} progressions.")

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
        # Keep the existing arc's ID
        arc_id = existing_arc.id

        # Update characters
        existing_characters = set(existing_arc.characters.split(", ") if existing_arc.characters else [])
        new_characters = set(new_arc.characters.split(", ") if new_arc.characters else [])
        existing_arc.characters = ", ".join(existing_characters.union(new_characters))

        # Update existing arc with new information from LLM decision
        existing_arc.title = new_arc.title
        existing_arc.description = new_arc.description
        existing_arc.arc_type = new_arc.arc_type
        existing_arc.episodic = new_arc.episodic
        existing_arc.series = series  # Ensure series is set

        # Update in database directly
        session.add(existing_arc)

        progressions_to_update = []

        for new_progression in new_arc.progressions:
            # Check if a progression already exists for this arc in the same episode
            existing_progression = self.db_manager.get_single_arc_episode_progression(arc_id = arc_id, series = series, season = season, episode = episode, session = session)
            if existing_progression:
                # Replace the existing progression's content with the new one
                existing_progression.content = new_progression.content
                # Keep the same ordinal position
                session.add(existing_progression)
                progressions_to_update.append(existing_progression)
                logger.info(f"Replaced progression for arc '{existing_arc.title}' in episode {series} S{season}E{episode}.")
            else:
                # Add new progression with next ordinal position
                new_progression.main_arc_id = arc_id  # Use existing arc's ID
                new_progression.narrative_arc = existing_arc  # Associate with existing arc
                new_progression.id = str(uuid.uuid4())  # New ID for progression

                # Determine the next ordinal position
                existing_progressions = self.db_manager.get_arc_progressions(main_arc_id=arc_id, session = session)
                existing_progressions = existing_progressions or []  # Ensure it's a list
                max_ordinal = max([p.ordinal_position or 0 for p in existing_progressions], default=0)
                new_progression.ordinal_position = max_ordinal + 1

                # Explicitly add the progression to the session before adding it to the relationship
                session.add(new_progression)
                existing_arc.progressions.append(new_progression)

                progressions_to_update.append(new_progression)
                logger.info(f"Added new progression for arc '{existing_arc.title}' in episode {series} S{season}E{episode} with ordinal position {new_progression.ordinal_position}.")

        # Update embeddings for the existing arc with its new content and updated progressions
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
                progression = ArcProgression(
                    id=str(uuid.uuid4()),
                    content=arc_data['single_episode_progression_string'],
                    series=series,
                    season=season,
                    episode=episode
                )
                narrative_arc = NarrativeArc(
                    id=str(uuid.uuid4()),
                    title=arc_data['title'],
                    arc_type=arc_data['arc_type'],
                    description=arc_data['description'],
                    episodic=arc_data['episodic'],
                    characters=arc_data['characters'],
                    series=series,
                    progressions=[progression]
                )
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
