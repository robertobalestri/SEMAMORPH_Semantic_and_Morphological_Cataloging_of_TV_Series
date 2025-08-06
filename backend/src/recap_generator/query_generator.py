"""
Query Generator Service for recap generation.

This service generates targeted vector database queries for finding relevant historical events
based on narrative arcs present in the current episode. Follows the proper LLM workflow:
one LLM call per narrative arc to generate semantically rich queries.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..ai_models.ai_models import get_llm, LLMType
from ..utils.logger_utils import setup_logging
from ..utils.llm_utils import clean_llm_json_response
from .exceptions.recap_exceptions import LLMServiceError, MissingInputFilesError
from .models.event_models import QueryResult
from ..path_handler import PathHandler

logger = setup_logging(__name__)


class QueryGeneratorService:
    """Service for generating vector database queries per narrative arc in the current episode."""
    
    def __init__(self, path_handler: PathHandler):
        self.path_handler = path_handler
        self.llm = get_llm(LLMType.INTELLIGENT)
        
    def generate_queries_for_narrative_arcs(self) -> List[Dict[str, Any]]:
        """
        Generate targeted vector database queries for each narrative arc present in the episode.
        
        Returns:
            List of query dictionaries, one per narrative arc with generated queries
        """
        try:
            logger.info(f"🔍 Generating queries for narrative arcs in episode: {self.path_handler.get_episode_code()}")
            
            # Load required input files
            season_summary = self._load_season_summary()
            plot_content = self._load_plot_content()
            running_plotlines = self._load_running_plotlines()
            
            # Generate queries for each narrative arc
            arc_queries = []
            plotlines = running_plotlines.get('running_plotlines', [])
            
            logger.info(f"📊 Processing {len(plotlines)} narrative arcs")
            
            for i, plotline in enumerate(plotlines, 1):
                arc_title = plotline.get('title', f'Arc_{i}')
                arc_id = plotline.get('narrative_arc_id')
                
                logger.info(f"🎯 Generating queries for arc {i}/{len(plotlines)}: '{arc_title}'")
                
                try:
                    queries = self._generate_queries_for_arc(
                        plotline, season_summary, plot_content
                    )
                    
                    arc_query_data = {
                        'narrative_arc_id': arc_id,
                        'arc_title': arc_title,
                        'arc_description': plotline.get('description', ''),
                        'queries': queries,
                        'total_queries': len(queries)
                    }
                    
                    arc_queries.append(arc_query_data)
                    logger.info(f"✅ Generated {len(queries)} queries for arc '{arc_title}'")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to generate queries for arc '{arc_title}': {e}")
                    # Continue processing other arcs
                    
            logger.info(f"✅ Completed query generation for {len(arc_queries)} arcs")
            return arc_queries
            
        except Exception as e:
            raise LLMServiceError(
                f"Query generation failed: {e}",
                service_name="QueryGeneratorService.generate_queries_for_narrative_arcs",
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def _generate_queries_for_arc(self, plotline: Dict[str, Any], season_summary: str, plot_content: str) -> List[Dict[str, Any]]:
        """
        Generate vector database queries for a specific narrative arc.
        
        Args:
            plotline: Dictionary containing arc title, description, and narrative_arc_id
            season_summary: Full season summary content
            plot_content: Current episode plot content
            
        Returns:
            List of query dictionaries for this specific arc
        """
        try:
            # Create prompt for this specific arc
            arc_prompt = self._create_arc_query_prompt(plotline, season_summary, plot_content)
            
            # Get LLM response
            start_time = datetime.now()
            response = self.llm.invoke(arc_prompt)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse response using the existing utility
            queries_data = clean_llm_json_response(response.content)
            
            # Extract queries list from the response
            if isinstance(queries_data, list) and len(queries_data) > 0:
                # If it's a list, take the first item which should contain the queries
                if 'queries' in queries_data[0]:
                    queries = queries_data[0]['queries']
                else:
                    # If it's directly a list of queries
                    queries = queries_data
            else:
                queries = []
            
            # Validate and enrich each query
            validated_queries = []
            for i, query in enumerate(queries):
                if isinstance(query, dict) and 'query_text' in query and query['query_text'].strip():
                    # Ensure required fields with defaults
                    validated_query = {
                        'query_text': query['query_text'].strip(),
                        'purpose': query.get('purpose', ''),
                        'temporal_preference': query.get('temporal_preference', 'any'),
                        'expected_event_type': query.get('expected_event_type', 'general'),
                        'characters_involved': query.get('characters_involved', []),
                        'narrative_arc_id': plotline.get('narrative_arc_id'),
                        'arc_title': plotline.get('title', ''),
                        'query_rank': i + 1,
                        'processing_time_seconds': processing_time
                    }
                    validated_queries.append(validated_query)
            
            if not validated_queries:
                logger.warning(f"⚠️ No valid queries generated for arc '{plotline.get('title', '')}'")
                # Return a fallback query
                validated_queries = [{
                    'query_text': f"events related to {plotline.get('title', 'narrative arc')}",
                    'purpose': 'general context for narrative arc',
                    'temporal_preference': 'any',
                    'expected_event_type': 'general',
                    'characters_involved': [],
                    'narrative_arc_id': plotline.get('narrative_arc_id'),
                    'arc_title': plotline.get('title', ''),
                    'query_rank': 1,
                    'processing_time_seconds': processing_time
                }]
            
            logger.debug(f"🎯 Generated {len(validated_queries)} queries for arc in {processing_time:.2f}s")
            return validated_queries
            
        except Exception as e:
            logger.error(f"❌ Failed to generate queries for arc '{plotline.get('title', '')}': {e}")
            raise
    
    def _create_arc_query_prompt(self, plotline: Dict[str, Any], season_summary: str, plot_content: str) -> str:
        """Create prompt for generating queries for a specific narrative arc."""
        
        arc_title = plotline.get('title', 'Unknown Arc')
        arc_description = plotline.get('description', '')
        
        # Include season summary if available
        season_context = ""
        if season_summary:
            season_context = f"""
**Season Summary Context:**
{season_summary}

"""
        
        return f"""You are generating vector database search queries to find relevant historical events from previous episodes that provide context for a specific narrative arc in the current episode.

{season_context}**Current Episode Plot:**
{plot_content}

**Focus Narrative Arc:**
- **Title:** {arc_title}
- **Description:** {arc_description}

**Your Task:**
Generate 3-5 specific search queries that will find events from previous episodes that:
1. Show the development and evolution of this narrative arc
2. Provide background context for current developments in this arc
3. Help viewers understand character motivations within this arc
4. Reveal past conflicts, relationships, or decisions that impact this arc
5. Show how this arc has influenced or been influenced by other storylines

**Query Guidelines:**
- Use natural language that matches how events would be described in episode transcripts
- Focus on character interactions, conflicts, and developments specific to this arc
- Include emotional context and relationship dynamics
- Target both recent developments and foundational events
- Make queries specific enough to find relevant events but broad enough to capture related context
- Consider character names, locations, and specific conflicts mentioned in the arc description

**Output Format (JSON):**
```json
{{
    "queries": [
        {{
            "query_text": "specific search query text for finding relevant events",
            "purpose": "what specific context or background this query aims to provide",
            "temporal_preference": "recent/any/early_series",
            "expected_event_type": "relationship_development/conflict_origin/character_growth/plot_advancement",
            "characters_involved": ["character1", "character2"]
        }}
    ]
}}
```

Focus on generating queries that will help viewers understand the full context and development of the '{arc_title}' narrative arc."""
    
    def _load_plot_content(self) -> str:
        """Load current episode plot content."""
        plot_path = self.path_handler.get_plot_possible_speakers_path()
        
        if not plot_path or not os.path.exists(plot_path):
            raise MissingInputFilesError(
                [plot_path],
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
        
        try:
            with open(plot_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                raise MissingInputFilesError(
                    [f"Empty plot file: {plot_path}"],
                    series=self.path_handler.get_series(),
                    season=self.path_handler.get_season(), 
                    episode=self.path_handler.get_episode()
                )
            
            logger.debug(f"📖 Loaded plot content: {len(content)} characters")
            return content
            
        except Exception as e:
            raise MissingInputFilesError(
                [f"Failed to load plot file: {plot_path} - {e}"],
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def _load_running_plotlines(self) -> Dict[str, Any]:
        """Load running plotlines data and resolve narrative arc IDs."""
        plotlines_path = self.path_handler.get_present_running_plotlines_path()
        
        if not plotlines_path or not os.path.exists(plotlines_path):
            raise MissingInputFilesError(
                [plotlines_path],
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
        
        try:
            with open(plotlines_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Resolve narrative arc IDs from the database
            plotlines_with_ids = []
            
            if isinstance(data, list):
                # Direct list of plotlines
                for plotline in data:
                    if isinstance(plotline, dict):
                        title = plotline.get('title', '')
                        arc_id = self._get_narrative_arc_id_by_title(title)
                        plotline['narrative_arc_id'] = arc_id
                        plotlines_with_ids.append(plotline)
                
                result = {
                    'running_plotlines': plotlines_with_ids,
                    'total_arcs': len(plotlines_with_ids)
                }
            else:
                # Assume dictionary format
                plotlines = data.get('running_plotlines', [])
                for plotline in plotlines:
                    if isinstance(plotline, dict):
                        title = plotline.get('title', '')
                        arc_id = self._get_narrative_arc_id_by_title(title)
                        plotline['narrative_arc_id'] = arc_id
                        plotlines_with_ids.append(plotline)
                
                result = {
                    'running_plotlines': plotlines_with_ids,
                    'total_arcs': len(plotlines_with_ids)
                }
            
            logger.debug(f"📈 Loaded {len(plotlines_with_ids)} running plotlines with narrative arc IDs")
            return result
            
        except json.JSONDecodeError as e:
            raise MissingInputFilesError(
                [f"Invalid JSON in plotlines file: {plotlines_path} - {e}"],
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
        except Exception as e:
            raise MissingInputFilesError(
                [f"Failed to load plotlines file: {plotlines_path} - {e}"],
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def _get_narrative_arc_id_by_title(self, title: str) -> Optional[str]:
        """Get narrative arc ID from database by title."""
        try:
            import sqlite3
            
            db_path = "narrative_storage/narrative.db"
            if not os.path.exists(db_path):
                logger.warning(f"⚠️ Narrative database not found: {db_path}")
                return None
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM narrativearc WHERE title = ?", (title,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                arc_id = result[0]
                logger.debug(f"📍 Found arc ID for '{title}': {arc_id}")
                return arc_id
            else:
                logger.warning(f"⚠️ No arc ID found for title: '{title}'")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to lookup arc ID for '{title}': {e}")
            return None
    
    def _load_season_summary(self) -> str:
        """Load season summary content for broader context."""
        summary_path = self.path_handler.get_season_summary_path()
        
        if not summary_path or not os.path.exists(summary_path):
            logger.warning(f"⚠️ Season summary file not found: {summary_path}")
            return ""
        
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:
                logger.debug(f"📖 Loaded season summary ({len(content)} characters)")
                return content
            else:
                logger.warning(f"⚠️ Season summary file is empty: {summary_path}")
                return ""
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load season summary: {e}")
            return ""
