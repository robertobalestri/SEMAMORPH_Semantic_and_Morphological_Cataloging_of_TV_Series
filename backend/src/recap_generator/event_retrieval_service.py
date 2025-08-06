"""
Event Retrieval Service for recap generation.

This service executes vector database queries to find relevant historical events,
ranks and filters them, and selects the optimal set for recap inclusion.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..ai_models.ai_models import get_llm, LLMType  
from ..utils.logger_utils import setup_logging
from .exceptions.recap_exceptions import LLMServiceError, VectorSearchError
from .models.event_models import (
    VectorEvent, EventRanking, QueryResult, EventSelectionResult
)
from ..narrative_storage_management.vector_store_service import VectorStoreService
from ..path_handler import PathHandler

logger = setup_logging(__name__)


class EventRetrievalService:
    """Service for retrieving and ranking historical events for recap generation."""
    
    def __init__(self, path_handler: PathHandler, vector_store_service: VectorStoreService = None):
        self.path_handler = path_handler
        self.vector_store = vector_store_service or VectorStoreService()
        self.llm = get_llm(LLMType.INTELLIGENT)
        
    def search_vector_database(self, queries: List[Dict[str, Any]]) -> List[QueryResult]:
        """
        Execute vector database queries to find relevant events.
        
        Args:
            queries: List of query dictionaries with text, weights, and metadata
            
        Returns:
            List of QueryResult objects with search results
        """
        try:
            logger.info(f"🔍 Executing {len(queries)} vector database queries")
            
            query_results = []
            total_search_time = 0
            
            series = self.path_handler.get_series()
            
            for i, query_data in enumerate(queries):
                query_text = query_data.get('query_text', query_data.get('query', ''))
                arc_weight = query_data.get('weight', 0.5)
                target_arc = query_data.get('target_arc')
                narrative_arc_id = query_data.get('narrative_arc_id')  # NEW: Get arc ID from query
                
                logger.debug(f"🎯 Query {i+1}: {query_text[:50]}... (weight: {arc_weight:.2f}, arc_id: {narrative_arc_id})")
                
                start_time = datetime.now()
                
                try:
                    # Get narrative arc IDs for filtering
                    arc_ids = None
                    if narrative_arc_id:
                        arc_ids = [narrative_arc_id]  # Filter by specific narrative arc
                        logger.debug(f"🔍 Filtering by narrative arc ID: {narrative_arc_id}")
                    
                    # Execute similarity search for events with arc filtering
                    # IMPORTANT: Recap should only contain previous episodes, not the current one
                    current_episode = self.path_handler.get_episode()
                    current_season = self.path_handler.get_season()
                    
                    search_results = self.vector_store.find_similar_events(
                        query=query_text,
                        n_results=20,  # Get more results for better filtering
                        series=series,
                        min_confidence=0.3,  # Filter out low-confidence events
                        narrative_arc_ids=arc_ids,  # Filter by narrative arc
                        exclude_episodes=[(current_season, current_episode)]  # EXCLUDE current episode at vector store level
                    )
                    
                    search_time = (datetime.now() - start_time).total_seconds() * 1000
                    total_search_time += search_time
                    
                    # Convert search results to VectorEvent objects
                    vector_events = []
                    for result in search_results:
                        try:
                            event = self._create_vector_event_from_search_result(result)
                            if event:
                                vector_events.append(event)
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to parse search result: {e}")
                            continue
                    
                    query_result = QueryResult(
                        query=query_text,
                        narrative_arc_id=narrative_arc_id,  # Use the actual arc ID
                        arc_weight=arc_weight,
                        events=vector_events,
                        total_results=len(search_results),
                        search_time_ms=search_time
                    )
                    
                    query_results.append(query_result)
                    
                    logger.debug(f"✅ Query {i+1} found {len(vector_events)} valid events in {search_time:.0f}ms")
                    
                except Exception as e:
                    logger.error(f"❌ Query {i+1} failed: {e}")
                    # Add empty result to maintain query order
                    query_result = QueryResult(
                        query=query_text,
                        narrative_arc_id=target_arc,
                        arc_weight=arc_weight,
                        events=[],
                        total_results=0,
                        search_time_ms=0
                    )
                    query_results.append(query_result)
            
            logger.info(f"✅ Vector search completed: {len(query_results)} queries in {total_search_time:.0f}ms total")
            
            return query_results
            
        except Exception as e:
            raise VectorSearchError(
                f"Vector database search failed: {e}",
                collection_name="narrative_arcs",
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode(),
                original_error=e
            )
    
    def rank_events_by_relevance(self, 
                                query_results: List[QueryResult],
                                max_events: int = 15) -> List[EventRanking]:
        """
        Rank all found events by relevance and importance for recap inclusion.
        
        Args:
            query_results: Results from vector database queries
            max_events: Maximum number of events to rank
            
        Returns:
            List of EventRanking objects sorted by final score
        """
        try:
            logger.info("📊 Ranking events by relevance")
            
            # Collect all unique events from query results
            all_events = {}  # event_id -> (event, best_query_score)
            
            for query_result in query_results:
                for event in query_result.events:
                    event_id = event.id
                    
                    # Calculate query-specific score
                    query_score = self._calculate_query_score(event, query_result)
                    
                    # Keep the best score for each event
                    if event_id not in all_events or query_score > all_events[event_id][1]:
                        all_events[event_id] = (event, query_score)
            
            logger.info(f"📈 Found {len(all_events)} unique events across all queries")
            
            # Limit to prevent excessive processing
            if len(all_events) > max_events * 3:
                # Keep top events by query score
                sorted_events = sorted(all_events.items(), key=lambda x: x[1][1], reverse=True)
                all_events = dict(sorted_events[:max_events * 3])
                logger.info(f"🔪 Limited to top {len(all_events)} events for ranking")
            
            # Generate comprehensive rankings using LLM
            event_rankings = self._generate_comprehensive_rankings(
                list(all_events.values()),
                query_results
            )
            
            # Sort by final score
            event_rankings.sort(key=lambda x: x.final_score, reverse=True)
            
            # Assign ranking positions
            for i, ranking in enumerate(event_rankings):
                ranking.ranking_position = i + 1
            
            logger.info(f"✅ Event ranking completed: {len(event_rankings)} events ranked")
            logger.info(f"📊 Top 5 scores: {[f'{r.final_score:.3f}' for r in event_rankings[:5]]}")
            
            return event_rankings
            
        except Exception as e:
            raise LLMServiceError(
                f"Event ranking failed: {e}",
                service_name="EventRetrievalService.rank_events_by_relevance",
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def select_final_events(self, 
                           event_rankings: List[EventRanking],
                           target_count: int = 8,
                           min_score: float = 0.6) -> EventSelectionResult:
        """
        Select the final set of events for recap inclusion with arc balancing.
        
        Args:
            event_rankings: Ranked events from rank_events_by_relevance
            target_count: Target number of events to select
            min_score: Minimum score threshold for inclusion
            
        Returns:
            EventSelectionResult with selected events and metadata
        """
        try:
            logger.info(f"🎯 Selecting final events: target={target_count}, min_score={min_score}")
            
            start_time = datetime.now()
            
            # Filter by minimum score
            qualifying_events = [
                ranking for ranking in event_rankings 
                if ranking.final_score >= min_score
            ]
            
            logger.info(f"📋 {len(qualifying_events)}/{len(event_rankings)} events qualify (score >= {min_score})")
            
            if len(qualifying_events) == 0:
                logger.warning("⚠️ No events meet minimum score threshold, relaxing criteria")
                # Relax threshold if no events qualify
                min_score = max(0.3, min_score - 0.2)
                qualifying_events = [
                    ranking for ranking in event_rankings 
                    if ranking.final_score >= min_score
                ][:target_count * 2]  # Limit fallback candidates
            
            # Apply arc balancing to select diverse events
            selected_rankings = self._balance_arc_distribution(
                qualifying_events,
                target_count
            )
            
            # Extract selected events
            selected_events = [ranking.event for ranking in selected_rankings]
            
            # Calculate arc distribution
            arc_distribution = {}
            for ranking in selected_rankings:
                arc_title = ranking.event.arc_title
                arc_distribution[arc_title] = arc_distribution.get(arc_title, 0) + 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = EventSelectionResult(
                query_results=[],  # Will be set by caller
                ranked_events=event_rankings,
                selected_events=selected_events,
                selection_criteria={
                    'target_count': target_count,
                    'min_score': min_score,
                    'actual_min_score': min(ranking.final_score for ranking in selected_rankings) if selected_rankings else 0,
                    'arc_balancing_enabled': True
                },
                total_events_considered=len(event_rankings),
                selection_time_seconds=processing_time,
                arc_distribution=arc_distribution
            )
            
            logger.info(f"✅ Event selection completed: {len(selected_events)}/{target_count} events selected")
            logger.info(f"📊 Arc distribution: {arc_distribution}")
            logger.info(f"⏱️ Selection took {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            raise LLMServiceError(
                f"Event selection failed: {e}",
                service_name="EventRetrievalService.select_final_events",
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def _create_vector_event_from_search_result(self, result: Dict[str, Any]) -> Optional[VectorEvent]:
        """Create VectorEvent object from vector search result."""
        try:
            metadata = result.get('metadata', {})
            
            # Required fields validation
            required_fields = ['id', 'series', 'season', 'episode']
            for field in required_fields:
                if field not in metadata:
                    logger.warning(f"⚠️ Missing required field '{field}' in search result")
                    return None
            
            return VectorEvent(
                id=metadata['id'],
                content=result.get('page_content', ''),
                series=metadata['series'],
                season=metadata['season'],
                episode=metadata['episode'],
                start_timestamp=metadata.get('start_timestamp'),
                end_timestamp=metadata.get('end_timestamp'),
                cosine_distance=result.get('cosine_distance', 1.0),
                narrative_arc_id=metadata.get('main_arc_id', ''),
                arc_title=metadata.get('arc_title', ''),
                arc_type=metadata.get('arc_type', ''),
                main_characters=metadata.get('main_characters', []),
                ordinal_position=metadata.get('ordinal_position', 1),
                confidence_score=metadata.get('confidence_score'),
                extraction_method=metadata.get('extraction_method')
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create VectorEvent: {e}")
            return None
    
    def _calculate_query_score(self, event: VectorEvent, query_result: QueryResult) -> float:
        """Calculate score for an event within a specific query context."""
        # Base similarity score
        score = event.similarity_score
        
        # Weight by query importance
        score *= query_result.arc_weight
        
        # Bonus for timestamp availability
        if event.has_valid_timestamps:
            score += 0.1
        
        # Bonus for high confidence extraction
        if event.confidence_score and event.confidence_score > 0.8:
            score += 0.05
        
        return min(1.0, score)
    
    def _generate_comprehensive_rankings(self, 
                                       events_with_scores: List[Tuple[VectorEvent, float]],
                                       query_results: List[QueryResult]) -> List[EventRanking]:
        """Generate comprehensive rankings using LLM analysis."""
        
        event_rankings = []
        
        # For now, create rankings based on available data
        # In a more sophisticated implementation, this would use LLM to assess
        # narrative importance, character significance, etc.
        
        for event, query_score in events_with_scores:
            ranking = EventRanking(
                event=event,
                relevance_score=query_score,
                importance_score=self._assess_importance(event),
                character_significance=self._assess_character_significance(event),
                arc_priority=self._assess_arc_priority(event),
                temporal_relevance=self._assess_temporal_relevance(event),
                dialogue_quality=self._assess_dialogue_quality(event),
                final_score=0.0,  # Will be calculated
                selection_reasons=[],
                exclusion_reasons=[]
            )
            
            # Calculate final score as weighted average
            ranking.final_score = (
                ranking.relevance_score * 0.3 +
                ranking.importance_score * 0.2 +
                ranking.character_significance * 0.2 +
                ranking.arc_priority * 0.15 +
                ranking.temporal_relevance * 0.1 +
                ranking.dialogue_quality * 0.05
            )
            
            # Generate selection reasons
            if ranking.final_score >= 0.8:
                ranking.selection_reasons.append("High overall relevance")
            if ranking.relevance_score >= 0.9:
                ranking.selection_reasons.append("Strong query match")
            if ranking.character_significance >= 0.8:
                ranking.selection_reasons.append("Involves important characters")
            if event.has_valid_timestamps:
                ranking.selection_reasons.append("Has precise timestamps")
            
            event_rankings.append(ranking)
        
        return event_rankings
    
    def _assess_importance(self, event: VectorEvent) -> float:
        """Assess the overall importance of an event."""
        score = 0.5  # Base score
        
        # Higher ordinal position suggests more important plot point
        if event.ordinal_position == 1:
            score += 0.2  # First event in arc progression
        
        # Longer content suggests more detailed/important event
        content_length = len(event.content)
        if content_length > 200:
            score += 0.1
        elif content_length < 50:
            score -= 0.1
        
        # Arc type importance
        if event.arc_type in ['Soap Arc', 'Genre-Specific Arc']:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _assess_character_significance(self, event: VectorEvent) -> float:
        """Assess the significance of characters involved."""
        if not event.main_characters:
            return 0.3
        
        # More characters = potentially more significant
        char_count = len(event.main_characters)
        if char_count >= 3:
            return 0.8
        elif char_count == 2:
            return 0.7
        else:
            return 0.6
    
    def _assess_arc_priority(self, event: VectorEvent) -> float:
        """Assess the priority of the narrative arc."""
        # Soap arcs are often important for character development
        if event.arc_type == 'Soap Arc':
            return 0.8
        elif event.arc_type == 'Genre-Specific Arc':
            return 0.7
        elif event.arc_type == 'Anthology Arc':
            return 0.4  # Less likely to need context
        else:
            return 0.5
    
    def _assess_temporal_relevance(self, event: VectorEvent) -> float:
        """Assess how temporally relevant the event is."""
        # For now, assume all events are equally relevant
        # In practice, this would consider episode distance from current episode
        return 0.6
    
    def _assess_dialogue_quality(self, event: VectorEvent) -> float:
        """Assess the quality of dialogue/content in the event."""
        if not event.content:
            return 0.2
        
        # Simple heuristics for dialogue quality
        content = event.content.lower()
        
        # Look for dialogue indicators
        has_quotes = '"' in event.content or "'" in event.content
        has_questions = '?' in event.content
        has_emotional_words = any(word in content for word in [
            'love', 'hate', 'angry', 'sad', 'happy', 'afraid', 'worried'
        ])
        
        score = 0.5
        if has_quotes:
            score += 0.2
        if has_questions:
            score += 0.1
        if has_emotional_words:
            score += 0.1
        
        return min(1.0, score)
    
    def _balance_arc_distribution(self, 
                                 qualified_events: List[EventRanking],
                                 target_count: int) -> List[EventRanking]:
        """Balance event selection across different narrative arcs."""
        
        if len(qualified_events) <= target_count:
            return qualified_events
        
        # Group events by arc
        arc_groups = {}
        for ranking in qualified_events:
            arc_title = ranking.event.arc_title
            if arc_title not in arc_groups:
                arc_groups[arc_title] = []
            arc_groups[arc_title].append(ranking)
        
        # Sort arcs by their best event score
        sorted_arcs = sorted(
            arc_groups.items(),
            key=lambda x: max(r.final_score for r in x[1]),
            reverse=True
        )
        
        selected = []
        events_per_arc = max(1, target_count // len(arc_groups))
        remaining_slots = target_count
        
        # First pass: take best events from each arc
        for arc_title, arc_events in sorted_arcs:
            if remaining_slots <= 0:
                break
            
            arc_events.sort(key=lambda x: x.final_score, reverse=True)
            take_count = min(events_per_arc, len(arc_events), remaining_slots)
            
            selected.extend(arc_events[:take_count])
            remaining_slots -= take_count
        
        # Second pass: fill remaining slots with highest-scoring events
        if remaining_slots > 0:
            remaining_events = [
                ranking for ranking in qualified_events
                if ranking not in selected
            ]
            remaining_events.sort(key=lambda x: x.final_score, reverse=True)
            selected.extend(remaining_events[:remaining_slots])
        
        # Final sort by score
        selected.sort(key=lambda x: x.final_score, reverse=True)
        
        return selected[:target_count]
