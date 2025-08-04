export enum ArcType {
  SoapArc = 'Soap Arc',
  GenreSpecificArc = 'Genre-Specific Arc',
  AnthologyArc = 'Anthology Arc'
}

export interface Event {
  id: string;
  progression_id: string;
  content: string;
  series: string;
  season: string;
  episode: string;
  start_timestamp?: number;
  end_timestamp?: number;
  ordinal_position: number;
  confidence_score?: number;
  extraction_method?: string;
  characters_involved: string[];
}

export interface ArcProgression {
  id: string;
  content: string;
  series: string;
  season: string;
  episode: string;
  ordinal_position: number;
  interfering_characters: string[];
  events?: Event[];  // New: array of individual events
}

export interface NarrativeArc {
  id: string;
  title: string;
  description: string;
  arc_type: ArcType;
  main_characters: string[];
  series: string;
  progressions: ArcProgression[];
}

export interface ArcCluster {
  cluster_id: number;
  arcs: Array<{
    id: string;
    title: string;
    type: string;
    metadata: Record<string, unknown>;
    cluster_probability: number;
  }>;
  average_distance: number;
  size: number;
  average_probability: number;
  cluster_persistence?: number;
}

export interface ApiResponse<T> {
  data: T;
  error?: string;
}

export interface ProgressionMapping {
  season: string;
  episode: string;
  content: string;
  interfering_characters: string[];
  arc_id?: string;
  arc_title?: string;
  series?: string;
}

// Add a new type for creating arcs
export interface CreateArcData extends Omit<Partial<NarrativeArc>, 'progressions'> {
  progressions?: Omit<Partial<ArcProgression>, 'id'>[];
}

// Event-related interfaces
export interface EventCreateRequest {
  progression_id: string;
  content: string;
  series: string;
  season: string;
  episode: string;
  start_timestamp?: number;
  end_timestamp?: number;
  ordinal_position: number;
  confidence_score?: number;
  extraction_method?: string;
}

export interface EventUpdateRequest {
  content?: string;
  start_timestamp?: number;
  end_timestamp?: number;
  ordinal_position?: number;
  confidence_score?: number;
  extraction_method?: string;
}

export interface EventExtractionRequest {
  force_reextraction: boolean;
}

export interface EventExtractionResult {
  message: string;
  success: boolean;
  events_extracted: number;
  error_message?: string;
  validation_results?: {
    is_valid: boolean;
    issues: string[];
    suggestions: string[];
    overall_quality_score: number;
  };
  extracted_events?: Event[];
}

export interface EventStatistics {
  total_events: number;
  average_confidence: number;
  extraction_methods: Record<string, number>;
  total_duration: number;
} 