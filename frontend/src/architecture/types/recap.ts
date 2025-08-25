export interface RecapEvent {
  id: string;
  content: string;
  series: string;
  season: string;
  episode: string;
  start_time: string;
  end_time: string;
  narrative_arc_id: string;
  arc_title: string;
  relevance_score: number;
  selected_subtitles: string[];
  debug_info: Record<string, any>;
}

export interface RecapClip {
  event_id: string;
  file_path: string;
  start_seconds: number;
  end_seconds: number;
  duration: number;
  subtitle_lines: string[];
  arc_title: string;
}

export interface RecapQuery {
  query_text: string;
  purpose: string;
  narrative_arc_id: string;
  arc_title: string;
}

export interface RecapGenerationRequest {
  series: string;
  season: string;
  episode: string;
}

export interface RecapGenerationResponse {
  video_path: string;
  events: RecapEvent[];
  clips: RecapClip[];
  total_duration: number;
  success: boolean;
  error_message?: string;
  queries: RecapQuery[];
  ranking_details: Record<string, RecapEvent[]>;
}

export interface RecapGenerationJob {
  job_id: string;
  series: string;
  season: string;
  episode: string;
  status: 'running' | 'completed' | 'failed';
  progress: number; // 0.0 to 1.0
  current_step: string;
  result?: RecapGenerationResponse;
  error_message?: string;
  created_at: string;
  updated_at: string;
}