// Core domain types
import { ArcType } from './arc';

export { ArcType };
export type { ArcType as ArcTypeEnum } from './arc';
export type { Episode } from './episode';

// Export all types from arc.ts
export type {
  ArcProgression,
  NarrativeArc,
  ArcCluster,
  ProgressionMapping,
  ApiResponse,
} from './arc';

// Add Character and VectorStoreEntry types
export interface Character {
  entity_name: string;
  best_appellation: string;
  series: string;
  appellations: string[];
}

export interface VectorStoreEntry {
  id: string;
  content: string;
  metadata: Record<string, any>;
  embedding?: number[];
  distance?: number;
}