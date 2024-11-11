export const ARC_TYPES = {
  'Soap Arc': '#F687B3',          // pink
  'Genre-Specific Arc': '#ED8936', // orange
  'Anthology Arc': '#48BB78',      // green
} as const;

export type ArcType = keyof typeof ARC_TYPES;