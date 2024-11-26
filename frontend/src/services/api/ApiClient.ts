import type { 
  ApiResponse, 
  NarrativeArc, 
  Character, 
  VectorStoreEntry, 
  ArcProgression, 
  ArcCluster 
} from '@/architecture/types';

// Add interface for arc creation data
interface ArcCreateData extends Partial<NarrativeArc> {
  initial_progression?: {
    content: string;
    season: string;
    episode: string;
    interfering_characters: string[];
  };
}

interface GenerateProgressionResponse {
  content: string;
  interfering_characters: string[];
}

interface ApiProgressionResponse {
  data: GenerateProgressionResponse | null;
  error?: string;
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8000/api') {
    this.baseUrl = baseUrl;
  }

  public async request<T>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      return {
        data: null as any,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  // Arc endpoints
  async getArcs(series: string): Promise<ApiResponse<NarrativeArc[]>> {
    return this.request<NarrativeArc[]>(`/arcs/series/${series}`);
  }

  async getArcById(arcId: string): Promise<ApiResponse<NarrativeArc>> {
    return this.request<NarrativeArc>(`/arcs/${arcId}`);
  }

  async createArc(arcData: ArcCreateData): Promise<ApiResponse<NarrativeArc>> {
    const formattedData = {
      title: arcData.title,
      description: arcData.description,
      arc_type: arcData.arc_type,
      main_characters: arcData.main_characters,
      series: arcData.series,
      initial_progression: arcData.initial_progression && {
        content: arcData.initial_progression.content,
        season: arcData.initial_progression.season,
        episode: arcData.initial_progression.episode,
        interfering_characters: arcData.initial_progression.interfering_characters
      }
    };

    console.log('Creating arc with data:', formattedData);

    return this.request<NarrativeArc>('/arcs', {
      method: 'POST',
      body: JSON.stringify(formattedData),
    });
  }

  async updateArc(
    arcId: string,
    updateData: {
      title?: string;
      description?: string;
      arc_type?: string;
      main_characters?: string[] | string;
    }
  ): Promise<ApiResponse<NarrativeArc>> {
    const formattedData = {
      ...updateData,
      main_characters: typeof updateData.main_characters === 'string'
        ? (updateData.main_characters as string).split(';')
        : updateData.main_characters
    };

    return this.request<NarrativeArc>(
      `/arcs/${arcId}`,
      {
        method: 'PATCH',
        body: JSON.stringify(formattedData)
      }
    );
  }

  async deleteArc(arcId: string): Promise<ApiResponse<void>> {
    return this.request<void>(`/arcs/${arcId}`, {
      method: 'DELETE',
    });
  }

  async mergeArcs(
    arc1Id: string,
    arc2Id: string,
    mergedData: Partial<NarrativeArc>
  ): Promise<ApiResponse<NarrativeArc>> {
    return this.request<NarrativeArc>('/arcs/merge', {
      method: 'POST',
      body: JSON.stringify({
        arc_id_1: arc1Id,
        arc_id_2: arc2Id,
        ...mergedData,
      }),
    });
  }

  // Character endpoints
  async getCharacters(series: string): Promise<ApiResponse<Character[]>> {
    return this.request<Character[]>(`/characters/${series}`);
  }

  async createCharacter(series: string, characterData: Partial<Character>): Promise<ApiResponse<Character>> {
    return this.request<Character>(`/characters/${series}`, {
      method: 'POST',
      body: JSON.stringify(characterData),
    });
  }

  async updateCharacter(series: string, characterData: Partial<Character>): Promise<ApiResponse<Character>> {
    return this.request<Character>(`/characters/${series}`, {
      method: 'PATCH',
      body: JSON.stringify(characterData),
    });
  }

  async deleteCharacter(series: string, entityName: string): Promise<ApiResponse<void>> {
    return this.request<void>(`/characters/${series}/${entityName}`, {
      method: 'DELETE',
    });
  }

  async mergeCharacters(
    series: string,
    data: {
      character1_id: string;
      character2_id: string;
      keep_character: 'character1' | 'character2';
    }
  ): Promise<ApiResponse<void>> {
    return this.request<void>(`/characters/${series}/merge`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Vector store endpoints
  async searchVectorStore(
    series: string, 
    query?: string
  ): Promise<ApiResponse<VectorStoreEntry[]>> {
    const endpoint = query 
      ? `/vector-store/${series}?query=${encodeURIComponent(query)}`
      : `/vector-store/${series}`;
    return this.request<VectorStoreEntry[]>(endpoint);
  }

  async getArcClusters(
    series: string,
    params: {
      threshold?: number;
      min_cluster_size?: number;
      max_clusters?: number;
    } = {}
  ): Promise<ApiResponse<ArcCluster[]>> {
    const queryParams = new URLSearchParams();
    if (params.threshold) queryParams.append('threshold', params.threshold.toString());
    if (params.min_cluster_size) queryParams.append('min_cluster_size', params.min_cluster_size.toString());
    if (params.max_clusters) queryParams.append('max_clusters', params.max_clusters.toString());

    const endpoint = `/vector-store/${series}/clusters${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    return this.request<ArcCluster[]>(endpoint);
  }

  async compareArcs(arcIds: string[]): Promise<ApiResponse<{ distance: number }>> {
    return this.request<{ distance: number }>('/vector-store/compare', {
      method: 'POST',
      body: JSON.stringify(arcIds),
    });
  }

  async updateProgression(progressionId: string, data: Partial<ArcProgression>): Promise<ApiResponse<ArcProgression>> {
    const formattedData = {
      content: data.content,
      interfering_characters: data.interfering_characters || []
    };

    return this.request<ArcProgression>(`/progressions/${progressionId}`, {
      method: 'PATCH',
      body: JSON.stringify(formattedData),
    });
  }

  async createProgression(data: {
    arc_id: string;
    content: string;
    series: string;
    season: string;
    episode: string;
    interfering_characters: string[];
  }): Promise<ApiResponse<ArcProgression>> {
    return this.request<ArcProgression>('/progressions', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async deleteProgression(progressionId: string): Promise<ApiResponse<void>> {
    return this.request<void>(
      `/progressions/${progressionId}`,
      {
        method: 'DELETE'
      }
    );
  }

  async generateProgression(
    arcId: string | null,
    series: string,
    season: string,
    episode: string,
    title?: string,
    description?: string
  ): Promise<ApiProgressionResponse> {
    const response = await this.request<GenerateProgressionResponse>(
      `/progressions/generate?series=${series}&season=${season}&episode=${episode}`,
      {
        method: 'POST',
        body: JSON.stringify({
          arc_id: arcId,
          arc_title: title || null,
          arc_description: description || null,
          delete_existing: true
        }),
      }
    );

    // Check for NO_PROGRESSION
    if (response.data?.content === "NO_PROGRESSION") {
      return {
        data: null,
        error: "No progression found for this arc in this episode"
      };
    }

    // Check if we have valid content
    if (!response.data?.content) {
      return {
        data: null,
        error: "Failed to generate progression content"
      };
    }

    return { data: response.data };
  }
} 