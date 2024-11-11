import {
  Box,
  VStack,
  Text,
  Input,
  Button,
  useToast,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Badge,
  Spinner,
  HStack,
  Divider,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Checkbox,
  Tag,
  FormControl,
  FormLabel,
  SimpleGrid,
} from '@chakra-ui/react';
import { SearchIcon } from '@chakra-ui/icons';
import { useState, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { PCA } from 'ml-pca';
import { Matrix } from 'ml-matrix';
import ArcFilters from './ArcFilters';
import { ARC_TYPES, ArcType } from '../types/ArcTypes';

interface VectorStoreEntry {
  id: string;
  content: string;
  metadata: {
    progression_title?: string;
    title?: string;
    arc_type: string;
    description?: string;
    main_characters?: string;
    interfering_characters?: string;
    series: string;
    doc_type: string;
    season?: string;
    episode?: string;
    ordinal_position?: number;
    main_arc_id?: string;
    parent_arc_title?: string;
    id?: string;
  };
  embedding?: number[];
  distance?: number;
}

interface VectorStoreExplorerProps {
  series: string;
}

const VectorStoreExplorer: React.FC<VectorStoreExplorerProps> = ({ series }) => {
  const [entries, setEntries] = useState<VectorStoreEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  // Filter states
  const [selectedSeason, setSelectedSeason] = useState('');
  const [selectedCharacters, setSelectedCharacters] = useState<string[]>([]);
  const [includeInterferingCharacters, setIncludeInterferingCharacters] = useState(false);
  const [selectedEpisodes, setSelectedEpisodes] = useState<string[]>([]);

  // Derived state for available seasons and episodes
  const seasons = useMemo(() => {
    return [...new Set(entries
      .filter(e => e.metadata.season)
      .map(e => e.metadata.season!)
      .sort()
    )];
  }, [entries]);

  const episodes = useMemo(() => {
    const episodeSet = new Set<string>();
    const uniqueEpisodes: { season: string; episode: string; }[] = [];
    
    entries
      .filter(e => e.metadata.season && e.metadata.episode)
      .forEach(e => {
        const key = `${e.metadata.season}-${e.metadata.episode}`;
        if (!episodeSet.has(key)) {
          episodeSet.add(key);
          uniqueEpisodes.push({
            season: e.metadata.season!,
            episode: e.metadata.episode!
          });
        }
      });

    return uniqueEpisodes.sort((a, b) => {
      const seasonA = parseInt(a.season.replace('S', ''));
      const seasonB = parseInt(b.season.replace('S', ''));
      if (seasonA !== seasonB) return seasonA - seasonB;
      
      const episodeA = parseInt(a.episode.replace('E', ''));
      const episodeB = parseInt(b.episode.replace('E', '') || '0');
      return episodeA - episodeB;
    });
  }, [entries]);

  // Get all available characters
  const allCharacters = useMemo(() => {
    const characters = new Set<string>();
    entries.forEach(entry => {
      if (entry.metadata.main_characters) {
        entry.metadata.main_characters.split(', ').forEach(char => characters.add(char));
      }
      if (entry.metadata.interfering_characters) {
        entry.metadata.interfering_characters.split(', ').forEach(char => characters.add(char));
      }
    });
    return Array.from(characters).sort();
  }, [entries]);

  // Add new state for main arcs filter
  const [showOnlyMainArcs, setShowOnlyMainArcs] = useState(false);

  // Add arc type filter state
  const [selectedArcTypes, setSelectedArcTypes] = useState<ArcType[]>(Object.keys(ARC_TYPES) as ArcType[]);

  // Update filteredEntries to include main arcs filter
  const filteredEntries = useMemo(() => {
    return entries.filter(entry => {
      // Main arcs filter
      if (showOnlyMainArcs && entry.metadata.doc_type !== 'main') {
        return false;
      }

      // Season filter
      if (selectedSeason && entry.metadata.season) {
        if (entry.metadata.season !== selectedSeason) {
          return false;
        }
      }

      // Episode filter
      if (selectedEpisodes.length > 0) {
        if (entry.metadata.doc_type === 'progression') {
          // For progressions, check if they match the selected episode
          if (!entry.metadata.season || !entry.metadata.episode) return false;
          const episodeKey = `${entry.metadata.season}-${entry.metadata.episode}`;
          if (!selectedEpisodes.includes(episodeKey)) {
            return false;
          }
        } else if (entry.metadata.doc_type === 'main') {
          // For main arcs, check if they have any progressions in the selected episodes
          const arcId = entry.metadata.id;
          const hasProgressionInSelectedEpisode = entries.some(e => 
            e.metadata.doc_type === 'progression' &&
            e.metadata.main_arc_id === arcId &&
            e.metadata.season &&
            e.metadata.episode &&
            selectedEpisodes.includes(`${e.metadata.season}-${e.metadata.episode}`)
          );
          if (!hasProgressionInSelectedEpisode) {
            return false;
          }
        }
      }

      // Character filter
      if (selectedCharacters.length > 0) {
        const mainChars = entry.metadata.main_characters?.split(', ') || [];
        const interferingChars = includeInterferingCharacters 
          ? (entry.metadata.interfering_characters?.split(', ') || [])
          : [];
        const allChars = [...mainChars, ...interferingChars];
        
        if (!selectedCharacters.some(char => allChars.includes(char))) {
          return false;
        }
      }

      // Arc type filter
      if (!selectedArcTypes.includes(entry.metadata.arc_type as ArcType)) {
        return false;
      }

      return true;
    });
  }, [entries, selectedSeason, selectedEpisodes, selectedCharacters, includeInterferingCharacters, showOnlyMainArcs, selectedArcTypes]);

  const calculatePCA = (embeddings: number[][]) => {
    try {
      const pca = new PCA(embeddings);
      const result = pca.predict(embeddings, { nComponents: 3 });
      return Array.from(result.to2DArray());
    } catch (error) {
      console.error('Error calculating PCA:', error);
      return null;
    }
  };

  // Update the getArcTypeColor function
  const getArcTypeColor = (arcType: string, isProgression: boolean = false) => {
    const typeColors = {
      'Soap Arc': {
        main: '#F687B3',      // pink
        progression: '#FAB3D0'  // lighter pink
      },
      'Genre-Specific Arc': {
        main: '#ED8936',      // orange
        progression: '#F6AD6A'  // lighter orange
      },
      'Anthology Arc': {
        main: '#48BB78',      // green
        progression: '#7AD49B'  // lighter green
      },
      'default': {
        main: '#A0AEC0',      // gray
        progression: '#CBD5E0'  // lighter gray
      }
    };

    const colorSet = typeColors[arcType as keyof typeof typeColors] || typeColors.default;
    return isProgression ? colorSet.progression : colorSet.main;
  };

  const prepareVisualizationData = (filteredEntries: VectorStoreEntry[]) => {
    const embeddings = filteredEntries
      .filter(entry => entry.embedding)
      .map(entry => entry.embedding!);

    if (embeddings.length < 3) {
      return null;
    }

    const pcaResult = calculatePCA(embeddings);
    if (!pcaResult) {
      return null;
    }

    // Create a map of main arc types for progressions
    const mainArcTypes = new Map<string, string>();
    filteredEntries.forEach(entry => {
      if (entry.metadata.doc_type === 'main') {
        mainArcTypes.set(entry.metadata.id!, entry.metadata.arc_type);
      }
    });

    return {
      x: pcaResult.map(p => p[0]),
      y: pcaResult.map(p => p[1]),
      z: pcaResult.map(p => p[2]),
      type: 'scatter3d' as const,
      mode: 'markers' as const,
      marker: {
        size: 8,
        symbol: filteredEntries.map(entry => 
          entry.metadata.doc_type === 'main' ? 'circle' : 'square'
        ),
        color: filteredEntries.map(entry => {
          let arcType = entry.metadata.arc_type;
          if (entry.metadata.doc_type === 'progression') {
            arcType = mainArcTypes.get(entry.metadata.main_arc_id!) || arcType;
          }
          return getArcTypeColor(arcType, entry.metadata.doc_type === 'progression');
        }),
      },
      text: filteredEntries.map(entry => {
        if (entry.metadata.doc_type === 'main') {
          return `Title: ${entry.metadata.title}<br>Type: ${entry.metadata.arc_type}<br>Characters: ${entry.metadata.main_characters}`;
        } else {
          const mainArcType = mainArcTypes.get(entry.metadata.main_arc_id!);
          return `Progression for: ${entry.metadata.parent_arc_title}<br>Arc Type: ${mainArcType}<br>S${entry.metadata.season}E${entry.metadata.episode}<br>Characters: ${entry.metadata.interfering_characters}`;
        }
      }),
      hoverinfo: 'text' as const
    };
  };

  const fetchEntries = async (query?: string) => {
    try {
      setIsLoading(true);
      const url = `http://localhost:8000/api/vector-store/${series}${query ? `?query=${encodeURIComponent(query)}` : ''}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error('Failed to fetch vector store entries');
      }

      const data = await response.json();
      setEntries(data);
      return data;
    } catch (error) {
      toast({
        title: 'Error',
        description: String(error),
        status: 'error',
        duration: 5000,
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (series) {
      fetchEntries().then(() => {
        // Set initial season if available
        if (seasons.length > 0) {
          setSelectedSeason(seasons[0]);
        }
      });
    }
  }, [series]);

  // Update the renderListView function
  const renderListView = () => {
    // Group entries by arc
    const groupedEntries = filteredEntries.reduce((acc, entry) => {
      if (entry.metadata.doc_type === 'main') {
        // Create a new group for main arc
        acc[entry.metadata.id!] = {
          arc: entry,
          progressions: []
        };
      } else if (entry.metadata.doc_type === 'progression') {
        // Add progression to its arc group
        const arcId = entry.metadata.main_arc_id!;
        if (!acc[arcId]) {
          acc[arcId] = {
            arc: null,
            progressions: []
          };
        }
        acc[arcId].progressions.push(entry);
      }
      return acc;
    }, {} as Record<string, { arc: VectorStoreEntry | null, progressions: VectorStoreEntry[] }>);

    return (
      <Accordion allowMultiple>
        {Object.entries(groupedEntries)
          .filter(([_, group]) => group.arc !== null) // Only show groups with main arcs
          .map(([arcId, group]) => (
            <AccordionItem key={arcId}>
              <h2>
                <AccordionButton>
                  <Box flex="1" textAlign="left">
                    <HStack spacing={2}>
                      <Badge 
                        colorScheme={
                          group.arc?.metadata.arc_type === 'Soap Arc' ? 'pink' :
                          group.arc?.metadata.arc_type === 'Genre-Specific Arc' ? 'orange' :
                          group.arc?.metadata.arc_type === 'Anthology Arc' ? 'green' :
                          'gray'
                        }
                      >
                        {group.arc?.metadata.arc_type}
                      </Badge>
                      <Text fontWeight="bold">
                        {group.arc?.metadata.title}
                      </Text>
                      <Text color="gray.500" fontSize="sm">
                        MC: {group.arc?.metadata.main_characters || 'None'}
                      </Text>
                    </HStack>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
              </h2>
              <AccordionPanel pb={4}>
                <VStack align="stretch" spacing={4}>
                  {/* Arc Details */}
                  <Box>
                    <Text fontWeight="bold">Arc Details:</Text>
                    <VStack align="stretch" spacing={1} pl={4}>
                      <Text>Description: {group.arc?.content}</Text>
                      <Text>Main Characters: {group.arc?.metadata.main_characters || 'None'}</Text>
                    </VStack>
                  </Box>

                  {/* Progressions */}
                  {group.progressions.length > 0 && (
                    <Box>
                      <Text fontWeight="bold" mb={2}>Progressions:</Text>
                      <Accordion allowMultiple>
                        {group.progressions
                          .sort((a, b) => {
                            const seasonA = parseInt(a.metadata.season?.replace('S', '') || '0');
                            const seasonB = parseInt(b.metadata.season?.replace('S', '') || '0');
                            if (seasonA !== seasonB) return seasonA - seasonB;
                            
                            const episodeA = parseInt(a.metadata.episode?.replace('E', '') || '0');
                            const episodeB = parseInt(b.metadata.episode?.replace('E', '') || '0');
                            return episodeA - episodeB;
                          })
                          .map(progression => (
                            <AccordionItem key={progression.id}>
                              <h3>
                                <AccordionButton>
                                  <Box flex="1" textAlign="left">
                                    <HStack spacing={2}>
                                      <Badge colorScheme="green">Progression</Badge>
                                      <Text>
                                        S{progression.metadata.season?.replace('S', '')}-
                                        E{progression.metadata.episode?.replace('E', '')}
                                      </Text>
                                      <Tag
                                        size="sm"
                                        bg={`${getArcTypeColor(group.arc?.metadata.arc_type || '')}50`}
                                        color={getArcTypeColor(group.arc?.metadata.arc_type || '')}
                                      >
                                        {group.arc?.metadata.arc_type}
                                      </Tag>
                                    </HStack>
                                  </Box>
                                  <AccordionIcon />
                                </AccordionButton>
                              </h3>
                              <AccordionPanel pb={4}>
                                <VStack align="stretch" spacing={2}>
                                  <Text>Content: {progression.content}</Text>
                                  <Text>
                                    Interfering Characters: {progression.metadata.interfering_characters || 'None'}
                                  </Text>
                                </VStack>
                              </AccordionPanel>
                            </AccordionItem>
                          ))}
                      </Accordion>
                    </Box>
                  )}
                </VStack>
              </AccordionPanel>
            </AccordionItem>
          ))}
      </Accordion>
    );
  };

  // Add this after setting entries
  useEffect(() => {
    console.log('Vector Store Entries:', entries);
    console.log('Available Seasons:', seasons);
    console.log('Available Episodes:', episodes);
  }, [entries, seasons, episodes]);

  // Optimize PCA calculation by memoizing it
  const visualizationData = useMemo(() => {
    return prepareVisualizationData(filteredEntries);
  }, [filteredEntries]);

  // Add arc type filter component
  const renderArcTypeFilters = () => (
    <Box borderWidth={1} borderRadius="md" p={4}>
      <FormControl>
        <FormLabel fontWeight="bold">Arc Types</FormLabel>
        <SimpleGrid columns={3} spacing={2}>
          {(Object.keys(ARC_TYPES) as ArcType[]).map(arcType => (
            <Checkbox
              key={arcType}
              isChecked={selectedArcTypes.includes(arcType)}
              onChange={(e) => {
                if (e.target.checked) {
                  setSelectedArcTypes([...selectedArcTypes, arcType]);
                } else {
                  setSelectedArcTypes(selectedArcTypes.filter(t => t !== arcType));
                }
              }}
            >
              <HStack>
                <Box
                  w="3"
                  h="3"
                  borderRadius="full"
                  bg={ARC_TYPES[arcType]}
                />
                <Text fontSize="sm">{arcType}</Text>
              </HStack>
            </Checkbox>
          ))}
        </SimpleGrid>
      </FormControl>
    </Box>
  );

  return (
    <Box p={4}>
      <VStack spacing={4} align="stretch">
        {/* Search bar */}
        <HStack>
          <Input
            placeholder="Search vector store..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && fetchEntries(searchQuery)}
          />
          <Button
            leftIcon={<SearchIcon />}
            onClick={() => fetchEntries(searchQuery)}
            isLoading={isLoading}
          >
            Search
          </Button>
          <Button onClick={() => fetchEntries()} variant="outline">
            Reset
          </Button>
        </HStack>

        {/* Add Arc Type Filters */}
        {renderArcTypeFilters()}

        {/* Filters */}
        <ArcFilters
          seasons={seasons}
          selectedSeason={selectedSeason}
          onSeasonChange={setSelectedSeason}
          allCharacters={allCharacters}
          selectedCharacters={selectedCharacters}
          setSelectedCharacters={setSelectedCharacters}
          includeInterferingCharacters={includeInterferingCharacters}
          setIncludeInterferingCharacters={setIncludeInterferingCharacters}
          selectedEpisodes={selectedEpisodes}
          setSelectedEpisodes={setSelectedEpisodes}
          episodes={episodes}
        />

        {/* Add Main Arcs Filter */}
        <HStack>
          <Checkbox
            isChecked={showOnlyMainArcs}
            onChange={(e) => setShowOnlyMainArcs(e.target.checked)}
          >
            <Text>Show Only Main Arcs</Text>
          </Checkbox>
        </HStack>

        {/* Tabs for List/Visualization views */}
        <Tabs>
          <TabList>
            <Tab>List View</Tab>
            <Tab>3D Visualization</Tab>
          </TabList>

          <TabPanels>
            {/* List View Panel */}
            <TabPanel>
              {isLoading ? (
                <VStack justify="center" p={8}>
                  <Spinner />
                  <Text>Loading entries...</Text>
                </VStack>
              ) : filteredEntries.length > 0 ? (
                <Box maxH="600px" overflowY="auto">
                  {renderListView()}
                </Box>
              ) : (
                <Text textAlign="center" p={8}>No entries found</Text>
              )}
            </TabPanel>

            {/* Visualization Panel */}
            <TabPanel>
              <Box height="600px" borderWidth={1} borderRadius="lg">
                {isLoading ? (
                  <VStack justify="center" height="100%">
                    <Spinner />
                    <Text>Loading visualization...</Text>
                  </VStack>
                ) : (
                  <Plot
                    data={[visualizationData || { type: 'scatter3d', mode: 'markers', x: [], y: [], z: [] }]}
                    layout={{
                      title: '3D PCA of Narrative Arc Embeddings',
                      autosize: true,
                      scene: {
                        xaxis: { title: 'PCA Component 1' },
                        yaxis: { title: 'PCA Component 2' },
                        zaxis: { title: 'PCA Component 3' },
                        camera: {
                          eye: { x: 1.5, y: 1.5, z: 1.5 }
                        }
                      },
                      showlegend: true,
                      legend: {
                        x: 1,
                        y: 1,
                        title: { text: 'Arc Types' },
                        traceorder: 'normal'
                      },
                      margin: { l: 0, r: 0, t: 30, b: 0 }
                    }}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler
                    config={{
                      displayModeBar: true,
                      responsive: true,
                      scrollZoom: true,
                      displaylogo: false
                    }}
                  />
                )}
              </Box>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </VStack>
    </Box>
  );
};

export default VectorStoreExplorer;