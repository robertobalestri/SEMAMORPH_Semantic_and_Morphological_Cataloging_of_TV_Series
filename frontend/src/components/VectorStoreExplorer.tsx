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
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  useDisclosure,
  IconButton,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter,
  TagLabel,
  TagCloseButton,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Link,
} from '@chakra-ui/react';
import { SearchIcon, CompareArrowsIcon, RepeatIcon, AddIcon } from '@chakra-ui/icons';
import { useState, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { PCA } from 'ml-pca';
import { Matrix } from 'ml-matrix';
import ArcFilters from './ArcFilters';
import { ARC_TYPES, ArcType } from '../types/ArcTypes';
import ArcMergeModal from './ArcMergeModal';

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
  onArcUpdated?: () => void;
}

interface CosineDistance {
  arc1: {
    id: string;
    title: string;
    type: string;
  };
  arc2: {
    id: string;
    title: string;
    type: string;
  };
  distance: number;
}

// Add interface for grouped entries
interface GroupedEntry {
  arc: VectorStoreEntry | null;
  progressions: VectorStoreEntry[];
}

// Add this interface for similar arcs
interface SimilarArcPair {
  arc1: VectorStoreEntry;
  arc2: VectorStoreEntry;
  distance: number;
}

// Add interface for clusters
interface ArcCluster {
  cluster_id: number;
  arcs: Array<{
    id: string;
    title: string;
    type: string;
    metadata: any;
    cluster_probability: number;
  }>;
  average_distance: number;
  size: number;
  average_probability: number;
  cluster_persistence?: number;  // Make it optional since it might not always be present
}

// Add this interface for NarrativeArc
interface ArcProgression {
  id: string;
  content: string;
  series: string;
  season: string;
  episode: string;
  ordinal_position: number;
  interfering_characters: string[];
}

interface NarrativeArc {
  id: string;
  title: string;
  description: string;
  arc_type: string;
  main_characters: string[];
  series: string;
  progressions: ArcProgression[];
}

// Add this conversion function
const convertToNarrativeArc = (entry: VectorStoreEntry): NarrativeArc => {
  return {
    id: entry.id,
    title: entry.metadata.title || '',
    description: entry.metadata.description || '',
    arc_type: entry.metadata.arc_type,
    main_characters: entry.metadata.main_characters ? entry.metadata.main_characters.split('; ') : [],
    series: entry.metadata.series,
    progressions: [] // Initialize with empty progressions
  };
};

const DEBUG = true;
const log = (message: string, data?: any) => {
  if (DEBUG) {
    if (data) {
      console.log(`[VectorStoreExplorer] ${message}:`, data);
    } else {
      console.log(`[VectorStoreExplorer] ${message}`);
    }
  }
};

const VectorStoreExplorer: React.FC<VectorStoreExplorerProps> = ({ series, onArcUpdated }) => {
  const [entries, setEntries] = useState<VectorStoreEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  // Filter states
  const [selectedSeason, setSelectedSeason] = useState('');
  const [selectedCharacters, setSelectedCharacters] = useState<string[]>([]);
  const [includeInterferingCharacters, setIncludeInterferingCharacters] = useState(false);
  const [selectedEpisodes, setSelectedEpisodes] = useState<string[]>([]);

  // First define episodes
  const episodes = useMemo(() => {
    log('Calculating episodes');
    const startTime = performance.now();
    
    const episodeSet = new Set<string>();
    const uniqueEpisodes: { season: string; episode: string; }[] = [];
    
    entries
      .filter(e => e.metadata?.season && e.metadata?.episode)
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

    const result = uniqueEpisodes.sort((a, b) => {
      const seasonA = parseInt(a.season.replace('S', ''));
      const seasonB = parseInt(b.season.replace('S', ''));
      if (seasonA !== seasonB) return seasonA - seasonB;
      
      const episodeA = parseInt(a.episode.replace('E', ''));
      const episodeB = parseInt(b.episode.replace('E', '') || '0');
      return episodeA - episodeB;
    });

    const endTime = performance.now();
    log(`Episodes calculation took ${endTime - startTime}ms`, result);
    return result;
  }, [entries]);

  // Then use episodes in seasons
  const seasons = useMemo(() => {
    // Get seasons from episodes
    const episodeSeasons = new Set(episodes.map(ep => ep.season));
    
    // Get seasons from entries, safely handling missing metadata
    const entriesSeasons = new Set(
      entries
        .filter(e => e.metadata?.season)  // Safely check for metadata
        .map(e => e.metadata.season!)
    );
    
    // Combine both sets and sort
    return [...new Set([...episodeSeasons, ...entriesSeasons])]
      .filter(Boolean)  // Remove any undefined/null values
      .sort((a, b) => {
        const numA = parseInt(a.replace(/\D/g, ''));
        const numB = parseInt(b.replace(/\D/g, ''));
        return numA - numB;
      });
  }, [episodes, entries]);

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

  // Add new states for arc comparison
  const [cosineDistances, setCosineDistances] = useState<CosineDistance[]>([]);
  const { isOpen: isCompareOpen, onOpen: onCompareOpen, onClose: onCompareClose } = useDisclosure();

  // Update filteredEntries to include main arcs filter
  const filteredEntries = useMemo(() => {
    log('Filtering entries');
    const startTime = performance.now();
    
    let filtered = entries;
    log('Initial entries count', filtered.length);

    // Arc type filter
    filtered = filtered.filter(entry => selectedArcTypes.includes(entry.metadata.arc_type as ArcType));
    log('After arc type filter', filtered.length);

    // Season filter
    if (selectedSeason) {
      filtered = filtered.filter(entry => 
        entry.metadata.season === selectedSeason
      );
      log('After season filter', filtered.length);
    }

    // Character filter
    if (selectedCharacters.length > 0) {
      filtered = filtered.filter(entry => {
        const mainChars = entry.metadata.main_characters?.split(', ') || [];
        const interferingChars = includeInterferingCharacters 
          ? (entry.metadata.interfering_characters?.split(', ') || [])
          : [];
        const allChars = [...mainChars, ...interferingChars];
        
        return selectedCharacters.some(char => allChars.includes(char));
      });
      log('After character filter', filtered.length);
    }

    // Episode filter
    if (selectedEpisodes.length > 0) {
      filtered = filtered.filter(entry => {
        if (!entry.metadata.season || !entry.metadata.episode) return false;
        const episodeKey = `${entry.metadata.season}-${entry.metadata.episode}`;
        return selectedEpisodes.includes(episodeKey);
      });
      log('After episode filter', filtered.length);
    }

    const endTime = performance.now();
    log(`Filtering took ${endTime - startTime}ms`);
    return filtered;
  }, [entries, selectedSeason, selectedCharacters, selectedEpisodes, includeInterferingCharacters, selectedArcTypes]);

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
    log('Fetching entries', query);
    const startTime = performance.now();
    
    try {
      setIsLoading(true);
      const url = `http://localhost:8000/api/vector-store/${series}${query ? `?query=${encodeURIComponent(query)}` : ''}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error('Failed to fetch vector store entries');
      }

      const data = await response.json();
      log('Received entries', data.length);
      setEntries(data);
      
      const endTime = performance.now();
      log(`Fetch took ${endTime - startTime}ms`);
      return data;
    } catch (error) {
      log('Error fetching entries', error);
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
    if (series && !entries.length) {
      fetchEntries().then(() => {
        if (seasons.length > 0) {
          setSelectedSeason(seasons[0]);
        }
      });
    }
  }, [series]);

  // Add cluster-specific states
  const [clusters, setClusters] = useState<ArcCluster[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<ArcCluster | null>(null);
  const { isOpen: isClusterOpen, onOpen: onClusterOpen, onClose: onClusterClose } = useDisclosure();

  // Remove the old comparison functions and add cluster-specific ones
  const handleViewCluster = (cluster: ArcCluster) => {
    setSelectedCluster(cluster);
    onClusterOpen();
  };

  // Replace ComparisonModal with ClusterModal
  const ClusterModal = () => (
    <Modal isOpen={isClusterOpen} onClose={onClusterClose} size="xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Arc Cluster Details</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          {selectedCluster && (
            <VStack spacing={4} align="stretch">
              <Box>
                <Text fontWeight="bold">Cluster Size: {selectedCluster.size} arcs</Text>
                <Text>Average Similarity: {((1 - selectedCluster.average_distance) * 100).toFixed(1)}%</Text>
                {selectedCluster.cluster_persistence && (
                  <Text>Cluster Stability: {(selectedCluster.cluster_persistence * 100).toFixed(1)}%</Text>
                )}
              </Box>
              <Table variant="simple">
                <Thead>
                  <Tr>
                    <Th>Arc Title</Th>
                    <Th>Type</Th>
                    <Th>Cluster Probability</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {selectedCluster.arcs.map((arc) => (
                    <Tr key={arc.id}>
                      <Td>{arc.title}</Td>
                      <Td>
                        <Badge colorScheme="blue">{arc.type}</Badge>
                      </Td>
                      <Td>{(arc.cluster_probability * 100).toFixed(1)}%</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
              <Button
                colorScheme="blue"
                onClick={() => {
                  // Navigate to merge view or open merge modal
                  // You can implement this based on your needs
                }}
              >
                Merge Cluster Arcs
              </Button>
            </VStack>
          )}
        </ModalBody>
      </ModalContent>
    </Modal>
  );

  // Add states for merging
  const [selectedArcsForMerge, setSelectedArcsForMerge] = useState<[NarrativeArc | null, NarrativeArc | null]>([null, null]);
  const [showMergeModal, setShowMergeModal] = useState(false);

  // Function to handle arc selection for merging
  const handleSelectArcForMerge = async (arc: VectorStoreEntry) => {
    log('Selecting arc for merge', arc.metadata.title);
    const startTime = performance.now();
    
    try {
      const arcId = arc.metadata.id;
      if (!arcId) {
        throw new Error('No arc ID found in metadata');
      }

      console.log('Starting arc fetch for ID:', arcId);
      
      const response = await fetch(`http://localhost:8000/api/arcs/by-id/${arcId}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch arc data: ${errorText}`);
      }
      
      const arcData = await response.json();
      console.log('Received arc data:', arcData);

      if (!arcData.id || !arcData.title || !arcData.progressions) {
        throw new Error('Received invalid arc data structure');
      }

      const narrativeArc: NarrativeArc = {
        id: arcData.id,
        title: arcData.title,
        description: arcData.description,
        arc_type: arcData.arc_type,
        main_characters: arcData.main_characters || [],
        series: arcData.series,
        progressions: arcData.progressions.map((prog: any) => ({
          id: prog.id,
          content: prog.content,
          series: prog.series,
          season: prog.season,
          episode: prog.episode,
          ordinal_position: prog.ordinal_position,
          interfering_characters: prog.interfering_characters || []
        }))
      };

      setSelectedArcsForMerge(prev => {
        if (!prev[0]) {
          setArcsToMerge([narrativeArc, null]);
          return [narrativeArc, null];
        }
        if (!prev[1]) {
          setArcsToMerge([prev[0], narrativeArc]);
          setShowMergeModal(true);
          return [prev[0], narrativeArc];
        }
        setArcsToMerge([narrativeArc, null]);
        return [narrativeArc, null];
      });

    } catch (error) {
      console.error('Error in handleSelectArcForMerge:', error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to fetch arc data",
        status: "error",
        duration: 3000,
      });
    }

    const endTime = performance.now();
    log(`Arc selection took ${endTime - startTime}ms`);
  };

  // Function to handle merge completion
  const handleMergeComplete = async () => {
    try {
      setShowMergeModal(false);
      setSelectedArcsForMerge([null, null]);
      setArcsToMerge([null, null]);
      
      // First refresh local data
      await fetchEntries();
      await fetchClusters();
      
      // Then notify parent component to update
      if (onArcUpdated) {
        onArcUpdated();
      }
      
      toast({
        title: "Success",
        description: "Arcs merged successfully",
        status: "success",
        duration: 3000,
      });
    } catch (error) {
      console.error('Error in handleMergeComplete:', error);
      toast({
        title: "Error",
        description: "Failed to refresh data after merge",
        status: "error",
        duration: 3000,
      });
    }
  };

  // Update the SimilarArcsAlert component
  const SimilarArcsAlert = () => {
    const fetchClusters = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`http://localhost:8000/api/vector-store/${series}/clusters`);
        if (!response.ok) throw new Error('Failed to fetch clusters');
        const data = await response.json();
        setClusters(data);
      } catch (error) {
        console.error('Error fetching clusters:', error);
        toast({
          title: "Error",
          description: "Failed to fetch similar arc clusters",
          status: "error",
          duration: 3000,
        });
      } finally {
        setIsLoading(false);
      }
    };

    useEffect(() => {
      if (entries.length > 0 && clusters.length === 0) {
        fetchClusters();
      }
    }, [entries]);

    if (isLoading) {
      return (
        <Alert status="info" variant="subtle" flexDirection="column" alignItems="start" p={4} borderRadius="md" mb={4}>
          <AlertIcon />
          <AlertTitle mb={2}>Analyzing Arc Similarities</AlertTitle>
          <AlertDescription>
            <HStack>
              <Spinner size="sm" />
              <Text>Finding similar arcs...</Text>
            </HStack>
          </AlertDescription>
        </Alert>
      );
    }

    if (clusters.length === 0) return null;

    return (
      <VStack spacing={4} align="stretch">
        {clusters.map((cluster) => (
          <Box key={cluster.cluster_id} p={4} borderWidth={1} borderRadius="md">
            <Text fontWeight="bold" mb={2}>
              Cluster {cluster.cluster_id} - Similarity: {((1 - cluster.average_distance) * 100).toFixed(1)}%
            </Text>
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>Arc Title</Th>
                  <Th>Type</Th>
                  <Th>Cluster Probability</Th>
                  <Th>Actions</Th>
                </Tr>
              </Thead>
              <Tbody>
                {cluster.arcs.map((arc) => {
                  const isSelected = selectedArcsForMerge.some(selected => selected?.id === arc.id);
                  return (
                    <Tr key={arc.id}>
                      <Td>{arc.title}</Td>
                      <Td>
                        <Badge colorScheme="blue">{arc.type}</Badge>
                      </Td>
                      <Td>{(arc.cluster_probability * 100).toFixed(1)}%</Td>
                      <Td>
                        <Button
                          size="sm"
                          colorScheme={isSelected ? "green" : "blue"}
                          onClick={() => handleSelectArcForMerge({
                            id: arc.id,
                            content: '',
                            metadata: {
                              id: arc.id,
                              title: arc.title,
                              arc_type: arc.type,
                              ...arc.metadata
                            }
                          })}
                          isDisabled={
                            selectedArcsForMerge[0] !== null &&
                            selectedArcsForMerge[1] !== null &&
                            !isSelected
                          }
                        >
                          {isSelected ? "Selected" : "Select for Merge"}
                        </Button>
                      </Td>
                    </Tr>
                  );
                })}
              </Tbody>
            </Table>
          </Box>
        ))}

        {selectedArcsForMerge[0] && selectedArcsForMerge[1] && (
          <Button
            colorScheme="green"
            onClick={handleOpenMergeModal}
            size="lg"
            w="full"
          >
            Merge Selected Arcs
          </Button>
        )}

        {showMergeModal && arcsToMerge[0] && arcsToMerge[1] && (
          <ArcMergeModal
            isOpen={showMergeModal}
            onClose={() => {
              setShowMergeModal(false);
              setSelectedArcsForMerge([null, null]);
              setArcsToMerge([null, null]);
            }}
            arc1={arcsToMerge[0]}
            arc2={arcsToMerge[1]}
            availableCharacters={allCharacters}
            onMergeComplete={() => {
              setShowMergeModal(false);
              setSelectedArcsForMerge([null, null]);
              setArcsToMerge([null, null]);
              fetchClusters();
            }}
          />
        )}
      </VStack>
    );
  };

  // Add this function inside the VectorStoreExplorer component
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

  // Add state for arc comparison
  const [arcsToCompare, setArcsToCompare] = useState<[VectorStoreEntry | null, VectorStoreEntry | null]>([null, null]);
  const [cosineDistance, setCosineDistance] = useState<number | null>(null);

  // Add function to handle arc comparison
  const handleCompareArc = async (arc: VectorStoreEntry) => {
    log('Comparing arc', arc.metadata.title);
    const startTime = performance.now();
    
    setArcsToCompare(prev => {
      if (!prev[0]) {
        return [arc, null];
      }
      if (!prev[1] && prev[0].id !== arc.id) {
        // Calculate cosine distance when second arc is selected
        calculateCosineDistance(prev[0], arc);
        onCompareOpen(); // Open the comparison modal
        return [prev[0], arc];
      }
      return [arc, null];
    });

    const endTime = performance.now();
    log(`Arc comparison took ${endTime - startTime}ms`);
  };

  // Function to calculate cosine distance
  const calculateCosineDistance = async (arc1: VectorStoreEntry, arc2: VectorStoreEntry) => {
    try {
      const response = await fetch('http://localhost:8000/api/vector-store/compare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify([arc1.id, arc2.id])
      });

      if (!response.ok) {
        throw new Error('Failed to calculate cosine distance');
      }

      const data = await response.json();
      setCosineDistance(data.distance);
    } catch (error) {
      console.error('Error calculating cosine distance:', error);
      toast({
        title: "Error",
        description: "Failed to calculate cosine distance",
        status: "error",
        duration: 3000,
      });
    }
  };

  // Update the ComparisonModal component
  const ComparisonModal = () => (
    <Modal 
      isOpen={isCompareOpen} 
      onClose={() => {
        onCompareClose();
        setArcsToCompare([null, null]);  // Reset selected arcs
        setCosineDistance(null);  // Reset distance
      }}
    >
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Arc Comparison</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          {arcsToCompare[0] && arcsToCompare[1] && (
            <VStack spacing={4} align="stretch">
              <Box>
                <Text fontWeight="bold">Arc 1:</Text>
                <Text>{arcsToCompare[0].metadata.title}</Text>
              </Box>
              <Box>
                <Text fontWeight="bold">Arc 2:</Text>
                <Text>{arcsToCompare[1].metadata.title}</Text>
              </Box>
              {cosineDistance !== null && (
                <Box>
                  <Text fontWeight="bold">Similarity:</Text>
                  <Text>{((1 - cosineDistance) * 100).toFixed(1)}%</Text>
                </Box>
              )}
            </VStack>
          )}
        </ModalBody>
        <ModalFooter>
          <Button onClick={() => {
            onCompareClose();
            setArcsToCompare([null, null]);  // Reset selected arcs
            setCosineDistance(null);  // Reset distance
          }}>
            Close
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );

  // Update the renderListView function
  const renderListView = () => {
    log('Rendering list view');
    const startTime = performance.now();

    const mainArcs = Object.entries(filteredEntries)
      .filter(([_, entry]) => entry.metadata.doc_type === 'main');
    log('Main arcs count', mainArcs.length);

    const result = (
      <Accordion allowMultiple>
        {mainArcs.map(([id, entry]) => {
          const arcStartTime = performance.now();
          const isSelectedForCompare = arcsToCompare.some(arc => arc?.id === entry.id);
          const isSelectedForMerge = selectedArcsForMerge.some(selected => selected?.id === entry.id);
          
          // Find progressions for this arc
          const arcProgressions = filteredEntries.filter(
            prog => 
              prog.metadata.doc_type === 'progression' && 
              prog.metadata.main_arc_id === entry.id
          ).sort((a, b) => {
            const seasonA = parseInt(a.metadata.season?.replace('S', '') || '0');
            const seasonB = parseInt(b.metadata.season?.replace('S', '') || '0');
            if (seasonA !== seasonB) return seasonA - seasonB;
            
            const episodeA = parseInt(a.metadata.episode?.replace('E', '') || '0');
            const episodeB = parseInt(b.metadata.episode?.replace('E', '') || '0');
            return episodeA - episodeB;
          });

          log(`Found ${arcProgressions.length} progressions for arc ${entry.metadata.title}`);

          const arcEndTime = performance.now();
          log(`Arc ${entry.metadata.title} rendering took ${arcEndTime - arcStartTime}ms`);

          return (
            <AccordionItem key={id}>
              <Box position="relative">
                <AccordionButton>
                  <Box flex="1" textAlign="left">
                    <HStack spacing={2}>
                      <Badge 
                        colorScheme={
                          entry.metadata.arc_type === 'Soap Arc' ? 'pink' :
                          entry.metadata.arc_type === 'Genre-Specific Arc' ? 'orange' :
                          entry.metadata.arc_type === 'Anthology Arc' ? 'green' :
                          'gray'
                        }
                      >
                        {entry.metadata.arc_type}
                      </Badge>
                      <Text fontWeight="bold">
                        {entry.metadata.title}
                      </Text>
                    </HStack>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
                <HStack 
                  position="absolute" 
                  right="40px" 
                  top="8px" 
                  zIndex={2} 
                  spacing={2}
                  onClick={(e) => e.stopPropagation()}
                >
                  {/* Compare Button */}
                  <IconButton
                    aria-label="Compare arc"
                    icon={<RepeatIcon />}
                    size="sm"
                    colorScheme={isSelectedForCompare ? "green" : "teal"}
                    onClick={() => handleCompareArc(entry)}
                    title={isSelectedForCompare ? "Selected for comparison" : "Select for comparison"}
                  />
                  {/* Merge Button */}
                  <IconButton
                    aria-label="Select for merge"
                    icon={<AddIcon />}
                    size="sm"
                    colorScheme={isSelectedForMerge ? "green" : "blue"}
                    onClick={() => handleSelectArcForMerge(entry)}
                  />
                </HStack>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={4}>
                    {/* Arc Details */}
                    <Box>
                      <Text fontWeight="bold">Description:</Text>
                      <Text>{entry.metadata.description}</Text>
                      <Text fontWeight="bold" mt={2}>Main Characters:</Text>
                      <Text>{entry.metadata.main_characters || 'None'}</Text>
                    </Box>

                    {/* Progressions */}
                    {arcProgressions.length > 0 && (
                      <Box>
                        <Text fontWeight="bold" mb={2}>Progressions:</Text>
                        <Accordion allowMultiple>
                          {arcProgressions.map(progression => (
                            <AccordionItem key={progression.id}>
                              <AccordionButton>
                                <Box flex="1" textAlign="left">
                                  <HStack spacing={2}>
                                    <Badge colorScheme="purple">
                                      S{progression.metadata.season?.replace('S', '')}-
                                      E{progression.metadata.episode?.replace('E', '')}
                                    </Badge>
                                    <Text noOfLines={1}>
                                      {progression.content}
                                    </Text>
                                  </HStack>
                                </Box>
                                <AccordionIcon />
                              </AccordionButton>
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
              </Box>
            </AccordionItem>
          );
        })}
      </Accordion>
    );

    const endTime = performance.now();
    log(`List view rendering took ${endTime - startTime}ms`);
    return result;
  };

  // Add a reset comparison function
  const resetComparison = () => {
    setArcsToCompare([null, null]);
    setCosineDistance(null);
  };

  // Add this to the JSX where you want to show the comparison status
  {arcsToCompare[0] && (
    <Alert status="info" mb={4}>
      <AlertIcon />
      <Box flex="1">
        <AlertTitle>Comparing Arcs</AlertTitle>
        <AlertDescription>
          {arcsToCompare[0]?.metadata.title}
          {arcsToCompare[1] && ` ↔ ${arcsToCompare[1]?.metadata.title}`}
          {cosineDistance !== null && (
            <Text mt={2}>
              Distance: {(cosineDistance * 100).toFixed(1)}%
            </Text>
          )}
        </AlertDescription>
      </Box>
      <Button size="sm" onClick={resetComparison}>
        Reset
      </Button>
    </Alert>
  )}

  // Add this after other state declarations
  const [visualizationData, setVisualizationData] = useState<any>(null);

  // Update the calculateVisualization function
  const calculateVisualization = (entries: VectorStoreEntry[]) => {
    try {
      const entriesWithEmbeddings = entries.filter(entry => entry.embedding);
      
      if (entriesWithEmbeddings.length < 3) {
        return null;
      }

      const embeddings = entriesWithEmbeddings.map(entry => entry.embedding!);
      
      const pca = new PCA(embeddings);
      const result = pca.predict(embeddings, { nComponents: 3 });
      const pcaResult = Array.from(result.to2DArray());

      return {
        type: 'scatter3d',
        mode: 'markers',
        x: pcaResult.map(p => p[0]),
        y: pcaResult.map(p => p[1]),
        z: pcaResult.map(p => p[2]),
        text: entriesWithEmbeddings.map(entry => 
          `Title: ${entry.metadata.title || entry.metadata.progression_title}<br>` +
          `Type: ${entry.metadata.arc_type}<br>` +
          `Doc Type: ${entry.metadata.doc_type}<br>` +
          `Characters: ${entry.metadata.main_characters || 'None'}`
        ),
        marker: {
          size: 8,
          color: entriesWithEmbeddings.map(entry => {
            const isProgression = entry.metadata.doc_type === 'progression';
            return getArcTypeColor(entry.metadata.arc_type, isProgression);
          }),
          opacity: 0.8
        },
        hoverinfo: 'text'
      };
    } catch (error) {
      console.error('Error calculating PCA:', error);
      return null;
    }
  };

  // Update useEffect to recalculate visualization when filters change
  useEffect(() => {
    const visData = calculateVisualization(filteredEntries);
    setVisualizationData(visData);
  }, [filteredEntries, showOnlyMainArcs]);

  // Add this function to fetch complete arc data
  const fetchCompleteArc = async (arcId: string): Promise<NarrativeArc | null> => {
    try {
      const response = await fetch(`http://localhost:8000/api/arcs/${arcId}`);
      if (!response.ok) throw new Error('Failed to fetch complete arc data');
      return await response.json();
    } catch (error) {
      console.error('Error fetching complete arc:', error);
      toast({
        title: "Error",
        description: "Failed to fetch complete arc data",
        status: "error",
        duration: 3000,
      });
      return null;
    }
  };

  // Update the handleOpenMergeModal function
  const handleOpenMergeModal = () => {
    if (!selectedArcsForMerge[0] || !selectedArcsForMerge[1]) {
      toast({
        title: "Error",
        description: "Please select two arcs to merge",
        status: "error",
        duration: 3000,
      });
      return;
    }

    setShowMergeModal(true);
  };

  // Add state for storing fetched arcs
  const [arcsToMerge, setArcsToMerge] = useState<[NarrativeArc | null, NarrativeArc | null]>([null, null]);

  return (
    <Box p={4}>
      <VStack spacing={4} align="stretch">
        <Tabs>
          <TabList>
            <Tab>Similar Clusters</Tab>
            <Tab>Vector Store Explorer</Tab>
          </TabList>

          <TabPanels>
            {/* Clusters Tab */}
            <TabPanel>
              <VStack spacing={4} align="stretch">
                <SimilarArcsAlert />
                <ClusterModal />
              </VStack>
            </TabPanel>

            {/* Vector Store Explorer Tab */}
            <TabPanel>
              <VStack spacing={4} align="stretch">
                {/* Search Bar */}
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

                {/* Comparison Alert */}
                {arcsToCompare[0] && (
                  <Alert status="info" mb={4}>
                    <AlertIcon />
                    <Box flex="1">
                      <AlertTitle>Comparing Arcs</AlertTitle>
                      <AlertDescription>
                        {arcsToCompare[0]?.metadata.title}
                        {arcsToCompare[1] && ` ↔ ${arcsToCompare[1]?.metadata.title}`}
                        {cosineDistance !== null && (
                          <Text mt={2}>
                            Distance: {(cosineDistance * 100).toFixed(1)}%
                          </Text>
                        )}
                      </AlertDescription>
                    </Box>
                    <Button size="sm" onClick={resetComparison}>
                      Reset
                    </Button>
                  </Alert>
                )}

                {/* Arc Type Filters */}
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

                {/* Main Arcs Filter */}
                <HStack>
                  <Checkbox
                    isChecked={showOnlyMainArcs}
                    onChange={(e) => setShowOnlyMainArcs(e.target.checked)}
                  >
                    <Text>Show Only Main Arcs</Text>
                  </Checkbox>
                </HStack>

                {/* List View */}
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

                {/* 3D Visualization */}
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
              </VStack>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </VStack>
      <ComparisonModal />
    </Box>
  );
};

export default VectorStoreExplorer;