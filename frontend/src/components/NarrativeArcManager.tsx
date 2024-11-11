import {
  Box,
  Grid,
  GridItem,
  Text,
  VStack,
  HStack,
  Button,
  Checkbox,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Textarea,
  Tag,
  Tooltip,
  useColorModeValue,
  Select as ChakraSelect,
  Stack,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  FormControl,
  FormLabel,
  Switch,
  Input,
  useColorMode,
  useTheme,
  Wrap,
  WrapItem,
  SimpleGrid,
} from '@chakra-ui/react';
import { AddIcon, DeleteIcon, RepeatIcon } from '@chakra-ui/icons';
import { useState, useMemo, useEffect, useCallback } from 'react';
import ArcMergeModal from './ArcMergeModal';
import ArcProgressionEditModal from './ArcProgressionEditModal';
import NewArcModal from './NewArcModal';
import ArcFilters from './ArcFilters';
import { ARC_TYPES, ArcType } from '../types/ArcTypes';

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

interface NarrativeArcManagerProps {
  arcs: NarrativeArc[];
  episodes: { season: string; episode: string; }[];
  onArcUpdated: () => void;
  selectedSeason?: string;
  series: string;
}

// 1. Timeline component
const ArcTimeline: React.FC<{
  arcs: NarrativeArc[];
  episodes: { season: string; episode: string; }[];
  selectedSeason: string;
  onCellClick: (arc: NarrativeArc, season: string, episode: string) => void;
  isMergeMode: boolean;
  selectedForMerge: NarrativeArc[];
  onToggleMerge: (arc: NarrativeArc) => void;
  onEditArc: (arc: NarrativeArc) => void;
}> = ({ arcs, episodes, selectedSeason, onCellClick, isMergeMode, selectedForMerge, onToggleMerge, onEditArc }) => {
  // 1. All Chakra UI hooks first
  const bgColor = useColorModeValue('gray.50', 'gray.700');
  const cellBgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // 2. All useMemo hooks
  const seasonEpisodes = useMemo(() => {
    return episodes
      .filter(ep => ep.season === selectedSeason)
      .sort((a, b) => parseInt(a.episode.replace('E', '')) - parseInt(b.episode.replace('E', '')));
  }, [episodes, selectedSeason]);

  const getArcTypeColor = useCallback((arcType: string) => {
    const typeColors = {
      'Soap Arc': '#F687B3',      // pink
      'Genre-Specific Arc': '#ED8936', // orange
      'Episodic Arc': '#48BB78',     // green
    };
    return typeColors[arcType as keyof typeof typeColors] || '#A0AEC0';
  }, []);

  return (
    <Box overflowX="auto" mt={4}>
      <Box minWidth="fit-content" p={4}>
        <Grid
          templateColumns={`250px repeat(${seasonEpisodes.length}, 200px)`}
          gap={2}
        >
          {/* Header */}
          <GridItem p={2} bg={bgColor}>
            <Text fontWeight="bold">Narrative Arcs</Text>
          </GridItem>
          {seasonEpisodes.map(ep => (
            <GridItem
              key={ep.episode}
              p={2}
              bg={bgColor}
              textAlign="center"
            >
              <Text fontWeight="bold">Episode {ep.episode.replace('E', '')}</Text>
            </GridItem>
          ))}

          {/* Arc rows */}
          {arcs.map(arc => (
            <Box key={arc.id} display="contents">
              {/* Arc info */}
              <GridItem p={2} borderWidth={1} borderRadius="md" bg={cellBgColor} borderColor={borderColor}>
                <HStack alignItems="flex-start">
                  {isMergeMode && (
                    <Checkbox
                      isChecked={selectedForMerge.some(a => a.id === arc.id)}
                      onChange={() => onToggleMerge(arc)}
                    />
                  )}
                  <VStack align="start" spacing={1} width="100%">
                    <HStack width="100%" justify="space-between">
                      <Tooltip label={arc.description} placement="right">
                        <Text fontWeight="bold" noOfLines={2}>
                          {arc.title}
                        </Text>
                      </Tooltip>
                      <Button
                        size="xs"
                        colorScheme="blue"
                        onClick={() => onEditArc(arc)}
                      >
                        Edit
                      </Button>
                    </HStack>
                    <Text fontSize="xs" color="gray.500">
                      Main Ch: {arc.main_characters.join(', ')}
                    </Text>
                    <Tag size="sm" bg={getArcTypeColor(arc.arc_type)} color="white">
                      {arc.arc_type}
                    </Tag>
                  </VStack>
                </HStack>
              </GridItem>

              {/* Episodes */}
              {seasonEpisodes.map(ep => {
                const prog = arc.progressions.find(p => p.season === ep.season && p.episode === ep.episode);
                return (
                  <GridItem
                    key={`${arc.id}-${ep.episode}`}
                    p={2}
                    borderWidth={1}
                    borderRadius="md"
                    minH="100px"
                    cursor="pointer"
                    onClick={() => onCellClick(arc, ep.season, ep.episode)}
                    bg={prog ? `${getArcTypeColor(arc.arc_type)}15` : cellBgColor}
                    borderColor={borderColor}
                    _hover={{
                      bg: prog ? `${getArcTypeColor(arc.arc_type)}30` : bgColor
                    }}
                    transition="all 0.2s"
                  >
                    {prog && (
                      <VStack spacing={1} align="start" height="100%">
                        <Box 
                          w="100%" 
                          h="4px" 
                          bg={getArcTypeColor(arc.arc_type)}
                          borderRadius="full"
                        />
                        <Text 
                          fontSize="sm" 
                          noOfLines={4}
                          color="gray.600"
                          overflow="hidden"
                          textOverflow="ellipsis"
                        >
                          {prog.content}
                        </Text>
                        {prog.interfering_characters && prog.interfering_characters.length > 0 && (
                          <Text fontSize="xs" color="gray.500" mt="auto">
                            Characters: {prog.interfering_characters.join(', ')}
                          </Text>
                        )}
                      </VStack>
                    )}
                  </GridItem>
                );
              })}
            </Box>
          ))}
        </Grid>
      </Box>
    </Box>
  );
};

// Main component
const NarrativeArcManager: React.FC<NarrativeArcManagerProps> = ({
  arcs,
  episodes,
  onArcUpdated,
  series,
}) => {
  // All state declarations at the top
  const [selectedSeason, setSelectedSeason] = useState('');
  const [isMergeMode, setIsMergeMode] = useState(false);
  const [selectedForMerge, setSelectedForMerge] = useState<NarrativeArc[]>([]);
  const [showMergeModal, setShowMergeModal] = useState(false);
  const [isNewArcModalOpen, setIsNewArcModalOpen] = useState(false);
  const [editingArc, setEditingArc] = useState<NarrativeArc | null>(null);
  const [editArcTitle, setEditArcTitle] = useState('');
  const [editArcDescription, setEditArcDescription] = useState('');
  const [editArcType, setEditArcType] = useState('');
  const [editMainCharacters, setEditMainCharacters] = useState<string[]>([]);
  const [selectedCharacters, setSelectedCharacters] = useState<string[]>([]);
  const [selectedEpisodes, setSelectedEpisodes] = useState<string[]>([]);
  const [includeInterferingCharacters, setIncludeInterferingCharacters] = useState(false);
  const [selectedArcTypes, setSelectedArcTypes] = useState<ArcType[]>(Object.keys(ARC_TYPES) as ArcType[]);
  const [selectedCell, setSelectedCell] = useState<{
    arc: NarrativeArc;
    season: string;
    episode: string;
    content?: string;
    interferingCharacters?: string[];
  } | null>(null);

  const editArcDisclosure = useDisclosure();
  const { isOpen, onOpen, onClose } = useDisclosure();

  // Memoized values
  const allCharacters = useMemo(() => {
    const characters = new Set<string>();
    arcs.forEach(arc => {
      arc.main_characters.forEach(char => characters.add(char));
      arc.progressions.forEach(prog => {
        prog.interfering_characters.forEach(char => characters.add(char));
      });
    });
    return Array.from(characters).sort();
  }, [arcs]);

  // Single filteredArcs implementation
  const filteredArcs = useMemo(() => {
    let filtered = arcs;

    // Arc type filter
    filtered = filtered.filter(arc => selectedArcTypes.includes(arc.arc_type as ArcType));

    // Filter by season if selected
    if (selectedSeason) {
      filtered = filtered.filter(arc => 
        arc.progressions.some(prog => prog.season === selectedSeason)
      );
    }

    // Filter by characters
    if (selectedCharacters.length > 0) {
      filtered = filtered.filter(arc => {
        const isMainCharacter = selectedCharacters.some(char => arc.main_characters.includes(char));
        const isInterferingCharacter = includeInterferingCharacters && 
          arc.progressions.some(prog => 
            selectedCharacters.some(char => prog.interfering_characters.includes(char))
          );
        return isMainCharacter || isInterferingCharacter;
      });
    }

    // Fix episode filtering
    if (selectedEpisodes.length > 0) {
      filtered = filtered.filter(arc => {
        return arc.progressions.some(prog => {
          const episodeKey = `${prog.season}-${prog.episode}`;
          return selectedEpisodes.includes(episodeKey);
        });
      });
    }

    return filtered;
  }, [
    arcs, 
    selectedSeason, 
    selectedCharacters, 
    selectedEpisodes, 
    includeInterferingCharacters,
    selectedArcTypes
  ]);

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

  // Event handlers
  const handleCellClick = useCallback((arc: NarrativeArc, season: string, episode: string) => {
    const progression = arc.progressions.find(
      p => p.season === season && p.episode === episode
    );
    
    setSelectedCell({
      arc,
      season,
      episode,
      content: progression?.content || '',
      interferingCharacters: progression?.interfering_characters || []
    });
    onOpen();
  }, [onOpen]);

  const handleToggleMerge = useCallback((arc: NarrativeArc) => {
    setSelectedForMerge(prev => {
      if (prev.find(a => a.id === arc.id)) {
        return prev.filter(a => a.id !== arc.id);
      }
      if (prev.length < 2) {
        return [...prev, arc];
      }
      return prev;
    });
  }, []);

  const handleEditArc = useCallback((arc: NarrativeArc) => {
    setEditingArc(arc);
    setEditArcTitle(arc.title);
    setEditArcDescription(arc.description);
    setEditArcType(arc.arc_type);
    setEditMainCharacters(arc.main_characters);
    editArcDisclosure.onOpen();
  }, [editArcDisclosure]);

  // Add missing handlers
  const handleCreateArc = useCallback(async (arcData: any) => {
    try {
      const response = await fetch(`http://localhost:8000/api/arcs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(arcData),
      });

      if (!response.ok) throw new Error('Failed to create arc');

      // Get the newly created arc with its progression
      const newArc = await response.json();
      
      // Update the arcs data
      await onArcUpdated();  // Ensure this function refetches the arcs and episodes
      
      // Set the selected season to show the new progression
      if (arcData.initial_progression) {
        setSelectedSeason(arcData.initial_progression.season);
      }
      
      setIsNewArcModalOpen(false);
    } catch (error) {
      console.error('Error creating arc:', error);
    }
  }, [onArcUpdated]);

  const handleSaveProgression = useCallback(async (content: string, interferingCharacters: string[]) => {
    if (!selectedCell) return;
    
    const progression = selectedCell.arc.progressions.find(
      p => p.season === selectedCell.season && p.episode === selectedCell.episode
    );

    try {
      if (progression) {
        const response = await fetch(`http://localhost:8000/api/progressions/${progression.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content,
            interfering_characters: interferingCharacters
          }),
        });

        if (!response.ok) throw new Error('Failed to update progression');
      } else {
        const response = await fetch('http://localhost:8000/api/progressions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content,
            arc_id: selectedCell.arc.id,
            series: selectedCell.arc.series,
            season: selectedCell.season,
            episode: selectedCell.episode,
            interfering_characters: interferingCharacters
          }),
        });

        if (!response.ok) throw new Error('Failed to create progression');
      }

      onArcUpdated();
      onClose();
    } catch (error) {
      console.error('Error saving progression:', error);
    }
  }, [selectedCell, onArcUpdated, onClose]);

  const handleDeleteProgression = useCallback(async () => {
    if (!selectedCell) return;
    
    const progression = selectedCell.arc.progressions.find(
      p => p.season === selectedCell.season && p.episode === selectedCell.episode
    );

    if (!progression) return;

    try {
      const response = await fetch(`http://localhost:8000/api/progressions/${progression.id}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Failed to delete progression');

      onArcUpdated();
      onClose();
    } catch (error) {
      console.error('Error deleting progression:', error);
    }
  }, [selectedCell, onArcUpdated, onClose]);

  const handleDeleteArc = useCallback(async () => {
    if (!editingArc) return;

    try {
      const response = await fetch(`http://localhost:8000/api/arcs/${editingArc.id}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Failed to delete arc');

      onArcUpdated();
      editArcDisclosure.onClose();
    } catch (error) {
      console.error('Error deleting arc:', error);
    }
  }, [editingArc, onArcUpdated, editArcDisclosure]);

  const handleSaveArcChanges = useCallback(async () => {
    if (!editingArc) return;

    try {
      const response = await fetch(`http://localhost:8000/api/arcs/${editingArc.id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: editArcTitle,
          description: editArcDescription,
          arc_type: editArcType,
          main_characters: editMainCharacters,
        }),
      });

      if (!response.ok) throw new Error('Failed to update arc');

      onArcUpdated();
      editArcDisclosure.onClose();
    } catch (error) {
      console.error('Error updating arc:', error);
    }
  }, [editingArc, editArcTitle, editArcDescription, editArcType, editMainCharacters, onArcUpdated, editArcDisclosure]);

  // Add a useEffect to handle season changes
  useEffect(() => {
    if (selectedSeason) {
      // Force recalculation of seasonEpisodes when season changes
      const timer = setTimeout(() => {
        // This will trigger a re-render of the timeline
        setSelectedSeason(prev => prev);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [selectedSeason]);

  // Update the seasonEpisodes useMemo
  const seasonEpisodes = useMemo(() => {
    if (!selectedSeason) return [];

    // Get episodes from the episodes list
    const episodesList = episodes
      .filter(ep => ep.season === selectedSeason)
      .map(ep => ({ season: ep.season, episode: ep.episode }));

    // Get episodes from progressions
    const progressionEpisodes = arcs.flatMap(arc => 
      arc.progressions
        .filter(prog => prog.season === selectedSeason)
        .map(prog => ({ season: prog.season, episode: prog.episode }))
    );

    // Combine both lists and remove duplicates
    const uniqueEpisodes = [...new Set([
      ...episodesList.map(ep => ep.episode),
      ...progressionEpisodes.map(ep => ep.episode)
    ])];

    // Create array of all possible episodes for this season
    const allPossibleEpisodes = uniqueEpisodes.map(ep => ({
      season: selectedSeason,
      episode: ep
    }));

    // Sort by episode number
    return allPossibleEpisodes.sort((a, b) => 
      parseInt(a.episode.replace('E', '')) - parseInt(b.episode.replace('E', ''))
    );
  }, [episodes, arcs, selectedSeason]);

  // Update the seasons useMemo
  const seasons = useMemo(() => {
    // Get seasons from episodes
    const episodeSeasons = new Set(episodes.map(ep => ep.season));
    
    // Get seasons from progressions
    const progressionSeasons = new Set(
      arcs.flatMap(arc => arc.progressions.map(prog => prog.season))
    );
    
    // Combine both sets and sort
    const allSeasons = [...new Set([...episodeSeasons, ...progressionSeasons])].sort();
    
    return allSeasons;
  }, [episodes, arcs]);

  // Set initial season
  useEffect(() => {
    if (seasons.length > 0 && !selectedSeason) {
      setSelectedSeason(seasons[0]);
    }
  }, [seasons, selectedSeason]);

  return (
    <Box>
      <VStack spacing={4} align="stretch">
        {/* Arc Type Filters */}
        {renderArcTypeFilters()}

        {/* Arc Filters */}
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

        {/* Action Buttons */}
        <HStack spacing={4} justify="flex-end" mb={4}>
          <Button
            colorScheme="gray"
            onClick={onArcUpdated}
            leftIcon={<RepeatIcon />}
          >
            Refresh Arcs
          </Button>
          <Button
            colorScheme={isMergeMode ? "orange" : "gray"}
            onClick={() => setIsMergeMode(!isMergeMode)}
          >
            {isMergeMode ? "Cancel Merge" : "Merge Arcs"}
          </Button>
          {isMergeMode && (
            <Button
              colorScheme="blue"
              isDisabled={selectedForMerge.length !== 2}
              onClick={() => setShowMergeModal(true)}
            >
              Merge Selected ({selectedForMerge.length}/2)
            </Button>
          )}
          <Button
            leftIcon={<AddIcon />}
            colorScheme="green"
            onClick={() => setIsNewArcModalOpen(true)}
          >
            New Arc
          </Button>
        </HStack>

        {/* Timeline - Update to use filteredArcs instead of arcs */}
        <ArcTimeline
          arcs={filteredArcs}
          episodes={episodes}
          selectedSeason={selectedSeason}
          onCellClick={handleCellClick}
          isMergeMode={isMergeMode}
          selectedForMerge={selectedForMerge}
          onToggleMerge={handleToggleMerge}
          onEditArc={handleEditArc}
        />

        {/* Add Modals */}
        {/* Merge Modal */}
        {showMergeModal && selectedForMerge.length === 2 && (
          <ArcMergeModal
            isOpen={showMergeModal}
            onClose={() => {
              setShowMergeModal(false);
              setIsMergeMode(false);
              setSelectedForMerge([]);
            }}
            arc1={selectedForMerge[0]}
            arc2={selectedForMerge[1]}
            availableCharacters={allCharacters}
            onMergeComplete={() => {
              onArcUpdated();
              setShowMergeModal(false);
              setIsMergeMode(false);
              setSelectedForMerge([]);
            }}
          />
        )}

        {/* New Arc Modal */}
        <NewArcModal
          isOpen={isNewArcModalOpen}
          onClose={() => setIsNewArcModalOpen(false)}
          onSubmit={handleCreateArc}
          availableCharacters={allCharacters}
          series={arcs[0]?.series || ''}
        />

        {/* Progression Edit Modal */}
        {selectedCell && (
          <ArcProgressionEditModal
            isOpen={isOpen}
            onClose={onClose}
            arcTitle={selectedCell.arc.title}
            season={selectedCell.season}
            episode={selectedCell.episode}
            content={selectedCell.content || ''}
            interferingCharacters={selectedCell.interferingCharacters || []}
            availableCharacters={allCharacters}
            onSave={handleSaveProgression}
            allowCustomEpisode={false}
            availableSeasons={seasons}
            availableEpisodes={seasonEpisodes
              .filter(ep => ep.season === selectedCell.season)
              .map(ep => ep.episode)
            }
            showDelete={true}
            onDelete={handleDeleteProgression}
          />
        )}

        {/* Arc Edit Modal */}
        <Modal isOpen={editArcDisclosure.isOpen} onClose={editArcDisclosure.onClose} size="xl">
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Edit Arc</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <VStack spacing={4}>
                <FormControl>
                  <FormLabel>Title</FormLabel>
                  <Input
                    value={editArcTitle}
                    onChange={(e) => setEditArcTitle(e.target.value)}
                  />
                </FormControl>
                <FormControl>
                  <FormLabel>Description</FormLabel>
                  <Textarea
                    value={editArcDescription}
                    onChange={(e) => setEditArcDescription(e.target.value)}
                    rows={4}
                  />
                </FormControl>
                <FormControl>
                  <FormLabel>Arc Type</FormLabel>
                  <ChakraSelect
                    value={editArcType}
                    onChange={(e) => setEditArcType(e.target.value)}
                  >
                    <option value="Soap Arc">Soap Arc</option>
                    <option value="Genre-Specific Arc">Genre-Specific Arc</option>
                    <option value="Episodic Arc">Episodic Arc</option>
                  </ChakraSelect>
                </FormControl>
                <FormControl>
                  <FormLabel>Main Characters</FormLabel>
                  <Box maxH="200px" overflowY="auto" borderWidth={1} borderRadius="md" p={2}>
                    <VStack align="start" spacing={1}>
                      {allCharacters.map(char => (
                        <Checkbox
                          key={char}
                          isChecked={editMainCharacters.includes(char)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setEditMainCharacters(prev => [...prev, char]);
                            } else {
                              setEditMainCharacters(prev => prev.filter(c => c !== char));
                            }
                          }}
                        >
                          <Text fontSize="sm">{char}</Text>
                        </Checkbox>
                      ))}
                    </VStack>
                  </Box>
                </FormControl>
              </VStack>
            </ModalBody>
            <ModalFooter>
              <HStack spacing={4}>
                <Button
                  leftIcon={<DeleteIcon />}
                  colorScheme="red"
                  variant="ghost"
                  onClick={handleDeleteArc}
                >
                  Delete Arc
                </Button>
                <Button variant="ghost" onClick={editArcDisclosure.onClose}>
                  Cancel
                </Button>
                <Button colorScheme="blue" onClick={handleSaveArcChanges}>
                  Save
                </Button>
              </HStack>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </VStack>
    </Box>
  );
};

export default NarrativeArcManager; 