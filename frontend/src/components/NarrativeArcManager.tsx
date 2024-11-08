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
} from '@chakra-ui/react';
import { AddIcon } from '@chakra-ui/icons';
import { useState, useMemo } from 'react';
import ArcMergeModal from './ArcMergeModal';
import ArcProgressionEditModal from './ArcProgressionEditModal';

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
  selectedSeason: string;
  onArcUpdated: () => void;
}

const NarrativeArcManager: React.FC<NarrativeArcManagerProps> = ({
  arcs,
  episodes,
  selectedSeason,
  onArcUpdated,
}) => {
  const [isMergeMode, setIsMergeMode] = useState(false);
  const [selectedForMerge, setSelectedForMerge] = useState<NarrativeArc[]>([]);
  const [showMergeModal, setShowMergeModal] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [selectedCell, setSelectedCell] = useState<{
    arc: NarrativeArc;
    season: string;
    episode: string;
    content?: string;
    interferingCharacters?: string[];
  } | null>(null);
  const [selectedCharacters, setSelectedCharacters] = useState<string[]>([]);
  const [selectedEpisodes, setSelectedEpisodes] = useState<string[]>([]);
  const [includeInterferingCharacters, setIncludeInterferingCharacters] = useState(false);
  const [editingArc, setEditingArc] = useState<NarrativeArc | null>(null);
  const [editArcTitle, setEditArcTitle] = useState('');
  const [editArcDescription, setEditArcDescription] = useState('');
  const [editArcType, setEditArcType] = useState('');
  const [editMainCharacters, setEditMainCharacters] = useState<string[]>([]);
  const editArcDisclosure = useDisclosure();

  const seasonEpisodes = useMemo(() => {
    console.log('Filtering episodes for season:', selectedSeason);
    const filtered = episodes
      .filter(ep => ep.season === selectedSeason)
      .sort((a, b) => parseInt(a.episode.replace('E', '')) - parseInt(b.episode.replace('E', '')));
    console.log('Filtered episodes:', filtered);
    return filtered;
  }, [episodes, selectedSeason]);

  const toggleArcForMerge = (arc: NarrativeArc) => {
    setSelectedForMerge(prev => {
      if (prev.find(a => a.id === arc.id)) {
        return prev.filter(a => a.id !== arc.id);
      }
      if (prev.length < 2) {
        return [...prev, arc];
      }
      return prev;
    });
  };

  const handleCellClick = (arc: NarrativeArc, season: string, episode: string) => {
    const progression = arc.progressions.find(
      p => p.season === season && p.episode === episode
    );
    
    // Normalize season and episode format
    const normalizedSeason = season.startsWith('S') ? season : `S${season}`;
    const normalizedEpisode = episode.startsWith('E') ? episode : `E${episode}`;
    
    setSelectedCell({
      arc,
      season: normalizedSeason,
      episode: normalizedEpisode,
      content: progression?.content || '',
      interferingCharacters: progression?.interfering_characters || []
    });
    onOpen();
  };

  const handleSaveProgression = async (content: string, interferingCharacters: string[]) => {
    if (!selectedCell) return;

    try {
      const existingProg = selectedCell.arc.progressions.find(
        p => p.season === selectedCell.season && p.episode === selectedCell.episode
      );

      if (content.trim()) {
        if (existingProg) {
          // Update existing progression
          const response = await fetch(`http://localhost:8000/api/progressions/${existingProg.id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              content,
              interfering_characters: interferingCharacters
            }),
          });

          if (!response.ok) {
            throw new Error('Failed to update progression');
          }
        } else {
          // Create new progression
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

          if (!response.ok) {
            throw new Error('Failed to create progression');
          }
        }

        onArcUpdated();  // Refresh the data
        onClose();
      }
    } catch (error) {
      console.error('Error saving progression:', error);
      // You might want to add a toast notification here
    }
  };

  const getArcTypeColor = (arcType: string) => {
    const typeColors = {
      'Soap Arc': '#F687B3',      // pink
      'Genre-Specific Arc': '#ED8936', // orange
      'Episodic Arc': '#48BB78',     // green
      'seasonal': '#4299E1',      // blue
      'episodic': '#9F7AEA',      // purple
    };
    return typeColors[arcType as keyof typeof typeColors] || '#A0AEC0';
  };

  const allCharacters = useMemo(() => {
    const characters = new Set<string>();
    arcs.forEach(arc => {
      // Add main characters
      arc.main_characters.forEach(char => characters.add(char));
      // Add interfering characters from progressions
      arc.progressions.forEach(prog => {
        prog.interfering_characters.forEach(char => characters.add(char));
      });
    });
    return Array.from(characters).sort();
  }, [arcs]);

  const filteredArcs = useMemo(() => {
    return arcs.filter(arc => {
      // Filter by characters (both main and interfering if enabled)
      if (selectedCharacters.length > 0) {
        const isMainCharacter = selectedCharacters.some(char => arc.main_characters.includes(char));
        const isInterferingCharacter = includeInterferingCharacters && 
          arc.progressions.some(prog => 
            selectedCharacters.some(char => prog.interfering_characters.includes(char))
          );
      
        if (!isMainCharacter && !isInterferingCharacter) {
          return false;
        }
      }

      // Filter by episodes
      if (selectedEpisodes.length > 0) {
        if (!arc.progressions.some(prog => 
          selectedEpisodes.includes(`${prog.season}-${prog.episode}`))
        ) {
          return false;
        }
      }

      return true;
    });
  }, [arcs, selectedCharacters, selectedEpisodes, includeInterferingCharacters]);

  const handleEditArc = (arc: NarrativeArc) => {
    setEditingArc(arc);
    setEditArcTitle(arc.title);
    setEditArcDescription(arc.description);
    setEditArcType(arc.arc_type);
    setEditMainCharacters(arc.main_characters);
    editArcDisclosure.onOpen();
  };

  const handleSaveArcChanges = async () => {
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
  };

  const handleEditArcTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditArcTitle(e.target.value);
  };

  const handleEditArcTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setEditArcType(e.target.value);
  };

  return (
    <Box>
      {/* Filters section */}
      <Box bg={useColorModeValue('white', 'gray.800')} p={4} shadow="sm">
        <VStack spacing={4} align="stretch">
          {/* Merge buttons */}
          <HStack width="100%">
            <Button
              colorScheme={isMergeMode ? "orange" : "gray"}
              onClick={() => {
                setIsMergeMode(!isMergeMode);
                setSelectedForMerge([]);
              }}
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
          </HStack>

          {/* Filters */}
          <Grid templateColumns="repeat(2, 1fr)" gap={4}>
            {/* Characters Filter with Slider */}
            <FormControl>
              <HStack justify="space-between" mb={2}>
                <FormLabel mb={0}>Characters</FormLabel>
                <HStack>
                  <Text fontSize="sm" color="gray.500">Include in progressions</Text>
                  <Switch
                    isChecked={includeInterferingCharacters}
                    onChange={(e) => setIncludeInterferingCharacters(e.target.checked)}
                  />
                </HStack>
              </HStack>
              <Box maxH="200px" overflowY="auto" borderWidth={1} borderRadius="md" p={2}>
                <VStack align="start" spacing={1}>
                  {allCharacters.map(char => (
                    <Checkbox
                      key={char}
                      isChecked={selectedCharacters.includes(char)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedCharacters(prev => [...prev, char]);
                        } else {
                          setSelectedCharacters(prev => prev.filter(c => c !== char));
                        }
                      }}
                    >
                      <Text fontSize="sm">{char}</Text>
                    </Checkbox>
                  ))}
                </VStack>
              </Box>
              {selectedCharacters.length > 0 && (
                <HStack mt={2} flexWrap="wrap" spacing={2}>
                  <Text fontSize="sm" color="gray.500">Selected:</Text>
                  {selectedCharacters.map(char => (
                    <Tag 
                      key={char} 
                      size="sm"
                      colorScheme="blue"
                      cursor="pointer"
                      onClick={() => setSelectedCharacters(prev => prev.filter(c => c !== char))}
                    >
                      {char} ×
                    </Tag>
                  ))}
                  <Button
                    size="xs"
                    variant="ghost"
                    onClick={() => setSelectedCharacters([])}
                  >
                    Clear all
                  </Button>
                </HStack>
              )}
            </FormControl>

            {/* Episodes Filter */}
            <FormControl>
              <FormLabel>Episodes</FormLabel>
              <Box maxH="200px" overflowY="auto" borderWidth={1} borderRadius="md" p={2}>
                <VStack align="start" spacing={1}>
                  {seasonEpisodes.map(ep => (
                    <Checkbox
                      key={`${ep.season}-${ep.episode}`}
                      isChecked={selectedEpisodes.includes(`${ep.season}-${ep.episode}`)}
                      onChange={(e) => {
                        const epKey = `${ep.season}-${ep.episode}`;
                        if (e.target.checked) {
                          setSelectedEpisodes(prev => [...prev, epKey]);
                        } else {
                          setSelectedEpisodes(prev => prev.filter(e => e !== epKey));
                        }
                      }}
                    >
                      <Text fontSize="sm">Episode {ep.episode.replace('E', '')}</Text>
                    </Checkbox>
                  ))}
                </VStack>
              </Box>
              {selectedEpisodes.length > 0 && (
                <HStack mt={2} flexWrap="wrap" spacing={2}>
                  <Text fontSize="sm" color="gray.500">Selected:</Text>
                  {selectedEpisodes.map(ep => (
                    <Tag 
                      key={ep} 
                      size="sm"
                      colorScheme="blue"
                      cursor="pointer"
                      onClick={() => setSelectedEpisodes(prev => prev.filter(e => e !== ep))}
                    >
                      Ep {ep.split('-')[1].replace('E', '')} ×
                    </Tag>
                  ))}
                  <Button
                    size="xs"
                    variant="ghost"
                    onClick={() => setSelectedEpisodes([])}
                  >
                    Clear all
                  </Button>
                </HStack>
              )}
            </FormControl>
          </Grid>
        </VStack>
      </Box>

      {/* Timeline section */}
      <Box 
        mt={4} 
        bg={useColorModeValue('white', 'gray.800')} 
        shadow="sm"
        overflowX="auto"
        sx={{
          // Custom scrollbar styling
          '&::-webkit-scrollbar': {
            height: '8px',
            borderRadius: '8px',
            backgroundColor: `rgba(0, 0, 0, 0.05)`,
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: `rgba(0, 0, 0, 0.1)`,
            borderRadius: '8px',
            '&:hover': {
              backgroundColor: `rgba(0, 0, 0, 0.2)`,
            },
          },
        }}
      >
        <Box minWidth="fit-content" p={4}>
          <Grid
            templateColumns={`250px repeat(${seasonEpisodes.length}, 200px)`}
            gap={2}
          >
            {/* Header */}
            <GridItem p={2} bg={useColorModeValue('gray.50', 'gray.700')}>
              <Text fontWeight="bold">Narrative Arcs</Text>
            </GridItem>
            {seasonEpisodes.map(ep => (
              <GridItem
                key={ep.episode}
                p={2}
                bg={useColorModeValue('gray.50', 'gray.700')}
                textAlign="center"
              >
                <Text fontWeight="bold">
                  Episode {ep.episode.replace('E', '')}
                </Text>
              </GridItem>
            ))}

            {/* Arcs and Progressions */}
            {filteredArcs.map(arc => (
              <Box key={arc.id} display="contents">
                {/* Arc Title */}
                <GridItem p={2} borderWidth={1} borderRadius="md">
                  <HStack alignItems="flex-start">
                    {isMergeMode && (
                      <Checkbox
                        isChecked={!!selectedForMerge.find(a => a.id === arc.id)}
                        onChange={() => toggleArcForMerge(arc)}
                        onClick={(e) => e.stopPropagation()}
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
                          onClick={() => handleEditArc(arc)}
                        >
                          Edit
                        </Button>
                      </HStack>
                      <Text 
                        fontSize="xs" 
                        color="gray.500" 
                        whiteSpace="pre-wrap"
                        wordBreak="break-word"
                      >
                        Main Ch: {arc.main_characters.join(', ')}
                      </Text>
                      <Tag size="sm" bg={getArcTypeColor(arc.arc_type)} color="white">
                        {arc.arc_type}
                      </Tag>
                    </VStack>
                  </HStack>
                </GridItem>

                {/* Progressions */}
                {seasonEpisodes.map(ep => {
                  const prog = arc.progressions.find(
                    p => p.season === ep.season && p.episode === ep.episode
                  );
                  console.log(`Checking progression for arc ${arc.title}, episode ${ep.episode}:`, {
                    found: !!prog,
                    lookingFor: `${ep.season}-${ep.episode}`,
                    available: arc.progressions.map(p => `${p.season}-${p.episode}`)
                  });
                  return (
                    <GridItem
                      key={`${arc.id}-${ep.episode}`}
                      p={2}
                      borderWidth={1}
                      borderRadius="md"
                      minH="100px"
                      cursor="pointer"
                      onClick={() => handleCellClick(arc, ep.season, ep.episode)}
                      bg={prog ? `${getArcTypeColor(arc.arc_type)}15` : 'transparent'}
                      _hover={{
                        bg: prog ? `${getArcTypeColor(arc.arc_type)}30` : useColorModeValue('gray.50', 'gray.700')
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
                            color={useColorModeValue('gray.600', 'gray.300')}
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
          availableSeasons={seasonEpisodes.map(ep => ep.season).filter((v, i, a) => a.indexOf(v) === i)}
          availableEpisodes={seasonEpisodes
            .filter(ep => ep.season === selectedCell.season)
            .map(ep => ep.episode)
          }
        />
      )}

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
                  onChange={handleEditArcTitleChange}
                />
              </FormControl>
              <FormControl>
                <FormLabel>Description</FormLabel>
                <Textarea
                  value={editArcDescription}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setEditArcDescription(e.target.value)}
                  rows={4}
                />
              </FormControl>
              <FormControl>
                <FormLabel>Arc Type</FormLabel>
                <ChakraSelect
                  value={editArcType}
                  onChange={handleEditArcTypeChange}
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
                {editMainCharacters.length > 0 && (
                  <HStack mt={2} flexWrap="wrap" spacing={2}>
                    <Text fontSize="sm" color="gray.500">Selected:</Text>
                    {editMainCharacters.map(char => (
                      <Tag 
                        key={char} 
                        size="sm"
                        colorScheme="blue"
                        cursor="pointer"
                        onClick={() => setEditMainCharacters(prev => prev.filter(c => c !== char))}
                      >
                        {char} ×
                      </Tag>
                    ))}
                    <Button
                      size="xs"
                      variant="ghost"
                      onClick={() => setEditMainCharacters([])}
                    >
                      Clear all
                    </Button>
                  </HStack>
                )}
              </FormControl>
            </VStack>
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={handleSaveArcChanges}>
              Save
            </Button>
            <Button variant="ghost" onClick={editArcDisclosure.onClose}>
              Cancel
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default NarrativeArcManager; 