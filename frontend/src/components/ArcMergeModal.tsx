import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Button,
  VStack,
  HStack,
  Box,
  Text,
  Input,
  Textarea,
  Select,
  Grid,
  IconButton,
  useColorModeValue,
  FormControl,
  FormLabel,
  Checkbox,
  Tooltip,
  Tag,
  TagLabel,
  TagCloseButton,
} from '@chakra-ui/react';
import { AddIcon } from '@chakra-ui/icons';
import { useState, useMemo, useEffect } from 'react';
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

interface ArcMergeModalProps {
  isOpen: boolean;
  onClose: () => void;
  arc1: NarrativeArc;
  arc2: NarrativeArc;
  availableCharacters: string[];
  onMergeComplete: () => void;
}

const ArcMergeModal: React.FC<ArcMergeModalProps> = ({
  isOpen,
  onClose,
  arc1,
  arc2,
  availableCharacters,
  onMergeComplete,
}) => {
  // State declarations
  const [mergedTitle, setMergedTitle] = useState(arc1.title);
  const [mergedDescription, setMergedDescription] = useState(arc1.description);
  const [mergedArcType, setMergedArcType] = useState(arc1.arc_type);
  const [editMainCharacters, setEditMainCharacters] = useState<string[]>([...arc1.main_characters]);
  const [progressionMappings, setProgressionMappings] = useState<Record<string, { content: string; interferingCharacters: string[] }>>({});

  // State for new progression insertion
  const [showNewProgressionForm, setShowNewProgressionForm] = useState(false);
  const [newProgressionSeason, setNewProgressionSeason] = useState('');
  const [newProgressionEpisode, setNewProgressionEpisode] = useState('');
  const [newProgressionContent, setNewProgressionContent] = useState('');

  // Add state for custom episode input
  const [customSeason, setCustomSeason] = useState('');
  const [customEpisode, setCustomEpisode] = useState('');

  // Add state for progression interfering characters
  const [newProgressionInterferingChars, setNewProgressionInterferingChars] = useState<string[]>([]);

  // Combine all characters
  const allCharacters = useMemo(() => {
    const characters = new Set<string>();
    [...arc1.main_characters, ...arc2.main_characters].forEach(char => characters.add(char));
    [...arc1.progressions, ...arc2.progressions].forEach(prog => {
      prog.interfering_characters.forEach(char => characters.add(char));
    });
    return Array.from(characters).sort();
  }, [arc1, arc2]);

  // Get all possible episodes including custom ones
  const [allEpisodes, setAllEpisodes] = useState<string[]>([]);

  // Initialize episodes and progression mappings
  useEffect(() => {
    const episodeSet = new Set<string>();
    const initialMappings: Record<string, { content: string; interferingCharacters: string[] }> = {};
    
    // Add existing episodes from both arcs
    [...arc1.progressions, ...arc2.progressions].forEach(prog => {
      const episodeKey = `${prog.season}-${prog.episode}`;
      episodeSet.add(episodeKey);
    });

    // Sort episodes
    const sortedEpisodes = Array.from(episodeSet).sort((a, b) => {
      const [s1, e1] = a.split('-').map(x => parseInt(x.replace(/[SE]/g, '')));
      const [s2, e2] = b.split('-').map(x => parseInt(x.replace(/[SE]/g, '')));
      return s1 === s2 ? e1 - e2 : s1 - s2;
    });

    // Initialize mappings for all episodes
    sortedEpisodes.forEach(episode => {
      const [season, ep] = episode.split('-');
      const prog1 = arc1.progressions.find(p => p.season === season && p.episode === ep);
      const prog2 = arc2.progressions.find(p => p.season === season && p.episode === ep);
      
      const newKey = `new-${season}-${ep}`;
      initialMappings[newKey] = {
        content: prog1?.content || prog2?.content || '',
        interferingCharacters: prog1?.interfering_characters || prog2?.interfering_characters || []
      };
    });

    setAllEpisodes(sortedEpisodes);
    setProgressionMappings(initialMappings);
  }, [arc1, arc2]); // Only depend on the arcs

  // Calculate episode width based on total episodes
  const episodeWidth = useMemo(() => {
    return `${100 / allEpisodes.length}%`;
  }, [allEpisodes]);

  const handleMerge = async () => {
    try {
      const formattedProgressionMappings = Object.entries(progressionMappings)
        .filter(([_, data]) => data.content.trim())
        .map(([key, data]) => {
          const [_, season, episode] = key.split('-');
          return {
            season,
            episode,
            content: data.content,
            interfering_characters: data.interferingCharacters || []
          };
        });

      console.log('Sending merge request with mappings:', formattedProgressionMappings);

      const response = await fetch('http://localhost:8000/api/arcs/merge', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          arc_id_1: arc1.id,
          arc_id_2: arc2.id,
          merged_title: mergedTitle,
          merged_description: mergedDescription,
          merged_arc_type: mergedArcType,
          main_characters: editMainCharacters,
          progression_mappings: formattedProgressionMappings
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to merge arcs');
      }

      onMergeComplete();
      onClose();
    } catch (error) {
      console.error('Error merging arcs:', error);
    }
  };

  // Modified handleAddNewProgression
  const handleAddNewProgression = () => {
    const season = customSeason || newProgressionSeason;
    const episode = customEpisode || newProgressionEpisode;
    
    const normalizedSeason = `S${season.replace(/\D/g, '').padStart(2, '0')}`;
    const normalizedEpisode = `E${episode.replace(/\D/g, '').padStart(2, '0')}`;
    const newEpisodeKey = `${normalizedSeason}-${normalizedEpisode}`;
    
    const newKey = `new-${normalizedSeason}-${normalizedEpisode}`;

    // Get existing interfering characters if any
    const existingMapping = progressionMappings[newKey];
    const existingCharacters = existingMapping?.interferingCharacters || [];

    // Update progression mappings
    setProgressionMappings(prev => ({
      ...prev,
      [newKey]: {
        content: newProgressionContent,
        interferingCharacters: [...new Set([...existingCharacters, ...newProgressionInterferingChars])]
      }
    }));

    // Update episodes list if it's a new episode
    if (!allEpisodes.includes(newEpisodeKey)) {
      const newEpisodes = [...allEpisodes, newEpisodeKey].sort((a, b) => {
        const [s1, e1] = a.split('-').map(x => parseInt(x.replace(/[SE]/g, '')));
        const [s2, e2] = b.split('-').map(x => parseInt(x.replace(/[SE]/g, '')));
        return s1 === s2 ? e1 - e2 : s1 - s2;
      });
      setAllEpisodes(newEpisodes);
    }

    setShowNewProgressionForm(false);
    setNewProgressionContent('');
    setNewProgressionInterferingChars([]);
    setCustomSeason('');
    setCustomEpisode('');
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="full">
      <ModalOverlay />
      <ModalContent maxW="95vw" maxH="95vh">
        <ModalHeader>Merge Arcs</ModalHeader>
        <ModalCloseButton />
        <ModalBody overflowY="auto">
          <VStack spacing={6} align="stretch">
            {/* Arc Details Section */}
            <Box>
              <FormControl>
                <FormLabel>Merged Title</FormLabel>
                <Input
                  value={mergedTitle}
                  onChange={(e) => setMergedTitle(e.target.value)}
                />
              </FormControl>

              <FormControl mt={4}>
                <FormLabel>Merged Description</FormLabel>
                <Textarea
                  value={mergedDescription}
                  onChange={(e) => setMergedDescription(e.target.value)}
                  rows={4}
                />
              </FormControl>

              <FormControl mt={4}>
                <FormLabel>Arc Type</FormLabel>
                <Select
                  value={mergedArcType}
                  onChange={(e) => setMergedArcType(e.target.value)}
                >
                  <option value="Soap Arc">Soap Arc</option>
                  <option value="Genre-Specific Arc">Genre-Specific Arc</option>
                  <option value="Episodic Arc">Episodic Arc</option>
                </Select>
              </FormControl>
            </Box>

            {/* Main Characters Section */}
            <Box>
              <FormLabel>Main Characters</FormLabel>
              <Grid templateColumns="repeat(3, 1fr)" gap={2}>
                {availableCharacters.map(char => (
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
                    {char}
                  </Checkbox>
                ))}
              </Grid>
            </Box>

            {/* Progressions Timeline Section */}
            <Box>
              <HStack justify="space-between" mb={4}>
                <Text fontWeight="bold">Progressions Timeline</Text>
                <Button
                  leftIcon={<AddIcon />}
                  size="sm"
                  onClick={() => setShowNewProgressionForm(true)}
                >
                  Add Progression
                </Button>
              </HStack>

              {/* Timeline Header */}
              <Box overflowX="auto">
                <Box minWidth="fit-content">
                  <Grid templateColumns={`200px repeat(${allEpisodes.length}, ${episodeWidth})`} gap={1}>
                    <Box p={2} bg="gray.100">Arc</Box>
                    {allEpisodes.map(episode => {
                      const [season, ep] = episode.split('-');
                      return (
                        <Box key={episode} p={2} bg="gray.100" textAlign="center">
                          S{season.replace('S', '')}-E{ep.replace('E', '')}
                        </Box>
                      );
                    })}
                  </Grid>

                  {/* Arc 1 Row */}
                  <Grid templateColumns={`200px repeat(${allEpisodes.length}, ${episodeWidth})`} gap={1}>
                    <Box p={2} bg="blue.50">
                      <Text color="blue.500" fontWeight="bold">{arc1.title}</Text>
                    </Box>
                    {allEpisodes.map(episode => {
                      const [season, ep] = episode.split('-');
                      const prog = arc1.progressions.find(p => 
                        p.season === season && p.episode === ep
                      );
                      return (
                        <Box 
                          key={`arc1-${episode}`} 
                          p={2} 
                          bg={prog ? "blue.50" : "transparent"}
                          borderWidth={1}
                          borderColor="gray.200"
                          minHeight="80px"
                          maxHeight="80px"
                          overflow="hidden"
                        >
                          {prog && (
                            <Tooltip label={`${prog.content}\nInterfering: ${prog.interfering_characters.join(', ')}`}>
                              <Text fontSize="xs" noOfLines={3}>
                                {prog.content}
                              </Text>
                            </Tooltip>
                          )}
                        </Box>
                      );
                    })}
                  </Grid>

                  {/* Arc 2 Row */}
                  <Grid templateColumns={`200px repeat(${allEpisodes.length}, ${episodeWidth})`} gap={1}>
                    <Box p={2} bg="green.50">
                      <Text color="green.500" fontWeight="bold">{arc2.title}</Text>
                    </Box>
                    {allEpisodes.map(episode => {
                      const [season, ep] = episode.split('-');
                      const prog = arc2.progressions.find(p => 
                        p.season === season && p.episode === ep
                      );
                      return (
                        <Box 
                          key={`arc2-${episode}`} 
                          p={2} 
                          bg={prog ? "green.50" : "transparent"}
                          borderWidth={1}
                          borderColor="gray.200"
                          minHeight="80px"
                          maxHeight="80px"
                          overflow="hidden"
                        >
                          {prog && (
                            <Tooltip label={`${prog.content}\nInterfering: ${prog.interfering_characters.join(', ')}`}>
                              <Text fontSize="xs" noOfLines={3}>
                                {prog.content}
                              </Text>
                            </Tooltip>
                          )}
                        </Box>
                      );
                    })}
                  </Grid>

                  {/* Merged Row */}
                  <Grid templateColumns={`200px repeat(${allEpisodes.length}, ${episodeWidth})`} gap={1}>
                    <Box p={2} bg="purple.50">
                      <Text color="purple.500" fontWeight="bold">Merged</Text>
                    </Box>
                    {allEpisodes.map(episode => {
                      const [season, ep] = episode.split('-');
                      const newKey = `new-${season}-${ep}`;
                      const hasContent = !!progressionMappings[newKey];
                      return (
                        <Box 
                          key={`merged-${episode}`} 
                          p={2} 
                          bg={hasContent ? "purple.50" : "transparent"}
                          borderWidth={1}
                          borderColor="gray.200"
                          onClick={() => {
                            setNewProgressionSeason(season);
                            setNewProgressionEpisode(ep);
                            const existingMapping = progressionMappings[`new-${season}-${ep}`];
                            setNewProgressionContent(existingMapping?.content || '');
                            setNewProgressionInterferingChars(existingMapping?.interferingCharacters || []);
                            setShowNewProgressionForm(true);
                          }}
                          cursor="pointer"
                          _hover={{ bg: "purple.100" }}
                          minHeight="80px"
                          maxHeight="80px"
                          overflow="hidden"
                        >
                          {hasContent && (
                            <Tooltip label={
                              `${progressionMappings[`new-${season}-${ep}`]?.content}\n` +
                              `Interfering: ${progressionMappings[`new-${season}-${ep}`]?.interferingCharacters?.join(', ') || 'None'}`
                            }>
                              <VStack align="start" spacing={1}>
                                <Text fontSize="xs" noOfLines={2}>
                                  {progressionMappings[`new-${season}-${ep}`]?.content}
                                </Text>
                                {progressionMappings[`new-${season}-${ep}`]?.interferingCharacters?.length > 0 && (
                                  <Text fontSize="xs" color="gray.500" noOfLines={1}>
                                    Interfering: {progressionMappings[`new-${season}-${ep}`]?.interferingCharacters.join(', ')}
                                  </Text>
                                )}
                              </VStack>
                            </Tooltip>
                          )}
                        </Box>
                      );
                    })}
                  </Grid>
                </Box>
              </Box>
            </Box>
          </VStack>

          {/* Modified New Progression Form Modal */}
          <ArcProgressionEditModal 
            isOpen={showNewProgressionForm}
            onClose={() => setShowNewProgressionForm(false)}
            arcTitle={mergedTitle}
            season={newProgressionSeason}
            episode={newProgressionEpisode}
            content={newProgressionContent}
            interferingCharacters={newProgressionInterferingChars}
            availableCharacters={availableCharacters}
            onSave={(content, interferingChars) => {
              setNewProgressionContent(content);
              setNewProgressionInterferingChars(interferingChars);
              handleAddNewProgression();
            }}
            allowCustomEpisode={true}
            onSeasonChange={setNewProgressionSeason}
            onEpisodeChange={setNewProgressionEpisode}
            availableSeasons={[...new Set(allEpisodes.map(ep => ep.split('-')[0]))]}
            availableEpisodes={allEpisodes
              .filter(ep => ep.startsWith(newProgressionSeason + '-'))
              .map(ep => ep.split('-')[1])
            }
          />
        </ModalBody>

        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button colorScheme="blue" onClick={handleMerge}>
            Merge Arcs
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default ArcMergeModal; 