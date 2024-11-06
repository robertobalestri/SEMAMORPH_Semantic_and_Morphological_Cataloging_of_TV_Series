import {
  Box,
  Button,
  VStack,
  HStack,
  Text,
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Select,
  Textarea,
  FormControl,
  FormLabel,
  Input,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  IconButton,
} from '@chakra-ui/react';
import { DeleteIcon, EditIcon, AddIcon } from '@chakra-ui/icons';
import { useState } from 'react';
import { NarrativeArc, ArcProgression } from '../types';

interface ArcEditorProps {
  arcs: NarrativeArc[];
  series: string;
  season: string;
  episodes: { season: string; episode: string; }[];
  onUpdate: () => void;
}

interface ProgressionFormData {
  content: string;
  episode: string;
  interfering_characters: string;
}

// Add proper error type
interface ApiError {
  message: string;
}

export default function ArcEditor({ arcs, series, season, episodes, onUpdate }: ArcEditorProps) {
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [selectedArcs, setSelectedArcs] = useState<string[]>([]);
  const [selectedArc, setSelectedArc] = useState<NarrativeArc | null>(null);
  const [progressionData, setProgressionData] = useState<ProgressionFormData>({
    content: '',
    episode: '',
    interfering_characters: '',
  });
  const [editingProgression, setEditingProgression] = useState<ArcProgression | null>(null);

  const seasonEpisodes = episodes
    .filter(ep => ep.season === season)
    .sort((a, b) => parseInt(a.episode) - parseInt(b.episode));

  const handleMergeArcs = async () => {
    if (selectedArcs.length < 2) {
      toast({
        title: "Select at least two arcs to merge",
        status: "warning",
        duration: 3000,
      });
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/api/arcs/merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          arc_ids: selectedArcs,
          series: series,
        }),
      });

      if (!response.ok) throw new Error('Failed to merge arcs');

      toast({
        title: "Arcs merged successfully",
        status: "success",
        duration: 3000,
      });
      onUpdate();
      setSelectedArcs([]);
    } catch (err) {
      const error = err as ApiError;
      toast({
        title: "Failed to merge arcs",
        description: error.message || "An unknown error occurred",
        status: "error",
        duration: 3000,
      });
    }
  };

  const handleDeleteArc = async (arcId: string) => {
    if (!confirm('Are you sure you want to delete this arc and all its progressions?')) return;

    try {
      const response = await fetch(`http://localhost:8000/api/arcs/${arcId}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Failed to delete arc');

      toast({
        title: "Arc deleted successfully",
        status: "success",
        duration: 3000,
      });
      onUpdate();
    } catch (err) {
      const error = err as ApiError;
      toast({
        title: "Failed to delete arc",
        description: error.message || "An unknown error occurred",
        status: "error",
        duration: 3000,
      });
    }
  };

  const handleAddProgression = async () => {
    if (!selectedArc) return;

    try {
      const response = await fetch(`http://localhost:8000/api/arcs/${selectedArc.id}/progressions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...progressionData,
          series,
          season,
        }),
      });

      if (!response.ok) throw new Error('Failed to add progression');

      toast({
        title: "Progression added successfully",
        status: "success",
        duration: 3000,
      });
      onUpdate();
      onClose();
    } catch (err) {
      const error = err as ApiError;
      toast({
        title: "Failed to add progression",
        description: error.message || "An unknown error occurred",
        status: "error",
        duration: 3000,
      });
    }
  };

  const handleUpdateProgression = async (progression: ArcProgression) => {
    try {
      const response = await fetch(`http://localhost:8000/api/progressions/${progression.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(progression),
      });

      if (!response.ok) throw new Error('Failed to update progression');

      toast({
        title: "Progression updated successfully",
        status: "success",
        duration: 3000,
      });
      onUpdate();
    } catch (err) {
      const error = err as ApiError;
      toast({
        title: "Failed to update progression",
        description: error.message || "An unknown error occurred",
        status: "error",
        duration: 3000,
      });
    }
  };

  const handleSaveProgression = async () => {
    if (!selectedArc) return;

    try {
      if (editingProgression) {
        // Update existing progression
        await handleUpdateProgression({
          ...editingProgression,
          content: progressionData.content,
          episode: progressionData.episode,
          interfering_characters: progressionData.interfering_characters.split(';').map(c => c.trim()),
        });
      } else {
        // Add new progression
        await handleAddProgression();
      }
      onClose();
      setEditingProgression(null);
    } catch (err) {
      const error = err as ApiError;
      toast({
        title: "Failed to save progression",
        description: error.message || "An unknown error occurred",
        status: "error",
        duration: 3000,
      });
    }
  };

  return (
    <Box>
      <VStack spacing={4} align="stretch">
        <HStack>
          <Button
            colorScheme="blue"
            onClick={handleMergeArcs}
            isDisabled={selectedArcs.length < 2}
          >
            Merge Selected Arcs
          </Button>
        </HStack>

        <Accordion allowMultiple>
          {arcs.map(arc => (
            <AccordionItem key={arc.id}>
              <AccordionButton>
                <HStack flex="1">
                  <input
                    type="checkbox"
                    checked={selectedArcs.includes(arc.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedArcs([...selectedArcs, arc.id]);
                      } else {
                        setSelectedArcs(selectedArcs.filter(id => id !== arc.id));
                      }
                    }}
                  />
                  <Text flex="1">{arc.title}</Text>
                  <IconButton
                    aria-label="Delete arc"
                    icon={<DeleteIcon />}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteArc(arc.id);
                    }}
                    size="sm"
                    colorScheme="red"
                  />
                </HStack>
                <AccordionIcon />
              </AccordionButton>

              <AccordionPanel>
                <VStack align="stretch" spacing={4}>
                  <Button
                    leftIcon={<AddIcon />}
                    onClick={() => {
                      setSelectedArc(arc);
                      setProgressionData({
                        content: '',
                        episode: '',
                        interfering_characters: '',
                      });
                      onOpen();
                    }}
                  >
                    Add Progression
                  </Button>

                  {arc.progressions
                    .sort((a: ArcProgression, b: ArcProgression) => parseInt(a.episode) - parseInt(b.episode))
                    .map((prog: ArcProgression) => (
                      <Box key={prog.id} p={4} borderWidth={1} borderRadius="md">
                        <Text fontWeight="bold">Episode {prog.episode}</Text>
                        <Text mt={2}>{prog.content}</Text>
                        <Text mt={2} fontSize="sm" color="gray.600">
                          Interfering Characters: {prog.interfering_characters.join(', ')}
                        </Text>
                        <IconButton
                          aria-label="Edit progression"
                          icon={<EditIcon />}
                          size="sm"
                          mt={2}
                          onClick={() => {
                            setSelectedArc(arc);
                            setEditingProgression(prog);
                            setProgressionData({
                              content: prog.content,
                              episode: prog.episode,
                              interfering_characters: prog.interfering_characters.join('; '),
                            });
                            onOpen();
                          }}
                        />
                      </Box>
                    ))}
                </VStack>
              </AccordionPanel>
            </AccordionItem>
          ))}
        </Accordion>
      </VStack>

      {/* Progression Edit Modal */}
      <Modal isOpen={isOpen} onClose={() => {
        onClose();
        setEditingProgression(null);
      }}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>
            {editingProgression ? 'Edit Progression' : 'Add Progression'}
          </ModalHeader>
          <ModalBody>
            <VStack spacing={4}>
              <FormControl>
                <FormLabel>Episode</FormLabel>
                <Select
                  value={progressionData.episode}
                  onChange={(e) => setProgressionData({
                    ...progressionData,
                    episode: e.target.value,
                  })}
                >
                  {seasonEpisodes.map(ep => (
                    <option key={ep.episode} value={ep.episode}>
                      Episode {ep.episode}
                    </option>
                  ))}
                </Select>
              </FormControl>

              <FormControl>
                <FormLabel>Content</FormLabel>
                <Textarea
                  value={progressionData.content}
                  onChange={(e) => setProgressionData({
                    ...progressionData,
                    content: e.target.value,
                  })}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Interfering Characters (semicolon-separated)</FormLabel>
                <Input
                  value={progressionData.interfering_characters}
                  onChange={(e) => setProgressionData({
                    ...progressionData,
                    interfering_characters: e.target.value,
                  })}
                />
              </FormControl>
            </VStack>
          </ModalBody>

          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={handleSaveProgression}>
              Save
            </Button>
            <Button onClick={() => {
              onClose();
              setEditingProgression(null);
            }}>
              Cancel
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Box>
  );
} 