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
  Text,
  Textarea,
  FormControl,
  FormLabel,
  Box,
  Tag,
  TagLabel,
  TagCloseButton,
  Input,
  List,
  ListItem,
  Select,
  HStack,
  Checkbox,
} from '@chakra-ui/react';
import { useState, useEffect, useRef } from 'react';

interface ArcProgressionEditModalProps {
  isOpen: boolean;
  onClose: () => void;
  arcTitle: string;
  season: string;
  episode: string;
  content: string;
  interferingCharacters?: string[];
  availableCharacters: string[];
  onSave: (content: string, interferingCharacters: string[]) => void;
  allowCustomEpisode?: boolean;
  onSeasonChange?: (season: string) => void;
  onEpisodeChange?: (episode: string) => void;
  availableSeasons?: string[];
  availableEpisodes?: string[];
}

const ArcProgressionEditModal: React.FC<ArcProgressionEditModalProps> = ({
  isOpen,
  onClose,
  arcTitle,
  season,
  episode,
  content,
  interferingCharacters = [],
  availableCharacters,
  onSave,
  allowCustomEpisode = false,
  onSeasonChange,
  onEpisodeChange,
  availableSeasons = [],
  availableEpisodes = [],
}) => {
  const [editContent, setEditContent] = useState(content);
  const [selectedCharacters, setSelectedCharacters] = useState<string[]>(interferingCharacters);
  
  const [customSeason, setCustomSeason] = useState('');
  const [customEpisode, setCustomEpisode] = useState('');

  useEffect(() => {
    setEditContent(content);
    setSelectedCharacters(interferingCharacters);
  }, [content, interferingCharacters]);

  const handleSave = () => {
    onSave(editContent, selectedCharacters);
  };

  const handleClose = () => {
    setEditContent(content);
    setSelectedCharacters(interferingCharacters);
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} size="xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>
          {allowCustomEpisode ? 'Add New Progression' : `Edit Progression for ${arcTitle}`}
          {!allowCustomEpisode && (
            <Text fontSize="sm" color="gray.500">
              Season {season.replace('S', '')}, Episode {episode.replace('E', '')}
            </Text>
          )}
        </ModalHeader>
        <ModalCloseButton />
        
        <ModalBody>
          <VStack spacing={4} align="stretch">
            {allowCustomEpisode && (
              <>
                <FormControl>
                  <FormLabel>Season</FormLabel>
                  <HStack>
                    <Select
                      value={season}
                      onChange={(e) => onSeasonChange?.(e.target.value)}
                      placeholder="Select season"
                      flex="1"
                    >
                      {availableSeasons.map(s => (
                        <option key={s} value={s}>
                          Season {s.replace('S', '')}
                        </option>
                      ))}
                    </Select>
                    <Text>or</Text>
                    <Input
                      placeholder="Custom season (e.g., 1)"
                      value={customSeason}
                      onChange={(e) => {
                        setCustomSeason(e.target.value);
                        const normalized = `S${e.target.value.replace(/\D/g, '').padStart(2, '0')}`;
                        onSeasonChange?.(normalized);
                      }}
                      width="150px"
                    />
                  </HStack>
                </FormControl>

                <FormControl>
                  <FormLabel>Episode</FormLabel>
                  <HStack>
                    <Select
                      value={episode}
                      onChange={(e) => onEpisodeChange?.(e.target.value)}
                      placeholder="Select episode"
                      flex="1"
                      isDisabled={!season}
                    >
                      {availableEpisodes.map(ep => (
                        <option key={ep} value={ep}>
                          Episode {ep.replace('E', '')}
                        </option>
                      ))}
                    </Select>
                    <Text>or</Text>
                    <Input
                      placeholder="Custom episode (e.g., 10)"
                      value={customEpisode}
                      onChange={(e) => {
                        setCustomEpisode(e.target.value);
                        const normalized = `E${e.target.value.replace(/\D/g, '').padStart(2, '0')}`;
                        onEpisodeChange?.(normalized);
                      }}
                      width="150px"
                    />
                  </HStack>
                </FormControl>
              </>
            )}

            <FormControl>
              <FormLabel>Content</FormLabel>
              <Textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                rows={6}
                placeholder="Enter progression content..."
              />
            </FormControl>

            <FormControl>
              <FormLabel>Interfering Characters</FormLabel>
              <Box maxH="200px" overflowY="auto" borderWidth={1} borderRadius="md" p={2}>
                <VStack align="start" spacing={1}>
                  {availableCharacters.map(char => (
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
          </VStack>
        </ModalBody>

        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={handleClose}>
            Cancel
          </Button>
          <Button colorScheme="blue" onClick={handleSave}>
            Save
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default ArcProgressionEditModal; 