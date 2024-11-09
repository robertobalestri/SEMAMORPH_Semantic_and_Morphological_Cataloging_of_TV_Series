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
  FormControl,
  FormLabel,
  Input,
  Textarea,
  Select,
  Box,
  Text,
  Tag,
  TagLabel,
  TagCloseButton,
  Checkbox,
  HStack,
} from '@chakra-ui/react';
import { useState } from 'react';

interface NewArcModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (arcData: any) => void;
  availableCharacters: string[];
  series: string;
}

const NewArcModal: React.FC<NewArcModalProps> = ({
  isOpen,
  onClose,
  onSubmit,
  availableCharacters,
  series
}) => {
  const [newArcTitle, setNewArcTitle] = useState('');
  const [newArcDescription, setNewArcDescription] = useState('');
  const [newArcType, setNewArcType] = useState('Soap Arc');
  const [newArcMainCharacters, setNewArcMainCharacters] = useState<string[]>([]);
  const [initialProgressionContent, setInitialProgressionContent] = useState('');
  const [initialProgressionSeason, setInitialProgressionSeason] = useState('');
  const [initialProgressionEpisode, setInitialProgressionEpisode] = useState('');
  const [initialProgressionCharacters, setInitialProgressionCharacters] = useState<string[]>([]);

  const handleSubmit = () => {
    onSubmit({
      title: newArcTitle,
      description: newArcDescription,
      arc_type: newArcType,
      main_characters: newArcMainCharacters,
      series: series,
      initial_progression: {
        content: initialProgressionContent,
        season: initialProgressionSeason,
        episode: initialProgressionEpisode,
        interfering_characters: initialProgressionCharacters
      }
    });

    // Reset form
    setNewArcTitle('');
    setNewArcDescription('');
    setNewArcType('Soap Arc');
    setNewArcMainCharacters([]);
    setInitialProgressionContent('');
    setInitialProgressionSeason('');
    setInitialProgressionEpisode('');
    setInitialProgressionCharacters([]);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Create New Arc</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4}>
            <FormControl isRequired>
              <FormLabel>Title</FormLabel>
              <Input
                value={newArcTitle}
                onChange={(e) => setNewArcTitle(e.target.value)}
                placeholder="Enter arc title"
              />
            </FormControl>

            <FormControl isRequired>
              <FormLabel>Description</FormLabel>
              <Textarea
                value={newArcDescription}
                onChange={(e) => setNewArcDescription(e.target.value)}
                placeholder="Enter arc description"
                rows={4}
              />
            </FormControl>

            <FormControl isRequired>
              <FormLabel>Arc Type</FormLabel>
              <Select
                value={newArcType}
                onChange={(e) => setNewArcType(e.target.value)}
              >
                <option value="Soap Arc">Soap Arc</option>
                <option value="Genre-Specific Arc">Genre-Specific Arc</option>
                <option value="Episodic Arc">Episodic Arc</option>
              </Select>
            </FormControl>

            <FormControl>
              <FormLabel>Main Characters</FormLabel>
              <Box maxH="200px" overflowY="auto" borderWidth={1} borderRadius="md" p={2}>
                <VStack align="start" spacing={1}>
                  {availableCharacters.map(char => (
                    <Checkbox
                      key={char}
                      isChecked={newArcMainCharacters.includes(char)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setNewArcMainCharacters(prev => [...prev, char]);
                        } else {
                          setNewArcMainCharacters(prev => prev.filter(c => c !== char));
                        }
                      }}
                    >
                      <Text fontSize="sm">{char}</Text>
                    </Checkbox>
                  ))}
                </VStack>
              </Box>
              {newArcMainCharacters.length > 0 && (
                <HStack mt={2} flexWrap="wrap" spacing={2}>
                  <Text fontSize="sm" color="gray.500">Selected:</Text>
                  {newArcMainCharacters.map(char => (
                    <Tag 
                      key={char} 
                      size="sm"
                      colorScheme="blue"
                      cursor="pointer"
                      onClick={() => setNewArcMainCharacters(prev => prev.filter(c => c !== char))}
                    >
                      {char} ×
                    </Tag>
                  ))}
                  <Button
                    size="xs"
                    variant="ghost"
                    onClick={() => setNewArcMainCharacters([])}
                  >
                    Clear all
                  </Button>
                </HStack>
              )}
            </FormControl>

            {/* Initial Progression Section */}
            <Box borderWidth={1} borderRadius="md" p={4}>
              <VStack spacing={4}>
                <FormControl isRequired>
                  <FormLabel>Season</FormLabel>
                  <Input
                    placeholder="Enter season number"
                    value={initialProgressionSeason.replace('S', '')}
                    onChange={(e) => {
                      const value = e.target.value.replace(/\D/g, '');
                      if (value) {
                        setInitialProgressionSeason(`S${value}`);
                      } else {
                        setInitialProgressionSeason('');
                      }
                    }}
                  />
                </FormControl>

                <FormControl isRequired>
                  <FormLabel>Episode</FormLabel>
                  <Input
                    placeholder="Enter episode number"
                    value={initialProgressionEpisode.replace('E', '')}
                    onChange={(e) => {
                      const value = e.target.value.replace(/\D/g, '');
                      if (value) {
                        setInitialProgressionEpisode(`E${value}`);
                      } else {
                        setInitialProgressionEpisode('');
                      }
                    }}
                  />
                </FormControl>

                <FormControl isRequired>
                  <FormLabel>Content</FormLabel>
                  <Textarea
                    value={initialProgressionContent}
                    onChange={(e) => setInitialProgressionContent(e.target.value)}
                    placeholder="Enter progression content"
                    rows={4}
                  />
                </FormControl>

                <FormControl>
                  <FormLabel>Interfering Characters</FormLabel>
                  <Box maxH="200px" overflowY="auto" borderWidth={1} borderRadius="md" p={2}>
                    <VStack align="start" spacing={1}>
                      {availableCharacters.map(char => (
                        <Checkbox
                          key={char}
                          isChecked={initialProgressionCharacters.includes(char)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setInitialProgressionCharacters(prev => [...prev, char]);
                            } else {
                              setInitialProgressionCharacters(prev => prev.filter(c => c !== char));
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
            </Box>
          </VStack>
        </ModalBody>

        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button 
            colorScheme="blue" 
            onClick={handleSubmit}
            isDisabled={
              !newArcTitle.trim() || 
              !newArcDescription.trim() || 
              !initialProgressionContent.trim() || 
              !initialProgressionSeason || 
              !initialProgressionEpisode
            }
          >
            Create Arc
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default NewArcModal;