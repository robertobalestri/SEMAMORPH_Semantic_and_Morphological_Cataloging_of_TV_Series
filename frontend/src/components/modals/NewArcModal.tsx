import React, { useState } from 'react';
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
  HStack,
  Tag,
  TagLabel,
  TagCloseButton,
} from '@chakra-ui/react';
import { ArcType } from '@/architecture/types';
import type { NarrativeArc } from '@/architecture/types';

interface NewArcModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (arcData: Partial<NarrativeArc>) => void;
  availableCharacters: string[];
  series: string;
}

export const NewArcModal: React.FC<NewArcModalProps> = ({
  isOpen,
  onClose,
  onSubmit,
  availableCharacters,
  series,
}) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [arcType, setArcType] = useState<ArcType>(ArcType.SoapArc);
  const [mainCharacters, setMainCharacters] = useState<string[]>([]);
  const [progressionContent, setProgressionContent] = useState('');
  const [progressionSeason, setProgressionSeason] = useState('');
  const [progressionEpisode, setProgressionEpisode] = useState('');
  const [interferingCharacters, setInterferingCharacters] = useState<string[]>([]);

  const handleSubmit = () => {
    const arcData: Partial<NarrativeArc> = {
      title: title.trim(),
      description: description.trim(),
      arc_type: arcType,
      main_characters: mainCharacters.join(';'),
      series,
      initial_progression: {
        content: progressionContent,
        season: `S${progressionSeason.padStart(2, '0')}`,
        episode: `E${progressionEpisode.padStart(2, '0')}`,
        interfering_characters: interferingCharacters.join(';')
      }
    };

    onSubmit(arcData);
    resetForm();
  };

  const resetForm = () => {
    setTitle('');
    setDescription('');
    setArcType(ArcType.SoapArc);
    setMainCharacters([]);
    setProgressionContent('');
    setProgressionSeason('');
    setProgressionEpisode('');
    setInterferingCharacters([]);
  };

  const isFormValid = (): boolean => {
    return (
      title.trim() !== '' &&
      description.trim() !== '' &&
      progressionContent.trim() !== '' &&
      progressionSeason !== '' &&
      progressionEpisode !== ''
    );
  };

  const handleCharacterSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedChar = e.target.value;
    if (selectedChar && !mainCharacters.includes(selectedChar)) {
      setMainCharacters([...mainCharacters, selectedChar]);
    }
  };

  const handleInterferingCharacterSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedChar = e.target.value;
    if (selectedChar && !interferingCharacters.includes(selectedChar)) {
      setInterferingCharacters([...interferingCharacters, selectedChar]);
    }
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
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Enter arc title"
              />
            </FormControl>

            <FormControl isRequired>
              <FormLabel>Description</FormLabel>
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Enter arc description"
                rows={4}
              />
            </FormControl>

            <FormControl isRequired>
              <FormLabel>Arc Type</FormLabel>
              <Select
                value={arcType}
                onChange={(e) => setArcType(e.target.value as ArcType)}
              >
                {Object.values(ArcType).map((type) => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </Select>
            </FormControl>

            <FormControl>
              <FormLabel>Main Characters</FormLabel>
              <Select
                placeholder="Select characters"
                onChange={handleCharacterSelect}
                value=""
              >
                {availableCharacters
                  .filter(char => !mainCharacters.includes(char))
                  .map(char => (
                    <option key={char} value={char}>
                      {char}
                    </option>
                  ))}
              </Select>
              {mainCharacters.length > 0 && (
                <Box mt={2}>
                  <HStack spacing={2} wrap="wrap">
                    {mainCharacters.map(char => (
                      <Tag
                        key={char}
                        size="md"
                        variant="solid"
                        colorScheme="blue"
                        cursor="pointer"
                      >
                        <TagLabel>{char}</TagLabel>
                        <TagCloseButton 
                          onClick={() => setMainCharacters(prev => 
                            prev.filter(c => c !== char)
                          )}
                        />
                      </Tag>
                    ))}
                  </HStack>
                </Box>
              )}
            </FormControl>

            <Box borderWidth={1} borderRadius="md" p={4}>
              <VStack spacing={4}>
                <FormControl isRequired>
                  <FormLabel>Season</FormLabel>
                  <Input
                    placeholder="Enter season number (e.g., 1)"
                    value={progressionSeason}
                    onChange={(e) => {
                      const value = e.target.value.replace(/\D/g, '');
                      setProgressionSeason(value);
                    }}
                  />
                </FormControl>

                <FormControl isRequired>
                  <FormLabel>Episode</FormLabel>
                  <Input
                    placeholder="Enter episode number (e.g., 1)"
                    value={progressionEpisode}
                    onChange={(e) => {
                      const value = e.target.value.replace(/\D/g, '');
                      setProgressionEpisode(value);
                    }}
                  />
                </FormControl>

                <FormControl isRequired>
                  <FormLabel>Content</FormLabel>
                  <Textarea
                    value={progressionContent}
                    onChange={(e) => setProgressionContent(e.target.value)}
                    placeholder="Enter progression content"
                    rows={4}
                  />
                </FormControl>

                <FormControl>
                  <FormLabel>Interfering Characters</FormLabel>
                  <Select
                    placeholder="Select characters"
                    onChange={handleInterferingCharacterSelect}
                    value=""
                  >
                    {availableCharacters.map(char => (
                      <option key={char} value={char}>
                        {char}
                      </option>
                    ))}
                  </Select>
                  {interferingCharacters.length > 0 && (
                    <HStack mt={2} flexWrap="wrap" spacing={2}>
                      {interferingCharacters.map(char => (
                        <Tag 
                          key={char} 
                          size="sm"
                          colorScheme="purple"
                          cursor="pointer"
                          onClick={() => setInterferingCharacters(prev => prev.filter(c => c !== char))}
                        >
                          <TagLabel>{char}</TagLabel>
                          <TagCloseButton />
                        </Tag>
                      ))}
                      <Button
                        size="xs"
                        variant="ghost"
                        onClick={() => setInterferingCharacters([])}
                      >
                        Clear all
                      </Button>
                    </HStack>
                  )}
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
            isDisabled={!isFormValid()}
          >
            Create Arc
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}; 