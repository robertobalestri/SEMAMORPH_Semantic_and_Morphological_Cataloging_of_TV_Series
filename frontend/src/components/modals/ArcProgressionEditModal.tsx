import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Button,
  FormControl,
  FormLabel,
  Textarea,
  VStack,
  Box,
  Checkbox,
  Text,
  useColorModeValue,
} from '@chakra-ui/react';
import type { ProgressionMapping } from '@/architecture/types';
import { DeleteIcon } from '@chakra-ui/icons';

interface ArcProgressionEditModalProps {
  isOpen: boolean;
  onClose: () => void;
  progression: ProgressionMapping | null;
  onSave: (updatedProgression: ProgressionMapping) => void;
  onDelete?: () => void;
  availableCharacters: string[];
}

export const ArcProgressionEditModal: React.FC<ArcProgressionEditModalProps> = ({
  isOpen,
  onClose,
  progression,
  onSave,
  onDelete,
  availableCharacters,
}) => {
  const [content, setContent] = useState('');
  const [interferingCharacters, setInterferingCharacters] = useState<string[]>([]);
  const bgColor = useColorModeValue('gray.50', 'gray.700');

  useEffect(() => {
    if (progression) {
      setContent(progression.content);
      setInterferingCharacters(progression.interfering_characters || []);
    }
  }, [progression]);

  const handleSave = () => {
    if (progression) {
      onSave({
        ...progression,
        content,
        interfering_characters: interferingCharacters,
      });
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Edit Progression</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4}>
            <FormControl>
              <FormLabel>Content</FormLabel>
              <Textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                rows={6}
              />
            </FormControl>

            <FormControl>
              <FormLabel>Interfering Characters</FormLabel>
              <Box 
                maxH="200px" 
                overflowY="auto" 
                borderWidth={1} 
                borderRadius="md" 
                p={2}
                bg={bgColor}
              >
                <VStack align="start" spacing={1}>
                  {availableCharacters.map(char => (
                    <Checkbox
                      key={char}
                      isChecked={interferingCharacters.includes(char)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setInterferingCharacters([...interferingCharacters, char]);
                        } else {
                          setInterferingCharacters(interferingCharacters.filter(c => c !== char));
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
          {onDelete && (
            <Button
              leftIcon={<DeleteIcon />}
              colorScheme="red"
              variant="ghost"
              mr="auto"
              onClick={onDelete}
            >
              Delete
            </Button>
          )}
          <Button variant="ghost" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button 
            colorScheme="blue" 
            onClick={handleSave}
            isDisabled={!content.trim()}
          >
            Save Changes
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}; 