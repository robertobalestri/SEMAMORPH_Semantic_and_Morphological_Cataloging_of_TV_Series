import {
  Box,
  VStack,
  HStack,
  Button,
  Text,
  useColorModeValue,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  IconButton,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  FormControl,
  FormLabel,
  Input,
  useDisclosure,
  Tag,
  TagLabel,
  TagCloseButton,
  Checkbox,
} from '@chakra-ui/react';
import { AddIcon, EditIcon, DeleteIcon } from '@chakra-ui/icons';
import { useState, useEffect } from 'react';

interface Character {
  entity_name: string;
  best_appellation: string;
  series: string;
  appellations: string[];
}

interface CharacterManagerProps {
  series: string;
  onCharacterUpdated?: () => void;
}

const CharacterManager: React.FC<CharacterManagerProps> = ({ series, onCharacterUpdated }) => {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedCharacters, setSelectedCharacters] = useState<Character[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isMergeMode, setIsMergeMode] = useState(false);
  
  // States for character editing/creation
  const [editingCharacter, setEditingCharacter] = useState<Character | null>(null);
  const [newAppellation, setNewAppellation] = useState('');
  const [entityName, setEntityName] = useState('');
  const [bestAppellation, setBestAppellation] = useState('');
  const [appellations, setAppellations] = useState<string[]>([]);
  
  const { isOpen, onOpen, onClose } = useDisclosure();

  const fetchCharacters = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/characters/${series}`);
      if (!response.ok) throw new Error('Failed to fetch characters');
      const data = await response.json();
      setCharacters(data);
    } catch (error) {
      console.error('Error fetching characters:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (series) {
      fetchCharacters();
    }
  }, [series]);

  const handleAddCharacter = () => {
    setEditingCharacter(null);
    setEntityName('');
    setBestAppellation('');
    setAppellations([]);
    onOpen();
  };

  const handleEditCharacter = (character: Character) => {
    setEditingCharacter(character);
    setEntityName(character.entity_name);
    setBestAppellation(character.best_appellation);
    setAppellations(character.appellations);
    onOpen();
  };

  const handleDeleteCharacter = async (entityName: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/characters/${series}/${entityName}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete character');
      fetchCharacters();
      if (onCharacterUpdated) {
        onCharacterUpdated();
      }
    } catch (error) {
      console.error('Error deleting character:', error);
    }
  };

  const handleSubmit = async () => {
    const characterData = {
      entity_name: entityName,
      best_appellation: bestAppellation,
      series,
      appellations,
    };

    try {
      const response = await fetch(`http://localhost:8000/api/characters/${series}`, {
        method: editingCharacter ? 'PATCH' : 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(characterData),
      });

      if (!response.ok) throw new Error('Failed to save character');
      
      fetchCharacters();
      onClose();
    } catch (error) {
      console.error('Error saving character:', error);
    }
  };

  const handleMergeCharacters = async () => {
    if (selectedCharacters.length !== 2) return;

    try {
      const response = await fetch(`http://localhost:8000/api/characters/${series}/merge`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          character1_id: selectedCharacters[0],
          character2_id: selectedCharacters[1],
        }),
      });

      if (!response.ok) throw new Error('Failed to merge characters');
      
      setSelectedCharacters([]);
      setIsMergeMode(false);
      fetchCharacters();
      if (onCharacterUpdated) {
        onCharacterUpdated();
      }
    } catch (error) {
      console.error('Error merging characters:', error);
    }
  };

  return (
    <Box p={4}>
      <VStack spacing={4} align="stretch">
        <HStack justify="space-between">
          <Button
            leftIcon={<AddIcon />}
            colorScheme="green"
            onClick={handleAddCharacter}
            isDisabled={isMergeMode}
          >
            Add Character
          </Button>
          <HStack>
            <Button
              colorScheme={isMergeMode ? "orange" : "gray"}
              onClick={() => {
                setIsMergeMode(!isMergeMode);
                setSelectedCharacters([]);
              }}
            >
              {isMergeMode ? "Cancel Merge" : "Merge Characters"}
            </Button>
            {isMergeMode && (
              <Button
                colorScheme="blue"
                isDisabled={selectedCharacters.length !== 2}
                onClick={handleMergeCharacters}
              >
                Merge Selected ({selectedCharacters.length}/2)
              </Button>
            )}
          </HStack>
        </HStack>

        <Table variant="simple">
          <Thead>
            <Tr>
              {isMergeMode && <Th width="50px"></Th>}
              <Th>Entity Name</Th>
              <Th>Best Appellation</Th>
              <Th>Appellations</Th>
              <Th width="100px">Actions</Th>
            </Tr>
          </Thead>
          <Tbody>
            {characters.map((character) => (
              <Tr key={character.entity_name}>
                {isMergeMode && (
                  <Td>
                    <Checkbox
                      isChecked={selectedCharacters.some(c => c.entity_name === character.entity_name)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          if (selectedCharacters.length < 2) {
                            setSelectedCharacters([...selectedCharacters, character]);
                          }
                        } else {
                          setSelectedCharacters(selectedCharacters.filter(
                            c => c.entity_name !== character.entity_name
                          ));
                        }
                      }}
                    />
                  </Td>
                )}
                <Td>{character.entity_name}</Td>
                <Td>{character.best_appellation}</Td>
                <Td>
                  <HStack spacing={2} wrap="wrap">
                    {character.appellations.map((appellation, index) => (
                      <Tag key={index} size="sm" colorScheme="blue">
                        {appellation}
                      </Tag>
                    ))}
                  </HStack>
                </Td>
                <Td>
                  <HStack spacing={2}>
                    <IconButton
                      aria-label="Edit character"
                      icon={<EditIcon />}
                      size="sm"
                      onClick={() => handleEditCharacter(character)}
                      isDisabled={isMergeMode}
                    />
                    <IconButton
                      aria-label="Delete character"
                      icon={<DeleteIcon />}
                      size="sm"
                      colorScheme="red"
                      onClick={() => handleDeleteCharacter(character.entity_name)}
                      isDisabled={isMergeMode}
                    />
                  </HStack>
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>

        {/* Edit/Create Character Modal */}
        <Modal isOpen={isOpen} onClose={onClose} size="xl">
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>
              {editingCharacter ? 'Edit Character' : 'Add Character'}
            </ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <VStack spacing={4}>
                <FormControl isRequired>
                  <FormLabel>Entity Name</FormLabel>
                  <Input
                    value={entityName}
                    onChange={(e) => setEntityName(e.target.value)}
                    placeholder="Enter entity name"
                    isReadOnly={!!editingCharacter}
                  />
                </FormControl>

                <FormControl isRequired>
                  <FormLabel>Best Appellation</FormLabel>
                  <Input
                    value={bestAppellation}
                    onChange={(e) => setBestAppellation(e.target.value)}
                    placeholder="Enter best appellation"
                  />
                </FormControl>

                <FormControl>
                  <FormLabel>Appellations</FormLabel>
                  <VStack align="stretch" spacing={2}>
                    <HStack>
                      <Input
                        value={newAppellation}
                        onChange={(e) => setNewAppellation(e.target.value)}
                        placeholder="Enter new appellation"
                      />
                      <Button
                        onClick={() => {
                          if (newAppellation.trim()) {
                            setAppellations([...appellations, newAppellation.trim()]);
                            setNewAppellation('');
                          }
                        }}
                      >
                        Add
                      </Button>
                    </HStack>
                    <Box>
                      {appellations.map((appellation, index) => (
                        <Tag
                          key={index}
                          size="md"
                          borderRadius="full"
                          variant="solid"
                          colorScheme="blue"
                          m={1}
                        >
                          <TagLabel>{appellation}</TagLabel>
                          <TagCloseButton
                            onClick={() => {
                              setAppellations(appellations.filter((_, i) => i !== index));
                            }}
                          />
                        </Tag>
                      ))}
                    </Box>
                  </VStack>
                </FormControl>
              </VStack>
            </ModalBody>
            <ModalFooter>
              <Button variant="ghost" mr={3} onClick={onClose}>
                Cancel
              </Button>
              <Button
                colorScheme="blue"
                onClick={handleSubmit}
                isDisabled={!entityName || !bestAppellation || appellations.length === 0}
              >
                {editingCharacter ? 'Save' : 'Create'}
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </VStack>
    </Box>
  );
};

export default CharacterManager; 