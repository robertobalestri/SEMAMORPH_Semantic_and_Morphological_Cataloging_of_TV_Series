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
} from '@chakra-ui/react';
import { SearchIcon } from '@chakra-ui/icons';
import { useState, useEffect, useMemo } from 'react';

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
  distance?: number;
}

interface VectorStoreExplorerProps {
  series: string;
}

const VectorStoreExplorer: React.FC<VectorStoreExplorerProps> = ({ series }) => {
  const [entries, setEntries] = useState<VectorStoreEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const fetchEntries = async (query?: string) => {
    try {
      setIsLoading(true);
      const url = `http://localhost:8000/api/vector-store/${series}${query ? `?query=${encodeURIComponent(query)}` : ''}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error('Failed to fetch vector store entries');
      }

      const data = await response.json();
      setEntries(data);
    } catch (error) {
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
    if (series) {
      fetchEntries();
    }
  }, [series]);

  const handleSearch = () => {
    if (searchQuery.trim()) {
      fetchEntries(searchQuery);
    }
  };

  const handleReset = () => {
    setSearchQuery('');
    fetchEntries();
  };

  const sortedEntries = useMemo(() => {
    // First, group entries by arc
    const groupedEntries = entries.reduce((acc, entry) => {
      const arcId = entry.metadata.doc_type === 'main' ? 
        entry.metadata.id : 
        entry.metadata.main_arc_id;
      
      if (!arcId) return acc;
      
      if (!acc[arcId]) {
        acc[arcId] = [];
      }
      acc[arcId].push(entry);
      return acc;
    }, {} as Record<string, VectorStoreEntry[]>);

    // Sort entries within each group and flatten
    return Object.values(groupedEntries).flatMap(group => {
      // Sort entries within the group
      return group.sort((a, b) => {
        // Main doc always comes first in its group
        if (a.metadata.doc_type !== b.metadata.doc_type) {
          return a.metadata.doc_type === 'main' ? -1 : 1;
        }

        // If both are progressions, sort by ordinal position
        if (a.metadata.doc_type === 'progression' && b.metadata.doc_type === 'progression') {
          const posA = a.metadata.ordinal_position || 0;
          const posB = b.metadata.ordinal_position || 0;
          return posA - posB;
        }

        return 0;
      });
    });
  }, [entries]);

  return (
    <Box p={4}>
      <VStack spacing={4} align="stretch">
        <HStack>
          <Input
            placeholder="Search vector store..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <Button
            leftIcon={<SearchIcon />}
            onClick={handleSearch}
            isLoading={isLoading}
          >
            Search
          </Button>
          <Button onClick={handleReset} variant="outline">
            Reset
          </Button>
        </HStack>

        {isLoading ? (
          <Box textAlign="center" py={4}>
            <Spinner />
          </Box>
        ) : (
          <Accordion allowMultiple>
            {sortedEntries.map((entry) => (
              <AccordionItem key={entry.id}>
                <h2>
                  <AccordionButton>
                    <Box flex="1" textAlign="left">
                      <HStack>
                        <Badge colorScheme={entry.metadata.doc_type === 'main' ? "blue" : "green"}>
                          {entry.metadata.doc_type}
                        </Badge>
                        <Text fontWeight="bold">
                          {entry.metadata.doc_type === 'main' 
                            ? `${entry.metadata.title} | MC: ${entry.metadata.main_characters || 'None'}`
                            : entry.metadata.progression_title || entry.metadata.title ||
                              `${entry.metadata.parent_arc_title} - S${entry.metadata.season?.replace('S', '')}-E${entry.metadata.episode?.replace('E', '')}` +
                              `${entry.metadata.interfering_characters ? ` | ${entry.metadata.interfering_characters}` : ''}`
                          }
                        </Text>
                      </HStack>
                    </Box>
                    <AccordionIcon />
                  </AccordionButton>
                </h2>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={3}>
                    <Box>
                      <Text fontWeight="bold">Content:</Text>
                      <Text>{entry.content}</Text>
                    </Box>
                    <Divider />
                    <Box>
                      <Text fontWeight="bold">Metadata:</Text>
                      <VStack align="stretch" spacing={1}>
                        <Text>Arc Type: {entry.metadata.arc_type}</Text>
                        {entry.metadata.doc_type === 'main' ? (
                          entry.metadata.main_characters && (
                            <Text>Main Characters: {entry.metadata.main_characters}</Text>
                          )
                        ) : (
                          <>
                            <Text>Parent Arc: {entry.metadata.parent_arc_title}</Text>
                            <Text>Episode: S{entry.metadata.season?.replace('S', '')}-E{entry.metadata.episode?.replace('E', '')}</Text>
                            <Text>Interfering Characters: {entry.metadata.interfering_characters || 'None'}</Text>
                          </>
                        )}
                        {entry.metadata.ordinal_position !== undefined && (
                          <Text>Position: {entry.metadata.ordinal_position}</Text>
                        )}
                        <Text>ID: {entry.id}</Text>
                        {entry.metadata.main_arc_id && (
                          <Text>Main Arc ID: {entry.metadata.main_arc_id}</Text>
                        )}
                      </VStack>
                    </Box>
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            ))}
          </Accordion>
        )}
      </VStack>
    </Box>
  );
};

export default VectorStoreExplorer; 