import React, { useState } from 'react';
import {
  Box,
  Text,
  HStack,
  VStack,
  Badge,
  Input,
  Button,
  Card,
  CardBody,
  useColorModeValue,
  Select,
  FormControl,
  FormLabel,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Spinner,
  Alert,
  AlertIcon,
  SimpleGrid,
} from '@chakra-ui/react';
import { FiSearch, FiClock, FiUsers, FiTarget } from 'react-icons/fi';

interface EventSearchResult {
  event_id: string;
  content: string;
  similarity_score: number;
  series: string;
  season: string;
  episode: string;
  start_timestamp?: number;
  end_timestamp?: number;
  confidence_score?: number;
  characters: string[];
  arc_title?: string;
  extraction_method?: string;
}

interface ProgressionSearchResult {
  progression_id: string;
  content: string;
  similarity_score: number;
  series: string;
  season: string;
  episode: string;
  arc_title?: string;
  arc_type?: string;
  interfering_characters: string[];
}

interface VectorSearchProps {
  defaultSeries?: string;
  defaultSeason?: string;
  defaultEpisode?: string;
  onEventSelect?: (event: EventSearchResult) => void;
  onProgressionSelect?: (progression: ProgressionSearchResult) => void;
}

export const VectorSearch: React.FC<VectorSearchProps> = ({
  defaultSeries = '',
  defaultSeason = '',
  defaultEpisode = '',
  onEventSelect,
  onProgressionSelect
}) => {
  const [query, setQuery] = useState('');
  const [series, setSeries] = useState(defaultSeries);
  const [season, setSeason] = useState(defaultSeason);
  const [episode, setEpisode] = useState(defaultEpisode);
  const [minConfidence, setMinConfidence] = useState<number>(0);
  const [startTime, setStartTime] = useState<number | undefined>();
  const [endTime, setEndTime] = useState<number | undefined>();
  const [numResults, setNumResults] = useState(10);
  
  const [eventResults, setEventResults] = useState<EventSearchResult[]>([]);
  const [progressionResults, setProgressionResults] = useState<ProgressionSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchType, setSearchType] = useState<'events' | 'progressions'>('events');

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    try {
      setLoading(true);
      setError(null);

      if (searchType === 'events') {
        // Search events
        const params = new URLSearchParams({
          query: query.trim(),
          n_results: numResults.toString()
        });

        if (series) params.append('series', series);
        if (season) params.append('season', season);
        if (episode) params.append('episode', episode);
        if (minConfidence > 0) params.append('min_confidence', minConfidence.toString());
        if (startTime !== undefined) params.append('start_time', startTime.toString());
        if (endTime !== undefined) params.append('end_time', endTime.toString());

        const response = await fetch(`/api/search/similar-events?${params}`);
        if (!response.ok) throw new Error(`Search failed: ${response.statusText}`);
        
        const results = await response.json();
        setEventResults(results);
        setProgressionResults([]);
      } else {
        // Search progressions
        const params = new URLSearchParams({
          query: query.trim(),
          n_results: numResults.toString()
        });

        if (series) params.append('series', series);
        if (season) params.append('season', season);
        if (episode) params.append('episode', episode);

        const response = await fetch(`/api/search/similar-progressions?${params}`);
        if (!response.ok) throw new Error(`Search failed: ${response.statusText}`);
        
        const results = await response.json();
        setProgressionResults(results);
        setEventResults([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <VStack spacing={6} align="stretch">
      {/* Search Form */}
      <Card>
        <CardBody>
          <VStack spacing={4}>
            <HStack spacing={4} w="full">
              <FormControl flex={2}>
                <FormLabel>Search Query</FormLabel>
                <Input
                  placeholder="Describe the narrative event or progression..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                />
              </FormControl>
              <FormControl maxW="150px">
                <FormLabel>Search Type</FormLabel>
                <Select value={searchType} onChange={(e) => setSearchType(e.target.value as any)}>
                  <option value="events">Events</option>
                  <option value="progressions">Progressions</option>
                </Select>
              </FormControl>
              <Button
                colorScheme="blue"
                leftIcon={<FiSearch />}
                onClick={handleSearch}
                isLoading={loading}
                alignSelf="end"
              >
                Search
              </Button>
            </HStack>

            {/* Filters */}
            <SimpleGrid columns={{ base: 2, md: 4, lg: 6 }} spacing={4} w="full">
              <FormControl>
                <FormLabel>Series</FormLabel>
                <Input
                  placeholder="e.g., GA"
                  value={series}
                  onChange={(e) => setSeries(e.target.value)}
                />
              </FormControl>
              <FormControl>
                <FormLabel>Season</FormLabel>
                <Input
                  placeholder="e.g., S01"
                  value={season}
                  onChange={(e) => setSeason(e.target.value)}
                />
              </FormControl>
              <FormControl>
                <FormLabel>Episode</FormLabel>
                <Input
                  placeholder="e.g., E01"
                  value={episode}
                  onChange={(e) => setEpisode(e.target.value)}
                />
              </FormControl>
              <FormControl>
                <FormLabel>Results</FormLabel>
                <NumberInput value={numResults} onChange={(_, val) => setNumResults(val || 10)} min={1} max={50}>
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>

              {searchType === 'events' && (
                <>
                  <FormControl>
                    <FormLabel>Min Confidence</FormLabel>
                    <NumberInput 
                      value={minConfidence} 
                      onChange={(_, val) => setMinConfidence(val || 0)} 
                      min={0} 
                      max={1} 
                      step={0.1}
                    >
                      <NumberInputField />
                      <NumberInputStepper>
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                      </NumberInputStepper>
                    </NumberInput>
                  </FormControl>
                  <FormControl>
                    <FormLabel>Time Range (s)</FormLabel>
                    <HStack>
                      <NumberInput 
                        value={startTime} 
                        onChange={(_, val) => setStartTime(val)} 
                        min={0}
                      >
                        <NumberInputField placeholder="Start" />
                      </NumberInput>
                      <NumberInput 
                        value={endTime} 
                        onChange={(_, val) => setEndTime(val)} 
                        min={startTime || 0}
                      >
                        <NumberInputField placeholder="End" />
                      </NumberInput>
                    </HStack>
                  </FormControl>
                </>
              )}
            </SimpleGrid>
          </VStack>
        </CardBody>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert status="error">
          <AlertIcon />
          {error}
        </Alert>
      )}

      {/* Loading */}
      {loading && (
        <Card>
          <CardBody>
            <VStack spacing={4}>
              <Spinner size="xl" />
              <Text>Searching for similar {searchType}...</Text>
            </VStack>
          </CardBody>
        </Card>
      )}

      {/* Results */}
      {!loading && (eventResults.length > 0 || progressionResults.length > 0) && (
        <Card>
          <CardBody>
            <Text fontSize="lg" fontWeight="bold" mb={4}>
              Search Results ({searchType === 'events' ? eventResults.length : progressionResults.length})
            </Text>

            {searchType === 'events' && (
              <VStack spacing={3} align="stretch">
                {eventResults.map((event, index) => (
                  <Box
                    key={event.event_id}
                    p={4}
                    bg={useColorModeValue('gray.50', 'gray.700')}
                    borderRadius="md"
                    cursor="pointer"
                    onClick={() => onEventSelect?.(event)}
                    _hover={{ bg: useColorModeValue('gray.100', 'gray.600') }}
                    borderLeft="4px solid"
                    borderLeftColor={`hsl(${120 * event.similarity_score}, 50%, 50%)`}
                  >
                    <VStack align="start" spacing={2}>
                      <HStack justify="space-between" w="full">
                        <Text fontSize="md" fontWeight="medium">{event.content}</Text>
                        <Badge colorScheme="green">{(event.similarity_score * 100).toFixed(1)}%</Badge>
                      </HStack>
                      
                      <HStack spacing={4} wrap="wrap">
                        <HStack spacing={1}>
                          <FiClock />
                          <Text fontSize="sm">
                            {event.start_timestamp !== undefined ? formatTime(event.start_timestamp) : 'No timestamp'}
                            {event.end_timestamp !== undefined && ` - ${formatTime(event.end_timestamp)}`}
                          </Text>
                        </HStack>
                        
                        {event.characters.length > 0 && (
                          <HStack spacing={1}>
                            <FiUsers />
                            <Text fontSize="sm">{event.characters.join(', ')}</Text>
                          </HStack>
                        )}
                        
                        {event.confidence_score !== undefined && (
                          <HStack spacing={1}>
                            <FiTarget />
                            <Text fontSize="sm">Confidence: {(event.confidence_score * 100).toFixed(0)}%</Text>
                          </HStack>
                        )}
                      </HStack>
                      
                      <HStack spacing={2} wrap="wrap">
                        <Badge colorScheme="blue">{event.series} {event.season} {event.episode}</Badge>
                        {event.arc_title && <Badge variant="outline">{event.arc_title}</Badge>}
                        {event.extraction_method && <Badge colorScheme="purple" variant="outline">{event.extraction_method}</Badge>}
                      </HStack>
                    </VStack>
                  </Box>
                ))}
              </VStack>
            )}

            {searchType === 'progressions' && (
              <VStack spacing={3} align="stretch">
                {progressionResults.map((progression, index) => (
                  <Box
                    key={progression.progression_id}
                    p={4}
                    bg={useColorModeValue('gray.50', 'gray.700')}
                    borderRadius="md"
                    cursor="pointer"
                    onClick={() => onProgressionSelect?.(progression)}
                    _hover={{ bg: useColorModeValue('gray.100', 'gray.600') }}
                    borderLeft="4px solid"
                    borderLeftColor={`hsl(${120 * progression.similarity_score}, 50%, 50%)`}
                  >
                    <VStack align="start" spacing={2}>
                      <HStack justify="space-between" w="full">
                        <Text fontSize="md" fontWeight="medium">{progression.content}</Text>
                        <Badge colorScheme="green">{(progression.similarity_score * 100).toFixed(1)}%</Badge>
                      </HStack>
                      
                      {progression.interfering_characters.length > 0 && (
                        <HStack spacing={1}>
                          <FiUsers />
                          <Text fontSize="sm">Characters: {progression.interfering_characters.join(', ')}</Text>
                        </HStack>
                      )}
                      
                      <HStack spacing={2} wrap="wrap">
                        <Badge colorScheme="blue">{progression.series} {progression.season} {progression.episode}</Badge>
                        {progression.arc_title && <Badge variant="outline">{progression.arc_title}</Badge>}
                        {progression.arc_type && <Badge colorScheme="purple">{progression.arc_type}</Badge>}
                      </HStack>
                    </VStack>
                  </Box>
                ))}
              </VStack>
            )}
          </CardBody>
        </Card>
      )}

      {/* No Results */}
      {!loading && query && eventResults.length === 0 && progressionResults.length === 0 && (
        <Alert status="info">
          <AlertIcon />
          No {searchType} found for query: "{query}"
        </Alert>
      )}
    </VStack>
  );
};
