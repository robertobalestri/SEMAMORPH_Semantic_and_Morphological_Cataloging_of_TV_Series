import React, { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Select,
  Button,
  Text,
  Progress,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Heading,
  Card,
  CardHeader,
  CardBody,
  Badge,
  List,
  ListItem,
  ListIcon,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Code,
  Divider,
  Spinner,
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  Grid,
  GridItem,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
} from '@chakra-ui/react';
import { CheckCircleIcon, TimeIcon, InfoIcon, WarningIcon } from '@chakra-ui/icons';
import { ApiClient } from '@/services/api/ApiClient';
import type { 
  Episode, 
  RecapGenerationRequest, 
  RecapGenerationJob, 
  RecapGenerationResponse,
  RecapEvent,
  RecapClip,
  RecapQuery
} from '@/architecture/types';

interface RecapGeneratorProps {
  series: string;
  episodes: Episode[];
}

export const RecapGenerator: React.FC<RecapGeneratorProps> = ({ series, episodes }) => {
  const [selectedSeason, setSelectedSeason] = useState<string>('');
  const [selectedEpisode, setSelectedEpisode] = useState<string>('');
  const [currentJob, setCurrentJob] = useState<RecapGenerationJob | null>(null);
  const [jobHistory, setJobHistory] = useState<RecapGenerationJob[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();
  const api = new ApiClient();
  
  const { isOpen: isVideoOpen, onOpen: onVideoOpen, onClose: onVideoClose } = useDisclosure();
  const { isOpen: isDetailsOpen, onOpen: onDetailsOpen, onClose: onDetailsClose } = useDisclosure();

  // Get unique seasons from episodes
  const seasons = Array.from(new Set(episodes.map(ep => ep.season))).sort();
  
  // Get episodes for selected season
  const seasonEpisodes = episodes
    .filter(ep => ep.season === selectedSeason)
    .sort((a, b) => a.episode.localeCompare(b.episode));

  useEffect(() => {
    loadJobHistory();
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    
    if (currentJob && currentJob.status === 'running') {
      interval = setInterval(async () => {
        try {
          const response = await api.request<RecapGenerationJob>(`/recap-generation/jobs/${currentJob.job_id}`);
          if (!response.error && response.data) {
            setCurrentJob(response.data);
            
            if (response.data.status !== 'running') {
              if (interval) clearInterval(interval);
              loadJobHistory();
              
              if (response.data.status === 'completed') {
                toast({
                  title: 'Recap Generated Successfully!',
                  description: 'Your "Previously On" recap is ready to view.',
                  status: 'success',
                  duration: 5000,
                  isClosable: true,
                });
              } else if (response.data.status === 'failed') {
                toast({
                  title: 'Recap Generation Failed',
                  description: response.data.error_message || 'Unknown error occurred',
                  status: 'error',
                  duration: 8000,
                  isClosable: true,
                });
              }
            }
          }
        } catch (error) {
          console.error('Error polling job status:', error);
        }
      }, 2000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [currentJob]);

  const loadJobHistory = async () => {
    try {
      const response = await api.request<RecapGenerationJob[]>('/recap-generation/jobs');
      if (!response.error && response.data) {
        setJobHistory(response.data.filter(job => job.series === series));
      }
    } catch (error) {
      console.error('Error loading job history:', error);
    }
  };

  const startRecapGeneration = async () => {
    if (!selectedSeason || !selectedEpisode) {
      toast({
        title: 'Missing Selection',
        description: 'Please select both season and episode.',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsLoading(true);
    
    try {
      const request: RecapGenerationRequest = {
        series,
        season: selectedSeason,
        episode: selectedEpisode
      };
      
      const response = await api.request<RecapGenerationJob>('/recap-generation/start', {
        method: 'POST',
        body: JSON.stringify(request)
      });
      
      if (!response.error && response.data) {
        setCurrentJob(response.data);
        toast({
          title: 'Recap Generation Started',
          description: 'Your recap is being generated. This may take a few minutes.',
          status: 'info',
          duration: 4000,
          isClosable: true,
        });
      } else {
        throw new Error(response.error || 'Failed to start recap generation');
      }
    } catch (error) {
      toast({
        title: 'Error Starting Recap Generation',
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const canGenerate = selectedSeason && selectedEpisode && !currentJob?.status.includes('running') && !isLoading;

  return (
    <Box p={6}>
      <VStack spacing={6} align="stretch">
        <Heading size="lg">Recap Generator</Heading>
        
        <Text color="gray.600">
          Generate "Previously On" recaps for processed episodes. Select an episode and click generate to create a video recap with key moments from previous episodes.
        </Text>

        {/* Episode Selection */}
        <Card>
          <CardHeader>
            <Heading size="md">Episode Selection</Heading>
          </CardHeader>
          <CardBody>
            <Grid templateColumns="repeat(2, 1fr)" gap={4}>
              <GridItem>
                <Text mb={2} fontWeight="medium">Season</Text>
                <Select
                  placeholder="Select season"
                  value={selectedSeason}
                  onChange={(e) => {
                    setSelectedSeason(e.target.value);
                    setSelectedEpisode(''); // Reset episode when season changes
                  }}
                >
                  {seasons.map(season => (
                    <option key={season} value={season}>
                      {season}
                    </option>
                  ))}
                </Select>
              </GridItem>
              
              <GridItem>
                <Text mb={2} fontWeight="medium">Episode</Text>
                <Select
                  placeholder="Select episode"
                  value={selectedEpisode}
                  onChange={(e) => setSelectedEpisode(e.target.value)}
                  isDisabled={!selectedSeason}
                >
                  {seasonEpisodes.map(episode => (
                    <option key={episode.episode} value={episode.episode}>
                      {episode.episode} - {episode.title || 'Untitled'}
                    </option>
                  ))}
                </Select>
              </GridItem>
            </Grid>
            
            <Box mt={4}>
              <Button
                colorScheme="blue"
                size="lg"
                onClick={startRecapGeneration}
                isDisabled={!canGenerate}
                isLoading={isLoading}
                leftIcon={<TimeIcon />}
              >
                Generate Recap
              </Button>
            </Box>
          </CardBody>
        </Card>

        {/* Current Job Status */}
        {currentJob && (
          <Card>
            <CardHeader>
              <HStack justify="space-between">
                <Heading size="md">Current Generation</Heading>
                <Badge 
                  colorScheme={
                    currentJob.status === 'completed' ? 'green' : 
                    currentJob.status === 'failed' ? 'red' : 'blue'
                  }
                >
                  {currentJob.status.toUpperCase()}
                </Badge>
              </HStack>
            </CardHeader>
            <CardBody>
              <VStack spacing={4} align="stretch">
                <Box>
                  <Text fontWeight="medium" mb={2}>
                    {series} {currentJob.season} {currentJob.episode}
                  </Text>
                  <Text color="gray.600">{currentJob.current_step}</Text>
                </Box>
                
                <Progress 
                  value={currentJob.progress * 100} 
                  colorScheme={currentJob.status === 'failed' ? 'red' : 'blue'}
                  hasStripe={currentJob.status === 'running'}
                  isAnimated={currentJob.status === 'running'}
                />
                
                <Text fontSize="sm" color="gray.500">
                  {Math.round(currentJob.progress * 100)}% complete
                </Text>

                {currentJob.status === 'completed' && currentJob.result && (
                  <VStack spacing={3} align="stretch">
                    <Alert status="success">
                      <AlertIcon />
                      <Box>
                        <AlertTitle>Recap Generated Successfully!</AlertTitle>
                        <AlertDescription>
                          Duration: {currentJob.result.total_duration.toFixed(1)}s • 
                          Events: {currentJob.result.events.length} • 
                          Clips: {currentJob.result.clips.length}
                        </AlertDescription>
                      </Box>
                    </Alert>
                    
                    <HStack>
                      <Button colorScheme="green" onClick={onVideoOpen}>
                        View Video
                      </Button>
                      <Button variant="outline" onClick={onDetailsOpen}>
                        View Details
                      </Button>
                    </HStack>
                  </VStack>
                )}

                {currentJob.status === 'failed' && (
                  <Alert status="error">
                    <AlertIcon />
                    <Box>
                      <AlertTitle>Generation Failed</AlertTitle>
                      <AlertDescription>
                        {currentJob.error_message || 'Unknown error occurred'}
                      </AlertDescription>
                    </Box>
                  </Alert>
                )}
              </VStack>
            </CardBody>
          </Card>
        )}

        {/* Job History */}
        {jobHistory.length > 0 && (
          <Card>
            <CardHeader>
              <Heading size="md">Recent Generations</Heading>
            </CardHeader>
            <CardBody>
              <VStack spacing={3} align="stretch">
                {jobHistory.slice(-5).reverse().map(job => (
                  <Box key={job.job_id} p={3} borderWidth={1} borderRadius="md">
                    <HStack justify="space-between">
                      <VStack align="start" spacing={1}>
                        <Text fontWeight="medium">
                          {job.series} {job.season} {job.episode}
                        </Text>
                        <Text fontSize="sm" color="gray.600">
                          {new Date(job.created_at).toLocaleString()}
                        </Text>
                      </VStack>
                      <HStack>
                        <Badge 
                          colorScheme={
                            job.status === 'completed' ? 'green' : 
                            job.status === 'failed' ? 'red' : 'blue'
                          }
                        >
                          {job.status}
                        </Badge>
                        {job.status === 'completed' && job.result && (
                          <Button 
                            size="sm" 
                            variant="outline"
                            onClick={() => {
                              setCurrentJob(job);
                              onDetailsOpen();
                            }}
                          >
                            View
                          </Button>
                        )}
                      </HStack>
                    </HStack>
                  </Box>
                ))}
              </VStack>
            </CardBody>
          </Card>
        )}
      </VStack>

      {/* Video Modal */}
      <Modal isOpen={isVideoOpen} onClose={onVideoClose} size="6xl">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Recap Video</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            {currentJob?.result?.video_path ? (
              <Box>
                <video
                  controls
                  style={{ width: '100%', maxHeight: '70vh' }}
                  src={`/api/static/${currentJob.result.video_path}`}
                >
                  Your browser does not support the video tag.
                </video>
                <Text mt={2} fontSize="sm" color="gray.600">
                  Duration: {currentJob.result.total_duration.toFixed(1)}s
                </Text>
              </Box>
            ) : (
              <Text>Video not available</Text>
            )}
          </ModalBody>
        </ModalContent>
      </Modal>

      {/* Details Modal */}
      <Modal isOpen={isDetailsOpen} onClose={onDetailsClose} size="6xl">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Recap Details</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            {currentJob?.result && (
              <RecapDetails result={currentJob.result} />
            )}
          </ModalBody>
        </ModalContent>
      </Modal>
    </Box>
  );
};

interface RecapDetailsProps {
  result: RecapGenerationResponse;
}

const RecapDetails: React.FC<RecapDetailsProps> = ({ result }) => {
  return (
    <VStack spacing={6} align="stretch">
      {/* Summary Stats */}
      <Grid templateColumns="repeat(4, 1fr)" gap={4}>
        <Stat>
          <StatLabel>Total Duration</StatLabel>
          <StatNumber>{result.total_duration.toFixed(1)}s</StatNumber>
        </Stat>
        <Stat>
          <StatLabel>Events</StatLabel>
          <StatNumber>{result.events.length}</StatNumber>
        </Stat>
        <Stat>
          <StatLabel>Video Clips</StatLabel>
          <StatNumber>{result.clips.length}</StatNumber>
        </Stat>
        <Stat>
          <StatLabel>Queries Used</StatLabel>
          <StatNumber>{result.queries.length}</StatNumber>
        </Stat>
      </Grid>

      <Accordion defaultIndex={[0]} allowMultiple>
        {/* Queries Section */}
        <AccordionItem>
          <AccordionButton>
            <Box flex="1" textAlign="left">
              <Heading size="sm">Generated Queries ({result.queries.length})</Heading>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel>
            <VStack spacing={3} align="stretch">
              {result.queries.map((query, index) => (
                <Box key={index} p={3} borderWidth={1} borderRadius="md">
                  <Text fontWeight="medium" mb={1}>{query.arc_title}</Text>
                  <Code p={2} borderRadius="md" fontSize="sm">
                    {query.query_text}
                  </Code>
                  <Text fontSize="xs" color="gray.600" mt={1}>
                    Purpose: {query.purpose}
                  </Text>
                </Box>
              ))}
            </VStack>
          </AccordionPanel>
        </AccordionItem>

        {/* Selected Events Section */}
        <AccordionItem>
          <AccordionButton>
            <Box flex="1" textAlign="left">
              <Heading size="sm">Selected Events ({result.events.length})</Heading>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel>
            <VStack spacing={4} align="stretch">
              {result.events.map((event, index) => (
                <Box key={event.id} p={4} borderWidth={1} borderRadius="md">
                  <HStack justify="space-between" mb={2}>
                    <Badge colorScheme="blue">{event.arc_title}</Badge>
                    <Text fontSize="sm" color="gray.600">
                      {event.series} {event.season} {event.episode}
                    </Text>
                  </HStack>
                  
                  <Text fontSize="sm" mb={2}>{event.content}</Text>
                  
                  <HStack spacing={4} fontSize="xs" color="gray.600">
                    <Text>⏰ {event.start_time} - {event.end_time}</Text>
                    <Text>📊 Score: {event.relevance_score.toFixed(2)}</Text>
                  </HStack>
                  
                  {event.selected_subtitles.length > 0 && (
                    <Box mt={2}>
                      <Text fontSize="xs" fontWeight="medium" mb={1}>Selected Subtitles:</Text>
                      <List spacing={1}>
                        {event.selected_subtitles.map((subtitle, i) => (
                          <ListItem key={i} fontSize="xs">
                            <ListIcon as={InfoIcon} color="blue.500" />
                            {subtitle}
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}
                </Box>
              ))}
            </VStack>
          </AccordionPanel>
        </AccordionItem>

        {/* Video Clips Section */}
        <AccordionItem>
          <AccordionButton>
            <Box flex="1" textAlign="left">
              <Heading size="sm">Video Clips ({result.clips.length})</Heading>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel>
            <VStack spacing={3} align="stretch">
              {result.clips.map((clip, index) => (
                <Box key={clip.event_id} p={3} borderWidth={1} borderRadius="md">
                  <HStack justify="space-between" mb={2}>
                    <Badge colorScheme="green">{clip.arc_title}</Badge>
                    <Text fontSize="sm">{clip.duration.toFixed(1)}s</Text>
                  </HStack>
                  
                  <Text fontSize="sm" fontFamily="mono" color="gray.600" mb={1}>
                    {clip.file_path}
                  </Text>
                  
                  <Text fontSize="xs">
                    ⏱️ {clip.start_seconds.toFixed(1)}s - {clip.end_seconds.toFixed(1)}s
                  </Text>
                  
                  {clip.subtitle_lines.length > 0 && (
                    <Box mt={2}>
                      <Text fontSize="xs" fontWeight="medium">Dialogue:</Text>
                      {clip.subtitle_lines.map((line, i) => (
                        <Text key={i} fontSize="xs" color="gray.700">
                          "{line}"
                        </Text>
                      ))}
                    </Box>
                  )}
                </Box>
              ))}
            </VStack>
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </VStack>
  );
};