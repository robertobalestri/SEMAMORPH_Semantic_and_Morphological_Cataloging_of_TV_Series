import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  VStack,
  HStack,
  Button,
  Text,
  Card,
  CardBody,
  CardHeader,
  Progress,
  Badge,
  Select,
  Checkbox,
  CheckboxGroup,
  Stack,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Spinner,
  useToast,
  Heading,
  Divider,
  IconButton,
  Tooltip
} from '@chakra-ui/react';
import { FiPlay, FiTrash2, FiRefreshCw } from 'react-icons/fi';
import { ApiClient } from '../../services/api/ApiClient';
import { isApiSuccess } from '../../architecture/types/api';
import { useDebounce } from '../../hooks/useDebounce';

interface ProcessingJob {
  id: string;
  series: string;
  season: string;
  episodes: string[];
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  progress: Record<string, string>;
}

interface ProcessingManagerProps {
  series?: string; // Make series optional
}

export const ProcessingManager: React.FC<ProcessingManagerProps> = ({ series }) => {
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [availableEpisodes, setAvailableEpisodes] = useState<string[]>([]);
  const [availableSeries, setAvailableSeries] = useState<string[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<string>(series || '');
  const [selectedSeason, setSelectedSeason] = useState<string>('S01');
  const [selectedEpisodes, setSelectedEpisodes] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [lastFetchKey, setLastFetchKey] = useState<string>('');
  const [hasError, setHasError] = useState<boolean>(false);
  
  const api = new ApiClient();
  const toast = useToast();
  const abortControllerRef = useRef<AbortController | null>(null);
  
  // Debounce the season/series changes to prevent excessive API calls
  const debouncedSeason = useDebounce(selectedSeason, 500);
  const debouncedSeries = useDebounce(selectedSeries, 500);

  const fetchJobs = useCallback(async () => {
    try {
      const response = await api.request<ProcessingJob[]>('/processing/jobs');
      if (isApiSuccess(response)) {
        // Filter jobs for current series (if any is selected)
        const seriesJobs = selectedSeries 
          ? response.data.filter((job: ProcessingJob) => job.series === selectedSeries)
          : response.data;
        setJobs(seriesJobs);
      }
    } catch (error) {
      console.error('Error fetching processing jobs:', error);
    }
  }, [selectedSeries]);

  const fetchAvailableEpisodes = useCallback(async (series: string, season: string) => {
    const fetchKey = `${series}-${season}`;
    
    // Prevent duplicate requests
    if (fetchKey === lastFetchKey || !season) {
      return;
    }
    
    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Create new abort controller
    abortControllerRef.current = new AbortController();
    
    try {
      setIsLoading(true);
      setHasError(false);
      setLastFetchKey(fetchKey);
      
      console.log(`Fetching episodes for ${series}/${season}`);
      
      const response = await api.request<string[]>(
        `/available-episodes/${series}/${season}`,
        { signal: abortControllerRef.current.signal }
      );
      
      if (isApiSuccess(response)) {
        console.log(`Found episodes:`, response.data);
        // Ensure we always have an array, even if response.data is null/undefined
        const episodes = Array.isArray(response.data) ? response.data : [];
        setAvailableEpisodes(episodes);
        setSelectedEpisodes([]); // Reset selected episodes when changing season
      } else {
        console.error('API call was not successful:', response);
        setAvailableEpisodes([]);
        setHasError(true);
        toast({
          title: 'Error',
          description: 'Failed to fetch available episodes',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error('Error fetching available episodes:', error);
        setAvailableEpisodes([]);
        setHasError(true);
        toast({
          title: 'Error',
          description: 'Failed to fetch available episodes',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    } finally {
      setIsLoading(false);
    }
  }, [toast, lastFetchKey]);

  const fetchAvailableSeries = useCallback(async () => {
    try {
      const response = await api.request<string[]>('/available-series');
      if (isApiSuccess(response)) {
        setAvailableSeries(response.data);
        // If no series is selected and we have available series, select the first one
        if (!selectedSeries && response.data.length > 0) {
          setSelectedSeries(response.data[0]);
        }
      }
    } catch (error) {
      console.error('Error fetching available series:', error);
      setAvailableSeries(['GA', 'FIABA']); // Default fallback
      if (!selectedSeries) {
        setSelectedSeries('GA');
      }
    }
  }, [selectedSeries]);

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 3000); // Reduced frequency to 3 seconds
    return () => clearInterval(interval);
  }, [fetchJobs]);

  useEffect(() => {
    // Fetch episodes when debounced series or season changes
    if (debouncedSeries && debouncedSeason) {
      fetchAvailableEpisodes(debouncedSeries, debouncedSeason);
    }
  }, [debouncedSeries, debouncedSeason, fetchAvailableEpisodes]);

  useEffect(() => {
    fetchAvailableSeries();
  }, [fetchAvailableSeries]);

  // Update selectedSeries when series prop changes
  useEffect(() => {
    if (series && series !== selectedSeries) {
      setSelectedSeries(series);
    }
  }, [series, selectedSeries]);

  useEffect(() => {
    fetchAvailableSeries();
  }, [fetchAvailableSeries]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const startProcessing = async () => {
    if (!selectedSeries) {
      toast({
        title: 'No Series Selected',
        description: 'Please select a series to process',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (selectedEpisodes.length === 0) {
      toast({
        title: 'No Episodes Selected',
        description: 'Please select at least one episode to process',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    try {
      setIsStarting(true);
      const response = await api.request<ProcessingJob>('/processing/episodes', {
        method: 'POST',
        body: JSON.stringify({
          series: selectedSeries,
          season: selectedSeason,
          episodes: selectedEpisodes
        })
      });

      if (isApiSuccess(response)) {
        setSelectedEpisodes([]);
        toast({
          title: 'Processing Started',
          description: `Started processing ${selectedEpisodes.length} episodes`,
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        fetchJobs();
      } else {
        throw new Error('Failed to start processing');
      }
    } catch (error) {
      console.error('Error starting processing:', error);
      toast({
        title: 'Error',
        description: 'Failed to start processing',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsStarting(false);
    }
  };

  const deleteJob = async (jobId: string) => {
    try {
      const response = await api.request(`/processing/jobs/${jobId}`, {
        method: 'DELETE'
      });

      if (isApiSuccess(response)) {
        toast({
          title: 'Job Deleted',
          description: 'Processing job deleted successfully',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        fetchJobs();
      }
    } catch (error) {
      console.error('Error deleting job:', error);
      toast({
        title: 'Error',
        description: 'Failed to delete job',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'gray';
      case 'running': return 'blue';
      case 'completed': return 'green';
      case 'failed': return 'red';
      default: return 'gray';
    }
  };

  const getProgressPercent = (job: ProcessingJob) => {
    const total = job.episodes.length;
    const completed = Object.values(job.progress).filter(status => 
      status === 'completed' || status === 'failed'
    ).length;
    return total > 0 ? (completed / total) * 100 : 0;
  };

  return (
    <VStack spacing={6} align="stretch">
      <Card>
        <CardHeader>
          <Heading size="md">Start New Processing</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={4} align="stretch">
            {!series && (
              <Box>
                <Text mb={2} fontWeight="medium">Series:</Text>
                <Select
                  placeholder="Select series to process"
                  value={selectedSeries}
                  onChange={(e) => setSelectedSeries(e.target.value)}
                >
                  {availableSeries.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </Select>
              </Box>
            )}
            
            <HStack>
              <Box flex={1}>
                <Text mb={2} fontWeight="medium">Season:</Text>
                <Select
                  value={selectedSeason}
                  onChange={(e) => setSelectedSeason(e.target.value)}
                  isDisabled={!selectedSeries}
                >
                  <option value="S01">Season 1</option>
                  <option value="S02">Season 2</option>
                  <option value="S03">Season 3</option>
                  <option value="S04">Season 4</option>
                  <option value="S05">Season 5</option>
                </Select>
              </Box>
            </HStack>

            {!selectedSeries ? (
              <Alert status="info">
                <AlertIcon />
                <AlertDescription>
                  Please select a series above to begin processing episodes.
                </AlertDescription>
              </Alert>
            ) : isLoading ? (
              <Box textAlign="center" py={4}>
                <Spinner />
                <Text mt={2}>Loading available episodes...</Text>
              </Box>
            ) : (
              <Box>
                <Text mb={2} fontWeight="medium">
                  Available Episodes ({(availableEpisodes || []).length}):
                </Text>
                {(availableEpisodes || []).length > 0 ? (
                  <CheckboxGroup
                    value={selectedEpisodes}
                    onChange={(values) => setSelectedEpisodes(values as string[])}
                  >
                    <Stack direction="row" wrap="wrap" spacing={4}>
                      {(availableEpisodes || []).map((episode) => (
                        <Checkbox key={episode} value={episode}>
                          {episode}
                        </Checkbox>
                      ))}
                    </Stack>
                  </CheckboxGroup>
                ) : hasError ? (
                  <Alert status="error">
                    <AlertIcon />
                    <AlertDescription>
                      Failed to load episodes for {selectedSeries} {selectedSeason}. Please check if the API is running.
                    </AlertDescription>
                  </Alert>
                ) : (
                  <Alert status="info">
                    <AlertIcon />
                    <AlertDescription>
                      No episodes found for {selectedSeries} {selectedSeason}
                    </AlertDescription>
                  </Alert>
                )}
              </Box>
            )}

            <HStack justify="space-between">
              <Text fontSize="sm" color="gray.600">
                Selected: {selectedEpisodes.length} episodes
              </Text>
              <Button
                leftIcon={<FiPlay />}
                colorScheme="blue"
                isLoading={isStarting}
                loadingText="Starting..."
                isDisabled={!selectedSeries || selectedEpisodes.length === 0}
                onClick={startProcessing}
              >
                Start Processing
              </Button>
            </HStack>
          </VStack>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <HStack justify="space-between">
            <Heading size="md">Processing Jobs</Heading>
            <Tooltip label="Refresh">
              <IconButton
                icon={<FiRefreshCw />}
                size="sm"
                variant="ghost"
                onClick={fetchJobs}
                aria-label="Refresh jobs"
              />
            </Tooltip>
          </HStack>
        </CardHeader>
        <CardBody>
          {jobs.length === 0 ? (
            <Alert status="info">
              <AlertIcon />
              <AlertDescription>
                No processing jobs found for {series}
              </AlertDescription>
            </Alert>
          ) : (
            <VStack spacing={4} align="stretch">
              {jobs.map((job) => (
                <Card key={job.id} variant="outline">
                  <CardBody>
                    <VStack align="stretch" spacing={3}>
                      <HStack justify="space-between">
                        <VStack align="start" spacing={1}>
                          <HStack>
                            <Text fontWeight="medium">
                              {job.series} {job.season}
                            </Text>
                            <Badge colorScheme={getStatusColor(job.status)}>
                              {job.status}
                            </Badge>
                          </HStack>
                          <Text fontSize="sm" color="gray.600">
                            Episodes: {job.episodes.join(', ')}
                          </Text>
                          <Text fontSize="xs" color="gray.500">
                            Started: {new Date(job.created_at).toLocaleString()}
                          </Text>
                        </VStack>
                        <HStack>
                          {job.status !== 'running' && (
                            <Tooltip label="Delete Job">
                              <IconButton
                                icon={<FiTrash2 />}
                                size="sm"
                                variant="ghost"
                                colorScheme="red"
                                onClick={() => deleteJob(job.id)}
                                aria-label="Delete job"
                              />
                            </Tooltip>
                          )}
                        </HStack>
                      </HStack>

                      {job.status === 'running' && (
                        <Box>
                          <Text fontSize="sm" mb={2}>
                            Progress: {Math.round(getProgressPercent(job))}%
                          </Text>
                          <Progress 
                            value={getProgressPercent(job)} 
                            colorScheme="blue" 
                            size="sm"
                          />
                        </Box>
                      )}

                      {job.error_message && (
                        <Alert status="error" size="sm">
                          <AlertIcon />
                          <AlertTitle fontSize="sm">Error:</AlertTitle>
                          <AlertDescription fontSize="sm">
                            {job.error_message}
                          </AlertDescription>
                        </Alert>
                      )}

                      <Divider />
                      
                      <Box>
                        <Text fontSize="sm" fontWeight="medium" mb={2}>
                          Episode Progress:
                        </Text>
                        <Stack direction="row" wrap="wrap" spacing={2}>
                          {job.episodes.map((episode) => (
                            <Badge
                              key={episode}
                              colorScheme={getStatusColor(job.progress[episode] || 'pending')}
                              variant="outline"
                            >
                              {episode}: {job.progress[episode] || 'pending'}
                            </Badge>
                          ))}
                        </Stack>
                      </Box>
                    </VStack>
                  </CardBody>
                </Card>
              ))}
            </VStack>
          )}
        </CardBody>
      </Card>
    </VStack>
  );
};
