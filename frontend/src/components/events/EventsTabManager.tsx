import React, { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Heading,
  Select,
  Text,
  Badge,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Alert,
  AlertIcon,
  Flex,
  Card,
  CardBody,
  Grid,
  GridItem,
  useColorModeValue,
} from '@chakra-ui/react';
import { CalendarIcon, TimeIcon, ViewIcon } from '@chakra-ui/icons';
import EventTimeline from './EventTimeline';
import TimelineComparison from './TimelineComparison';
import { TemporalVisualization } from './TemporalVisualization';
import type { Episode } from '../../architecture/types';

interface EventsTabManagerProps {
  series: string;
  episodes: Episode[];
}

interface EpisodeInfo {
  season: string;
  episode: string;
  title?: string;
}

export const EventsTabManager: React.FC<EventsTabManagerProps> = ({
  series,
  episodes = []
}) => {
  const [selectedEpisode, setSelectedEpisode] = useState<EpisodeInfo | null>(null);
  const [availableEpisodes, setAvailableEpisodes] = useState<EpisodeInfo[]>([]);
  
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const selectBgColor = useColorModeValue('white', 'gray.700');

  // Process episodes into a more usable format
  useEffect(() => {
    const processedEpisodes: EpisodeInfo[] = [];
    
    if (episodes && episodes.length > 0) {
      episodes.forEach(ep => {
        processedEpisodes.push({
          season: ep.season,
          episode: ep.episode,
          title: `${ep.season}${ep.episode}` // Remove title property since it doesn't exist
        });
      });
    }
    
    // Sort episodes by season and episode
    processedEpisodes.sort((a, b) => {
      const seasonA = parseInt(a.season.replace('S', ''));
      const seasonB = parseInt(b.season.replace('S', ''));
      if (seasonA !== seasonB) return seasonA - seasonB;
      
      const episodeA = parseInt(a.episode.replace('E', ''));
      const episodeB = parseInt(b.episode.replace('E', ''));
      return episodeA - episodeB;
    });
    
    setAvailableEpisodes(processedEpisodes);
    
    // Auto-select first episode only if we don't already have one selected
    if (processedEpisodes.length > 0 && selectedEpisode === null) {
      setSelectedEpisode(processedEpisodes[0]);
    }
  }, [episodes]); // Remove selectedEpisode from dependencies to prevent infinite loop

  const handleEpisodeChange = (value: string) => {
    if (!value) {
      setSelectedEpisode(null);
      return;
    }
    
    const [season, episode] = value.split('-');
    const episodeInfo = availableEpisodes.find(ep => 
      ep.season === season && ep.episode === episode
    );
    
    if (episodeInfo) {
      setSelectedEpisode(episodeInfo);
    }
  };

  const getEpisodeSelectValue = (ep: EpisodeInfo) => {
    return `${ep.season}-${ep.episode}`;
  };

  const formatEpisodeLabel = (ep: EpisodeInfo) => {
    return `${ep.season}${ep.episode}${ep.title ? ` - ${ep.title}` : ''}`;
  };

  if (!series) {
    return (
      <Box p={6}>
        <Alert status="info">
          <AlertIcon />
          Please select a series to view events.
        </Alert>
      </Box>
    );
  }

  if (availableEpisodes.length === 0) {
    return (
      <Box p={6}>
        <VStack spacing={4}>
          <Alert status="warning">
            <AlertIcon />
            No episodes found for series "{series}". 
          </Alert>
          <Text color="gray.600">
            Episodes need to be processed first to generate events. 
            Use the Processing tab to process episodes for this series.
          </Text>
        </VStack>
      </Box>
    );
  }

  return (
    <Box p={6}>
      <VStack spacing={6} align="stretch">
        {/* Header */}
        <Card bg={bgColor} borderColor={borderColor}>
          <CardBody>
            <VStack spacing={4} align="stretch">
              <Flex justify="space-between" align="center">
                <Heading size="lg" color="purple.500">
                  📊 Events Dashboard
                </Heading>
                <HStack>
                  <Badge colorScheme="blue" variant="subtle">
                    {availableEpisodes.length} episodes
                  </Badge>
                  <Badge colorScheme="green" variant="subtle">
                    {series}
                  </Badge>
                </HStack>
              </Flex>
              
              <Text color="gray.600">
                Explore and visualize narrative events across episodes with timeline views, 
                temporal analysis, and semantic search capabilities.
              </Text>
            </VStack>
          </CardBody>
        </Card>

        {/* Episode Selection */}
        <Card bg={bgColor} borderColor={borderColor}>
          <CardBody>
            <VStack spacing={3} align="stretch">
              <HStack>
                <CalendarIcon color="purple.500" />
                <Text fontWeight="semibold">Select Episode</Text>
              </HStack>
              
              <Select
                placeholder="Choose an episode to analyze..."
                value={selectedEpisode ? getEpisodeSelectValue(selectedEpisode) : ''}
                onChange={(e) => handleEpisodeChange(e.target.value)}
                bg={selectBgColor}
              >
                {availableEpisodes.map((ep) => (
                  <option key={getEpisodeSelectValue(ep)} value={getEpisodeSelectValue(ep)}>
                    {formatEpisodeLabel(ep)}
                  </option>
                ))}
              </Select>
              
              {selectedEpisode && (
                <Box>
                  <Text fontSize="sm" color="gray.500">
                    Analyzing events for: <strong>{formatEpisodeLabel(selectedEpisode)}</strong>
                  </Text>
                </Box>
              )}
            </VStack>
          </CardBody>
        </Card>

        {/* Events Content */}
        {selectedEpisode ? (
          <Box>
            <Tabs variant="enclosed" colorScheme="purple">
              <TabList>
                <Tab>
                  <HStack>
                    <TimeIcon />
                    <Text>Event Timeline</Text>
                  </HStack>
                </Tab>
                <Tab>
                  <HStack>
                    <ViewIcon />
                    <Text>Timeline Comparison</Text>
                  </HStack>
                </Tab>
                <Tab>
                  <HStack>
                    <CalendarIcon />
                    <Text>Temporal Analysis</Text>
                  </HStack>
                </Tab>
              </TabList>
              
              <TabPanels>
                {/* Event Timeline Tab */}
                <TabPanel p={0}>
                  <Box mt={4}>
                    {selectedEpisode && (
                      <EventTimeline
                        series={series}
                        season={selectedEpisode.season}
                        episode={selectedEpisode.episode}
                        showAnalytics={true}
                        height={600}
                      />
                    )}
                  </Box>
                </TabPanel>
                
                {/* Timeline Comparison Tab */}
                <TabPanel p={0}>
                  <Box mt={4}>
                    {selectedEpisode && (
                      <TimelineComparison
                        series={series}
                        season={selectedEpisode.season}
                        episode={selectedEpisode.episode}
                      />
                    )}
                  </Box>
                </TabPanel>
                
                {/* Temporal Analysis Tab */}
                <TabPanel p={0}>
                  <Box mt={4}>
                    {selectedEpisode && (
                      <TemporalVisualization
                        series={series}
                        season={selectedEpisode.season}
                        episode={selectedEpisode.episode}
                      />
                    )}
                  </Box>
                </TabPanel>
              </TabPanels>
            </Tabs>
          </Box>
        ) : (
          <Card bg={bgColor} borderColor={borderColor}>
            <CardBody>
              <VStack spacing={4}>
                <TimeIcon boxSize={12} color="gray.400" />
                <Text color="gray.500" textAlign="center">
                  Select an episode above to explore its events timeline
                </Text>
                <Text fontSize="sm" color="gray.400" textAlign="center">
                  Events are automatically extracted from processed episodes and can be 
                  visualized using multiple timeline views and analysis tools.
                </Text>
              </VStack>
            </CardBody>
          </Card>
        )}

        {/* Quick Stats */}
        {selectedEpisode && (
          <Grid templateColumns="repeat(auto-fit, minmax(200px, 1fr))" gap={4}>
            <GridItem>
              <Card bg={bgColor} borderColor={borderColor} size="sm">
                <CardBody>
                  <Text fontSize="sm" color="gray.500">Episode</Text>
                  <Text fontWeight="bold" fontSize="lg">
                    {selectedEpisode.season}{selectedEpisode.episode}
                  </Text>
                </CardBody>
              </Card>
            </GridItem>
            
            <GridItem>
              <Card bg={bgColor} borderColor={borderColor} size="sm">
                <CardBody>
                  <Text fontSize="sm" color="gray.500">Series</Text>
                  <Text fontWeight="bold" fontSize="lg">
                    {series}
                  </Text>
                </CardBody>
              </Card>
            </GridItem>
            
            <GridItem>
              <Card bg={bgColor} borderColor={borderColor} size="sm">
                <CardBody>
                  <Text fontSize="sm" color="gray.500">Available Episodes</Text>
                  <Text fontWeight="bold" fontSize="lg">
                    {availableEpisodes.length}
                  </Text>
                </CardBody>
              </Card>
            </GridItem>
          </Grid>
        )}
      </VStack>
    </Box>
  );
};
