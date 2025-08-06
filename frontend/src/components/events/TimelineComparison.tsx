import React, { useState, useEffect } from 'react';
import {
  Box,
  Text,
  VStack,
  HStack,
  Card,
  CardBody,
  CardHeader,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Flex,
  Badge,
  Button,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  Divider,
  Grid,
  GridItem,
} from '@chakra-ui/react';
import EventTimeline from './EventTimeline';
import NarrativeArcTimeline from './NarrativeArcTimeline';
import { VectorSearch } from './VectorSearch';
import { NarrativeArc, Event } from '../../architecture/types/arc';

interface TimelineComparisonProps {
  series: string;
  season: string;
  episode: string;
}

interface TimelineEvent {
  id: string;
  content: string;
  start_timestamp: number;
  end_timestamp: number;
  confidence_score?: number;
  extraction_method?: string;
  characters_involved: string[];
  series: string;
  season: string;
  episode: string;
  ordinal_position: number;
  progression_id: string;
  // Progression context
  progression_content?: string;
  arc_title?: string;
  arc_id?: string;
}

const TimelineComparison: React.FC<TimelineComparisonProps> = ({
  series,
  season,
  episode
}) => {
  const [arcs, setArcs] = useState<NarrativeArc[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [syncTimelines, setSyncTimelines] = useState(true);
  
  const { isOpen: isEventModalOpen, onOpen: onEventModalOpen, onClose: onEventModalClose } = useDisclosure();

  useEffect(() => {
    loadNarrativeArcs();
  }, [series, season, episode]);

  const loadNarrativeArcs = async () => {
    try {
      const response = await fetch(`/api/arcs/${series}/${season}/${episode}`);
      if (response.ok) {
        const data: NarrativeArc[] = await response.json();
        setArcs(data);
      }
    } catch (error) {
      console.error('Failed to load narrative arcs:', error);
    }
  };

  const handleEventClick = (event: TimelineEvent) => {
    setSelectedEvent(event);
    onEventModalOpen();
  };

  const handleNarrativeEventClick = (event: Event) => {
    // Convert Event to TimelineEvent for display
    const timelineEvent: TimelineEvent = {
      id: event.id || '',
      content: event.content || '',
      start_timestamp: event.start_timestamp || 0,
      end_timestamp: event.end_timestamp || 0,
      confidence_score: event.confidence_score,
      extraction_method: event.extraction_method,
      characters_involved: event.characters_involved || [],
      series: series,
      season: season,
      episode: episode,
      ordinal_position: event.ordinal_position || 1,
      progression_id: event.progression_id || '',
      progression_content: undefined,
      arc_title: undefined,
      arc_id: undefined
    };
    setSelectedEvent(timelineEvent);
    onEventModalOpen();
  };

  const handleTimeSeek = (timestamp: number) => {
    setCurrentTime(timestamp);
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box p={6}>
      <VStack spacing={6} align="stretch">
        {/* Header */}
        <Card>
          <CardHeader>
            <Flex justify="space-between" align="center">
              <VStack align="start" spacing={1}>
                <Text fontSize="2xl" fontWeight="bold">Timeline Analysis</Text>
                <Text color="gray.600">
                  {series} • {season} • {episode}
                </Text>
              </VStack>
              <HStack>
                <Button
                  size="sm"
                  colorScheme={syncTimelines ? "blue" : "gray"}
                  onClick={() => setSyncTimelines(!syncTimelines)}
                >
                  {syncTimelines ? "🔗 Synced" : "🔓 Independent"}
                </Button>
                <Badge colorScheme="blue">
                  Current: {formatTime(currentTime)}
                </Badge>
              </HStack>
            </Flex>
          </CardHeader>
        </Card>

        {/* Timeline Views */}
        <Tabs variant="enclosed" colorScheme="blue">
          <TabList>
            <Tab>📚 Narrative Arcs & Progressions</Tab>
            <Tab>⚡ Individual Events</Tab>
            <Tab>🔍 Vector Search</Tab>
            <Tab>📊 Side-by-Side Comparison</Tab>
          </TabList>

          <TabPanels>
            {/* Narrative Arc Timeline */}
            <TabPanel p={0} pt={4}>
              <Card>
                <CardHeader>
                  <Text fontSize="lg" fontWeight="semibold">
                    Narrative Arc Timeline
                  </Text>
                  <Text fontSize="sm" color="gray.600">
                    Shows arcs, progressions, and their relationships over time
                  </Text>
                </CardHeader>
                <CardBody>
                  <NarrativeArcTimeline
                    arcs={arcs}
                    series={series}
                    season={season}
                    episode={episode}
                    height={500}
                    onEventClick={handleNarrativeEventClick}
                    onTimeSeek={syncTimelines ? handleTimeSeek : undefined}
                  />
                </CardBody>
              </Card>
            </TabPanel>

            {/* Event Timeline */}
            <TabPanel p={0} pt={4}>
              <Card>
                <CardHeader>
                  <Text fontSize="lg" fontWeight="semibold">
                    Event Timeline
                  </Text>
                  <Text fontSize="sm" color="gray.600">
                    Shows individual timestamped events with confidence scores and character involvement
                  </Text>
                </CardHeader>
                <CardBody>
                  <EventTimeline
                    series={series}
                    season={season}
                    episode={episode}
                    height={500}
                    onEventClick={handleEventClick}
                    onTimeSeek={syncTimelines ? handleTimeSeek : undefined}
                    showAnalytics={true}
                  />
                </CardBody>
              </Card>
            </TabPanel>

            {/* Vector Search */}
            <TabPanel p={0} pt={4}>
              <VectorSearch
                defaultSeries={series}
                defaultSeason={season}
                defaultEpisode={episode}
              />
            </TabPanel>

            {/* Side-by-Side Comparison */}
            <TabPanel p={0} pt={4}>
              <Grid templateColumns="1fr 1fr" gap={6} height="800px">
                <GridItem>
                  <Card height="100%">
                    <CardHeader>
                      <Text fontSize="md" fontWeight="semibold">
                        📚 Narrative Arc View
                      </Text>
                      <Text fontSize="sm" color="gray.600">
                        Arc-centric timeline showing story progressions
                      </Text>
                    </CardHeader>
                    <CardBody>
                      <NarrativeArcTimeline
                        arcs={arcs}
                        series={series}
                        season={season}
                        episode={episode}
                        height={350}
                        onEventClick={handleNarrativeEventClick}
                        onTimeSeek={handleTimeSeek}
                      />
                    </CardBody>
                  </Card>
                </GridItem>

                <GridItem>
                  <Card height="100%">
                    <CardHeader>
                      <Text fontSize="md" fontWeight="semibold">
                        ⚡ Event View
                      </Text>
                      <Text fontSize="sm" color="gray.600">
                        Event-centric timeline showing individual moments
                      </Text>
                    </CardHeader>
                    <CardBody>
                      <EventTimeline
                        series={series}
                        season={season}
                        episode={episode}
                        height={350}
                        onEventClick={handleEventClick}
                        onTimeSeek={handleTimeSeek}
                        showAnalytics={false}
                      />
                    </CardBody>
                  </Card>
                </GridItem>
              </Grid>

              {/* Sync Information */}
              <Card mt={4}>
                <CardBody>
                  <HStack justify="center" spacing={8}>
                    <VStack>
                      <Text fontSize="sm" fontWeight="semibold">Narrative Arc Granularity</Text>
                      <Text fontSize="xs" color="gray.600">Story-level progressions</Text>
                      <Badge colorScheme="purple">Arc Focus</Badge>
                    </VStack>
                    <Divider orientation="vertical" height="60px" />
                    <VStack>
                      <Text fontSize="sm" fontWeight="semibold">Current Sync Time</Text>
                      <Text fontSize="lg" fontWeight="bold" color="blue.500">
                        {formatTime(currentTime)}
                      </Text>
                    </VStack>
                    <Divider orientation="vertical" height="60px" />
                    <VStack>
                      <Text fontSize="sm" fontWeight="semibold">Event Granularity</Text>
                      <Text fontSize="xs" color="gray.600">Individual timestamped moments</Text>
                      <Badge colorScheme="green">Event Focus</Badge>
                    </VStack>
                  </HStack>
                </CardBody>
              </Card>
            </TabPanel>
          </TabPanels>
        </Tabs>

        {/* Event Detail Modal */}
        <Modal isOpen={isEventModalOpen} onClose={onEventModalClose} size="lg">
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Event Details</ModalHeader>
            <ModalCloseButton />
            <ModalBody pb={6}>
              {selectedEvent && (
                <VStack spacing={4} align="stretch">
                  <Box>
                    <Text fontSize="sm" color="gray.600" mb={1}>Content</Text>
                    <Text fontWeight="semibold">{selectedEvent.content}</Text>
                  </Box>
                  
                  <HStack spacing={6}>
                    <Box>
                      <Text fontSize="sm" color="gray.600" mb={1}>Start Time</Text>
                      <Text>{formatTime(selectedEvent.start_timestamp)}</Text>
                    </Box>
                    <Box>
                      <Text fontSize="sm" color="gray.600" mb={1}>End Time</Text>
                      <Text>{formatTime(selectedEvent.end_timestamp)}</Text>
                    </Box>
                    <Box>
                      <Text fontSize="sm" color="gray.600" mb={1}>Duration</Text>
                      <Text>{formatTime(selectedEvent.end_timestamp - selectedEvent.start_timestamp)}</Text>
                    </Box>
                  </HStack>

                  {selectedEvent.confidence_score !== undefined && (
                    <Box>
                      <Text fontSize="sm" color="gray.600" mb={1}>Confidence Score</Text>
                      <HStack>
                        <Badge
                          colorScheme={
                            selectedEvent.confidence_score >= 0.8 ? "green" :
                            selectedEvent.confidence_score >= 0.6 ? "yellow" : "red"
                          }
                        >
                          {Math.round(selectedEvent.confidence_score * 100)}%
                        </Badge>
                        <Text fontSize="sm" color="gray.600">
                          ({selectedEvent.confidence_score.toFixed(3)})
                        </Text>
                      </HStack>
                    </Box>
                  )}

                  {selectedEvent.extraction_method && (
                    <Box>
                      <Text fontSize="sm" color="gray.600" mb={1}>Extraction Method</Text>
                      <Badge>{selectedEvent.extraction_method}</Badge>
                    </Box>
                  )}

                  {selectedEvent.characters_involved && selectedEvent.characters_involved.length > 0 && (
                    <Box>
                      <Text fontSize="sm" color="gray.600" mb={1}>Characters Involved</Text>
                      <HStack wrap="wrap">
                        {selectedEvent.characters_involved.map(character => (
                          <Badge key={character} colorScheme="blue">{character}</Badge>
                        ))}
                      </HStack>
                    </Box>
                  )}

                  <Box>
                    <Text fontSize="sm" color="gray.600" mb={1}>Position</Text>
                    <Text>#{selectedEvent.ordinal_position}</Text>
                  </Box>
                </VStack>
              )}
            </ModalBody>
          </ModalContent>
        </Modal>
      </VStack>
    </Box>
  );
};

export default TimelineComparison;
