import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Text,
  HStack,
  VStack,
  Badge,
  Tooltip,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Flex,
  Card,
  CardBody,
  useColorModeValue,
  IconButton,
  Select,
  Switch,
  FormControl,
  FormLabel,
  Progress,
} from '@chakra-ui/react';
import { Event, NarrativeArc } from '../../architecture/types/arc';

interface EventTimelineProps {
  arcs: NarrativeArc[];
  series: string;
  season: string;
  episode: string;
  height?: number;
  onEventClick?: (event: Event) => void;
  onTimeSeek?: (timestamp: number) => void;
}

interface TimelineEvent extends Event {
  arcTitle: string;
  arcId: string;
  color: string;
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
};

const generateArcColor = (arcId: string): string => {
  const colors = [
    '#3182CE', '#38A169', '#D69E2E', '#E53E3E', '#805AD5',
    '#DD6B20', '#319795', '#C53030', '#2B6CB0', '#2F855A'
  ];
  let hash = 0;
  for (let i = 0; i < arcId.length; i++) {
    hash = arcId.charCodeAt(i) + ((hash << 5) - hash);
  }
  return colors[Math.abs(hash) % colors.length];
};

const NarrativeArcTimeline: React.FC<EventTimelineProps> = ({
  arcs,
  series,
  season,
  episode,
  height = 400,
  onEventClick,
  onTimeSeek
}) => {
  const [allEvents, setAllEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [showConfidenceFilter, setShowConfidenceFilter] = useState(false);
  const [minConfidence, setMinConfidence] = useState(0);
  const [selectedArcIds, setSelectedArcIds] = useState<Set<string>>(new Set());

  // Load all events for the episode
  useEffect(() => {
    loadAllEvents();
  }, [series, season, episode]);

  // Auto-play functionality
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && maxTimestamp > 0) {
      interval = setInterval(() => {
        setCurrentTime(prev => {
          const next = prev + playbackSpeed;
          if (next >= maxTimestamp) {
            setIsPlaying(false);
            return maxTimestamp;
          }
          return next;
        });
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed]);

  const loadAllEvents = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/events/episode/${series}/${season}/${episode}`);
      if (response.ok) {
        const events: Event[] = await response.json();
        
        // Map events to timeline events with arc information
        const timelineEvents: TimelineEvent[] = [];
        
        for (const arc of arcs) {
          const arcColor = generateArcColor(arc.id);
          for (const progression of arc.progressions) {
            const arcEvents = events.filter(e => e.progression_id === progression.id);
            for (const event of arcEvents) {
              timelineEvents.push({
                ...event,
                arcTitle: arc.title,
                arcId: arc.id,
                color: arcColor
              });
            }
          }
        }
        
        setAllEvents(timelineEvents);
        
        // Initialize arc selection to all arcs
        setSelectedArcIds(new Set(arcs.map(arc => arc.id)));
      }
    } catch (error) {
      console.error('Failed to load events:', error);
    } finally {
      setLoading(false);
    }
  };

  // Calculate timeline bounds
  const { minTimestamp, maxTimestamp } = useMemo(() => {
    const timestamps = allEvents
      .filter(e => e.start_timestamp !== undefined && e.start_timestamp !== null)
      .flatMap(e => [e.start_timestamp!, e.end_timestamp!].filter(t => t !== undefined && t !== null));
    
    if (timestamps.length === 0) return { minTimestamp: 0, maxTimestamp: 3600 };
    
    return {
      minTimestamp: Math.min(...timestamps),
      maxTimestamp: Math.max(...timestamps)
    };
  }, [allEvents]);

  // Filter events based on current filters
  const filteredEvents = useMemo(() => {
    return allEvents.filter(event => {
      // Arc filter
      if (!selectedArcIds.has(event.arcId)) return false;
      
      // Confidence filter
      if (showConfidenceFilter && event.confidence_score !== undefined) {
        if (event.confidence_score < minConfidence) return false;
      }
      
      // Must have valid timestamps
      return event.start_timestamp !== undefined && event.end_timestamp !== undefined;
    });
  }, [allEvents, selectedArcIds, showConfidenceFilter, minConfidence]);

  // Get events at current time
  const currentEvents = useMemo(() => {
    return filteredEvents.filter(event => 
      event.start_timestamp! <= currentTime && event.end_timestamp! >= currentTime
    );
  }, [filteredEvents, currentTime]);

  const handleTimeSeek = (timestamp: number) => {
    setCurrentTime(timestamp);
    onTimeSeek?.(timestamp);
  };

  const toggleArcSelection = (arcId: string) => {
    const newSelection = new Set(selectedArcIds);
    if (newSelection.has(arcId)) {
      newSelection.delete(arcId);
    } else {
      newSelection.add(arcId);
    }
    setSelectedArcIds(newSelection);
  };

  const timelineDuration = maxTimestamp - minTimestamp;
  const timelineWidth = 800; // Fixed width for the timeline

  if (loading) {
    return <Progress size="sm" isIndeterminate />;
  }

  return (
    <Box>
      <VStack spacing={4} align="stretch">
        {/* Timeline Controls */}
        <Card size="sm">
          <CardBody>
            <VStack spacing={3}>
              {/* Playback Controls */}
              <Flex justify="space-between" align="center" w="100%">
                <HStack>
                  <IconButton
                    aria-label={isPlaying ? "Pause" : "Play"}
                    icon={<Text fontSize="sm">{isPlaying ? "⏸️" : "▶️"}</Text>}
                    onClick={() => setIsPlaying(!isPlaying)}
                    size="sm"
                    isDisabled={maxTimestamp === 0}
                  />
                  <Text fontSize="sm" fontWeight="semibold">
                    {formatTime(currentTime)} / {formatTime(maxTimestamp)}
                  </Text>
                  <Select
                    size="sm"
                    width="auto"
                    value={playbackSpeed}
                    onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                  >
                    <option value={0.5}>0.5x</option>
                    <option value={1}>1x</option>
                    <option value={2}>2x</option>
                    <option value={5}>5x</option>
                    <option value={10}>10x</option>
                  </Select>
                </HStack>

                <HStack>
                  <FormControl display="flex" alignItems="center">
                    <FormLabel htmlFor="confidence-filter" mb="0" fontSize="sm">
                      Confidence Filter
                    </FormLabel>
                    <Switch
                      id="confidence-filter"
                      isChecked={showConfidenceFilter}
                      onChange={(e) => setShowConfidenceFilter(e.target.checked)}
                      size="sm"
                    />
                  </FormControl>
                </HStack>
              </Flex>

              {/* Time Scrubber */}
              <Box w="100%">
                <Slider
                  value={currentTime}
                  min={minTimestamp}
                  max={maxTimestamp}
                  onChange={handleTimeSeek}
                  step={0.1}
                >
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb />
                </Slider>
              </Box>

              {/* Confidence Filter */}
              {showConfidenceFilter && (
                <Box w="100%">
                  <Text fontSize="sm" mb={2}>Minimum Confidence: {minConfidence.toFixed(2)}</Text>
                  <Slider
                    value={minConfidence}
                    min={0}
                    max={1}
                    step={0.1}
                    onChange={setMinConfidence}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </Box>
              )}
            </VStack>
          </CardBody>
        </Card>

        {/* Arc Filter */}
        <Card size="sm">
          <CardBody>
            <Text fontSize="sm" fontWeight="semibold" mb={2}>Filter by Narrative Arc:</Text>
            <Flex wrap="wrap" gap={2}>
              {arcs.map(arc => (
                <Badge
                  key={arc.id}
                  colorScheme={selectedArcIds.has(arc.id) ? "blue" : "gray"}
                  cursor="pointer"
                  onClick={() => toggleArcSelection(arc.id)}
                  px={3}
                  py={1}
                >
                  {arc.title}
                </Badge>
              ))}
            </Flex>
          </CardBody>
        </Card>

        {/* Timeline Visualization */}
        <Card>
          <CardBody>
            <Box position="relative" height={`${height}px`} overflow="auto">
              {filteredEvents.length === 0 ? (
                <Flex justify="center" align="center" height="100%">
                  <Text color="gray.500">No events to display</Text>
                </Flex>
              ) : (
                <Box position="relative" width={`${timelineWidth}px`} height="100%">
                  {/* Current Time Indicator */}
                  <Box
                    position="absolute"
                    left={`${((currentTime - minTimestamp) / timelineDuration) * 100}%`}
                    top="0"
                    bottom="0"
                    width="2px"
                    bg="red.500"
                    zIndex={10}
                  />

                  {/* Events */}
                  {filteredEvents.map((event, index) => {
                    const left = ((event.start_timestamp! - minTimestamp) / timelineDuration) * 100;
                    const width = ((event.end_timestamp! - event.start_timestamp!) / timelineDuration) * 100;
                    const top = (index % 10) * 35; // Stack events to avoid overlap

                    return (
                      <Tooltip
                        key={event.id}
                        label={
                          <Box>
                            <Text fontWeight="bold">{event.arcTitle}</Text>
                            <Text>{event.content}</Text>
                            <Text fontSize="xs">
                              {formatTime(event.start_timestamp!)} - {formatTime(event.end_timestamp!)}
                            </Text>
                            {event.confidence_score && (
                              <Text fontSize="xs">
                                Confidence: {Math.round(event.confidence_score * 100)}%
                              </Text>
                            )}
                          </Box>
                        }
                        placement="top"
                      >
                        <Box
                          position="absolute"
                          left={`${left}%`}
                          top={`${top}px`}
                          width={`${Math.max(width, 0.5)}%`}
                          height="30px"
                          bg={event.color}
                          borderRadius="md"
                          cursor="pointer"
                          opacity={currentEvents.includes(event) ? 1 : 0.7}
                          border={currentEvents.includes(event) ? "2px solid white" : "none"}
                          _hover={{ opacity: 1, transform: "translateY(-2px)" }}
                          transition="all 0.2s"
                          onClick={() => {
                            handleTimeSeek(event.start_timestamp!);
                            onEventClick?.(event);
                          }}
                        >
                          <Text
                            fontSize="xs"
                            color="white"
                            p={1}
                            noOfLines={2}
                            fontWeight="semibold"
                          >
                            {event.content}
                          </Text>
                        </Box>
                      </Tooltip>
                    );
                  })}
                </Box>
              )}
            </Box>
          </CardBody>
        </Card>

        {/* Current Events Display */}
        {currentEvents.length > 0 && (
          <Card>
            <CardBody>
              <Text fontSize="sm" fontWeight="semibold" mb={2}>
                Currently Active Events at {formatTime(currentTime)}:
              </Text>
              <VStack spacing={2} align="stretch">
                {currentEvents.map(event => (
                  <Box
                    key={event.id}
                    p={3}
                    bg={useColorModeValue('gray.50', 'gray.700')}
                    borderRadius="md"
                    borderLeft={`4px solid ${event.color}`}
                  >
                    <HStack justify="space-between">
                      <Box>
                        <Text fontWeight="semibold" fontSize="sm">{event.arcTitle}</Text>
                        <Text fontSize="sm">{event.content}</Text>
                      </Box>
                      {event.confidence_score && (
                        <Badge
                          colorScheme={event.confidence_score > 0.7 ? "green" : event.confidence_score > 0.4 ? "yellow" : "red"}
                        >
                          {Math.round(event.confidence_score * 100)}%
                        </Badge>
                      )}
                    </HStack>
                  </Box>
                ))}
              </VStack>
            </CardBody>
          </Card>
        )}
      </VStack>
    </Box>
  );
};

export default NarrativeArcTimeline;
