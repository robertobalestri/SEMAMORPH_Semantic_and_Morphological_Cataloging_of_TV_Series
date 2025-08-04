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
  Divider,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Tag,
  TagLabel,
  TagCloseButton,
  Wrap,
  WrapItem,
  Button,
  Input,
  InputGroup,
  InputLeftElement,
} from '@chakra-ui/react';
import { SearchIcon, TriangleUpIcon, MinusIcon, RepeatIcon } from '@chakra-ui/icons';

// Event-specific interface (linked to progressions)
interface TimelineEvent {
  id: string;
  content: string;
  start_timestamp: string | number; // Can be string (HH:MM:SS,mmm) or number (seconds)
  end_timestamp: string | number;   // Can be string (HH:MM:SS,mmm) or number (seconds)
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

interface EventTimelineProps {
  series: string;
  season: string;
  episode: string;
  height?: number;
  onEventClick?: (event: TimelineEvent) => void;
  onTimeSeek?: (timestamp: number) => void;
  autoPlay?: boolean;
  showAnalytics?: boolean;
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
};

const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
};

// Convert timestamp to seconds (handles both string HH:MM:SS,mmm and number formats)
const timestampToSeconds = (timestamp: string | number): number => {
  if (typeof timestamp === 'number') {
    return timestamp;
  }
  
  if (typeof timestamp === 'string') {
    try {
      // Parse HH:MM:SS,mmm format
      const [timePart, msPart] = timestamp.split(',');
      const [hours, minutes, seconds] = timePart.split(':').map(Number);
      const milliseconds = parseInt(msPart || '0');
      return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000;
    } catch (error) {
      console.warn('Failed to parse timestamp:', timestamp);
      return 0;
    }
  }
  
  return 0;
};

const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return 'green';
  if (confidence >= 0.6) return 'yellow';
  if (confidence >= 0.4) return 'orange';
  return 'red';
};

const getExtractionMethodColor = (method: string): string => {
  switch (method) {
    case 'scene_matching': return 'blue';
    case 'multiagent_extraction': return 'purple';
    case 'manual_annotation': return 'green';
    default: return 'gray';
  }
};

const EventTimeline: React.FC<EventTimelineProps> = ({
  series,
  season,
  episode,
  height = 500,
  onEventClick,
  onTimeSeek,
  autoPlay = false,
  showAnalytics = true
}) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  
  // Filters
  const [searchTerm, setSearchTerm] = useState('');
  const [minConfidence, setMinConfidence] = useState(0);
  const [selectedCharacters, setSelectedCharacters] = useState<Set<string>>(new Set());
  const [selectedMethods, setSelectedMethods] = useState<Set<string>>(new Set());
  const [selectedProgressions, setSelectedProgressions] = useState<Set<string>>(new Set());
  const [selectedArcs, setSelectedArcs] = useState<Set<string>>(new Set());
  const [showHighConfidenceOnly, setShowHighConfidenceOnly] = useState(false);
  const [groupByProgression, setGroupByProgression] = useState(true);

  // Load events for the episode
  useEffect(() => {
    loadEvents();
  }, [series, season, episode]);

  // Auto-play functionality
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && maxTimestamp > 0) {
      interval = setInterval(() => {
        setCurrentTime(prev => {
          const next = prev + (playbackSpeed * 0.1);
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

  const loadEvents = async () => {
    setLoading(true);
    try {
      // Load events with progression and arc context
      const response = await fetch(`/api/events/episode/${series}/${season}/${episode}?include_context=true`);
      if (response.ok) {
        const data: TimelineEvent[] = await response.json();
        
        // Deduplicate events by ID to prevent duplicate key errors
        const uniqueEventsMap = new Map<string, TimelineEvent>();
        data.forEach(event => {
          if (event.id && event.start_timestamp !== undefined && event.end_timestamp !== undefined) {
            uniqueEventsMap.set(event.id, event);
          }
        });
        
        // Sort by start timestamp (convert to seconds for proper sorting)
        const sortedEvents = Array.from(uniqueEventsMap.values())
          .sort((a, b) => timestampToSeconds(a.start_timestamp) - timestampToSeconds(b.start_timestamp));
        
        setEvents(sortedEvents);
        
        // Auto-select all characters, methods, progressions, and arcs initially
        const allCharacters = new Set<string>();
        const allMethods = new Set<string>();
        const allProgressions = new Set<string>();
        const allArcs = new Set<string>();
        
        sortedEvents.forEach(event => {
          event.characters_involved?.forEach(char => allCharacters.add(char));
          if (event.extraction_method) allMethods.add(event.extraction_method);
          if (event.progression_id) allProgressions.add(event.progression_id);
          if (event.arc_id) allArcs.add(event.arc_id);
        });
        
        setSelectedCharacters(allCharacters);
        setSelectedMethods(allMethods);
        setSelectedProgressions(allProgressions);
        setSelectedArcs(allArcs);
      }
    } catch (error) {
      console.error('Failed to load events:', error);
    } finally {
      setLoading(false);
    }
  };

  // Calculate timeline bounds
  const { minTimestamp, maxTimestamp, duration } = useMemo(() => {
    if (events.length === 0) return { minTimestamp: 0, maxTimestamp: 3600, duration: 3600 };
    
    const timestamps = events.flatMap(e => [
      timestampToSeconds(e.start_timestamp), 
      timestampToSeconds(e.end_timestamp)
    ]);
    const min = Math.min(...timestamps);
    const max = Math.max(...timestamps);
    
    return {
      minTimestamp: min,
      maxTimestamp: max,
      duration: max - min
    };
  }, [events]);

  // Filter events
  const filteredEvents = useMemo(() => {
    return events.filter(event => {
      // Search filter
      if (searchTerm && !event.content.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }
      
      // Confidence filter
      if (event.confidence_score !== undefined) {
        if (event.confidence_score < minConfidence) return false;
        if (showHighConfidenceOnly && event.confidence_score < 0.8) return false;
      }
      
      // Character filter
      if (selectedCharacters.size > 0) {
        const hasSelectedCharacter = event.characters_involved?.some(char => 
          selectedCharacters.has(char)
        );
        if (!hasSelectedCharacter) return false;
      }
      
      // Method filter
      if (selectedMethods.size > 0 && event.extraction_method) {
        if (!selectedMethods.has(event.extraction_method)) return false;
      }
      
      // Progression filter
      if (selectedProgressions.size > 0) {
        if (!selectedProgressions.has(event.progression_id)) return false;
      }
      
      // Arc filter
      if (selectedArcs.size > 0 && event.arc_id) {
        if (!selectedArcs.has(event.arc_id)) return false;
      }
      
      return true;
    });
  }, [events, searchTerm, minConfidence, selectedCharacters, selectedMethods, selectedProgressions, selectedArcs, showHighConfidenceOnly]);

  // Get active events at current time
  const activeEvents = useMemo(() => {
    return filteredEvents.filter(event => {
      const startSeconds = timestampToSeconds(event.start_timestamp);
      const endSeconds = timestampToSeconds(event.end_timestamp);
      return startSeconds <= currentTime && endSeconds >= currentTime;
    });
  }, [filteredEvents, currentTime]);

  // Analytics
  const analytics = useMemo(() => {
    if (filteredEvents.length === 0) return null;
    
    const avgConfidence = filteredEvents
      .filter(e => e.confidence_score !== undefined)
      .reduce((sum, e) => sum + e.confidence_score!, 0) / filteredEvents.length;
    
    const avgDuration = filteredEvents
      .reduce((sum, e) => {
        const startSeconds = timestampToSeconds(e.start_timestamp);
        const endSeconds = timestampToSeconds(e.end_timestamp);
        return sum + (endSeconds - startSeconds);
      }, 0) / filteredEvents.length;
    
    const density = filteredEvents.length / (duration / 60); // events per minute
    
    const methodCounts = filteredEvents.reduce((acc, e) => {
      const method = e.extraction_method || 'unknown';
      acc[method] = (acc[method] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    return {
      totalEvents: filteredEvents.length,
      avgConfidence,
      avgDuration,
      density,
      methodCounts
    };
  }, [filteredEvents, duration]);

  const handleTimeSeek = (timestamp: number) => {
    setCurrentTime(timestamp);
    onTimeSeek?.(timestamp);
  };

  const resetPlayback = () => {
    setCurrentTime(minTimestamp);
    setIsPlaying(false);
  };

  const toggleCharacterFilter = (character: string) => {
    const newSelection = new Set(selectedCharacters);
    if (newSelection.has(character)) {
      newSelection.delete(character);
    } else {
      newSelection.add(character);
    }
    setSelectedCharacters(newSelection);
  };

  const toggleMethodFilter = (method: string) => {
    const newSelection = new Set(selectedMethods);
    if (newSelection.has(method)) {
      newSelection.delete(method);
    } else {
      newSelection.add(method);
    }
    setSelectedMethods(newSelection);
  };

  const toggleProgressionFilter = (progressionId: string) => {
    const newSelection = new Set(selectedProgressions);
    if (newSelection.has(progressionId)) {
      newSelection.delete(progressionId);
    } else {
      newSelection.add(progressionId);
    }
    setSelectedProgressions(newSelection);
  };

  const toggleArcFilter = (arcId: string) => {
    const newSelection = new Set(selectedArcs);
    if (newSelection.has(arcId)) {
      newSelection.delete(arcId);
    } else {
      newSelection.add(arcId);
    }
    setSelectedArcs(newSelection);
  };

  if (loading) {
    return <Progress size="sm" isIndeterminate />;
  }

  return (
    <Box>
      <VStack spacing={4} align="stretch">
        {/* Header */}
        <Card>
          <CardBody>
            <Flex justify="space-between" align="center">
              <VStack align="start" spacing={1}>
                <Text fontSize="lg" fontWeight="bold">Event Timeline</Text>
                <Text fontSize="sm" color="gray.600">
                  {series} • {season} • {episode}
                </Text>
              </VStack>
              <HStack>
                <Text fontSize="sm" color="gray.600">
                  {filteredEvents.length} events • {formatDuration(duration)}
                </Text>
              </HStack>
            </Flex>
          </CardBody>
        </Card>

        {/* Controls */}
        <Card>
          <CardBody>
            <VStack spacing={4}>
              {/* Playback Controls */}
              <Flex justify="space-between" align="center" w="100%">
                <HStack spacing={3}>
                  <IconButton
                    aria-label={isPlaying ? "Pause" : "Play"}
                    icon={isPlaying ? <MinusIcon /> : <TriangleUpIcon />}
                    onClick={() => setIsPlaying(!isPlaying)}
                    colorScheme="blue"
                    isDisabled={filteredEvents.length === 0}
                  />
                  <IconButton
                    aria-label="Reset"
                    icon={<RepeatIcon />}
                    onClick={resetPlayback}
                    variant="outline"
                  />
                  <VStack spacing={0} align="start">
                    <Text fontSize="sm" fontWeight="semibold">
                      {formatTime(currentTime)}
                    </Text>
                    <Text fontSize="xs" color="gray.500">
                      of {formatTime(maxTimestamp)}
                    </Text>
                  </VStack>
                  <Select
                    size="sm"
                    width="80px"
                    value={playbackSpeed}
                    onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                  >
                    <option value={0.1}>0.1x</option>
                    <option value={0.5}>0.5x</option>
                    <option value={1}>1x</option>
                    <option value={2}>2x</option>
                    <option value={5}>5x</option>
                  </Select>
                </HStack>

                <HStack>
                  <FormControl display="flex" alignItems="center">
                    <FormLabel htmlFor="high-confidence" mb="0" fontSize="sm">
                      High Confidence Only
                    </FormLabel>
                    <Switch
                      id="high-confidence"
                      isChecked={showHighConfidenceOnly}
                      onChange={(e) => setShowHighConfidenceOnly(e.target.checked)}
                      size="sm"
                    />
                  </FormControl>
                  <FormControl display="flex" alignItems="center">
                    <FormLabel htmlFor="group-progression" mb="0" fontSize="sm">
                      Group by Progression
                    </FormLabel>
                    <Switch
                      id="group-progression"
                      isChecked={groupByProgression}
                      onChange={(e) => setGroupByProgression(e.target.checked)}
                      size="sm"
                    />
                  </FormControl>
                </HStack>
              </Flex>

              {/* Time Scrubber */}
              <Box w="100%">
                <Slider
                  value={isNaN(currentTime) ? 0 : currentTime}
                  min={isNaN(minTimestamp) ? 0 : minTimestamp}
                  max={isNaN(maxTimestamp) ? 3600 : maxTimestamp}
                  onChange={handleTimeSeek}
                  step={0.1}
                >
                  <SliderTrack bg="gray.200">
                    <SliderFilledTrack bg="blue.400" />
                  </SliderTrack>
                  <SliderThumb boxSize={4} />
                </Slider>
              </Box>

              {/* Search and Filters */}
              <HStack w="100%" spacing={4}>
                <InputGroup size="sm" flex={1}>
                  <InputLeftElement pointerEvents="none">
                    <SearchIcon color="gray.300" />
                  </InputLeftElement>
                  <Input
                    placeholder="Search events..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                </InputGroup>
                <Box>
                  <Text fontSize="xs" mb={1}>Min Confidence: {(isNaN(minConfidence) ? 0 : minConfidence).toFixed(1)}</Text>
                  <Slider
                    value={isNaN(minConfidence) ? 0 : minConfidence}
                    min={0}
                    max={1}
                    step={0.1}
                    onChange={setMinConfidence}
                    width="100px"
                    size="sm"
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb boxSize={3} />
                  </Slider>
                </Box>
              </HStack>
            </VStack>
          </CardBody>
        </Card>

        {/* Character and Method Filters */}
        <Card>
          <CardBody>
            <VStack spacing={3} align="stretch">
              <Box>
                <Text fontSize="sm" fontWeight="semibold" mb={2}>Characters:</Text>
                <Wrap>
                  {Array.from(new Set(events.flatMap(e => e.characters_involved || []))).map(character => (
                    <WrapItem key={character}>
                      <Tag
                        size="sm"
                        colorScheme={selectedCharacters.has(character) ? "blue" : "gray"}
                        cursor="pointer"
                        onClick={() => toggleCharacterFilter(character)}
                      >
                        <TagLabel>{character}</TagLabel>
                        {selectedCharacters.has(character) && (
                          <TagCloseButton onClick={(e) => {
                            e.stopPropagation();
                            toggleCharacterFilter(character);
                          }} />
                        )}
                      </Tag>
                    </WrapItem>
                  ))}
                </Wrap>
              </Box>
              
              <Box>
                <Text fontSize="sm" fontWeight="semibold" mb={2}>Extraction Methods:</Text>
                <Wrap>
                  {Array.from(new Set(events.map(e => e.extraction_method).filter(Boolean))).map(method => (
                    <WrapItem key={method}>
                      <Tag
                        size="sm"
                        colorScheme={selectedMethods.has(method!) ? getExtractionMethodColor(method!) : "gray"}
                        cursor="pointer"
                        onClick={() => toggleMethodFilter(method!)}
                      >
                        <TagLabel>{method}</TagLabel>
                        {selectedMethods.has(method!) && (
                          <TagCloseButton onClick={(e) => {
                            e.stopPropagation();
                            toggleMethodFilter(method!);
                          }} />
                        )}
                      </Tag>
                    </WrapItem>
                  ))}
                </Wrap>
              </Box>
              
              <Box>
                <Text fontSize="sm" fontWeight="semibold" mb={2}>Narrative Arcs:</Text>
                <Wrap>
                  {Array.from(
                    events.reduce((arcMap, e) => {
                      if (e.arc_id && !arcMap.has(e.arc_id)) {
                        arcMap.set(e.arc_id, { id: e.arc_id, title: e.arc_title });
                      }
                      return arcMap;
                    }, new Map<string, { id: string; title?: string }>()).values()
                  ).map(arc => (
                    <WrapItem key={arc.id}>
                      <Tag
                        size="sm"
                        colorScheme={selectedArcs.has(arc.id) ? "purple" : "gray"}
                        cursor="pointer"
                        onClick={() => toggleArcFilter(arc.id)}
                      >
                        <TagLabel>{arc.title || `Arc ${arc.id.slice(0, 8)}`}</TagLabel>
                        {selectedArcs.has(arc.id) && (
                          <TagCloseButton onClick={(e) => {
                            e.stopPropagation();
                            toggleArcFilter(arc.id);
                          }} />
                        )}
                      </Tag>
                    </WrapItem>
                  ))}
                </Wrap>
              </Box>
              
              <Box>
                <Text fontSize="sm" fontWeight="semibold" mb={2}>Progressions:</Text>
                <Wrap>
                  {Array.from(
                    events.reduce((progressionMap, e) => {
                      if (e.progression_id && !progressionMap.has(e.progression_id)) {
                        progressionMap.set(e.progression_id, { id: e.progression_id, content: e.progression_content });
                      }
                      return progressionMap;
                    }, new Map<string, { id: string; content?: string }>()).values()
                  ).map(progression => (
                    <WrapItem key={progression.id}>
                      <Tag
                        size="sm"
                        colorScheme={selectedProgressions.has(progression.id) ? "orange" : "gray"}
                        cursor="pointer"
                        onClick={() => toggleProgressionFilter(progression.id)}
                      >
                        <TagLabel>{progression.content?.slice(0, 30) || `Progression ${progression.id.slice(0, 8)}`}...</TagLabel>
                        {selectedProgressions.has(progression.id) && (
                          <TagCloseButton onClick={(e) => {
                            e.stopPropagation();
                            toggleProgressionFilter(progression.id);
                          }} />
                        )}
                      </Tag>
                    </WrapItem>
                  ))}
                </Wrap>
              </Box>
            </VStack>
          </CardBody>
        </Card>

        {/* Analytics */}
        {showAnalytics && analytics && (
          <Card>
            <CardBody>
              <Text fontSize="sm" fontWeight="semibold" mb={3}>Event Analytics</Text>
              <HStack spacing={6} wrap="wrap">
                <Stat size="sm">
                  <StatLabel>Total Events</StatLabel>
                  <StatNumber>{analytics.totalEvents}</StatNumber>
                </Stat>
                <Stat size="sm">
                  <StatLabel>Avg Confidence</StatLabel>
                  <StatNumber>{(analytics.avgConfidence * 100).toFixed(0)}%</StatNumber>
                </Stat>
                <Stat size="sm">
                  <StatLabel>Avg Duration</StatLabel>
                  <StatNumber>{formatDuration(analytics.avgDuration)}</StatNumber>
                </Stat>
                <Stat size="sm">
                  <StatLabel>Event Density</StatLabel>
                  <StatNumber>{analytics.density.toFixed(1)}</StatNumber>
                  <StatHelpText>events/min</StatHelpText>
                </Stat>
              </HStack>
            </CardBody>
          </Card>
        )}

        {/* Timeline Visualization */}
        <Card>
          <CardBody>
            <Box position="relative" height={`${height}px`} overflow="auto" bg="gray.50" borderRadius="md">
              {filteredEvents.length === 0 ? (
                <Flex justify="center" align="center" height="100%">
                  <Text color="gray.500">No events match the current filters</Text>
                </Flex>
              ) : (
                <Box position="relative" width="100%" height="100%" p={4}>
                  {/* Time markers */}
                  {Array.from({ length: Math.ceil(duration / 60) }, (_, i) => {
                    const time = minTimestamp + (i * 60);
                    const left = ((time - minTimestamp) / duration) * 100;
                    return (
                      <Box
                        key={i}
                        position="absolute"
                        left={`${left}%`}
                        top="0"
                        bottom="0"
                        width="1px"
                        bg="gray.300"
                        _before={{
                          content: `"${formatTime(time)}"`,
                          position: "absolute",
                          top: "-20px",
                          left: "-20px",
                          fontSize: "xs",
                          color: "gray.500",
                          whiteSpace: "nowrap"
                        }}
                      />
                    );
                  })}

                  {/* Current Time Indicator */}
                  <Box
                    position="absolute"
                    left={`${((currentTime - minTimestamp) / duration) * 100}%`}
                    top="0"
                    bottom="0"
                    width="3px"
                    bg="red.500"
                    zIndex={20}
                    _before={{
                      content: `"${formatTime(currentTime)}"`,
                      position: "absolute",
                      top: "-25px",
                      left: "-30px",
                      bg: "red.500",
                      color: "white",
                      fontSize: "xs",
                      px: 2,
                      py: 1,
                      borderRadius: "md",
                      whiteSpace: "nowrap"
                    }}
                  />

                  {/* Events */}
                  {filteredEvents.map((event, index) => {
                    const startSeconds = timestampToSeconds(event.start_timestamp);
                    const endSeconds = timestampToSeconds(event.end_timestamp);
                    const left = ((startSeconds - minTimestamp) / duration) * 100;
                    const width = Math.max(((endSeconds - startSeconds) / duration) * 100, 0.2);
                    const top = 40 + (index % 15) * 32; // Stack events with spacing
                    const isActive = activeEvents.includes(event);

                    return (
                      <Tooltip
                        key={event.id}
                        label={
                          <Box maxW="300px">
                            {event.arc_title && (
                              <Text fontWeight="bold" mb={1} color="purple.200">
                                📚 {event.arc_title}
                              </Text>
                            )}
                            {event.progression_content && (
                              <Text fontSize="xs" mb={2} color="orange.200">
                                🔗 {event.progression_content.slice(0, 60)}...
                              </Text>
                            )}
                            <Text fontWeight="bold" mb={1}>{event.content}</Text>
                            <Text fontSize="xs" mb={1}>
                              {formatTime(startSeconds)} - {formatTime(endSeconds)}
                              ({formatDuration(endSeconds - startSeconds)})
                            </Text>
                            {event.characters_involved && event.characters_involved.length > 0 && (
                              <Text fontSize="xs" mb={1}>
                                Characters: {event.characters_involved.join(', ')}
                              </Text>
                            )}
                            {event.confidence_score !== undefined && (
                              <Text fontSize="xs" mb={1}>
                                Confidence: {Math.round(event.confidence_score * 100)}%
                              </Text>
                            )}
                            {event.extraction_method && (
                              <Text fontSize="xs">
                                Method: {event.extraction_method}
                              </Text>
                            )}
                            <Text fontSize="xs" mt={1} color="gray.300">
                              Position: #{event.ordinal_position} in progression
                            </Text>
                          </Box>
                        }
                        placement="top"
                      >
                        <Box
                          position="absolute"
                          left={`${left}%`}
                          top={`${top}px`}
                          width={`${width}%`}
                          height="28px"
                          bg={event.confidence_score ? `${getConfidenceColor(event.confidence_score)}.400` : 'gray.400'}
                          borderRadius="md"
                          cursor="pointer"
                          opacity={isActive ? 1 : 0.8}
                          border={isActive ? "3px solid white" : "1px solid white"}
                          boxShadow={isActive ? "0 0 10px rgba(0,0,0,0.3)" : "0 1px 3px rgba(0,0,0,0.1)"}
                          _hover={{ 
                            opacity: 1, 
                            transform: "translateY(-2px)",
                            boxShadow: "0 4px 12px rgba(0,0,0,0.2)"
                          }}
                          transition="all 0.2s"
                          onClick={() => {
                            handleTimeSeek(startSeconds);
                            onEventClick?.(event);
                          }}
                        >
                          <Text
                            fontSize="xs"
                            color="white"
                            p={1}
                            noOfLines={2}
                            fontWeight="semibold"
                            textShadow="0 1px 2px rgba(0,0,0,0.5)"
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

        {/* Active Events Display */}
        {activeEvents.length > 0 && (
          <Card>
            <CardBody>
              <Text fontSize="sm" fontWeight="semibold" mb={3}>
                Active Events at {formatTime(currentTime)}:
              </Text>
              <VStack spacing={3} align="stretch">
                {activeEvents.map(event => (
                  <Box
                    key={event.id}
                    p={4}
                    bg={useColorModeValue('blue.50', 'blue.900')}
                    borderRadius="md"
                    borderLeft={`4px solid ${event.confidence_score ? getConfidenceColor(event.confidence_score) : 'gray'}.400`}
                    cursor="pointer"
                    onClick={() => onEventClick?.(event)}
                    _hover={{ bg: useColorModeValue('blue.100', 'blue.800') }}
                  >
                    <VStack align="stretch" spacing={2}>
                      {event.arc_title && (
                        <HStack>
                          <Badge colorScheme="purple" size="sm">📚 {event.arc_title}</Badge>
                        </HStack>
                      )}
                      
                      <HStack justify="space-between">
                        <Box flex={1}>
                          <Text fontWeight="semibold">{event.content}</Text>
                          {event.progression_content && (
                            <Text fontSize="xs" color="gray.600" mt={1}>
                              🔗 Progression: {event.progression_content.slice(0, 80)}...
                            </Text>
                          )}
                        </Box>
                        <HStack>
                          {event.confidence_score && (
                            <Badge colorScheme={getConfidenceColor(event.confidence_score)}>
                              {Math.round(event.confidence_score * 100)}%
                            </Badge>
                          )}
                          {event.extraction_method && (
                            <Badge colorScheme={getExtractionMethodColor(event.extraction_method)}>
                              {event.extraction_method}
                            </Badge>
                          )}
                        </HStack>
                      </HStack>
                      
                      <HStack justify="space-between" fontSize="sm" color="gray.600">
                        <Text>
                          {formatTime(timestampToSeconds(event.start_timestamp))} - {formatTime(timestampToSeconds(event.end_timestamp))}
                        </Text>
                        <Text>
                          Position #{event.ordinal_position}
                        </Text>
                        {event.characters_involved && event.characters_involved.length > 0 && (
                          <Text>
                            Characters: {event.characters_involved.join(', ')}
                          </Text>
                        )}
                      </HStack>
                    </VStack>
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

export default EventTimeline;
