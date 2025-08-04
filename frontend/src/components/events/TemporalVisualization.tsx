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
  Spinner,
  Alert,
  AlertIcon,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  SimpleGrid,
  Divider,
  Input,
  Button,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
} from '@chakra-ui/react';
import { FiPlay, FiPause, FiSkipBack, FiSkipForward, FiSearch, FiFilter } from 'react-icons/fi';
import Plot from 'react-plotly.js';

interface TimelineEvent {
  id: string;
  content: string;
  start_timestamp: number;
  end_timestamp: number;
  duration?: number;
  confidence_score?: number;
  extraction_method?: string;
  characters: string[];
  ordinal_position: number;
  arc_id?: string;
  arc_title?: string;
  arc_type?: string;
  progression_id: string;
}

interface ArcData {
  id: string;
  title: string;
  arc_type: string;
  description: string;
  event_count: number;
  total_duration: number;
  events: TimelineEvent[];
}

interface EpisodeTimelineData {
  episode_info: {
    series: string;
    season: string;
    episode: string;
    total_events: number;
    timestamped_events: number;
    total_duration: number;
    arcs_count: number;
  };
  timeline_events: TimelineEvent[];
  arcs: Record<string, ArcData>;
  statistics: {
    events_per_minute: number;
    average_event_duration: number;
    confidence_distribution: {
      high: number;
      medium: number;
      low: number;
    };
  };
}

interface TemporalVisualizationProps {
  series: string;
  season: string;
  episode: string;
  onEventClick?: (event: TimelineEvent) => void;
  height?: number;
}

export const TemporalVisualization: React.FC<TemporalVisualizationProps> = ({
  series,
  season,
  episode,
  onEventClick,
  height = 600
}) => {
  const [timelineData, setTimelineData] = useState<EpisodeTimelineData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [selectedArcTypes, setSelectedArcTypes] = useState<string[]>([]);
  const [confidenceFilter, setConfidenceFilter] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'timeline' | 'density' | 'confidence'>('timeline');

  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Load timeline data
  useEffect(() => {
    loadTimelineData();
  }, [series, season, episode]);

  // Playback control
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && timelineData) {
      interval = setInterval(() => {
        setCurrentTime(prev => {
          const newTime = prev + playbackSpeed;
          if (newTime >= timelineData.episode_info.total_duration) {
            setIsPlaying(false);
            return timelineData.episode_info.total_duration;
          }
          return newTime;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, timelineData]);

  const loadTimelineData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`/api/temporal/timeline/${series}/${season}/${episode}`);
      if (!response.ok) {
        throw new Error(`Failed to load timeline data: ${response.statusText}`);
      }
      const data = await response.json();
      setTimelineData(data);
      
      // Initialize arc type filter with all available types
      const arcTypes = Object.values(data.arcs).map((arc: any) => arc.arc_type);
      setSelectedArcTypes([...new Set(arcTypes)]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load timeline data');
    } finally {
      setLoading(false);
    }
  };

  // Filter events based on current filters
  const filteredEvents = useMemo(() => {
    if (!timelineData) return [];
    
    return timelineData.timeline_events.filter(event => {
      // Arc type filter
      if (selectedArcTypes.length > 0 && event.arc_type && !selectedArcTypes.includes(event.arc_type)) {
        return false;
      }
      
      // Confidence filter
      if (event.confidence_score && event.confidence_score < confidenceFilter) {
        return false;
      }
      
      // Search filter
      if (searchQuery && !event.content.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !event.characters.some(char => char.toLowerCase().includes(searchQuery.toLowerCase()))) {
        return false;
      }
      
      return true;
    });
  }, [timelineData, selectedArcTypes, confidenceFilter, searchQuery]);

  // Create timeline visualization data
  const createTimelineData = () => {
    if (!filteredEvents.length) return null;

    const traces: any[] = [];
    const arcColors: Record<string, string> = {};
    const colorPalette = ['#3182CE', '#38A169', '#D69E2E', '#E53E3E', '#805AD5', '#DD6B20', '#0BC5EA'];
    
    // Assign colors to arc types
    Object.values(timelineData?.arcs || {}).forEach((arc, index) => {
      arcColors[arc.arc_type] = colorPalette[index % colorPalette.length];
    });

    if (viewMode === 'timeline') {
      // Create timeline bars for each event
      const x: number[] = [];
      const y: string[] = [];
      const widths: number[] = [];
      const colors: string[] = [];
      const text: string[] = [];

      filteredEvents.forEach((event, index) => {
        if (event.start_timestamp !== null && event.end_timestamp !== null) {
          x.push(event.start_timestamp);
          y.push(`Event ${index + 1}`);
          widths.push(event.end_timestamp - event.start_timestamp);
          colors.push(arcColors[event.arc_type || 'default'] || '#718096');
          text.push(`${event.content.substring(0, 50)}...\\n${event.characters.join(', ')}`);
        }
      });

      traces.push({
        type: 'bar',
        x: widths,
        y: y,
        base: x,
        orientation: 'h',
        marker: { color: colors },
        text: text,
        textposition: 'none',
        hovertemplate: '%{text}<br>Start: %{base:.1f}s<br>Duration: %{x:.1f}s<extra></extra>',
        name: 'Events'
      });
    } else if (viewMode === 'density') {
      // Create event density over time
      const timeSlots = 60; // 1-minute slots
      const slotDuration = (timelineData?.episode_info.total_duration || 0) / timeSlots;
      const density = new Array(timeSlots).fill(0);
      
      filteredEvents.forEach(event => {
        if (event.start_timestamp !== null) {
          const slot = Math.floor(event.start_timestamp / slotDuration);
          if (slot < timeSlots) density[slot]++;
        }
      });

      traces.push({
        type: 'scatter',
        mode: 'lines+markers',
        x: density.map((_, i) => i * slotDuration),
        y: density,
        fill: 'tonexty',
        name: 'Event Density'
      });
    } else if (viewMode === 'confidence') {
      // Create confidence distribution over time
      const x: number[] = [];
      const y: number[] = [];
      const colors: number[] = [];

      filteredEvents.forEach(event => {
        if (event.start_timestamp !== null && event.confidence_score !== null) {
          x.push(event.start_timestamp);
          y.push(event.confidence_score);
          colors.push(event.confidence_score);
        }
      });

      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: x,
        y: y,
        marker: {
          color: colors,
          colorscale: 'RdYlGn',
          size: 8,
          colorbar: { title: 'Confidence' }
        },
        text: filteredEvents.map(e => e.content.substring(0, 50) + '...'),
        hovertemplate: '%{text}<br>Time: %{x:.1f}s<br>Confidence: %{y:.2f}<extra></extra>',
        name: 'Events'
      });
    }

    return traces;
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (loading) {
    return (
      <Card>
        <CardBody>
          <VStack spacing={4}>
            <Spinner size="xl" />
            <Text>Loading temporal visualization...</Text>
          </VStack>
        </CardBody>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert status="error">
        <AlertIcon />
        {error}
      </Alert>
    );
  }

  if (!timelineData) {
    return (
      <Alert status="info">
        <AlertIcon />
        No timeline data available for {series} {season} {episode}
      </Alert>
    );
  }

  const plotData = createTimelineData();

  return (
    <VStack spacing={6} align="stretch">
      {/* Episode Statistics */}
      <Card>
        <CardBody>
          <Text fontSize="lg" fontWeight="bold" mb={4}>
            Episode Timeline: {series} {season} {episode}
          </Text>
          <SimpleGrid columns={{ base: 2, md: 4 }} spacing={4}>
            <Stat>
              <StatLabel>Total Events</StatLabel>
              <StatNumber>{timelineData.episode_info.total_events}</StatNumber>
              <StatHelpText>{timelineData.episode_info.timestamped_events} timestamped</StatHelpText>
            </Stat>
            <Stat>
              <StatLabel>Duration</StatLabel>
              <StatNumber>{formatTime(timelineData.episode_info.total_duration)}</StatNumber>
              <StatHelpText>{timelineData.statistics.events_per_minute.toFixed(1)} events/min</StatHelpText>
            </Stat>
            <Stat>
              <StatLabel>Narrative Arcs</StatLabel>
              <StatNumber>{timelineData.episode_info.arcs_count}</StatNumber>
              <StatHelpText>Active storylines</StatHelpText>
            </Stat>
            <Stat>
              <StatLabel>Avg Event Duration</StatLabel>
              <StatNumber>{timelineData.statistics.average_event_duration.toFixed(1)}s</StatNumber>
              <StatHelpText>Per timestamped event</StatHelpText>
            </Stat>
          </SimpleGrid>
        </CardBody>
      </Card>

      {/* Filters and Controls */}
      <Card>
        <CardBody>
          <VStack spacing={4}>
            <HStack spacing={4} wrap="wrap">
              <FormControl maxW="200px">
                <FormLabel>View Mode</FormLabel>
                <Select value={viewMode} onChange={(e) => setViewMode(e.target.value as any)}>
                  <option value="timeline">Timeline</option>
                  <option value="density">Event Density</option>
                  <option value="confidence">Confidence Plot</option>
                </Select>
              </FormControl>

              <FormControl maxW="200px">
                <FormLabel>Min Confidence</FormLabel>
                <Slider
                  value={confidenceFilter}
                  onChange={setConfidenceFilter}
                  min={0}
                  max={1}
                  step={0.1}
                >
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb />
                </Slider>
                <Text fontSize="sm">{confidenceFilter.toFixed(1)}</Text>
              </FormControl>

              <FormControl maxW="300px">
                <FormLabel>Search Events</FormLabel>
                <Input
                  placeholder="Search content or characters..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </FormControl>
            </HStack>

            {/* Playback Controls */}
            <HStack spacing={4}>
              <IconButton
                aria-label="Skip back"
                icon={<FiSkipBack />}
                onClick={() => setCurrentTime(Math.max(0, currentTime - 30))}
              />
              <IconButton
                aria-label={isPlaying ? "Pause" : "Play"}
                icon={isPlaying ? <FiPause /> : <FiPlay />}
                onClick={() => setIsPlaying(!isPlaying)}
              />
              <IconButton
                aria-label="Skip forward"
                icon={<FiSkipForward />}
                onClick={() => setCurrentTime(Math.min(timelineData.episode_info.total_duration, currentTime + 30))}
              />
              <Text minW="80px">{formatTime(currentTime)}</Text>
              <Slider
                value={currentTime}
                onChange={setCurrentTime}
                min={0}
                max={timelineData.episode_info.total_duration}
                flex={1}
              >
                <SliderTrack>
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb />
              </Slider>
              <Select value={playbackSpeed} onChange={(e) => setPlaybackSpeed(Number(e.target.value))} maxW="100px">
                <option value={0.5}>0.5x</option>
                <option value={1}>1x</option>
                <option value={2}>2x</option>
                <option value={5}>5x</option>
              </Select>
            </HStack>
          </VStack>
        </CardBody>
      </Card>

      {/* Main Visualization */}
      <Card>
        <CardBody>
          {plotData && (
            <Plot
              data={plotData}
              layout={{
                height: height,
                title: `Event ${viewMode.charAt(0).toUpperCase() + viewMode.slice(1)} - ${series} ${season} ${episode}`,
                xaxis: { 
                  title: 'Time (seconds)',
                  range: viewMode === 'timeline' ? [Math.max(0, currentTime - 60), currentTime + 60] : undefined
                },
                yaxis: { 
                  title: viewMode === 'timeline' ? 'Events' : 
                          viewMode === 'density' ? 'Events per Minute' : 'Confidence Score'
                },
                margin: { l: 100, r: 50, t: 50, b: 50 },
                showlegend: false,
                shapes: viewMode === 'timeline' ? [{
                  type: 'line',
                  x0: currentTime,
                  x1: currentTime,
                  y0: 0,
                  y1: 1,
                  yref: 'paper',
                  line: { color: 'red', width: 2 }
                }] : undefined
              }}
              config={{ responsive: true }}
              onHover={(data) => {
                if (data.points && data.points[0]) {
                  const pointIndex = data.points[0].pointIndex;
                  const event = filteredEvents[pointIndex];
                  if (event && onEventClick) {
                    onEventClick(event);
                  }
                }
              }}
            />
          )}
        </CardBody>
      </Card>

      {/* Event List at Current Time */}
      <Card>
        <CardBody>
          <Text fontSize="md" fontWeight="bold" mb={3}>
            Events at {formatTime(currentTime)} (±30s)
          </Text>
          <VStack spacing={2} align="stretch" maxH="200px" overflowY="auto">
            {filteredEvents
              .filter(event => 
                event.start_timestamp !== null && 
                Math.abs(event.start_timestamp - currentTime) <= 30
              )
              .map(event => (
                <Box
                  key={event.id}
                  p={3}
                  bg={useColorModeValue('gray.50', 'gray.700')}
                  borderRadius="md"
                  cursor="pointer"
                  onClick={() => onEventClick?.(event)}
                  _hover={{ bg: useColorModeValue('gray.100', 'gray.600') }}
                >
                  <HStack justify="space-between">
                    <VStack align="start" spacing={1}>
                      <Text fontSize="sm" fontWeight="medium">{event.content}</Text>
                      <HStack spacing={2}>
                        <Badge colorScheme="blue">{event.arc_type}</Badge>
                        {event.characters.map(char => (
                          <Badge key={char} variant="outline">{char}</Badge>
                        ))}
                      </HStack>
                    </VStack>
                    <VStack align="end" spacing={1}>
                      <Text fontSize="xs">{formatTime(event.start_timestamp || 0)}</Text>
                      {event.confidence_score && (
                        <Badge colorScheme={event.confidence_score >= 0.8 ? 'green' : event.confidence_score >= 0.5 ? 'yellow' : 'red'}>
                          {(event.confidence_score * 100).toFixed(0)}%
                        </Badge>
                      )}
                    </VStack>
                  </HStack>
                </Box>
              ))
            }
          </VStack>
        </CardBody>
      </Card>
    </VStack>
  );
};
