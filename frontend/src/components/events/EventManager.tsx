import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  VStack,
  HStack,
  Text,
  Badge,
  Card,
  CardBody,
  CardHeader,
  Heading,
  Input,
  Textarea,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  useDisclosure,
  useToast,
  IconButton,
  Tooltip,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Flex,
  Divider,
  Alert,
  AlertIcon,
  Progress,
} from '@chakra-ui/react';
import { EditIcon, DeleteIcon, TimeIcon, RepeatIcon } from '@chakra-ui/icons';
import { Event, EventCreateRequest, EventUpdateRequest, EventExtractionRequest } from '../../architecture/types/arc';

interface EventManagerProps {
  progressionId: string;
  series: string;
  season: string;
  episode: string;
  onEventsUpdated?: () => void;
}

interface EventEditModalProps {
  event: Event | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: (eventData: EventCreateRequest | EventUpdateRequest, eventId?: string) => void;
  progressionId: string;
  series: string;
  season: string;
  episode: string;
}

const formatTimestamp = (seconds?: number): string => {
  if (!seconds) return '--:--';
  
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const EventEditModal: React.FC<EventEditModalProps> = ({
  event,
  isOpen,
  onClose,
  onSave,
  progressionId,
  series,
  season,
  episode
}) => {
  const [content, setContent] = useState('');
  const [startTimestamp, setStartTimestamp] = useState<number | undefined>();
  const [endTimestamp, setEndTimestamp] = useState<number | undefined>();
  const [ordinalPosition, setOrdinalPosition] = useState(1);
  const [confidenceScore, setConfidenceScore] = useState<number | undefined>();
  const [extractionMethod, setExtractionMethod] = useState('');

  useEffect(() => {
    if (event) {
      setContent(event.content);
      setStartTimestamp(event.start_timestamp);
      setEndTimestamp(event.end_timestamp);
      setOrdinalPosition(event.ordinal_position);
      setConfidenceScore(event.confidence_score);
      setExtractionMethod(event.extraction_method || '');
    } else {
      // Reset for new event
      setContent('');
      setStartTimestamp(undefined);
      setEndTimestamp(undefined);
      setOrdinalPosition(1);
      setConfidenceScore(undefined);
      setExtractionMethod('manual');
    }
  }, [event, isOpen]);

  const handleSave = () => {
    if (event) {
      // Update existing event
      const updateData: EventUpdateRequest = {
        content,
        start_timestamp: startTimestamp,
        end_timestamp: endTimestamp,
        ordinal_position: ordinalPosition,
        confidence_score: confidenceScore,
        extraction_method: extractionMethod
      };
      onSave(updateData, event.id);
    } else {
      // Create new event
      const createData: EventCreateRequest = {
        progression_id: progressionId,
        content,
        series,
        season,
        episode,
        start_timestamp: startTimestamp,
        end_timestamp: endTimestamp,
        ordinal_position: ordinalPosition,
        confidence_score: confidenceScore,
        extraction_method: extractionMethod
      };
      onSave(createData);
    }
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{event ? 'Edit Event' : 'Create New Event'}</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4}>
            <Box w="100%">
              <Text mb={2} fontWeight="semibold">Event Content</Text>
              <Textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Describe what happens in this event..."
                rows={4}
              />
            </Box>

            <HStack w="100%" spacing={4}>
              <Box flex={1}>
                <Text mb={2} fontWeight="semibold">Start Time (seconds)</Text>
                <NumberInput
                  value={startTimestamp || ''}
                  onChange={(valueString) => setStartTimestamp(parseFloat(valueString) || undefined)}
                  min={0}
                  precision={1}
                >
                  <NumberInputField placeholder="0.0" />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
                <Text fontSize="sm" color="gray.500">
                  {formatTimestamp(startTimestamp)}
                </Text>
              </Box>

              <Box flex={1}>
                <Text mb={2} fontWeight="semibold">End Time (seconds)</Text>
                <NumberInput
                  value={endTimestamp || ''}
                  onChange={(valueString) => setEndTimestamp(parseFloat(valueString) || undefined)}
                  min={startTimestamp || 0}
                  precision={1}
                >
                  <NumberInputField placeholder="0.0" />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
                <Text fontSize="sm" color="gray.500">
                  {formatTimestamp(endTimestamp)}
                </Text>
              </Box>
            </HStack>

            <HStack w="100%" spacing={4}>
              <Box flex={1}>
                <Text mb={2} fontWeight="semibold">Position</Text>
                <NumberInput
                  value={ordinalPosition}
                  onChange={(valueString) => setOrdinalPosition(parseInt(valueString) || 1)}
                  min={1}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </Box>

              <Box flex={1}>
                <Text mb={2} fontWeight="semibold">Confidence Score</Text>
                <NumberInput
                  value={confidenceScore || ''}
                  onChange={(valueString) => setConfidenceScore(parseFloat(valueString) || undefined)}
                  min={0}
                  max={1}
                  step={0.1}
                  precision={2}
                >
                  <NumberInputField placeholder="0.0 - 1.0" />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </Box>
            </HStack>

            <Box w="100%">
              <Text mb={2} fontWeight="semibold">Extraction Method</Text>
              <Input
                value={extractionMethod}
                onChange={(e) => setExtractionMethod(e.target.value)}
                placeholder="e.g., manual, scene_matching, llm_extraction"
              />
            </Box>
          </VStack>
        </ModalBody>

        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button colorScheme="blue" onClick={handleSave} isDisabled={!content.trim()}>
            {event ? 'Update' : 'Create'} Event
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

const EventManager: React.FC<EventManagerProps> = ({
  progressionId,
  series,
  season,
  episode,
  onEventsUpdated
}) => {
  const [events, setEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();

  // Load events on component mount
  useEffect(() => {
    loadEvents();
  }, [progressionId]);

  const loadEvents = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/events/progression/${progressionId}`);
      if (response.ok) {
        const eventsData = await response.json();
        setEvents(eventsData);
      } else {
        throw new Error('Failed to load events');
      }
    } catch (error) {
      toast({
        title: 'Error Loading Events',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleExtractEvents = async () => {
    setExtracting(true);
    try {
      const requestData: EventExtractionRequest = {
        force_reextraction: events.length > 0 // Force if events already exist
      };

      const response = await fetch(`/api/progressions/${progressionId}/extract-events`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          toast({
            title: 'Events Extracted Successfully',
            description: `Extracted ${result.events_extracted} events`,
            status: 'success',
            duration: 5000,
            isClosable: true,
          });
          await loadEvents();
          onEventsUpdated?.();
        } else {
          throw new Error(result.error_message || 'Extraction failed');
        }
      } else {
        throw new Error('Failed to extract events');
      }
    } catch (error) {
      toast({
        title: 'Event Extraction Failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setExtracting(false);
    }
  };

  const handleSaveEvent = async (eventData: EventCreateRequest | EventUpdateRequest, eventId?: string) => {
    try {
      let response;
      
      if (eventId) {
        // Update existing event
        response = await fetch(`/api/events/${eventId}`, {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(eventData),
        });
      } else {
        // Create new event
        response = await fetch('/api/events', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(eventData),
        });
      }

      if (response.ok) {
        toast({
          title: eventId ? 'Event Updated' : 'Event Created',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        await loadEvents();
        onEventsUpdated?.();
      } else {
        throw new Error('Failed to save event');
      }
    } catch (error) {
      toast({
        title: 'Error Saving Event',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleDeleteEvent = async (eventId: string) => {
    if (!confirm('Are you sure you want to delete this event?')) return;

    try {
      const response = await fetch(`/api/events/${eventId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        toast({
          title: 'Event Deleted',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        await loadEvents();
        onEventsUpdated?.();
      } else {
        throw new Error('Failed to delete event');
      }
    } catch (error) {
      toast({
        title: 'Error Deleting Event',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const openEditModal = (event: Event | null = null) => {
    setSelectedEvent(event);
    onOpen();
  };

  if (loading) {
    return <Progress size="sm" isIndeterminate />;
  }

  return (
    <Box>
      <VStack spacing={4} align="stretch">
        {/* Header */}
        <Flex justify="space-between" align="center">
          <Heading size="md">Events Timeline</Heading>
          <HStack>
            <Tooltip label="Extract events from progression content using AI">
              <Button
                leftIcon={<RepeatIcon />}
                onClick={handleExtractEvents}
                isLoading={extracting}
                loadingText="Extracting..."
                size="sm"
                colorScheme="purple"
              >
                Extract Events
              </Button>
            </Tooltip>
            <Button leftIcon={<TimeIcon />} onClick={() => openEditModal()} size="sm" colorScheme="blue">
              Add Event
            </Button>
          </HStack>
        </Flex>

        {/* Events List or Empty State */}
        {events.length === 0 ? (
          <Alert status="info">
            <AlertIcon />
            No events found for this progression. Use "Extract Events" to automatically generate events from the progression content, or "Add Event" to create them manually.
          </Alert>
        ) : (
          <VStack spacing={3} align="stretch">
            {events
              .sort((a, b) => a.ordinal_position - b.ordinal_position)
              .map((event, index) => (
                <Card key={event.id} size="sm">
                  <CardBody>
                    <Flex justify="space-between" align="flex-start">
                      <Box flex={1}>
                        <HStack mb={2}>
                          <Badge colorScheme="blue">#{event.ordinal_position}</Badge>
                          {event.start_timestamp && (
                            <Badge colorScheme="green">
                              {formatTimestamp(event.start_timestamp)} - {formatTimestamp(event.end_timestamp)}
                            </Badge>
                          )}
                          {event.confidence_score && (
                            <Badge 
                              colorScheme={event.confidence_score > 0.7 ? "green" : event.confidence_score > 0.4 ? "yellow" : "red"}
                            >
                              {Math.round(event.confidence_score * 100)}% confidence
                            </Badge>
                          )}
                        </HStack>
                        <Text fontSize="sm">{event.content}</Text>
                        {event.extraction_method && (
                          <Text fontSize="xs" color="gray.500" mt={1}>
                            Method: {event.extraction_method}
                          </Text>
                        )}
                      </Box>
                      <HStack spacing={1}>
                        <IconButton
                          aria-label="Edit event"
                          icon={<EditIcon />}
                          size="sm"
                          variant="ghost"
                          onClick={() => openEditModal(event)}
                        />
                        <IconButton
                          aria-label="Delete event"
                          icon={<DeleteIcon />}
                          size="sm"
                          variant="ghost"
                          colorScheme="red"
                          onClick={() => handleDeleteEvent(event.id)}
                        />
                      </HStack>
                    </Flex>
                  </CardBody>
                </Card>
              ))}
          </VStack>
        )}
      </VStack>

      {/* Edit/Create Modal */}
      <EventEditModal
        event={selectedEvent}
        isOpen={isOpen}
        onClose={onClose}
        onSave={handleSaveEvent}
        progressionId={progressionId}
        series={series}
        season={season}
        episode={episode}
      />
    </Box>
  );
};

export default EventManager;
