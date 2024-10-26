import { useState, useEffect } from 'react';
import { 
  ChakraProvider, 
  Box, 
  Grid, 
  Select, 
  Heading, 
  Text,
  useColorModeValue,
  VStack,
  HStack,
  Badge,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
} from '@chakra-ui/react';
import { VerticalTimeline, VerticalTimelineElement } from 'react-vertical-timeline-component';
import 'react-vertical-timeline-component/style.min.css';

interface ArcProgression {
  id: string;
  content: string;
  series: string;
  season: string;
  episode: string;
  ordinal_position: number;
  interfering_episode_characters: string;
}

interface NarrativeArc {
  id: string;
  title: string;
  description: string;
  arc_type: string;
  episodic: boolean;
  main_characters: string;
  series: string;
  progressions: ArcProgression[];
}

function App() {
  const [series, setSeries] = useState<string[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<string>('');
  const [episodes, setEpisodes] = useState<{ season: string; episode: string; }[]>([]);
  const [selectedEpisode, setSelectedEpisode] = useState<string>('');
  const [arcs, setArcs] = useState<NarrativeArc[]>([]);
  
  const bgColor = useColorModeValue('white', 'gray.800');
  const textColor = useColorModeValue('gray.800', 'white');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  useEffect(() => {
    fetch('/api/series')
      .then(res => res.json())
      .then(data => setSeries(data))
      .catch(error => console.error('Error fetching series:', error));
  }, []);

  useEffect(() => {
    if (selectedSeries) {
      fetch(`/api/episodes/${selectedSeries}`)
        .then(res => res.json())
        .then(data => setEpisodes(data))
        .catch(error => console.error('Error fetching episodes:', error));
    }
  }, [selectedSeries]);

  useEffect(() => {
    if (selectedSeries && selectedEpisode) {
      const [season, episode] = selectedEpisode.split('E');
      // Remove any leading zeros from the episode number before sending
      const episodeNumber = parseInt(episode, 10).toString();
      
      fetch(`/api/arcs/${selectedSeries}/${season}/${episodeNumber}`)
        .then(res => {
          if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
          }
          return res.json();
        })
        .then(data => setArcs(data))
        .catch(error => {
          console.error('Error fetching arcs:', error);
          setArcs([]); // Reset arcs on error
        });
    }
  }, [selectedSeries, selectedEpisode]);

  return (
    <ChakraProvider>
      <Box p={5} minH="100vh" bg={useColorModeValue('gray.50', 'gray.900')}>
        <Grid templateColumns="repeat(1, 1fr)" gap={6}>
          <Box bg={bgColor} p={6} borderRadius="lg" shadow="base">
            <Heading mb={4} color={textColor}>Narrative Arcs Dashboard</Heading>
            <HStack spacing={4} mb={6}>
              <Select
                placeholder="Select series"
                value={selectedSeries}
                onChange={(e) => setSelectedSeries(e.target.value)}
                bg={bgColor}
              >
                {series.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </Select>
              <Select
                placeholder="Select episode"
                value={selectedEpisode}
                onChange={(e) => setSelectedEpisode(e.target.value)}
                isDisabled={!selectedSeries}
                bg={bgColor}
              >
                {episodes.map(ep => {
                  // Format episode number to ensure it has leading zeros
                  const formattedEpisode = ep.episode;  // Episode is already padded from backend
                  return (
                    <option 
                      key={`${ep.season}E${formattedEpisode}`} 
                      value={`${ep.season}E${formattedEpisode}`}
                    >
                      {`${ep.season}E${formattedEpisode}`}
                    </option>
                  );
                })}
              </Select>
            </HStack>
          </Box>

          {arcs.length > 0 && (
            <Box bg={bgColor} p={6} borderRadius="lg" shadow="base">
              <VerticalTimeline>
                {arcs.map((arc) => (
                  <VerticalTimelineElement
                    key={arc.id}
                    className="vertical-timeline-element"
                    contentStyle={{
                      background: bgColor,
                      color: textColor,
                      borderRadius: '8px',
                      boxShadow: 'lg',
                      border: `1px solid ${borderColor}`,
                    }}
                    contentArrowStyle={{ borderRight: `7px solid ${bgColor}` }}
                    date={arc.episodic ? 'Episodic Arc' : 'Season Arc'}
                    iconStyle={{ 
                      background: arc.episodic ? '#48BB78' : '#4299E1',
                      color: '#fff',
                      boxShadow: '0 0 0 4px #fff',
                    }}
                  >
                    <VStack align="stretch" spacing={3}>
                      <Heading size="md" color={textColor}>{arc.title}</Heading>
                      <HStack wrap="wrap" spacing={2}>
                        <Badge colorScheme={arc.episodic ? 'green' : 'blue'}>
                          {arc.arc_type}
                        </Badge>
                        <Badge colorScheme="purple">
                          {arc.main_characters}
                        </Badge>
                      </HStack>
                      <Text color={textColor}>{arc.description}</Text>
                      
                      <Accordion allowMultiple>
                        {arc.progressions.map((prog) => (
                          <AccordionItem key={prog.id}>
                            <AccordionButton>
                              <Box flex="1" textAlign="left">
                                Progression #{prog.ordinal_position}
                              </Box>
                              <AccordionIcon />
                            </AccordionButton>
                            <AccordionPanel>
                              <VStack align="stretch" spacing={2}>
                                <Text>{prog.content}</Text>
                                {prog.interfering_episode_characters && (
                                  <Text fontSize="sm" color="gray.500">
                                    Interfering characters: {prog.interfering_episode_characters}
                                  </Text>
                                )}
                              </VStack>
                            </AccordionPanel>
                          </AccordionItem>
                        ))}
                      </Accordion>
                    </VStack>
                  </VerticalTimelineElement>
                ))}
              </VerticalTimeline>
            </Box>
          )}
        </Grid>
      </Box>
    </ChakraProvider>
  );
}

export default App;
