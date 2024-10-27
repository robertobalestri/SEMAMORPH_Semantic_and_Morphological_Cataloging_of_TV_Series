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
  Flex,
  Spacer,
  Tag,
  TagLabel,
  Divider,
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
  interfering_episode_characters: string[];  // Changed to string[]
}

interface NarrativeArc {
  id: string;
  title: string;
  description: string;
  arc_type: string;
  episodic: boolean;
  main_characters: string[];  // Changed to string[]
  series: string;
  progressions: ArcProgression[];
}

function App() {
  const [series, setSeries] = useState<string[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<string>('');
  const [episodes, setEpisodes] = useState<{ season: string; episode: string; }[]>([]);
  const [selectedSeason, setSelectedSeason] = useState<string>('');
  const [selectedEpisode, setSelectedEpisode] = useState<string>('');
  const [arcs, setArcs] = useState<NarrativeArc[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  useEffect(() => {
    fetch('http://localhost:8000/api/series')
      .then(response => response.json())
      .then(data => setSeries(data));
  }, []);

  useEffect(() => {
    if (selectedSeries) {
      fetch(`http://localhost:8000/api/episodes/${selectedSeries}`)
        .then(response => response.json())
        .then(data => setEpisodes(data));
    }
  }, [selectedSeries]);

  useEffect(() => {
    if (selectedSeries && selectedSeason && selectedEpisode) {
      setIsLoading(true);
      const formattedEpisode = `E${selectedEpisode.padStart(2, '0')}`;  // Ensure episode is prefixed with 'E'
      
      fetch(`http://localhost:8000/api/arcs/${selectedSeries}/${selectedSeason}/${formattedEpisode}`)
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          console.log('Received arcs:', data);
          setArcs(data);
        })
        .catch(error => {
          console.error('Error fetching arcs:', error);
          setArcs([]);
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [selectedSeries, selectedSeason, selectedEpisode]);

  const getArcTypeColor = (arcType: string) => {
    const typeColors = {
      'Character Arc': 'purple',
      'Plot Arc': 'blue',
      'Theme Arc': 'green',
      'Genre-Specific Arc': 'orange',
      'Relationship Arc': 'pink',
    };
    return typeColors[arcType as keyof typeof typeColors] || 'gray';
  };

  // Get unique seasons and sort them
  const seasons = [...new Set(episodes.map(ep => ep.season))].sort((a, b) => {
    // Remove 'S' prefix and convert to numbers for proper sorting
    const numA = parseInt(a.replace('S', ''));
    const numB = parseInt(b.replace('S', ''));
    return numA - numB;
  });

  // Get episodes for selected season and sort them
  const episodesForSeason = episodes
    .filter(ep => ep.season === selectedSeason)
    .map(ep => ep.episode)
    .sort((a, b) => {
      // Remove 'E' prefix and convert to numbers for proper sorting
      const numA = parseInt(a.replace('E', ''));
      const numB = parseInt(b.replace('E', ''));
      return numA - numB;
    });

  return (
    <ChakraProvider>
      <Box p={5} minH="100vh" bg={useColorModeValue('gray.50', 'gray.900')}>
        <Grid gap={6} maxW="1200px" mx="auto">
          <VStack spacing={4} align="stretch">
            <Heading as="h1" size="xl" mb={6} textAlign="center">
              Narrative Arcs Visualization
            </Heading>

            <HStack spacing={4}>
              <Select
                placeholder="Select series"
                value={selectedSeries}
                onChange={(e) => {
                  setSelectedSeries(e.target.value);
                  setSelectedSeason('');
                  setSelectedEpisode('');
                }}
              >
                {series.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </Select>

              <Select
                placeholder="Select season"
                value={selectedSeason}
                onChange={(e) => {
                  setSelectedSeason(e.target.value);
                  setSelectedEpisode('');
                }}
                isDisabled={!selectedSeries}
              >
                {seasons.map(season => (
                  <option key={`season-${season}`} value={season}>
                    Season {season.replace('S', '')}
                  </option>
                ))}
              </Select>

              <Select
                placeholder="Select episode"
                value={selectedEpisode}
                onChange={(e) => setSelectedEpisode(e.target.value)}
                isDisabled={!selectedSeason}
              >
                {episodesForSeason.map(episode => (
                  <option key={`episode-${episode}`} value={episode}>
                    Episode {episode.replace('E', '')}
                  </option>
                ))}
              </Select>
            </HStack>

            {isLoading ? (
              <Box mt={8} textAlign="center">
                <Text>Loading arcs...</Text>
              </Box>
            ) : arcs.length > 0 ? (
              <Box mt={8}>
                <VerticalTimeline>
                  {arcs.map((arc) => (
                    <VerticalTimelineElement
                      key={arc.id}
                      contentStyle={{
                        background: bgColor,
                        boxShadow: '0 3px 0 ' + borderColor,
                        padding: '2em',
                      }}
                      contentArrowStyle={{ borderRight: `7px solid ${bgColor}` }}
                      iconStyle={{ background: `var(--chakra-colors-${getArcTypeColor(arc.arc_type)}-500)`, color: '#fff' }}
                    >
                      <VStack align="stretch" spacing={4}>
                        <Flex align="center">
                          <Heading size="md">{arc.title}</Heading>
                          <Spacer />
                          <Tag size="md" colorScheme={getArcTypeColor(arc.arc_type)} ml={2}>
                            <TagLabel>{arc.arc_type}</TagLabel>
                          </Tag>
                        </Flex>

                        <Text>{arc.description}</Text>

                        <Box>
                          <Text fontWeight="bold" mb={2}>Main Characters:</Text>
                          <Flex gap={2} flexWrap="wrap">
                            {arc.main_characters.map((char, idx) => (
                              <Tag key={idx} size="md" colorScheme="teal">
                                <TagLabel>{char.trim()}</TagLabel>
                              </Tag>
                            ))}
                          </Flex>
                        </Box>

                        <Divider />

                        <Accordion allowMultiple>
                          {arc.progressions.map((prog) => (
                            <AccordionItem key={prog.id} border="none">
                              <AccordionButton 
                                _hover={{ bg: 'gray.100' }}
                                borderRadius="md"
                              >
                                <Box flex="1" textAlign="left">
                                  <Text fontWeight="bold">Progression #{prog.ordinal_position}</Text>
                                </Box>
                                <AccordionIcon />
                              </AccordionButton>
                              <AccordionPanel pb={4}>
                                <VStack align="stretch" spacing={3}>
                                  <Text>{prog.content}</Text>
                                  {prog.interfering_episode_characters && prog.interfering_episode_characters.length > 0 && (
                                    <Box>
                                      <Text fontWeight="bold" mb={2}>Interfering Characters:</Text>
                                      <Flex gap={2} flexWrap="wrap">
                                        {prog.interfering_episode_characters.map((char, idx) => (
                                          <Tag key={idx} size="sm" colorScheme="purple">
                                            <TagLabel>{char.trim()}</TagLabel>
                                          </Tag>
                                        ))}
                                      </Flex>
                                    </Box>
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
            ) : (
              <Box mt={8} textAlign="center">
                <Text>No arcs found for this episode.</Text>
              </Box>
            )}
          </VStack>
        </Grid>
      </Box>
    </ChakraProvider>
  );
}

export default App;
