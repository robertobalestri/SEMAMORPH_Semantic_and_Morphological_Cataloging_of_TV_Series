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
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
} from '@chakra-ui/react';
import NarrativeArcManager from './components/NarrativeArcManager';
import VectorStoreExplorer from './components/VectorStoreExplorer';

interface ArcProgression {
  id: string;
  content: string;
  series: string;
  season: string;
  episode: string;
  ordinal_position: number;
  interfering_characters: string[];
}

interface NarrativeArc {
  id: string;
  title: string;
  description: string;
  arc_type: string;
  main_characters: string[];
  series: string;
  progressions: ArcProgression[];
}

interface SeasonArcs {
  series: string;
  season: string;
  arcs: NarrativeArc[];
}

function App() {
  const [series, setSeries] = useState<string[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<string>('');
  const [episodes, setEpisodes] = useState<{ season: string; episode: string; }[]>([]);
  const [selectedSeason, setSelectedSeason] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [seasonArcs, setSeasonArcs] = useState<SeasonArcs | null>(null);

  const fetchOptions = {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    mode: 'cors' as RequestMode,
  };

  useEffect(() => {
    fetch('http://localhost:8000/api/series', fetchOptions)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => setSeries(data))
      .catch(error => {
        console.error('Error fetching series:', error);
        setSeries([]);
      });
  }, []);

  useEffect(() => {
    if (selectedSeries) {
      fetch(`http://localhost:8000/api/episodes/${selectedSeries}`, fetchOptions)
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => setEpisodes(data))
        .catch(error => {
          console.error('Error fetching episodes:', error);
          setEpisodes([]);
        });
    }
  }, [selectedSeries]);

  useEffect(() => {
    if (selectedSeries && selectedSeason) {
      setIsLoading(true);
      console.log('Fetching arcs for:', { selectedSeries, selectedSeason });
      fetch(`http://localhost:8000/api/arcs/${selectedSeries}`, fetchOptions)
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          console.log('Received arcs data:', data);
          const seasonArcsData = data.filter((arc: NarrativeArc) =>
            arc.progressions.some(prog => prog.season === selectedSeason)
          );
          console.log('Filtered season arcs:', seasonArcsData);
          setSeasonArcs({
            series: selectedSeries,
            season: selectedSeason,
            arcs: seasonArcsData
          });
        })
        .catch(error => {
          console.error('Error fetching season arcs:', error);
          setSeasonArcs(null);
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [selectedSeries, selectedSeason]);

  // Get unique seasons and sort them
  const seasons = [...new Set(episodes.map(ep => ep.season))].sort((a, b) => {
    const numA = parseInt(a.replace('S', ''));
    const numB = parseInt(b.replace('S', ''));
    return numA - numB;
  });

  const handleArcUpdated = () => {
    // Refresh arcs data when an arc is updated
    if (selectedSeries && selectedSeason) {
      setIsLoading(true);
      fetch(`http://localhost:8000/api/arcs/${selectedSeries}`, fetchOptions)
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          const seasonArcsData = data.filter((arc: NarrativeArc) =>
            arc.progressions.some(prog => prog.season === selectedSeason)
          );
          setSeasonArcs({
            series: selectedSeries,
            season: selectedSeason,
            arcs: seasonArcsData
          });
        })
        .catch(error => {
          console.error('Error fetching season arcs:', error);
          setSeasonArcs(null);
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  };

  return (
    <ChakraProvider>
      <Box minH="100vh" bg={useColorModeValue('gray.50', 'gray.900')} position="relative">
        <Box maxW="100%" mx="auto">
          <VStack spacing={4} align="stretch">
            <Box px={4} py={5} bg={useColorModeValue('white', 'gray.800')} shadow="sm">
              <Heading as="h1" size="xl" textAlign="center">
                Narrative Arcs Dashboard
              </Heading>
            </Box>

            <Box px={4}>
              <HStack spacing={4} width="100%">
                <Select
                  placeholder="Select series"
                  value={selectedSeries}
                  onChange={(e) => {
                    setSelectedSeries(e.target.value);
                    setSelectedSeason('');
                  }}
                >
                  {series.map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </Select>

                <Select
                  placeholder="Select season"
                  value={selectedSeason}
                  onChange={(e) => setSelectedSeason(e.target.value)}
                  isDisabled={!selectedSeries}
                >
                  {seasons.map(season => (
                    <option key={`season-${season}`} value={season}>
                      Season {season.replace('S', '')}
                    </option>
                  ))}
                </Select>
              </HStack>
            </Box>

            {isLoading ? (
              <Box textAlign="center" p={8}>
                <Text>Loading season overview...</Text>
              </Box>
            ) : seasonArcs ? (
              <Box width="100%" overflowX="hidden">
                <Tabs isFitted variant="enclosed">
                  <TabList>
                    <Tab>Narrative Arcs</Tab>
                    <Tab>Vector Store</Tab>
                  </TabList>
                  <TabPanels>
                    <TabPanel p={0}>
                      <NarrativeArcManager 
                        arcs={seasonArcs.arcs}
                        episodes={episodes}
                        selectedSeason={selectedSeason}
                        onArcUpdated={handleArcUpdated}
                      />
                    </TabPanel>
                    <TabPanel>
                      <VectorStoreExplorer series={selectedSeries} />
                    </TabPanel>
                  </TabPanels>
                </Tabs>
              </Box>
            ) : (
              <Box textAlign="center" p={8}>
                <Text>Select a series and season to view the overview.</Text>
              </Box>
            )}
          </VStack>
        </Box>
      </Box>
    </ChakraProvider>
  );
}

export default App;
