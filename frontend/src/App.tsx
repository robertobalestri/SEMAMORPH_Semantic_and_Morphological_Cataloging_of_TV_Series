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
import CharacterManager from './components/CharacterManager';

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

function App() {
  const [series, setSeries] = useState<string[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<string>('');
  const [episodes, setEpisodes] = useState<{ season: string; episode: string; }[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [arcs, setArcs] = useState<NarrativeArc[]>([]);

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
      setIsLoading(true);
      Promise.all([
        // Fetch episodes
        fetch(`http://localhost:8000/api/episodes/${selectedSeries}`, fetchOptions)
          .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
          }),
        // Fetch arcs
        fetch(`http://localhost:8000/api/arcs/${selectedSeries}`, fetchOptions)
          .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
          })
      ])
      .then(([episodesData, arcsData]) => {
        setEpisodes(episodesData);
        setArcs(arcsData);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        setEpisodes([]);
        setArcs([]);
      })
      .finally(() => {
        setIsLoading(false);
      });
    }
  }, [selectedSeries]);

  const handleArcUpdated = () => {
    if (selectedSeries) {
      setIsLoading(true);
      Promise.all([
        fetch(`http://localhost:8000/api/episodes/${selectedSeries}`, fetchOptions)
          .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
          }),
        fetch(`http://localhost:8000/api/arcs/${selectedSeries}`, fetchOptions)
          .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
          })
      ])
      .then(([episodesData, arcsData]) => {
        setEpisodes(episodesData);
        setArcs(arcsData);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        setEpisodes([]);
        setArcs([]);
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
              <Select
                placeholder="Select series"
                value={selectedSeries}
                onChange={(e) => setSelectedSeries(e.target.value)}
              >
                {series.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </Select>
            </Box>

            {isLoading ? (
              <Box textAlign="center" p={8}>
                <Text>Loading data...</Text>
              </Box>
            ) : selectedSeries ? (
              <Box width="100%" overflowX="hidden">
                <Tabs isFitted variant="enclosed">
                  <TabList>
                    <Tab>Narrative Arcs</Tab>
                    <Tab>Vector Store</Tab>
                    <Tab>Characters</Tab>
                  </TabList>
                  <TabPanels>
                    <TabPanel p={0}>
                      <NarrativeArcManager 
                        arcs={arcs}
                        episodes={episodes}
                        onArcUpdated={handleArcUpdated}
                      />
                    </TabPanel>
                    <TabPanel>
                      <VectorStoreExplorer series={selectedSeries} />
                    </TabPanel>
                    <TabPanel>
                      <CharacterManager 
                        series={selectedSeries} 
                        onCharacterUpdated={handleArcUpdated}
                      />
                    </TabPanel>
                  </TabPanels>
                </Tabs>
              </Box>
            ) : (
              <Box textAlign="center" p={8}>
                <Text>Select a series to view the dashboard.</Text>
              </Box>
            )}
          </VStack>
        </Box>
      </Box>
    </ChakraProvider>
  );
}

export default App;
