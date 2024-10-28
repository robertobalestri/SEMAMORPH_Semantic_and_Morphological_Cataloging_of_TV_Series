import { 
  Box, 
  useColorModeValue, 
  Text, 
  VStack, 
  HStack, 
  Tag, 
  Tooltip,
  Select,
  Checkbox,
  CheckboxGroup,
  Stack,
} from '@chakra-ui/react';
import { useState, useMemo } from 'react';

interface ArcProgression {
  id: string;
  content: string;
  series: string;
  season: string;
  episode: string;
  ordinal_position: number;
  interfering_episode_characters: string[];
}

interface NarrativeArc {
  id: string;
  title: string;
  description: string;
  arc_type: string;
  episodic: boolean;
  main_characters: string[];
  series: string;
  progressions: ArcProgression[];
}

interface NarrativeGanttProps {
  arcs: NarrativeArc[];
  episodes: { season: string; episode: string; }[];
  selectedSeason: string;
}

const NarrativeGantt: React.FC<NarrativeGanttProps> = ({ arcs, episodes, selectedSeason }) => {
  const [selectedArcTypes, setSelectedArcTypes] = useState<string[]>([]);
  const [selectedCharacters, setSelectedCharacters] = useState<string[]>([]);

  const getArcTypeColor = (arcType: string) => {
    // Updated color mapping to match database arc types exactly
    const typeColors = {
      'Soap Arc': '#F687B3',      // pink
      'Genre-Specific Arc': '#ED8936', // orange
      'Episodic Arc': '#48BB78',     // green
      'seasonal': '#4299E1',      // blue - for seasonal arcs
      'episodic': '#9F7AEA',      // purple - for episodic arcs
    };
    
    console.log('Arc type:', arcType); // Debug log
    const color = typeColors[arcType as keyof typeof typeColors];
    console.log('Color chosen:', color); // Debug log
    
    return color || '#A0AEC0';
  };

  // Get unique arc types and characters
  const arcTypes = useMemo(() => 
    [...new Set(arcs.map(arc => arc.arc_type))],
    [arcs]
  );

  const allCharacters = useMemo(() => 
    [...new Set(arcs.flatMap(arc => arc.main_characters))],
    [arcs]
  );

  const seasonEpisodes = useMemo(() => {
    return episodes
      .filter(ep => ep.season === selectedSeason)
      .sort((a, b) => parseInt(a.episode.replace('E', '')) - parseInt(b.episode.replace('E', '')));
  }, [episodes, selectedSeason]);

  const filteredArcs = useMemo(() => {
    return arcs.filter(arc => {
      const matchesArcType = selectedArcTypes.length === 0 || selectedArcTypes.includes(arc.arc_type);
      const matchesCharacters = selectedCharacters.length === 0 || 
        selectedCharacters.some(char => arc.main_characters.includes(char));
      return matchesArcType && matchesCharacters;
    });
  }, [arcs, selectedArcTypes, selectedCharacters]);

  return (
    <VStack spacing={4} align="stretch">
      {/* Filters */}
      <HStack spacing={4} p={4} bg={useColorModeValue('white', 'gray.800')} borderRadius="lg" shadow="base">
        <Box flex={1}>
          <Text fontWeight="bold" mb={2}>Arc Types</Text>
          <CheckboxGroup 
            colorScheme="blue" 
            value={selectedArcTypes}
            onChange={(values) => setSelectedArcTypes(values as string[])}
          >
            <Stack direction={['column', 'row']} spacing={[2, 4]} wrap="wrap">
              {arcTypes.map(type => (
                <Checkbox 
                  key={type} 
                  value={type}
                  borderColor={getArcTypeColor(type)}
                >
                  <HStack>
                    <Text>{type}</Text>
                    <Box w="12px" h="12px" borderRadius="full" bg={getArcTypeColor(type)} />
                  </HStack>
                </Checkbox>
              ))}
            </Stack>
          </CheckboxGroup>
        </Box>
        <Box flex={1}>
          <Text fontWeight="bold" mb={2}>Characters</Text>
          <CheckboxGroup 
            colorScheme="blue" 
            value={selectedCharacters}
            onChange={(values) => setSelectedCharacters(values as string[])}
          >
            <Stack direction={['column', 'row']} spacing={[2, 4]} wrap="wrap">
              {allCharacters.map(char => (
                <Checkbox key={char} value={char}>
                  {char}
                </Checkbox>
              ))}
            </Stack>
          </CheckboxGroup>
        </Box>
      </HStack>

      {/* Gantt Chart */}
      <Box 
        h="600px" 
        w="100%" 
        position="relative" 
        bg={useColorModeValue('white', 'gray.800')} 
        p={4} 
        borderRadius="lg" 
        shadow="base"
        overflowX="auto"
        overflowY="auto"
      >
        <Box display="grid" gridTemplateColumns={`250px repeat(${seasonEpisodes.length}, 120px)`}>
          {/* Header */}
          <Box 
            borderBottom="1px" 
            borderRight="1px"
            borderColor="gray.200" 
            p={2} 
            fontWeight="bold"
            bg={useColorModeValue('gray.50', 'gray.700')}
          >
            Narrative Arcs
          </Box>
          {seasonEpisodes.map(ep => (
            <Box 
              key={ep.episode} 
              borderBottom="1px" 
              borderRight="1px"
              borderColor="gray.200" 
              p={2} 
              fontWeight="bold"
              textAlign="center"
              bg={useColorModeValue('gray.50', 'gray.700')}
            >
              Episode {ep.episode.replace('E', '')}
            </Box>
          ))}

          {/* Arcs */}
          {filteredArcs.map(arc => (
            <Box key={arc.id} display="contents">
              <Box 
                borderBottom="1px" 
                borderRight="1px"
                borderColor="gray.200" 
                p={2}
                bg={useColorModeValue('white', 'gray.800')}
              >
                <VStack align="start" spacing={1}>
                  <Tooltip label={arc.description} placement="right">
                    <Text fontSize="sm" fontWeight="semibold" noOfLines={2}>
                      {arc.title}
                    </Text>
                  </Tooltip>
                  <Tag size="sm" bg={getArcTypeColor(arc.arc_type)} color="white">
                    {arc.arc_type}
                  </Tag>
                </VStack>
              </Box>
              {seasonEpisodes.map(ep => {
                const epNumber = ep.episode.replace('E', '').padStart(2, '0');
                const progression = arc.progressions.find(
                  prog => prog.season === selectedSeason && prog.episode.replace('E', '').padStart(2, '0') === epNumber
                );
                
                return (
                  <Tooltip 
                    key={`${arc.id}-${ep.episode}`}
                    label={progression ? progression.content : 'No progression in this episode'}
                    placement="top"
                    hasArrow
                  >
                    <Box 
                      borderBottom="1px" 
                      borderRight="1px"
                      borderColor="gray.200"
                      p={2}
                      minHeight="80px"
                      bg={progression ? `${getArcTypeColor(arc.arc_type)}15` : 'transparent'}
                      _hover={{
                        bg: progression ? `${getArcTypeColor(arc.arc_type)}30` : useColorModeValue('gray.50', 'gray.700')
                      }}
                      transition="all 0.2s"
                      cursor="pointer"
                    >
                      {progression && (
                        <VStack spacing={1} align="start">
                          <Box 
                            w="100%" 
                            h="4px" 
                            bg={getArcTypeColor(arc.arc_type)}
                            borderRadius="full"
                          />
                          <Text 
                            fontSize="xs" 
                            noOfLines={3}
                            color={useColorModeValue('gray.600', 'gray.300')}
                          >
                            {progression.content}
                          </Text>
                        </VStack>
                      )}
                    </Box>
                  </Tooltip>
                );
              })}
            </Box>
          ))}
        </Box>
      </Box>
    </VStack>
  );
};

export default NarrativeGantt;
