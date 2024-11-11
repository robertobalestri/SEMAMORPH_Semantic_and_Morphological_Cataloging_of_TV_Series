import {
  Grid,
  Box,
  VStack,
  HStack,
  FormControl,
  FormLabel,
  Switch,
  Select as ChakraSelect,
  Checkbox,
  Text,
  SimpleGrid,
} from '@chakra-ui/react';
import React from 'react';

interface ArcFiltersProps {
  seasons: string[];
  selectedSeason: string;
  onSeasonChange: (season: string) => void;
  allCharacters: string[];
  selectedCharacters: string[];
  setSelectedCharacters: (chars: string[]) => void;
  includeInterferingCharacters: boolean;
  setIncludeInterferingCharacters: (include: boolean) => void;
  selectedEpisodes: string[];
  setSelectedEpisodes: (episodes: string[]) => void;
  episodes: { season: string; episode: string; }[];
}

const ArcFilters: React.FC<ArcFiltersProps> = ({
  seasons,
  selectedSeason,
  onSeasonChange,
  allCharacters,
  selectedCharacters,
  setSelectedCharacters,
  includeInterferingCharacters,
  setIncludeInterferingCharacters,
  selectedEpisodes,
  setSelectedEpisodes,
  episodes,
}) => {
  return (
    <Grid templateColumns="repeat(2, 1fr)" gap={4} mb={4}>
      {/* Character Filter */}
      <Box borderWidth={1} borderRadius="md" p={4}>
        <VStack align="stretch" spacing={3}>
          <FormControl>
            <FormLabel fontWeight="bold">Filter by Characters</FormLabel>
            <Box maxH="200px" overflowY="auto" borderWidth={1} borderRadius="md" p={2} bg="white">
              <VStack align="start" spacing={1}>
                {allCharacters.map(char => (
                  <Checkbox
                    key={char}
                    isChecked={selectedCharacters.includes(char)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedCharacters([...selectedCharacters, char]);
                      } else {
                        setSelectedCharacters(selectedCharacters.filter(c => c !== char));
                      }
                    }}
                  >
                    <Text fontSize="sm">{char}</Text>
                  </Checkbox>
                ))}
              </VStack>
            </Box>
            <FormControl mt={2}>
              <HStack>
                <Switch
                  id="include-interfering"
                  isChecked={includeInterferingCharacters}
                  onChange={(e) => setIncludeInterferingCharacters(e.target.checked)}
                />
                <FormLabel htmlFor="include-interfering" mb={0}>
                  Include Interfering Characters
                </FormLabel>
              </HStack>
            </FormControl>
          </FormControl>
        </VStack>
      </Box>

      {/* Episode Filter */}
      <Box borderWidth={1} borderRadius="md" p={4}>
        <VStack align="stretch" spacing={3}>
          <FormControl>
            <FormLabel fontWeight="bold">Season & Episodes</FormLabel>
            <ChakraSelect
              value={selectedSeason}
              onChange={(e) => onSeasonChange(e.target.value)}
              mb={3}
              placeholder="Select season"
            >
              {seasons.map(season => (
                <option key={season} value={season}>
                  Season {season.replace('S', '')}
                </option>
              ))}
            </ChakraSelect>
            <Box maxH="200px" overflowY="auto" borderWidth={1} borderRadius="md" p={2} bg="white">
              <SimpleGrid columns={3} spacing={2}>
                {episodes
                  .filter(ep => ep.season === selectedSeason)
                  .map(ep => {
                    const episodeKey = `${ep.season}-${ep.episode}`;
                    const episodeNumber = ep.episode.replace('E', '');
                    return (
                      <Checkbox
                        key={`${ep.season}-${ep.episode}-${Math.random()}`}
                        isChecked={selectedEpisodes.includes(episodeKey)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedEpisodes([...selectedEpisodes, episodeKey]);
                          } else {
                            setSelectedEpisodes(selectedEpisodes.filter(e => e !== episodeKey));
                          }
                        }}
                      >
                        <Text fontSize="sm">Ep {episodeNumber}</Text>
                      </Checkbox>
                    );
                  })}
              </SimpleGrid>
            </Box>
          </FormControl>
        </VStack>
      </Box>
    </Grid>
  );
};

export default ArcFilters; 