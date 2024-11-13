import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  VStack,
  HStack,
  Button,
  useDisclosure,
  useToast,
  Text,
  Badge,
  Grid,
  FormControl,
  FormLabel,
  SimpleGrid,
  Checkbox,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Input,
  Textarea,
  Select,
} from '@chakra-ui/react';
import { AddIcon, DeleteIcon, RepeatIcon } from '@chakra-ui/icons';
import { ArcTimeline } from './ArcTimeline';
import { NewArcModal } from '../modals/NewArcModal';
import { ArcMergeModal } from '../modals/ArcMergeModal';
import { ArcFilters } from '../filters/ArcFilters';
import { useArcStore } from '@/store/arcStore';
import { useApi } from '@/hooks/useApi';
import { ApiClient } from '@/services/api/ApiClient';
import type { NarrativeArc, Episode, ArcProgression, Character, ProgressionMapping } from '@/architecture/types';
import { ArcProgressionEditModal } from '../modals/ArcProgressionEditModal';
import { ArcType } from '@/architecture/types/arc';
import { ArcEditModal } from '../modals/ArcEditModal';

interface NarrativeArcManagerProps {
  series: string;
  arcs: NarrativeArc[];
  episodes: Episode[];
  onArcUpdated: () => void;
}

export const NarrativeArcManager: React.FC<NarrativeArcManagerProps> = ({
  series,
  arcs,
  episodes,
  onArcUpdated,
}) => {
  // State
  const [selectedSeason, setSelectedSeason] = useState('');
  const [isMergeMode, setIsMergeMode] = useState(false);
  const [selectedForMerge, setSelectedForMerge] = useState<NarrativeArc[]>([]);
  const [availableCharacters, setAvailableCharacters] = useState<string[]>([]);
  const [selectedCharacters, setSelectedCharacters] = useState<string[]>([]);
  const [includeInterferingCharacters, setIncludeInterferingCharacters] = useState(false);
  const [selectedArcTypes, setSelectedArcTypes] = useState<ArcType[]>([]);
  const [selectedProgression, setSelectedProgression] = useState<ProgressionMapping | null>(null);
  const [selectedArc, setSelectedArc] = useState<NarrativeArc | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState('');
  const [editingArc, setEditingArc] = useState<NarrativeArc | null>(null);
  const [editArcTitle, setEditArcTitle] = useState('');
  const [editArcDescription, setEditArcDescription] = useState('');
  const [editArcType, setEditArcType] = useState('');
  const [editMainCharacters, setEditMainCharacters] = useState<string[]>([]);
  const [characters, setCharacters] = useState<Character[]>([]);

  // Modal disclosures
  const {
    isOpen: isProgressionModalOpen,
    onOpen: onProgressionModalOpen,
    onClose: onProgressionModalClose
  } = useDisclosure();

  const {
    isOpen: isNewArcModalVisible,
    onOpen: openNewArcModal,
    onClose: closeNewArcModal
  } = useDisclosure();

  const {
    isOpen: isMergeModalOpen,
    onOpen: openMergeModal,
    onClose: closeMergeModal
  } = useDisclosure();

  const {
    isOpen: isArcEditModalOpen,
    onOpen: onArcEditModalOpen,
    onClose: onArcEditModalClose
  } = useDisclosure();

  const toast = useToast();
  const { request, isLoading } = useApi();
  const api = new ApiClient();

  // Get unique seasons from episodes
  const seasons = React.useMemo(() => {
    const uniqueSeasons = new Set(episodes.map(ep => ep.season));
    return Array.from(uniqueSeasons).sort();
  }, [episodes]);

  // Set initial season
  useEffect(() => {
    if (seasons.length > 0 && !selectedSeason) {
      setSelectedSeason(seasons[0]);
    }
  }, [seasons, selectedSeason]);

  // Fetch characters when component mounts
  useEffect(() => {
    const fetchCharacters = async () => {
      try {
        const response = await request(() => api.getCharacters(series));
        if (response) {
          // Make sure we're getting an array of characters
          const characterList = Array.isArray(response) ? response : [];
          setCharacters(characterList);
        }
      } catch (error) {
        console.error('Error fetching characters:', error);
        // Set empty array on error to prevent map errors
        setCharacters([]);
      }
    };

    if (series) {
      fetchCharacters();
    }
  }, [series]);

  // Update the filteredArcs useMemo to include arc type filtering
  const filteredArcs = useMemo(() => {
    let filtered = arcs;

    // Filter by arc type if any are selected
    if (selectedArcTypes.length > 0) {
      filtered = filtered.filter(arc => selectedArcTypes.includes(arc.arc_type as ArcType));
    }

    // Filter by characters if any are selected
    if (selectedCharacters.length > 0) {
      filtered = filtered.filter(arc => {
        // Check main characters
        const hasMainCharacter = arc.main_characters.some(char => 
          selectedCharacters.includes(char)
        );

        if (hasMainCharacter) return true;

        // Check interfering characters if enabled
        if (includeInterferingCharacters) {
          return arc.progressions.some(prog => 
            prog.interfering_characters.some(char => 
              selectedCharacters.includes(char)
            )
          );
        }

        return false;
      });
    }

    // Filter by episode if selected
    if (selectedSeason && selectedEpisode) {
      filtered = filtered.filter(arc => 
        arc.progressions.some(prog => 
          prog.season === selectedSeason && prog.episode === selectedEpisode
        )
      );
    }

    return filtered;
  }, [
    arcs, 
    selectedCharacters, 
    includeInterferingCharacters, 
    selectedArcTypes,  // Add this dependency
    selectedSeason,
    selectedEpisode
  ]);

  const handleCreateArc = async (arcData: Partial<NarrativeArc>) => {
    try {
      const response = await request(() => api.createArc(arcData));
      if (response) {
        toast({
          title: 'Arc created',
          status: 'success',
          duration: 3000,
        });
        onArcUpdated();
        closeNewArcModal();
      }
    } catch (error) {
      toast({
        title: 'Error creating arc',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
      });
    }
  };

  const handleMergeArcs = async (mergedArcData: Partial<NarrativeArc>) => {
    if (selectedForMerge.length !== 2) return;

    try {
      const response = await request(() => api.mergeArcs(
        selectedForMerge[0].id,
        selectedForMerge[1].id,
        mergedArcData
      ));

      if (response) {
        toast({
          title: 'Arcs merged successfully',
          status: 'success',
          duration: 3000,
        });
        setIsMergeMode(false);
        setSelectedForMerge([]);
        onArcUpdated();
        closeMergeModal();
      }
    } catch (error) {
      toast({
        title: 'Error merging arcs',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
      });
    }
  };

  const handleToggleMerge = (arc: NarrativeArc) => {
    setSelectedForMerge(prev => {
      if (prev.includes(arc)) {
        return prev.filter(a => a.id !== arc.id);
      }
      if (prev.length < 2) {
        return [...prev, arc];
      }
      return prev;
    });
  };

  const handleEditArc = (arc: NarrativeArc) => {
    setEditingArc(arc);
    setEditArcTitle(arc.title);
    setEditArcDescription(arc.description);
    setEditArcType(arc.arc_type);
    setEditMainCharacters(arc.main_characters);
    onArcEditModalOpen();
  };

  const handleSaveArcChanges = async () => {
    if (!editingArc) return;

    try {
      await request(() => api.updateArc(editingArc.id, {
        title: editArcTitle,
        description: editArcDescription,
        arc_type: editArcType,
        main_characters: editMainCharacters.join(';')
      }));

      toast({
        title: 'Arc updated',
        status: 'success',
        duration: 3000,
      });
      onArcUpdated();
      onArcEditModalClose();
    } catch (error) {
      toast({
        title: 'Error updating arc',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
      });
    }
  };

  const handleDeleteArc = async () => {
    if (!editingArc) return;

    try {
      await request(() => api.deleteArc(editingArc.id));
      toast({
        title: 'Arc deleted',
        status: 'success',
        duration: 3000,
      });
      onArcUpdated();
      onArcEditModalClose();
    } catch (error) {
      toast({
        title: 'Error deleting arc',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
      });
    }
  };

  const handleCellClick = async (arc: NarrativeArc, season: string, episode: string) => {
    const existingProgression = arc.progressions.find(
      (p: ArcProgression) => p.season === season && p.episode === episode
    );

    const progressionData: ProgressionMapping = {
      season,
      episode,
      content: existingProgression?.content || '',
      interfering_characters: existingProgression?.interfering_characters || []
    };

    setSelectedProgression(progressionData);
    setSelectedArc(arc);
    onProgressionModalOpen();
  };

  const handleProgressionSave = async (updatedProgression: ProgressionMapping) => {
    if (!selectedArc) return;

    try {
      const existingProgression = selectedArc.progressions.find(
        (p: ArcProgression) => p.season === updatedProgression.season && p.episode === updatedProgression.episode
      );

      if (existingProgression) {
        // Update existing progression
        await request(() =>
          api.updateProgression(existingProgression.id, {
            content: updatedProgression.content,
            interfering_characters: updatedProgression.interfering_characters
          })
        );
      } else {
        // Create new progression
        await request(() =>
          api.createProgression({
            content: updatedProgression.content,
            arc_id: selectedArc.id,
            series: series,
            season: updatedProgression.season,
            episode: updatedProgression.episode,
            interfering_characters: updatedProgression.interfering_characters
          })
        );
      }

      onArcUpdated();
      onProgressionModalClose();
      toast({
        title: 'Success',
        description: `Progression ${existingProgression ? 'updated' : 'created'} successfully`,
        status: 'success',
        duration: 3000,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to save progression',
        status: 'error',
        duration: 5000,
      });
    }
  };

  const handleProgressionDelete = async () => {
    if (!selectedArc || !selectedProgression) return;

    try {
      const existingProgression = selectedArc.progressions.find(
        (p: ArcProgression) => p.season === selectedProgression.season && p.episode === selectedProgression.episode
      );

      if (!existingProgression) {
        onProgressionModalClose();
        return;
      }

      await request(() => api.deleteProgression(existingProgression.id));

      onArcUpdated();
      onProgressionModalClose();
      toast({
        title: 'Success',
        description: 'Progression deleted successfully',
        status: 'success',
        duration: 3000,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to delete progression',
        status: 'error',
        duration: 5000,
      });
    }
  };

  // Update the renderArcTypeFilters function
  const renderArcTypeFilters = () => {
    const arcTypeColors = {
      'Soap Arc': '#F687B3',
      'Genre-Specific Arc': '#ED8936',
      'Anthology Arc': '#48BB78',
    };

    return (
      <Box borderWidth={1} borderRadius="md" p={4}>
        <FormControl>
          <FormLabel fontWeight="bold">Arc Types</FormLabel>
          <SimpleGrid columns={3} spacing={2}>
            {Object.values(ArcType).map((arcType) => (
              <Checkbox
                key={arcType}
                isChecked={selectedArcTypes.includes(arcType)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedArcTypes([...selectedArcTypes, arcType]);
                  } else {
                    setSelectedArcTypes(selectedArcTypes.filter(t => t !== arcType));
                  }
                }}
              >
                <HStack>
                  <Box
                    w="3"
                    h="3"
                    borderRadius="full"
                    bg={arcTypeColors[arcType as keyof typeof arcTypeColors]}
                  />
                  <Text fontSize="sm">{arcType}</Text>
                </HStack>
              </Checkbox>
            ))}
          </SimpleGrid>
        </FormControl>
      </Box>
    );
  };

  const handleUpdateArc = async (updatedArc: Partial<NarrativeArc>) => {
    try {
      await request(() => api.updateArc(updatedArc.id!, updatedArc));
      onArcUpdated();
      onArcEditModalClose();
      toast({
        title: 'Arc updated',
        status: 'success',
        duration: 3000,
      });
    } catch (error) {
      toast({
        title: 'Error updating arc',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
      });
    }
  };

  return (
    <Box>
      <VStack spacing={4} align="stretch">
        {/* Arc Type Filters */}
        {renderArcTypeFilters()}

        {/* Arc Filters */}
        <ArcFilters
          seasons={seasons}
          selectedSeason={selectedSeason}
          onSeasonChange={setSelectedSeason}
          selectedEpisode={selectedEpisode}
          onEpisodeChange={setSelectedEpisode}
          allCharacters={characters?.map(c => c.best_appellation) || []}
          selectedCharacters={selectedCharacters}
          setSelectedCharacters={setSelectedCharacters}
          includeInterferingCharacters={includeInterferingCharacters}
          setIncludeInterferingCharacters={setIncludeInterferingCharacters}
          episodes={episodes}
          selectedArcTypes={selectedArcTypes}
          setSelectedArcTypes={setSelectedArcTypes}
        />

        {/* Timeline Header with Actions */}
        <HStack justify="space-between" align="center" mb={2}>
          <Text fontSize="lg" fontWeight="bold">Narrative Arcs Timeline</Text>
          <HStack spacing={4}>
            <Button
              colorScheme={isMergeMode ? "orange" : "gray"}
              onClick={() => {
                setIsMergeMode(!isMergeMode);
                if (!isMergeMode) {
                  setSelectedForMerge([]);
                }
              }}
            >
              {isMergeMode ? "Cancel Merge" : "Merge Arcs"}
            </Button>
            {isMergeMode && (
              <Button
                colorScheme="blue"
                isDisabled={selectedForMerge.length !== 2}
                onClick={openMergeModal}
              >
                Merge Selected ({selectedForMerge.length}/2)
              </Button>
            )}
            <Button
              leftIcon={<AddIcon />}
              colorScheme="green"
              onClick={openNewArcModal}
            >
              New Arc
            </Button>
          </HStack>
        </HStack>

        {/* Timeline */}
        <ArcTimeline
          arcs={filteredArcs}
          episodes={episodes}
          selectedSeason={selectedSeason}
          selectedEpisode={selectedEpisode}
          onCellClick={handleCellClick}
          isMergeMode={isMergeMode}
          selectedForMerge={selectedForMerge}
          onToggleMerge={handleToggleMerge}
          onEditArc={handleEditArc}
        />

        {/* Modals */}
        <NewArcModal
          isOpen={isNewArcModalVisible}
          onClose={closeNewArcModal}
          onSubmit={handleCreateArc}
          availableCharacters={characters.map(c => c.best_appellation)}
          series={series}
        />

        {selectedForMerge.length === 2 && (
          <ArcMergeModal
            isOpen={isMergeModalOpen}
            onClose={closeMergeModal}
            arc1={selectedForMerge[0]}
            arc2={selectedForMerge[1]}
            onMergeComplete={handleMergeArcs}
          />
        )}

        <ArcProgressionEditModal
          isOpen={isProgressionModalOpen}
          onClose={onProgressionModalClose}
          progression={selectedProgression}
          onSave={handleProgressionSave}
          onDelete={handleProgressionDelete}
          availableCharacters={characters.map(c => c.best_appellation)}
        />

        <ArcEditModal
          isOpen={isArcEditModalOpen}
          onClose={onArcEditModalClose}
          arc={editingArc}
          onSave={handleUpdateArc}
          onDelete={handleDeleteArc}
          availableCharacters={characters.map(c => c.best_appellation)}
        />
      </VStack>
    </Box>
  );
};