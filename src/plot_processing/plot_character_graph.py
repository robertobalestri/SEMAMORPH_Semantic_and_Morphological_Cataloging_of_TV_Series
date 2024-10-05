import networkx as nx
from typing import List
from src.plot_processing.plot_processing_models import EntityLink

def build_character_graph(entities: List[EntityLink], text: str) -> nx.Graph:
    G = nx.Graph()
    
    # Add nodes for each character
    for entity in entities:
        G.add_node(entity.character)
    
    # Add edges based on co-occurrence in sentences
    sentences = text.split('.')
    for sentence in sentences:
        characters_in_sentence = [entity.character for entity in entities if entity.character in sentence]
        for i in range(len(characters_in_sentence)):
            for j in range(i+1, len(characters_in_sentence)):
                G.add_edge(characters_in_sentence[i], characters_in_sentence[j])
    
    return G