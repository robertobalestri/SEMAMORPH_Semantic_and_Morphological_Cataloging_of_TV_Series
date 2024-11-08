from src.narrative_storage.repositories import DatabaseSessionManager
from src.narrative_storage.vector_store_service import VectorStoreService
from src.narrative_storage.narrative_models import NarrativeArc, ArcProgression
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import logging
import pandas as pd
from sqlmodel import select
from sqlalchemy.orm import selectinload
from src.utils.logger_utils import setup_logging

logger = setup_logging(__name__)

def visualize_narrative_arcs():
    # Initialize services
    db_manager = DatabaseSessionManager()
    vector_store_service = VectorStoreService()
    
    # Get all documents from vector store
    results = vector_store_service.collection.get(
        include=['embeddings', 'documents', 'metadatas']
    )
    
    if not results or not results['ids']:
        logger.warning("No documents found in vector store")
        return
        
    embeddings = results['embeddings']
    documents = results['documents']
    metadatas = results['metadatas']
    ids = results['ids']

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)

    # Print statistics
    print(f"Total number of embeddings: {len(embeddings)}")
    print(f"Number of documents: {len(documents)}")
    print(f"Number of unique titles: {len(set(meta.get('title', '') for meta in metadatas))}")
    print(f"Document types: {set(meta.get('doc_type', 'N/A') for meta in metadatas)}")

    # Check if we have enough data for PCA
    if len(embeddings) < 3:
        logger.warning("Not enough embeddings for PCA. Need at least 3.")
        return

    # Reduce dimensionality with PCA
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(embeddings_array)

    # Prepare visualization data
    text_labels = []
    doc_types = []

    with db_manager.session_scope() as session:
        # Fetch all arcs with their progressions in one query
        query = select(NarrativeArc).options(
            selectinload(NarrativeArc.progressions),
            selectinload(NarrativeArc.main_characters)
        )
        arcs = session.exec(query).all()
        arc_cache = {arc.id: arc for arc in arcs}

        for doc, meta, doc_id in zip(documents, metadatas, ids):
            doc_type = meta.get('doc_type', 'N/A')
            
            try:
                if doc_type == 'main':
                    arc = arc_cache.get(meta.get('id'))
                    if arc:
                        # Access relationships within session
                        main_characters = [char.best_appellation for char in arc.main_characters]
                        text_label = (
                            f"Title: {arc.title}<br>"
                            f"Type: {arc.arc_type}<br>"
                            f"Characters: {', '.join(main_characters)}<br>"
                            f"Description: {arc.description[:100]}..."
                        )
                    else:
                        text_label = (
                            f"Title: {meta.get('title', 'N/A')}<br>"
                            f"Type: {meta.get('arc_type', 'N/A')}<br>"
                            f"Characters: {meta.get('main_characters', 'N/A')}<br>"
                            f"Description: {meta.get('description', 'N/A')[:100]}..."
                        )
                    doc_types.append('Main Arc')
                    
                elif doc_type == 'progression':
                    # Get the parent arc to get the title
                    main_arc_id = meta.get('main_arc_id')
                    parent_arc = arc_cache.get(main_arc_id)
                    if parent_arc:
                        text_label = (
                            f"Progression for: {parent_arc.title}<br>"
                            f"S{meta.get('season', 'N/A')}E{meta.get('episode', 'N/A')}<br>"
                            f"Characters: {meta.get('interfering_episode_characters', 'N/A')}<br>"
                            f"Content: {doc[:100]}..."
                        )
                    else:
                        text_label = (
                            f"Progression (Arc not found)<br>"
                            f"S{meta.get('season', 'N/A')}E{meta.get('episode', 'N/A')}<br>"
                            f"Content: {doc[:100]}..."
                        )
                    doc_types.append('Progression')

                    logger.warning(f"Progression datas: {meta}")
                                                                    
                else:
                    text_label = f"Unknown Document Type: {doc_type}"
                    doc_types.append('Unknown')

            except Exception as e:
                logger.error(f"Error processing document: {e}")
                text_label = "Error processing document"
                doc_types.append('Error')

            text_labels.append(text_label)

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'PCA1': vis_dims[:, 0],
        'PCA2': vis_dims[:, 1],
        'PCA3': vis_dims[:, 2],
        'Type': doc_types,
        'Label': text_labels
    })

    # Create interactive 3D plot
    fig = px.scatter_3d(
        df,
        x='PCA1',
        y='PCA2',
        z='PCA3',
        color='Type',
        hover_data=['Label'],
        labels={
            'PCA1': 'PCA Component 1',
            'PCA2': 'PCA Component 2',
            'PCA3': 'PCA Component 3',
        },
        title='3D PCA of Narrative Arc Embeddings',
        color_discrete_map={
            'Main Arc': 'blue',
            'Progression': 'red',
            'Unknown': 'gray',
            'Error': 'black'
        }
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    showarrow=False,
                    x=vis_dims[i][0],
                    y=vis_dims[i][1],
                    z=vis_dims[i][2],
                    text=text_labels[i],
                    xanchor="left",
                    xshift=10,
                    opacity=0.7
                ) for i in range(len(vis_dims))
            ]
        ),
        showlegend=True,
        legend_title_text="Document Type"
    )

    # Show plot
    fig.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    visualize_narrative_arcs()
