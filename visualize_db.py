import sys
import os
# Use absolute imports from the src directory
from src.storage.vectorstore_collection import get_vectorstore_collection, CollectionType
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
from src.storage.database import NarrativeArc, ArcProgression, DatabaseManager

# Initialize the database manager
db_manager = DatabaseManager()

# Initialize the vectorstore collection for narrative arcs
vectorstore_collection = get_vectorstore_collection(collection_type=CollectionType.NARRATIVE_ARCS)

# Get embeddings, ids, documents, and metadata from the collection
embeddings = vectorstore_collection.get_all_embeddings()  # Update this line
ids = vectorstore_collection.get_all_ids()
documents = vectorstore_collection.get_all_documents()
metadatas = vectorstore_collection.get_all_metadatas()

# Fetch ArcProgressions from the database based on main arc IDs
arc_progressions = {}
with db_manager.session_scope() as session:  # Use the session_scope method from DatabaseManager
    for arc_id in ids:
        progressions = db_manager.get_arc_progressions(arc_id, session=session)  # Fetch progressions using db_manager
        arc_progressions[arc_id] = [(prog.get_title(session), prog) for prog in progressions]  # Store titles and progressions by arc ID

# Convert embeddings to numpy array
embeddings_array = np.array(embeddings)

# Print some statistics
print(f"Total number of embeddings: {len(embeddings)}")
print(f"Number of documents: {len(documents)}")
print(f"Number of unique titles: {len(set(meta.get('title', '') for meta in metadatas))}")
print(f"Document types: {set(meta.get('doc_type', 'N/A') for meta in metadatas)}")

# Check if we have enough data for PCA
if len(embeddings) < 3:
    print("Not enough embeddings for PCA. Need at least 3.")
    exit()

# Reduce the embedding dimensionality with PCA to 3 components for visualization
pca = PCA(n_components=3)
vis_dims = pca.fit_transform(embeddings_array)

# Prepare text for the scatter plot with relevant metadata
text_labels = []
colors = []
main_arcs = {}  # Dictionary to store main arcs by their id

for doc, meta, doc_id in zip(documents, metadatas, ids):
    
    doc_type = meta.get('doc_type', 'N/A')

    # Convert Characters string to list
    characters = meta.get('characters', '').split(';') if isinstance(meta.get('characters'), str) else meta.get('characters', [])
    
    try:
        if doc_type == 'main':
            print("\nDocument: ", doc)
            print("\nMetadata: ", meta)
            print("\n\n")

            # Create a NarrativeArc instance with updated attributes
            arc_data = {k: v for k, v in meta.items() if k in NarrativeArc.__annotations__}
            arc_data['characters'] = characters
            
            # Debugging output to check arc_data
            print(f"Creating NarrativeArc with data: {arc_data}")

            arc = NarrativeArc(**arc_data)  # Ensure to match the updated model structure
            main_arcs[arc.id] = arc  # Store the main arc in the dictionary
            text_label = f"Title: {arc.title}<br>Description: {arc.description[:20]}"
            colors.append('blue')
        elif doc_type == 'progression':
            # Use the arc_progressions dictionary to get the titles
            progression_titles = [title for title, _ in arc_progressions.get(meta.get('main_arc_id', ''), [])]
            text_label = f"Title: {progression_titles}<br> Progression: {doc[:50]}"  # Use doc for content
            colors.append('red')
        else:
            text_label = f"Unknown Document Type: {doc_type}<br>" + "<br>".join([f"{k}: {v}" for k, v in meta.items()])
            colors.append('gray')

    except Exception as e:
        print(f"Error processing document: {e}")
        text_label = "Error processing document"
        colors.append('gray')

    text_labels.append(text_label)

# Create an interactive 3D plot with Plotly
fig = px.scatter_3d(
    x=vis_dims[:, 0],
    y=vis_dims[:, 1],
    z=vis_dims[:, 2],
    color=colors,
    text=text_labels,  # Show only titles on the dots
    labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'z': 'PCA Component 3', 'color': 'Document Type'},
    title='3D PCA of Narrative Arc Embeddings'
)

# Update color legend
fig.update_layout(
    coloraxis_colorbar=dict(
        title="Document Type",
        tickvals=[0, 1, 2],
        ticktext=["Main", "Progression", "Unknown"],
    )
)

# Show the 3D PCA plot
fig.show()
