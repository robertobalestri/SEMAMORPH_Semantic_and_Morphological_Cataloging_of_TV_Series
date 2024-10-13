from langchain_chroma import Chroma
from vectorstore.vectorstore_collection import get_vectorstore_collection, CollectionType
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

# Initialize the vectorstore collection
vectorstore_collection = get_vectorstore_collection(collection_type=CollectionType.NARRATIVE_ARCS)

# Get embeddings and metadata
embeddings = vectorstore_collection.get_embeddings()
ids = vectorstore_collection.get_ids()
documents = vectorstore_collection.get_documents()
metadatas = vectorstore_collection.get_metadatas()

# Convert embeddings to numpy array
embeddings_array = np.array(embeddings)

# Print some statistics
print(f"Total number of embeddings: {len(embeddings)}")
print(f"Number of documents: {len(documents)}")
print(f"Number of unique titles: {len(set(meta['title'] for meta in metadatas if 'title' in meta))}")
print(f"Document types: {set(meta['doc_type'] for meta in metadatas if 'doc_type' in meta)}")

# Check if we have enough data for PCA
if len(embeddings) < 3:
    print("Not enough embeddings for PCA. Need at least 3.")
    exit()

# Reduce the embedding dimensionality
pca = PCA(n_components=3)
vis_dims = pca.fit_transform(embeddings_array)

# Prepare text for the scatter plot
text_labels = []
for doc, meta in zip(documents, metadatas):
    title = meta.get('title', 'N/A')
    doc_type = meta.get('doc_type', 'N/A')
    description = meta.get('description', 'N/A')
    progression = meta.get('progression', 'N/A')
    #the text label should contain the description or the progression only if they are not 'N/A'
    text_labels.append(f"Title: {title}<br>Type: {doc_type}<br>{'Description: ' + description if description != 'N/A' else ''}{'Progression: ' + progression if progression != 'N/A' else ''}")

# Create an interactive 3D plot
fig = px.scatter_3d(
    x=vis_dims[:, 0],
    y=vis_dims[:, 1],
    z=vis_dims[:, 2],
    text=text_labels,
    labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'z': 'PCA Component 3'},
    title='3D PCA of Narrative Arc Embeddings'
)

# Update hover information
fig.update_traces(hoverinfo="text", hovertext=text_labels)

# Show the plot
fig.show()