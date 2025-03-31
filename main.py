from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
load_dotenv()

EMBEDDING_MODEL="text-embedding-3-large" #"text-embedding-3-small" #"text-embedding-ada-002"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

def get_embedding(text: str) -> np.ndarray:

    try:
        embeddings = openai_embeddings.embed_query(text)
        return np.array(embeddings)
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def plot_embeddings_3d(embeddings_3d, labels):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(labels):
        x, y, z = embeddings_3d[i]
        text= f"{label} ({x:.2f}, {y:.2f}, {z:.2f})"
        ax.scatter(x, y, z, label=label)
        ax.text(x, y, z, text, fontsize=10)

    ax.set_title("3D Visualization of Similar Words Embeddings")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Words")  # Adjust position
    plt.tight_layout()
    plt.savefig("3d_plot_small.png", dpi=1000, bbox_inches='tight')
    
texts = [
    "nfl",
    "football",
    "soccer",
    "basketball",
    "baseball",
]


embeddings = []
for text in texts:
    embedding = get_embedding(text)
    if embedding is not None:
        embeddings.append(embedding)
        

embeddings = np.array(embeddings)

pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

plot_embeddings_3d(embeddings_3d, texts)