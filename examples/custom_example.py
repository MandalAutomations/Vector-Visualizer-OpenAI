#!/usr/bin/env python3
"""
Example script showing how to customize Vector-Visualizer for different domains.

This example demonstrates visualizing embeddings for programming languages.
You can modify the 'texts' list to explore any domain you're interested in.
"""

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Error: Please set your OPENAI_API_KEY in a .env file")
    exit(1)

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding for given text."""
    try:
        embeddings = openai_embeddings.embed_query(text)
        return np.array(embeddings)
    except Exception as e:
        print(f"Error generating embedding for '{text}': {e}")
        return None

def plot_embeddings_3d(embeddings_3d, labels, title="3D Visualization", filename="custom_plot.png"):
    """Create and save 3D plot of embeddings."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create different colors for each point
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    for i, label in enumerate(labels):
        x, y, z = embeddings_3d[i]
        text_label = f"{label}\n({x:.2f}, {y:.2f}, {z:.2f})"
        ax.scatter(x, y, z, label=label, color=colors[i], s=100)
        ax.text(x, y, z, text_label, fontsize=9)

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Terms")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Plot saved as: {filename}")

def main():
    """Main function to demonstrate custom embedding visualization."""
    
    # CUSTOMIZE THIS: Change to any words/concepts you want to explore
    texts = [
        "Python",
        "JavaScript", 
        "Java",
        "C++",
        "Ruby",
        "Go",
        "Rust",
        "TypeScript"
    ]
    
    print(f"Generating embeddings for: {', '.join(texts)}")
    print("This may take a few seconds...")
    
    # Generate embeddings
    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            print(f"Failed to generate embedding for: {text}")
    
    if len(embeddings) < 2:
        print("Error: Need at least 2 successful embeddings to create visualization")
        return
    
    # Convert to numpy array and apply PCA
    embeddings = np.array(embeddings)
    print(f"Original embedding dimensions: {embeddings.shape}")
    
    # Reduce to 3D using PCA
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Show explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by each component: {explained_variance}")
    print(f"Total explained variance: {sum(explained_variance):.2%}")
    
    # Create visualization
    plot_embeddings_3d(
        embeddings_3d, 
        texts[:len(embeddings)], 
        title="Programming Languages - Semantic Similarity",
        filename="programming_languages_3d.png"
    )

if __name__ == "__main__":
    main()