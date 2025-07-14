from flask import Flask, render_template_string
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.decomposition import PCA
import io
import base64

app = Flask(__name__)
@app.route('/')
def show_image():
    static_path = os.path.join(app.root_path, 'static', '3d_plot_small.png')
    if not os.path.exists(static_path):
        return f"Error: Image file not found at {static_path}"
    
    import matplotlib.pyplot as plt

    load_dotenv()

    EMBEDDING_MODEL = "text-embedding-ada-002"
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

    def create_3d_plot():
        texts = [
            "ai",
            "machine learning",
            "deep learning",
            "neural networks",
        ]
        
        embeddings = []
        for text in texts:
            embedding = get_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        print(f"Embeddings shape: {embeddings.shape}")
        
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, label in enumerate(texts):
            x, y, z = embeddings_3d[i]
            text_label = f"{label} ({x:.2f}, {y:.2f}, {z:.2f})"
            ax.scatter(x, y, z, label=label)
            ax.text(x, y, z, text_label, fontsize=10)
        
        ax.set_title("3D Visualization of Similar Words Embeddings")
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Words")
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        print(plot_url)
        return plot_url
    plot_url = create_3d_plot()

    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Plot Viewer</title>
        <style>
            body {{
                text-align: center;
                font-family: Arial, sans-serif;
            }}
            h1 {{
                text-align: center;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <h1>3D Plot Visualization</h1>
        <img src="data:image/png;base64,{plot_url}" alt="3D Plot">
    </body>
    </html>
    '''
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(debug=True)