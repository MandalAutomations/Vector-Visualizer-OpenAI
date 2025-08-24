# Embedding Visualization with LangChain + OpenAI

This repository demonstrates how to generate **text embeddings** using OpenAI models through [LangChain](https://www.langchain.com/), and then visualize the semantic similarity between words in **3D space** using PCA and Matplotlib.

## Features
- Generate embeddings for arbitrary text using OpenAI’s embedding models.
- Support for multiple models:
  - `text-embedding-3-large`
  - `text-embedding-3-small`
  - `text-embedding-ada-002` (default)
- Dimensionality reduction using **PCA**.
- Interactive **3D scatter plot** visualization of embeddings.

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```txt
langchain-openai
python-dotenv
numpy
matplotlib
scikit-learn
```

You will also need an [OpenAI API key](https://platform.openai.com/).

## Setup
1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/embedding-visualizer.git
   cd embedding-visualizer
   ```

2. Create a `.env` file in the root directory and add your OpenAI API key:

   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

3. Choose your embedding model by editing the `EMBEDDING_MODEL` variable in the script:
   ```python
   EMBEDDING_MODEL="text-embedding-ada-002"
   ```

## Usage
Run the script to generate embeddings and plot them:

```bash
python embeddings_plot.py
```

This will:
1. Generate embeddings for the hardcoded list of words:
   ```python
   texts = ["nfl", "football", "soccer", "basketball", "baseball"]
   ```
2. Reduce them to 3D space using PCA.
3. Save the visualization to `3d_plot_small.png`.

Example output:

📊 A 3D scatter plot showing the relative similarity of sports terms.

## Project Structure
```
.
├── embeddings_plot.py   # Main script
├── requirements.txt     # Dependencies
└── .env                 # API key (not committed)
```

## Customization
- To change the words being compared, edit the `texts` list in `embeddings_plot.py`.
- To try a different embedding model, set `EMBEDDING_MODEL` accordingly.
- To adjust plot resolution, modify the `dpi` parameter in:
  ```python
  plt.savefig("3d_plot_small.png", dpi=1000, bbox_inches='tight')
  ```

## Example Plot
![Alt Text](./static/3d_plot_small.png)
