# Vector-Visualizer

A Python application for visualizing word embeddings in 3D space using OpenAI's embedding models. This tool converts high-dimensional text embeddings into interactive 3D scatter plots, making it easy to explore semantic relationships between words and concepts.

## Features

- 🎯 **OpenAI Integration**: Uses OpenAI's text embedding models (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
- 📊 **3D Visualization**: Creates interactive 3D scatter plots showing semantic relationships
- 🔄 **Dimensionality Reduction**: Uses PCA to reduce high-dimensional embeddings to 3D space
- 🌐 **Web Interface**: Flask-based web application for easy visualization
- 💾 **Export Options**: Save plots as high-resolution PNG images
- 🎨 **Customizable**: Easily modify word lists and visualization parameters

## Screenshot

The application generates 3D visualizations like this:

![3D Plot Example](static/3d_plot_small.png)

*Example: 3D visualization of AI-related terms showing semantic clustering*

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MandalAutomations/Vector-Visualizer.git
   cd Vector-Visualizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Or export the environment variable:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

## Usage

### Method 1: Standalone Script

Run the standalone script to generate embeddings for sports-related terms:

```bash
python main.py
```

This will:
- Generate embeddings for: "nfl", "football", "soccer", "basketball", "baseball"
- Create a 3D plot using PCA
- Save the result as `3d_plot_small.png`

### Method 2: Web Application

Launch the Flask web application:

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

The web app will:
- Generate embeddings for AI-related terms: "ai", "machine learning", "deep learning", "neural networks"
- Display an interactive 3D plot in your browser
- Show coordinates for each point

### Method 3: Custom Domains

Use the example script to explore embeddings in any domain:

```bash
python examples/custom_example.py
```

This example visualizes programming languages but can be easily customized for any topic. Edit the `texts` list in the script to explore:
- Different programming languages
- Scientific concepts
- Brand names
- Movie genres
- Countries and cities
- Any domain you're interested in!

### Customizing Word Lists

#### For the standalone script (`main.py`):
```python
texts = [
    "your_word_1",
    "your_word_2", 
    "your_word_3",
    # Add more words...
]
```

#### For the web application (`app.py`):
```python
texts = [
    "your_word_1",
    "your_word_2",
    "your_word_3", 
    # Add more words...
]
```

## Configuration

### Embedding Models

You can switch between different OpenAI embedding models by changing the `EMBEDDING_MODEL` variable:

```python
# Available options:
EMBEDDING_MODEL = "text-embedding-ada-002"      # Cheaper, good performance
EMBEDDING_MODEL = "text-embedding-3-small"     # Better performance
EMBEDDING_MODEL = "text-embedding-3-large"     # Best performance, most expensive
```

### Visualization Parameters

Customize the plot appearance in the plotting functions:

```python
# Adjust figure size
fig = plt.figure(figsize=(10, 7))

# Modify DPI for higher resolution
plt.savefig("output.png", dpi=1000, bbox_inches='tight')

# Change PCA components (always 3 for 3D plots)
pca = PCA(n_components=3)
```

## Project Structure

```
Vector-Visualizer/
├── main.py              # Standalone script for sports terms
├── app.py               # Flask web application
├── src/
│   └── create_image.py  # Image generation utilities
├── examples/
│   └── custom_example.py # Example script for custom domains
├── static/
│   └── 3d_plot_small.png # Generated plot images
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: 3D plotting and visualization
- **scikit-learn**: PCA dimensionality reduction
- **flask**: Web application framework
- **langchain-openai**: OpenAI API integration
- **python-dotenv**: Environment variable management
- **pandas**: Data manipulation (optional)

## How It Works

1. **Text Input**: Provide a list of words or phrases
2. **Embedding Generation**: Use OpenAI's API to convert text to high-dimensional vectors
3. **Dimensionality Reduction**: Apply PCA to reduce embeddings to 3D coordinates
4. **Visualization**: Plot points in 3D space where similar concepts cluster together
5. **Interpretation**: Words with similar meanings appear closer in the 3D space

## API Costs

Using OpenAI's embedding API incurs costs based on the number of tokens processed:

- **text-embedding-ada-002**: $0.0001 per 1K tokens
- **text-embedding-3-small**: $0.00002 per 1K tokens  
- **text-embedding-3-large**: $0.00013 per 1K tokens

For typical word lists (5-20 words), costs are minimal (usually < $0.01 per run).

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**
   ```
   Error: OpenAI API key not found
   ```
   Solution: Ensure your `.env` file contains `OPENAI_API_KEY=your_key` or export the environment variable.

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain_openai'
   ```
   Solution: Install dependencies with `pip install -r requirements.txt`

3. **Empty Plot**
   ```
   Error generating embedding: [API Error]
   ```
   Solution: Check your OpenAI API key and account credits.

### Debug Mode

Enable debug mode in the Flask app for detailed error messages:

```python
if __name__ == '__main__':
    app.run(debug=True)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Use Cases

- **Semantic Analysis**: Explore relationships between concepts
- **Content Clustering**: Group similar topics or themes
- **Educational**: Visualize how AI understands language
- **Research**: Analyze semantic spaces in different domains
- **Marketing**: Understand brand positioning and competitor relationships

## License

This project is open source. Please check with the repository owner for specific licensing terms.

## Acknowledgments

- [OpenAI](https://openai.com/) for providing the embedding API
- [scikit-learn](https://scikit-learn.org/) for PCA implementation
- [matplotlib](https://matplotlib.org/) for 3D visualization capabilities
