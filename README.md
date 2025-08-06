# Semantic Book Recommender

An intelligent book recommendation system that uses semantic search and emotional analysis to suggest books based on user preferences.

## Features

- **Semantic Search**: Find books using natural language descriptions
- **Category Filtering**: Filter recommendations by book categories
- **Emotional Tone Analysis**: Get recommendations based on emotional content (Happy, Surprising, Angry, Suspenseful, Sad)
- **Interactive Dashboard**: User-friendly Gradio interface
- **AI-Powered**: Uses OpenAI embeddings and LangChain for semantic understanding

## Data

The system uses a dataset of 7,000+ books with metadata including:
- Book titles, authors, and descriptions
- Categories and genres
- Publication years and ratings
- Emotional analysis scores
- Book covers and thumbnails

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/KatyaMarush/book_recommender.git
   cd book-recommender
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**
   Run the Jupyter notebooks in the correct order to prepare the required data files:
   ```bash
   # 1. Data exploration and initial processing
   jupyter notebook data-exploration.ipynb
   
   # 2. Vector search preparation
   jupyter notebook vector-search.ipynb
   
   # 3. Text classification for categories
   jupyter notebook text-classification.ipynb
   
   # 4. Sentiment analysis for emotions
   jupyter notebook sentiment-analysis.ipynb
   ```
   This will create the necessary processed files: `books_cleaned.csv`, `books_with_categories.csv`, `books_with_emotions.csv`, and `tagged_description.txt`.

5. **Environment Variables**
   Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Web Dashboard
After running all notebooks, launch the interactive Gradio dashboard:
```bash
python gradio-dashboard.py
```

### Jupyter Notebooks (Run in Order)
1. **`data-exploration.ipynb`**: Initial data analysis and preprocessing
2. **`vector-search.ipynb`**: Vector search implementation  
3. **`text-classification.ipynb`**: Text classification models
4. **`sentiment-analysis.ipynb`**: Emotional content analysis

## Project Structure

```
book-recommender/
├── gradio-dashboard.py      # Main web interface
├── requirements.txt         # Python dependencies
├── config.py               # Configuration settings
├── test_recommender.py     # Unit tests
├── cover_placeholder.jpg   # Default book cover
├── *.ipynb                # Jupyter notebooks for analysis
└── data files (created by notebooks):
    ├── books_cleaned.csv
    ├── books_with_categories.csv
    ├── books_with_emotions.csv
    └── tagged_description.txt
```

## Technologies Used

- **Python**: Core programming language
- **Gradio**: Web interface framework
- **LangChain**: AI/ML framework for semantic search
- **OpenAI**: Embeddings and language models
- **Pandas**: Data manipulation
- **ChromaDB**: Vector database for similarity search
- **Transformers**: NLP models for text analysis

## How It Works

1. **Data Processing**: Books are analyzed for categories and emotional content
2. **Semantic Embeddings**: Book descriptions are converted to vector embeddings
3. **Query Processing**: User queries are embedded and compared to book vectors
4. **Filtering**: Results are filtered by category and emotional tone
5. **Ranking**: Books are ranked by relevance and emotional scores

## Example Usage

1. Run the notebooks in order to prepare data
2. Launch the Gradio dashboard: `python gradio-dashboard.py`
3. Enter a book description: "A story about forgiveness"
4. Select a category: "Fiction"
5. Choose emotional tone: "Happy"
6. Get personalized book recommendations with covers and descriptions

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests! 