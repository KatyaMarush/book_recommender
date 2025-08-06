import pandas as pd
import numpy as np
import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document

import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path(".")
BOOKS_FILE = DATA_DIR / "books_with_emotions.csv"
TAGGED_DESCRIPTIONS_FILE = DATA_DIR / "tagged_description.txt"
COVER_PLACEHOLDER = DATA_DIR / "cover_placeholder.jpg"

# Validate required files exist
def validate_files():
    """Validate that all required files exist."""
    required_files = [BOOKS_FILE, TAGGED_DESCRIPTIONS_FILE, COVER_PLACEHOLDER]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

def load_data() -> pd.DataFrame:
    """Load and preprocess book data."""
    try:
        books = pd.read_csv(BOOKS_FILE)
        
        # Process thumbnail URLs
        books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"
        books["large_thumbnail"] = np.where(
            books["large_thumbnail"].isna() | (books["large_thumbnail"] == "&fife=w800"),
            str(COVER_PLACEHOLDER),
            books["large_thumbnail"],
        )
        
        return books
    except Exception as e:
        logger.error(f"Error loading books data: {e}")
        raise

def load_documents() -> List[Document]:
    """Load tagged descriptions as documents."""
    try:
        with open(TAGGED_DESCRIPTIONS_FILE, "r", encoding="utf-8") as f:
            documents = [Document(page_content=line.strip()) for line in f if line.strip()]
        return documents
    except Exception as e:
        logger.error(f"Error loading tagged descriptions: {e}")
        raise

def initialize_vector_db(documents: List[Document]) -> Chroma:
    """Initialize the vector database."""
    try:
        return Chroma.from_documents(documents, OpenAIEmbeddings())
    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        raise

# Initialize data and database
try:
    validate_files()
    books = load_data()
    documents = load_documents()
    db_books = initialize_vector_db(documents)
    logger.info("Successfully initialized book recommender system")
except Exception as e:
    logger.error(f"Failed to initialize system: {e}")
    raise

def retrieve_semantic_recommendations(
    query: str,
    category: Optional[str] = None,
    tone: Optional[str] = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    """
    Retrieve semantic book recommendations based on query and filters.
    
    Args:
        query: Search query string
        category: Book category filter
        tone: Emotional tone filter
        initial_top_k: Number of initial results
        final_top_k: Number of final results
    
    Returns:
        DataFrame with recommended books
    """
    if not query.strip():
        return pd.DataFrame()
    
    try:
        # Get semantic search results
        recs = db_books.similarity_search(query, k=initial_top_k)
        books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
        book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

        # Apply category filter
        if category and category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
        else:
            book_recs = book_recs.head(final_top_k)

        # Apply tone-based sorting
        tone_mapping = {
            "Happy": "joy",
            "Surprising": "surprise", 
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness"
        }
        
        if tone and tone in tone_mapping:
            book_recs.sort_values(by=tone_mapping[tone], ascending=False, inplace=True)

        return book_recs
    except Exception as e:
        logger.error(f"Error in semantic recommendations: {e}")
        return pd.DataFrame()

def format_authors(authors: str) -> str:
    """Format authors string for display."""
    if not authors or pd.isna(authors):
        return "Unknown Author"
    
    authors_split = authors.split(";")
    if len(authors_split) == 1:
        return authors_split[0]
    elif len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    else:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"

def truncate_description(description: str, max_words: int = 30) -> str:
    """Truncate description to specified number of words."""
    if not description or pd.isna(description):
        return "No description available."
    
    words = description.split()
    if len(words) <= max_words:
        return description
    return " ".join(words[:max_words]) + "..."

def recommend_books(
    query: str,
    category: str,
    tone: str
) -> List[Tuple[str, str]]:
    """
    Generate book recommendations and format for display.
    
    Args:
        query: User search query
        category: Selected category
        tone: Selected emotional tone
    
    Returns:
        List of tuples (image_path, caption)
    """
    if not query.strip():
        return []
    
    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        results = []

        for _, row in recommendations.iterrows():
            # Format description
            truncated_description = truncate_description(row["description"])
            
            # Format authors
            authors_str = format_authors(row["authors"])
            
            # Create caption
            caption = f"{row['title']} by {authors_str}: {truncated_description}"
            results.append((row["large_thumbnail"], caption))

        return results
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []

# Prepare UI options
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")
    
    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
            max_lines=3
        )
        category_dropdown = gr.Dropdown(
            choices=categories, 
            label="Select a category:", 
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, 
            label="Select an emotional tone:", 
            value="All"
        )
        submit_button = gr.Button("Find recommendations", variant="primary")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(
        label="Recommended books", 
        columns=8, 
        rows=2,
        show_label=True
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()