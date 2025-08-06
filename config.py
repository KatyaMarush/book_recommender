"""
Configuration settings for the Book Recommender system.
"""
import os
from pathlib import Path
from typing import Dict, List

# Base directory
BASE_DIR = Path(__file__).parent

# Data files
DATA_FILES = {
    "books": BASE_DIR / "books_with_emotions.csv",
    "tagged_descriptions": BASE_DIR / "tagged_description.txt",
    "cover_placeholder": BASE_DIR / "cover_placeholder.jpg",
}

# Model configuration
MODEL_CONFIG = {
    "embedding_model": "text-embedding-ada-002",
    "max_tokens": 4096,
    "temperature": 0.7,
}

# Recommendation settings
RECOMMENDATION_CONFIG = {
    "initial_top_k": 50,
    "final_top_k": 16,
    "max_description_words": 30,
}

# Emotional tone mapping
TONE_MAPPING = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger", 
    "Suspenseful": "fear",
    "Sad": "sadness",
}

# UI Configuration
UI_CONFIG = {
    "gallery_columns": 8,
    "gallery_rows": 2,
    "textbox_max_lines": 3,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return True

def get_data_files() -> Dict[str, Path]:
    """Get validated data file paths."""
    missing_files = [name for name, path in DATA_FILES.items() if not path.exists()]
    
    if missing_files:
        raise FileNotFoundError(f"Missing required data files: {missing_files}")
    
    return DATA_FILES 