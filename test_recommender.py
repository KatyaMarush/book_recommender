"""
Tests for the Book Recommender system.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from gradio_dashboard import (
    format_authors,
    truncate_description,
    retrieve_semantic_recommendations
)

class TestBookRecommender:
    """Test cases for book recommender functions."""
    
    def test_format_authors_single(self):
        """Test formatting single author."""
        result = format_authors("John Doe")
        assert result == "John Doe"
    
    def test_format_authors_two(self):
        """Test formatting two authors."""
        result = format_authors("John Doe;Jane Smith")
        assert result == "John Doe and Jane Smith"
    
    def test_format_authors_multiple(self):
        """Test formatting multiple authors."""
        result = format_authors("John Doe;Jane Smith;Bob Johnson")
        assert result == "John Doe, Jane Smith, and Bob Johnson"
    
    def test_format_authors_empty(self):
        """Test formatting empty author string."""
        result = format_authors("")
        assert result == "Unknown Author"
    
    def test_format_authors_none(self):
        """Test formatting None author."""
        result = format_authors(None)
        assert result == "Unknown Author"
    
    def test_truncate_description_short(self):
        """Test truncating short description."""
        desc = "This is a short description."
        result = truncate_description(desc, max_words=5)
        assert result == "This is a short description."
    
    def test_truncate_description_long(self):
        """Test truncating long description."""
        desc = "This is a very long description that should be truncated to only the first few words for display purposes."
        result = truncate_description(desc, max_words=5)
        assert result == "This is a very long..."
    
    def test_truncate_description_empty(self):
        """Test truncating empty description."""
        result = truncate_description("")
        assert result == "No description available."
    
    def test_truncate_description_none(self):
        """Test truncating None description."""
        result = truncate_description(None)
        assert result == "No description available."
    
    @patch('gradio_dashboard.db_books')
    def test_retrieve_semantic_recommendations_empty_query(self, mock_db):
        """Test recommendations with empty query."""
        result = retrieve_semantic_recommendations("")
        assert result.empty
    
    @patch('gradio_dashboard.db_books')
    def test_retrieve_semantic_recommendations_whitespace_query(self, mock_db):
        """Test recommendations with whitespace-only query."""
        result = retrieve_semantic_recommendations("   ")
        assert result.empty

if __name__ == "__main__":
    pytest.main([__file__]) 