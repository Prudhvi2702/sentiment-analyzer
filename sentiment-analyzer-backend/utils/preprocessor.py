import re
import html
from typing import Optional

def preprocess_text(text: str) -> Optional[str]:
    """
    Preprocess text for sentiment analysis
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned and preprocessed text, or None if invalid
    """
    if not text or not isinstance(text, str):
        return None
    
    # Convert to string if not already
    text = str(text)
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Remove or replace problematic characters
    # Remove excessive punctuation but keep sentence structure
    text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations to single
    text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions to single
    text = re.sub(r'[.]{3,}', '...', text)  # Multiple dots to ellipsis
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove excessive whitespace again
    text = ' '.join(text.split())
    
    # Check if text is too short after preprocessing
    if len(text.strip()) < 2:
        return None
    
    # Limit text length to prevent model issues (DistilBERT has 512 token limit)
    if len(text) > 1000:  # Conservative limit for tokens
        text = text[:1000].rsplit(' ', 1)[0]  # Cut at word boundary
    
    return text.strip()

def clean_for_csv(text: str) -> str:
    """
    Clean text specifically for CSV processing
    
    Args:
        text: Raw text from CSV
        
    Returns:
        Cleaned text suitable for analysis
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Handle common CSV issues
    text = str(text)
    
    # Remove leading/trailing quotes that might come from CSV parsing
    text = text.strip('"').strip("'")
    
    # Handle escaped quotes
    text = text.replace('""', '"')
    text = text.replace("''", "'")
    
    # Apply standard preprocessing
    processed = preprocess_text(text)
    
    return processed if processed else ""

def extract_meaningful_words(text: str, min_word_length: int = 2) -> list:
    """
    Extract meaningful words from text for analysis
    
    Args:
        text: Input text
        min_word_length: Minimum length for words to be included
        
    Returns:
        List of meaningful words
    """
    if not text:
        return []
    
    # Common stop words that don't contribute to sentiment
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
        'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter meaningful words
    meaningful_words = [
        word for word in words 
        if len(word) >= min_word_length and word not in stop_words
    ]
    
    return meaningful_words

def validate_text_content(text: str) -> dict:
    """
    Validate text content for sentiment analysis suitability
    
    Args:
        text: Text to validate
        
    Returns:
        Dictionary with validation results and suggestions
    """
    if not text or not isinstance(text, str):
        return {
            'valid': False,
            'reason': 'Empty or invalid text',
            'suggestions': ['Provide non-empty text input']
        }
    
    original_length = len(text)
    processed_text = preprocess_text(text)
    
    if not processed_text:
        return {
            'valid': False,
            'reason': 'Text becomes empty after preprocessing',
            'suggestions': ['Provide more substantial text content', 'Check for meaningful words']
        }
    
    processed_length = len(processed_text)
    meaningful_words = extract_meaningful_words(processed_text)
    
    suggestions = []
    warnings = []
    
    # Check length
    if processed_length < 10:
        warnings.append('Text is very short, analysis may be less accurate')
        suggestions.append('Consider providing more detailed text')
    
    # Check meaningful content
    if len(meaningful_words) < 2:
        warnings.append('Limited meaningful words detected')
        suggestions.append('Ensure text contains descriptive words')
    
    # Check if mostly punctuation or numbers
    alpha_ratio = sum(c.isalpha() for c in processed_text) / len(processed_text)
    if alpha_ratio < 0.3:
        warnings.append('Text contains limited alphabetic content')
        suggestions.append('Provide more descriptive text with words')
    
    return {
        'valid': True,
        'processed_text': processed_text,
        'original_length': original_length,
        'processed_length': processed_length,
        'meaningful_words_count': len(meaningful_words),
        'alpha_ratio': round(alpha_ratio, 2),
        'warnings': warnings,
        'suggestions': suggestions
    }



