import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from transformers import pipeline
import torch
from services.s3_service import S3Service
from utils.preprocessor import preprocess_text

logger = logging.getLogger(__name__)

class SentimentService:
    """Service for handling sentiment analysis using Hugging Face Transformers"""
    
    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.classifier = None
        self.s3_service = S3Service()
        
    def load_model(self):
        """Load the sentiment analysis model"""
        try:
            # Check if CUDA is available for GPU acceleration
            device = 0 if torch.cuda.is_available() else -1
            
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                return_all_scores=True
            )
            
            logger.info(f"Sentiment model loaded successfully: {self.model_name}")
            logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")
    
    def analyze_single(self, text: str) -> Dict:
        """
        Analyze sentiment for a single text input
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment label, confidence score, and timestamp
        """
        try:
            if not self.classifier:
                self.load_model()
            
            # Preprocess text
            processed_text = preprocess_text(text)
            
            if not processed_text:
                raise ValueError("Invalid or empty text input")
            
            # Perform sentiment analysis
            results = self.classifier(processed_text)
            
            # Extract the best prediction
            best_result = max(results[0], key=lambda x: x['score'])
            
            return {
                'label': best_result['label'],
                'score': round(best_result['score'], 4),
                'timestamp': datetime.utcnow().isoformat(),
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Single sentiment analysis failed: {str(e)}")
            raise Exception(f"Sentiment analysis failed: {str(e)}")
    
    def analyze_batch(self, texts: List[str]) -> Dict:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        try:
            if not self.classifier:
                self.load_model()
            
            if not texts:
                raise ValueError("No texts provided for analysis")
            
            results = []
            
            # Process each text
            for i, text in enumerate(texts):
                try:
                    processed_text = preprocess_text(text)
                    
                    if not processed_text:
                        results.append({
                            'original_text': text,
                            'processed_text': '',
                            'sentiment': 'NEUTRAL',
                            'confidence': 0.5,
                            'error': 'Empty or invalid text'
                        })
                        continue
                    
                    # Analyze sentiment
                    sentiment_results = self.classifier(processed_text)
                    best_result = max(sentiment_results[0], key=lambda x: x['score'])
                    
                    results.append({
                        'index': i,
                        'original_text': text,
                        'processed_text': processed_text,
                        'sentiment': best_result['label'],
                        'confidence': round(best_result['score'], 4)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze text at index {i}: {str(e)}")
                    results.append({
                        'index': i,
                        'original_text': text,
                        'processed_text': '',
                        'sentiment': 'NEUTRAL',
                        'confidence': 0.5,
                        'error': str(e)
                    })
            
            return {
                'reviews': results,
                'total_processed': len(results),
                'timestamp': datetime.utcnow().isoformat(),
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {str(e)}")
            raise Exception(f"Batch analysis failed: {str(e)}")
    
    def analyze_batch_from_s3(self, s3_key: str, text_column: str = 'review') -> Optional[Dict]:
        """
        Analyze sentiment for a CSV file stored in S3
        
        Args:
            s3_key: S3 key of the CSV file
            text_column: Name of the column containing text to analyze
            
        Returns:
            Dictionary containing analysis results or None if failed
        """
        try:
            # Download and read CSV from S3
            csv_content = self.s3_service.download_file(s3_key)
            
            if not csv_content:
                raise Exception("Failed to download CSV from S3")
            
            # Read CSV content
            df = pd.read_csv(csv_content)
            
            if df.empty:
                raise Exception("CSV file is empty")
            
            # Check if text column exists
            if text_column not in df.columns:
                # Try to find a suitable text column
                possible_columns = ['review', 'text', 'comment', 'feedback', 'description']
                text_column = None
                
                for col in possible_columns:
                    if col in df.columns:
                        text_column = col
                        break
                
                if not text_column:
                    # Use the first column that contains string data
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            text_column = col
                            break
                
                if not text_column:
                    raise Exception("No suitable text column found in CSV")
            
            # Extract texts for analysis
            texts = df[text_column].astype(str).tolist()
            
            # Perform batch analysis
            results = self.analyze_batch(texts)
            
            # Add original CSV metadata
            results['csv_info'] = {
                'total_rows': len(df),
                'text_column': text_column,
                'columns': df.columns.tolist()
            }
            
            logger.info(f"Successfully processed CSV with {len(texts)} reviews from S3: {s3_key}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze CSV from S3: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'loaded': self.classifier is not None,
            'device': 'GPU' if torch.cuda.is_available() else 'CPU',
            'cuda_available': torch.cuda.is_available()
        }



