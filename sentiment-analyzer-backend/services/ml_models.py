"""
Additional ML Models for Sentiment Analysis
Includes TensorFlow and Scikit-learn implementations alongside Hugging Face
"""

import os
import pickle
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocessor import preprocess_text

logger = logging.getLogger(__name__)

class MLModelsService:
    """
    Service class that provides multiple ML approaches for sentiment analysis:
    1. Scikit-learn (TF-IDF + Logistic Regression)
    2. TensorFlow/Keras (LSTM Neural Network)
    3. Hugging Face (DistilBERT) - existing implementation
    """
    
    def __init__(self):
        self.sklearn_model = None
        self.sklearn_vectorizer = None
        self.tensorflow_model = None
        self.tf_tokenizer = None
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load pre-trained models if available
        self._load_sklearn_model()
        self._load_tensorflow_model()
    
    def _load_sklearn_model(self):
        """Load pre-trained Scikit-learn model and vectorizer"""
        try:
            model_path = os.path.join(self.models_dir, 'sklearn_sentiment_model.pkl')
            vectorizer_path = os.path.join(self.models_dir, 'sklearn_vectorizer.pkl')
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                with open(model_path, 'rb') as f:
                    self.sklearn_model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    self.sklearn_vectorizer = pickle.load(f)
                logger.info("Scikit-learn model loaded successfully")
            else:
                logger.warning("Scikit-learn model files not found. Use train_sklearn_model() to create them.")
                
        except Exception as e:
            logger.error(f"Error loading Scikit-learn model: {str(e)}")
    
    def _load_tensorflow_model(self):
        """Load pre-trained TensorFlow model"""
        try:
            model_path = os.path.join(self.models_dir, 'tensorflow_sentiment_model.h5')
            tokenizer_path = os.path.join(self.models_dir, 'tf_tokenizer.pkl')
            
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                self.tensorflow_model = load_model(model_path)
                with open(tokenizer_path, 'rb') as f:
                    self.tf_tokenizer = pickle.load(f)
                logger.info("TensorFlow model loaded successfully")
            else:
                logger.warning("TensorFlow model files not found. Use train_tensorflow_model() to create them.")
                
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {str(e)}")
    
    def train_sklearn_model(self, texts, labels):
        """
        Train a Scikit-learn model using TF-IDF + Logistic Regression
        
        Args:
            texts (list): List of review texts
            labels (list): List of sentiment labels (0 for negative, 1 for positive)
        """
        try:
            logger.info("Training Scikit-learn sentiment model...")
            
            # Preprocess texts
            processed_texts = [preprocess_text(text) for text in texts]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels, test_size=0.2, random_state=42
            )
            
            # Create TF-IDF vectorizer
            self.sklearn_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit vectorizer and transform texts
            X_train_tfidf = self.sklearn_vectorizer.fit_transform(X_train)
            X_test_tfidf = self.sklearn_vectorizer.transform(X_test)
            
            # Train Logistic Regression model
            self.sklearn_model = LogisticRegression(random_state=42, max_iter=1000)
            self.sklearn_model.fit(X_train_tfidf, y_train)
            
            # Evaluate model
            y_pred = self.sklearn_model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Scikit-learn model training completed. Accuracy: {accuracy:.4f}")
            
            # Save model and vectorizer
            model_path = os.path.join(self.models_dir, 'sklearn_sentiment_model.pkl')
            vectorizer_path = os.path.join(self.models_dir, 'sklearn_vectorizer.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.sklearn_model, f)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.sklearn_vectorizer, f)
                
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training Scikit-learn model: {str(e)}")
            return None
    
    def train_tensorflow_model(self, texts, labels, max_features=10000, max_length=100, epochs=5):
        """
        Train a TensorFlow/Keras LSTM model for sentiment analysis
        
        Args:
            texts (list): List of review texts
            labels (list): List of sentiment labels (0 for negative, 1 for positive)
            max_features (int): Maximum number of words to keep
            max_length (int): Maximum sequence length
            epochs (int): Number of training epochs
        """
        try:
            logger.info("Training TensorFlow sentiment model...")
            
            # Preprocess texts
            processed_texts = [preprocess_text(text) for text in texts]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels, test_size=0.2, random_state=42
            )
            
            # Create tokenizer
            self.tf_tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
            self.tf_tokenizer.fit_on_texts(X_train)
            
            # Convert texts to sequences
            X_train_seq = self.tf_tokenizer.texts_to_sequences(X_train)
            X_test_seq = self.tf_tokenizer.texts_to_sequences(X_test)
            
            # Pad sequences
            X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)
            
            # Build LSTM model
            self.tensorflow_model = Sequential([
                Embedding(max_features, 128, input_length=max_length),
                LSTM(64, dropout=0.2, recurrent_dropout=0.2),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            self.tensorflow_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = self.tensorflow_model.fit(
                X_train_pad, y_train,
                epochs=epochs,
                batch_size=32,
                validation_data=(X_test_pad, y_test),
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy = self.tensorflow_model.evaluate(X_test_pad, y_test, verbose=0)
            logger.info(f"TensorFlow model training completed. Test Accuracy: {test_accuracy:.4f}")
            
            # Save model and tokenizer
            model_path = os.path.join(self.models_dir, 'tensorflow_sentiment_model.h5')
            tokenizer_path = os.path.join(self.models_dir, 'tf_tokenizer.pkl')
            
            self.tensorflow_model.save(model_path)
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tf_tokenizer, f)
                
            return test_accuracy
            
        except Exception as e:
            logger.error(f"Error training TensorFlow model: {str(e)}")
            return None
    
    def predict_sklearn(self, text):
        """
        Predict sentiment using Scikit-learn model
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Prediction results
        """
        if not self.sklearn_model or not self.sklearn_vectorizer:
            return {"error": "Scikit-learn model not available"}
        
        try:
            processed_text = preprocess_text(text)
            text_tfidf = self.sklearn_vectorizer.transform([processed_text])
            
            # Get prediction and probability
            prediction = self.sklearn_model.predict(text_tfidf)[0]
            probabilities = self.sklearn_model.predict_proba(text_tfidf)[0]
            
            # Convert to sentiment label
            sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
            confidence = max(probabilities)
            
            return {
                "model": "scikit-learn",
                "text": text,
                "processed_text": processed_text,
                "sentiment": sentiment,
                "confidence": float(confidence),
                "probabilities": {
                    "negative": float(probabilities[0]),
                    "positive": float(probabilities[1])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Scikit-learn prediction: {str(e)}")
            return {"error": str(e)}
    
    def predict_tensorflow(self, text, max_length=100):
        """
        Predict sentiment using TensorFlow model
        
        Args:
            text (str): Text to analyze
            max_length (int): Maximum sequence length
            
        Returns:
            dict: Prediction results
        """
        if not self.tensorflow_model or not self.tf_tokenizer:
            return {"error": "TensorFlow model not available"}
        
        try:
            processed_text = preprocess_text(text)
            
            # Convert to sequence and pad
            sequence = self.tf_tokenizer.texts_to_sequences([processed_text])
            padded_sequence = pad_sequences(sequence, maxlen=max_length)
            
            # Get prediction
            prediction = self.tensorflow_model.predict(padded_sequence, verbose=0)[0][0]
            
            # Convert to sentiment label
            sentiment = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            return {
                "model": "tensorflow",
                "text": text,
                "processed_text": processed_text,
                "sentiment": sentiment,
                "confidence": float(confidence),
                "raw_prediction": float(prediction)
            }
            
        except Exception as e:
            logger.error(f"Error in TensorFlow prediction: {str(e)}")
            return {"error": str(e)}
    
    def compare_models(self, text):
        """
        Compare predictions from all available models
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Comparison results from all models
        """
        results = {
            "text": text,
            "models": {}
        }
        
        # Scikit-learn prediction
        sklearn_result = self.predict_sklearn(text)
        if "error" not in sklearn_result:
            results["models"]["scikit_learn"] = sklearn_result
        
        # TensorFlow prediction
        tensorflow_result = self.predict_tensorflow(text)
        if "error" not in tensorflow_result:
            results["models"]["tensorflow"] = tensorflow_result
        
        # You can add Hugging Face result here if needed
        # results["models"]["hugging_face"] = hugging_face_result
        
        return results

# Create global instance
ml_models_service = MLModelsService()