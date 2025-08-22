from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import os
from dotenv import load_dotenv
import logging
from datetime import timedelta

# Import our custom modules
from models.user import User
from services.sentiment_service import SentimentService
from services.s3_service import S3Service
from utils.validators import validate_email, validate_password
from utils.preprocessor import preprocess_text

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
CORS(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Initialize services
sentiment_service = SentimentService()
s3_service = S3Service()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory user storage (replace with database in production)
users = {}

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Sentiment Analyzer API is running',
        'version': '1.0.0'
    }), 200

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        
        # Validate input
        if not email or not password or not name:
            return jsonify({'error': 'Email, password, and name are required'}), 400
            
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
            
        if not validate_password(password):
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
            
        # Check if user already exists
        if email in users:
            return jsonify({'error': 'User already exists'}), 409
            
        # Hash password and create user
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        users[email] = User(email, password_hash, name)
        
        logger.info(f"New user registered: {email}")
        
        return jsonify({
            'message': 'User registered successfully',
            'user': {'email': email, 'name': name}
        }), 201
        
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
            
        # Check if user exists
        user = users.get(email)
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
            
        # Verify password
        if not bcrypt.check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid credentials'}), 401
            
        # Create access token
        access_token = create_access_token(identity=email)
        
        logger.info(f"User logged in: {email}")
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': {'email': email, 'name': user.name}
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/sentiment', methods=['POST'])
@jwt_required()
def analyze_sentiment():
    """Single review sentiment analysis endpoint"""
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
            
        # Preprocess text
        processed_text = preprocess_text(text)
        
        if not processed_text:
            return jsonify({'error': 'Invalid text input'}), 400
            
        # Analyze sentiment
        result = sentiment_service.analyze_single(processed_text)
        
        logger.info(f"Sentiment analysis completed for user: {current_user}")
        
        return jsonify({
            'text': text,
            'processed_text': processed_text,
            'sentiment': result['label'],
            'confidence': result['score'],
            'analysis_timestamp': result['timestamp']
        }), 200
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return jsonify({'error': 'Failed to analyze sentiment'}), 500

@app.route('/api/batch', methods=['POST'])
@jwt_required()
def batch_sentiment_analysis():
    """Batch CSV sentiment analysis endpoint"""
    try:
        current_user = get_jwt_identity()
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
            
        # Upload file to S3
        s3_key = s3_service.upload_file(file, current_user)
        
        if not s3_key:
            return jsonify({'error': 'Failed to upload file'}), 500
            
        # Process CSV for sentiment analysis
        results = sentiment_service.analyze_batch_from_s3(s3_key)
        
        if not results:
            return jsonify({'error': 'Failed to process CSV file'}), 500
            
        # Calculate summary statistics
        total_reviews = len(results['reviews'])
        positive_count = sum(1 for r in results['reviews'] if r['sentiment'] == 'POSITIVE')
        negative_count = sum(1 for r in results['reviews'] if r['sentiment'] == 'NEGATIVE')
        neutral_count = total_reviews - positive_count - negative_count
        
        logger.info(f"Batch analysis completed for user: {current_user}, reviews: {total_reviews}")
        
        return jsonify({
            'message': 'Batch analysis completed successfully',
            'file_name': file.filename,
            's3_key': s3_key,
            'summary': {
                'total_reviews': total_reviews,
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count,
                'positive_percentage': round((positive_count / total_reviews) * 100, 2),
                'negative_percentage': round((negative_count / total_reviews) * 100, 2),
                'neutral_percentage': round((neutral_count / total_reviews) * 100, 2)
            },
            'reviews': results['reviews'][:100],  # Limit to first 100 for response size
            'processing_timestamp': results['timestamp']
        }), 200
        
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        return jsonify({'error': 'Failed to process batch analysis'}), 500

@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile endpoint"""
    try:
        current_user = get_jwt_identity()
        user = users.get(current_user)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        return jsonify({
            'email': user.email,
            'name': user.name,
            'created_at': user.created_at.isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Profile error: {str(e)}")
        return jsonify({'error': 'Failed to get profile'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token'}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({'error': 'Authorization token is required'}), 401

if __name__ == '__main__':
    # Ensure model is loaded on startup
    sentiment_service.load_model()
    
    # Run the application
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug)



