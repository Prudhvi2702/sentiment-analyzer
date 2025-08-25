


# Sentiment Analyzer Backend API

A production-ready Flask backend for AI-powered product review sentiment analysis with JWT authentication and AWS S3 integration.

## 🏗️ Project Structure

```
sentiment-analyzer-backend/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .env                       # Your actual environment variables (create this)
├── README.md                  # This file
├── models/
│   ├── __init__.py
│   └── user.py                # User model for authentication
├── services/
│   ├── __init__.py
│   ├── sentiment_service.py   # Hugging Face sentiment analysis
│   └── s3_service.py          # AWS S3 file operations
└── utils/
    ├── __init__.py
    ├── validators.py          # Input validation utilities
    └── preprocessor.py        # Text preprocessing utilities
```

## 🚀 Quick Setup

### 1. Clone and Setup Environment

```bash
# Create project directory
mkdir sentiment-analyzer-backend
cd sentiment-analyzer-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Directory Structure

```bash
# Create required directories
mkdir -p models services utils

# Create __init__.py files
touch models/__init__.py services/__init__.py utils/__init__.py
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and update with your values:

```bash
cp .env.example .env
```

**Required Environment Variables:**

- `SECRET_KEY`: Flask secret key for sessions
- `JWT_SECRET_KEY`: JWT token signing key
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_S3_BUCKET_NAME`: S3 bucket for CSV storage
- `AWS_REGION`: AWS region (default: us-east-1)

### 4. AWS S3 Setup

1. Create an S3 bucket in your AWS console
2. Create an IAM user with S3 permissions:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:GetObject",
           "s3:PutObject",
           "s3:DeleteObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::your-bucket-name/*",
           "arn:aws:s3:::your-bucket-name"
         ]
       }
     ]
   }
   ```
3. Update your `.env` file with the credentials

## 🔥 Running the Application

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the application
python app.py
```

The API will be available at `http://localhost:5000`

## 📡 API Endpoints

### Authentication Endpoints

#### POST `/api/auth/signup`
Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "name": "John Doe"
}
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user": {
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

#### POST `/api/auth/login`
Authenticate user and receive JWT token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "message": "Login successful",
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

### Protected Endpoints (Require JWT Token)

#### POST `/api/sentiment`
Analyze sentiment of a single text review.

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Request Body:**
```json
{
  "text": "This product is absolutely amazing! I love it so much."
}
```

**Response:**
```json
{
  "text": "This product is absolutely amazing! I love it so much.",
  "processed_text": "this product is absolutely amazing! i love it so much.",
  "sentiment": "POSITIVE",
  "confidence": 0.9998,
  "analysis_timestamp": "2024-01-15T10:30:45.123456"
}
```

#### POST `/api/batch`
Upload CSV file and perform batch sentiment analysis.

**Headers:**
```
Authorization: Bearer <your_jwt_token>
Content-Type: multipart/form-data
```

**Request:**
- Form data with file upload
- File field name: `file`
- File type: CSV

**CSV Format Expected:**
```csv
review
"Great product, highly recommend!"
"Terrible quality, waste of money"
"Average product, nothing special"
```

**Response:**
```json
{
  "message": "Batch analysis completed successfully",
  "file_name": "reviews.csv",
  "s3_key": "uploads/user_example_com/20240115_103045_abc12345_reviews.csv",
  "summary": {
    "total_reviews": 100,
    "positive": 65,
    "negative": 20,
    "neutral": 15,
    "positive_percentage": 65.0,
    "negative_percentage": 20.0,
    "neutral_percentage": 15.0
  },
  "reviews": [
    {
      "index": 0,
      "original_text": "Great product, highly recommend!",
      "processed_text": "great product, highly recommend!",
      "sentiment": "POSITIVE",
      "confidence": 0.9985
    }
  ],
  "processing_timestamp": "2024-01-15T10:30:45.123456"
}
```

#### GET `/api/user/profile`
Get current user profile information.

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Response:**
```json
{
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2024-01-15T10:00:00.000000"
}
```

## 🧪 Testing with Postman

### 1. Setup Postman Collection

Create a new collection called "Sentiment Analyzer API" with the following requests:

### 2. Test Authentication

#### Test Signup:
- **Method:** POST
- **URL:** `http://localhost:5000/api/auth/signup`
- **Body:** Raw JSON
```json
{
  "email": "test@example.com",
  "password": "testpass123",
  "name": "Test User"
}
```

#### Test Login:
- **Method:** POST
- **URL:** `http://localhost:5000/api/auth/login`
- **Body:** Raw JSON
```json
{
  "email": "test@example.com",
  "password": "testpass123"
}
```

**Save the `access_token` from the response for subsequent requests.**

### 3. Test Sentiment Analysis

#### Test Single Sentiment:
- **Method:** POST
- **URL:** `http://localhost:5000/api/sentiment`
- **Headers:** 
  - `Authorization: Bearer <your_access_token>`
  - `Content-Type: application/json`
- **Body:** Raw JSON
```json
{
  "text": "This product exceeded my expectations! Absolutely fantastic quality and fast shipping."
}
```

#### Test Batch Analysis:
- **Method:** POST
- **URL:** `http://localhost:5000/api/batch`
- **Headers:** 
  - `Authorization: Bearer <your_access_token>`
- **Body:** Form-data
  - Key: `file`
  - Type: File
  - Value: Upload a CSV file with reviews

### 4. Test Protected Route

#### Test Profile:
- **Method:** GET
- **URL:** `http://localhost:5000/api/user/profile`
- **Headers:** 
  - `Authorization: Bearer <your_access_token>`

## 📁 Sample CSV Format

Create a test CSV file (`sample_reviews.csv`) with this content:

```csv
review
"Amazing product! Love it so much, highly recommended."
"Terrible quality. Broke after one day of use."
"It's okay, nothing special but does the job."
"Outstanding customer service and fast delivery!"
"Waste of money. Very disappointed with this purchase."
"Good value for money. Happy with my purchase."
"Poor packaging, item arrived damaged."
"Excellent quality and exactly as described."
"Not worth the price. Found better alternatives."
"Perfect! Exactly what I was looking for."
```

## 🔒 Security Features

- **JWT Authentication:** Secure token-based authentication
- **Password Hashing:** Bcrypt encryption for stored passwords
- **Input Validation:** Comprehensive validation for all inputs
- **Rate Limiting Ready:** Structured for easy rate limiting implementation
- **CORS Protection:** Configurable cross-origin resource sharing
- **Environment Variables:** Sensitive data stored securely

## 🚀 Production Deployment

### AWS EC2 Deployment

1. **Launch EC2 Instance:**
   - Ubuntu 20.04 LTS
   - Security group allowing HTTP (80) and HTTPS (443)
   - Key pair for SSH access

2. **Server Setup:**
```bash
# Connect to EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv nginx -y

# Clone your repository
git clone <your-repo-url>
cd sentiment-analyzer-backend

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create production environment file
cp .env.example .env
# Edit .env with production values
```

3. **Configure Gunicorn:**
```bash
# Install gunicorn (should be in requirements.txt)
pip install gunicorn

# Test gunicorn
gunicorn --bind 0.0.0.0:5000 app:app

# Create systemd service
sudo nano /etc/systemd/system/sentiment-api.service
```

**Service file content:**
```ini
[Unit]
Description=Sentiment Analyzer API
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/sentiment-analyzer-backend
Environment="PATH=/home/ubuntu/sentiment-analyzer-backend/venv/bin"
ExecStart=/home/ubuntu/sentiment-analyzer-backend/venv/bin/gunicorn --workers 3 --bind unix:sentiment-api.sock -m 007 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

4. **Configure Nginx:**
```bash
sudo nano /etc/nginx/sites-available/sentiment-api
```

**Nginx configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/ubuntu/sentiment-analyzer-backend/sentiment-api.sock;
    }
}
```

5. **Enable and Start Services:**
```bash
# Enable nginx site
sudo ln -s /etc/nginx/sites-available/sentiment-api /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx

# Enable and start API service
sudo systemctl daemon-reload
sudo systemctl enable sentiment-api
sudo systemctl start sentiment-api
sudo systemctl status sentiment-api
```

## 🐛 Troubleshooting

### Common Issues:

1. **Model Loading Errors:**
   - Ensure sufficient disk space (models are ~250MB)
   - Check internet connectivity for first-time download
   - Verify write permissions in the application directory

2. **S3 Connection Issues:**
   - Verify AWS credentials are correct
   - Check S3 bucket exists and is accessible
   - Ensure IAM permissions are properly set

3. **JWT Token Errors:**
   - Verify JWT_SECRET_KEY is set in environment
   - Check token expiration (24 hours by default)
   - Ensure Bearer prefix in Authorization header

4. **Memory Issues:**
   - Transformer models require ~1GB RAM minimum
   - Consider using CPU-only mode for smaller instances
   - Implement request queuing for high load

### Debug Mode:

Set `FLASK_ENV=development` in your `.env` file to enable debug mode with detailed error messages.

## 📈 Performance Optimization

- **Model Caching:** Models are cached after first load
- **GPU Support:** Automatically uses GPU if available
- **Batch Processing:** Optimized for large CSV files
- **Connection Pooling:** AWS SDK manages connection pooling
- **Memory Management:** Efficient text preprocessing pipeline

## 🔧 Next Steps

After successfully testing the backend:

1. **Add Database Integration:** Replace in-memory storage with PostgreSQL/MongoDB
2. **Implement Rate Limiting:** Add request rate limiting per user
3. **Add Logging:** Implement structured logging with rotation
4. **Monitoring:** Add health checks and metrics endpoints
5. **Caching:** Implement Redis for response caching
6. **API Documentation:** Generate OpenAPI/Swagger docs
7. **Frontend Integration:** Build React frontend
8. **CI/CD Pipeline:** Setup automated deployment

