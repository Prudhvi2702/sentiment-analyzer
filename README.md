# Sentiment Analyzer Project

A comprehensive AI-powered sentiment analysis system for product reviews with a production-ready Flask backend API.

## 🎯 Project Overview

This project provides a complete solution for analyzing sentiment in product reviews using state-of-the-art natural language processing models. The system includes:

- **Backend API**: Flask-based REST API with JWT authentication
- **AI Models**: Hugging Face transformers for accurate sentiment analysis
- **AWS Integration**: S3 storage for batch processing results
- **Production Ready**: Scalable architecture with security best practices

## 📁 Project Structure

```
sentiment-analyzer/
├── sentiment-analyzer-backend/     # Flask API backend
│   ├── app.py                      # Main Flask application
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # Detailed backend documentation
│   ├── .env.example               # Environment variables template
│   ├── models/                     # User authentication models
│   ├── services/                   # Core business logic
│   └── utils/                      # Utility functions
└── README.md                       # This file
```

## 🚀 Quick Start

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd sentiment-analyzer-backend
   ```

2. **Follow the detailed setup instructions in the [Backend README](sentiment-analyzer-backend/README.md)**

3. **Key features:**
   - JWT authentication system
   - Single and batch sentiment analysis
   - AWS S3 integration for file storage
   - Production deployment guides

## 🔧 Features

### Backend API
- **Authentication**: Secure JWT-based user authentication
- **Sentiment Analysis**: Real-time sentiment analysis using Hugging Face models
- **Batch Processing**: Upload CSV files for bulk analysis
- **File Storage**: AWS S3 integration for result storage
- **RESTful API**: Clean, documented API endpoints

### Security
- Password hashing with bcrypt
- JWT token authentication
- Input validation and sanitization
- Environment variable configuration
- CORS protection

### Scalability
- Modular architecture
- Service-oriented design
- Ready for containerization
- Production deployment guides

## 📚 Documentation

- **[Backend API Documentation](sentiment-analyzer-backend/README.md)** - Complete setup and API reference
- **API Endpoints**: Authentication, sentiment analysis, batch processing
- **Deployment Guides**: AWS EC2, Docker, and production setup
- **Testing**: Postman collection and sample data

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask (Python)
- **Authentication**: JWT (PyJWT)
- **AI/ML**: Hugging Face Transformers
- **Storage**: AWS S3
- **Security**: bcrypt, input validation
- **Deployment**: Gunicorn, Nginx

### Development
- **Version Control**: Git
- **Environment**: Python virtual environments
- **Testing**: Postman/curl for API testing

## 🔮 Future Enhancements

- [ ] Frontend React application
- [ ] Real-time analysis dashboard
- [ ] Advanced analytics and reporting
- [ ] Multi-language support
- [ ] Mobile application
- [ ] CI/CD pipeline
- [ ] Kubernetes deployment
- [ ] Advanced ML model fine-tuning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
1. Check the [Backend README](sentiment-analyzer-backend/README.md) for detailed documentation
2. Review the troubleshooting section
3. Open an issue on GitHub

---

**Ready for development and deployment!** 🚀
