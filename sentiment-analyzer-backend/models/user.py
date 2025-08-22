from datetime import datetime
from typing import Optional

class User:
    """User model for authentication and user management"""
    
    def __init__(self, email: str, password_hash: str, name: str):
        self.email = email
        self.password_hash = password_hash
        self.name = name
        self.created_at = datetime.utcnow()
        self.last_login = None
        
    def to_dict(self) -> dict:
        """Convert user object to dictionary"""
        return {
            'email': self.email,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        
    def __repr__(self):
        return f"<User {self.email}>"



