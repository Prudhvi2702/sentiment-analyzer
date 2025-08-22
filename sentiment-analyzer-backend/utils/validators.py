import re
from typing import Optional

def validate_email(email: str) -> bool:
    """
    Validate email format using regex
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid email format, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    return bool(re.match(pattern, email.strip()))

def validate_password(password: str) -> bool:
    """
    Validate password strength
    
    Args:
        password: Password string to validate
        
    Returns:
        True if password meets requirements, False otherwise
    """
    if not password or not isinstance(password, str):
        return False
    
    # Basic password requirements
    # At least 8 characters long
    if len(password) < 8:
        return False
    
    return True

def validate_password_strong(password: str) -> dict:
    """
    Validate password with detailed requirements
    
    Args:
        password: Password string to validate
        
    Returns:
        Dictionary with validation results and requirements
    """
    if not password or not isinstance(password, str):
        return {
            'valid': False,
            'requirements': {
                'length': False,
                'uppercase': False,
                'lowercase': False,
                'digit': False,
                'special': False
            },
            'message': 'Invalid password input'
        }
    
    requirements = {
        'length': len(password) >= 8,
        'uppercase': bool(re.search(r'[A-Z]', password)),
        'lowercase': bool(re.search(r'[a-z]', password)),
        'digit': bool(re.search(r'\d', password)),
        'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    }
    
    all_valid = all(requirements.values())
    
    messages = []
    if not requirements['length']:
        messages.append('At least 8 characters')
    if not requirements['uppercase']:
        messages.append('At least one uppercase letter')
    if not requirements['lowercase']:
        messages.append('At least one lowercase letter')
    if not requirements['digit']:
        messages.append('At least one digit')
    if not requirements['special']:
        messages.append('At least one special character')
    
    return {
        'valid': all_valid,
        'requirements': requirements,
        'message': 'Password is strong' if all_valid else f"Password must contain: {', '.join(messages)}"
    }

def validate_text_length(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """
    Validate text length
    
    Args:
        text: Text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        True if text length is within bounds, False otherwise
    """
    if not isinstance(text, str):
        return False
    
    text_length = len(text.strip())
    return min_length <= text_length <= max_length

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed_file"
    
    # Remove or replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:250] + ('.' + ext if ext else '')
    
    return sanitized

def validate_csv_file(file) -> dict:
    """
    Validate uploaded CSV file
    
    Args:
        file: Flask file object
        
    Returns:
        Dictionary with validation results
    """
    if not file:
        return {'valid': False, 'message': 'No file provided'}
    
    if not file.filename:
        return {'valid': False, 'message': 'No filename provided'}
    
    # Check file extension
    if not file.filename.lower().endswith('.csv'):
        return {'valid': False, 'message': 'File must be a CSV file'}
    
    # Check file size (limit to 50MB)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    max_size = 50 * 1024 * 1024  # 50MB
    if file_size > max_size:
        return {'valid': False, 'message': f'File size exceeds {max_size // (1024*1024)}MB limit'}
    
    if file_size == 0:
        return {'valid': False, 'message': 'File is empty'}
    
    return {'valid': True, 'message': 'File validation passed', 'size': file_size}



