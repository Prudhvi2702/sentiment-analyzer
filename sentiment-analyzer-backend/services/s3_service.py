import boto3
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict
from botocore.exceptions import ClientError, NoCredentialsError
from io import BytesIO, StringIO
import uuid

logger = logging.getLogger(__name__)

class S3Service:
    """Service for handling AWS S3 operations"""
    
    def __init__(self):
        self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=self.region
            )
            
            # Test connection
            self._test_connection()
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            self.s3_client = None
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            self.s3_client = None
    
    def _test_connection(self):
        """Test S3 connection and bucket access"""
        try:
            if not self.bucket_name:
                logger.warning("S3 bucket name not configured")
                return False
                
            # Try to list objects in the bucket (just to test access)
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 connection successful. Using bucket: {self.bucket_name}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket '{self.bucket_name}' not found")
            elif error_code == '403':
                logger.error(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                logger.error(f"S3 connection test failed: {str(e)}")
            return False
    
    def upload_file(self, file, user_email: str) -> Optional[str]:
        """
        Upload a file to S3
        
        Args:
            file: File object from Flask request
            user_email: Email of the user uploading the file
            
        Returns:
            S3 key of the uploaded file or None if failed
        """
        try:
            if not self.s3_client or not self.bucket_name:
                logger.error("S3 service not properly configured")
                return None
            
            # Generate unique filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            safe_email = user_email.replace('@', '_').replace('.', '_')
            
            # Create S3 key with organized folder structure
            s3_key = f"uploads/{safe_email}/{timestamp}_{unique_id}_{file.filename}"
            
            # Upload file to S3
            file.seek(0)  # Reset file pointer
            
            self.s3_client.upload_fileobj(
                file,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'text/csv',
                    'Metadata': {
                        'user_email': user_email,
                        'upload_timestamp': timestamp,
                        'original_filename': file.filename
                    }
                }
            )
            
            logger.info(f"File uploaded successfully to S3: {s3_key}")
            return s3_key
            
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {str(e)}")
            return None
    
    def download_file(self, s3_key: str) -> Optional[StringIO]:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 key of the file to download
            
        Returns:
            StringIO object containing file content or None if failed
        """
        try:
            if not self.s3_client or not self.bucket_name:
                logger.error("S3 service not properly configured")
                return None
            
            # Download file from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            
            # Read file content
            file_content = response['Body'].read().decode('utf-8')
            
            logger.info(f"File downloaded successfully from S3: {s3_key}")
            return StringIO(file_content)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"File not found in S3: {s3_key}")
            else:
                logger.error(f"Failed to download file from S3: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to download file from S3: {str(e)}")
            return None
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            s3_key: S3 key of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.s3_client or not self.bucket_name:
                logger.error("S3 service not properly configured")
                return False
            
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            
            logger.info(f"File deleted successfully from S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file from S3: {str(e)}")
            return False
    
    def list_user_files(self, user_email: str) -> List[Dict]:
        """
        List files uploaded by a specific user
        
        Args:
            user_email: Email of the user
            
        Returns:
            List of file information dictionaries
        """
        try:
            if not self.s3_client or not self.bucket_name:
                logger.error("S3 service not properly configured")
                return []
            
            safe_email = user_email.replace('@', '_').replace('.', '_')
            prefix = f"uploads/{safe_email}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Get file metadata
                    metadata_response = self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'original_filename': metadata_response.get('Metadata', {}).get('original_filename', 'Unknown'),
                        'upload_timestamp': metadata_response.get('Metadata', {}).get('upload_timestamp', 'Unknown')
                    })
            
            logger.info(f"Listed {len(files)} files for user: {user_email}")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files for user {user_email}: {str(e)}")
            return []
    
    def get_file_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for file access
        
        Args:
            s3_key: S3 key of the file
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL or None if failed
        """
        try:
            if not self.s3_client or not self.bucket_name:
                logger.error("S3 service not properly configured")
                return None
            
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for: {s3_key}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}")
            return None
    
    def is_configured(self) -> bool:
        """Check if S3 service is properly configured"""
        return self.s3_client is not None and self.bucket_name is not None



