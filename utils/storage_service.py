import requests
from requests_aws4auth import AWS4Auth
import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

access_key = os.getenv("STORJ_ACCESS_KEY")
secret_key = os.getenv("STORJ_SECRET_KEY")

# Storj credentials
STORJ_CONFIG = {
    'access_key': access_key,
    'secret_key': secret_key,
    'endpoint_url': 'https://gateway.storjshare.io',
    'bucket_name': 'safeclause-ai'
}


def upload_to_storj(file_path: str, original_filename: str = None) -> Tuple[bool, str]:
    """
    Upload a file to Storj S3 bucket.
    
    Args:
        file_path: Local file path to upload
        original_filename: Original filename to use as object key (optional)
        
    Returns:
        Tuple of (success: bool, url_or_error: str)
    """
    try:
        # Validate credentials first
        if not STORJ_CONFIG['access_key'] or not STORJ_CONFIG['secret_key']:
            error_msg = "Missing Storj credentials. Please check STORJ_ACCESS_KEY and STORJ_SECRET_KEY in environment."
            logger.error(error_msg)
            return False, error_msg
        
        # Use original filename if provided, otherwise extract from path
        object_key = original_filename if original_filename else os.path.basename(file_path)
        
        logger.info(f"Starting upload of {object_key} to Storj...")
        
        # Read file
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            logger.info(f"File read successfully, size: {len(file_data)} bytes")
        except Exception as e:
            error_msg = f"Failed to read file {file_path}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        
        # Create signed request
        try:
            auth = AWS4Auth(
                STORJ_CONFIG['access_key'],
                STORJ_CONFIG['secret_key'],
                'us-east-1',
                's3'
            )
            logger.info("AWS4Auth created successfully")
        except Exception as e:
            error_msg = f"Failed to create AWS4Auth: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        
        url = f"{STORJ_CONFIG['endpoint_url']}/{STORJ_CONFIG['bucket_name']}/{object_key}"
        logger.info(f"Upload URL: {url}")
        
        headers = {
            'Content-Length': str(len(file_data)),
            'Content-Type': 'application/octet-stream'
        }
        
        # Upload
        logger.info("Sending PUT request to Storj...")
        response = requests.put(url, data=file_data, auth=auth, headers=headers, timeout=60)
        
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            logger.info(f"Uploaded successfully: {object_key}")
            return True, url
        else:
            error_msg = f"Upload failed with status {response.status_code}: {response.text[:500]}"
            logger.error(error_msg)
            return False, error_msg
    
    except requests.exceptions.Timeout:
        error_msg = "Upload timed out. Please check your internet connection and try again."
        logger.error(error_msg)
        return False, error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Connection error. Please check your internet connection and Storj endpoint configuration."
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error uploading file: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def upload_multiple_files(file_paths: List[str]) -> List[Tuple[str, str, bool]]:
    """
    Upload multiple files to Storj.
    
    Args:
        file_paths: List of local file paths
        
    Returns:
        List of tuples (filename, url_or_error, success)
    """
    results = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        success, result = upload_to_storj(file_path)
        results.append((filename, result, success))
    return results
