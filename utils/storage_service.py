# storage_service.py
import requests
from requests_aws4auth import AWS4Auth
import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Storj credentials
STORJ_CONFIG = {
    'access_key': 'jxv52ooceheejwc2njollbeo7gea',
    'secret_key': 'jyfrizf7g7vekzje7f4wwxa5frdplzbqkt3dao5nabvsbvaxqm46w',
    'endpoint_url': 'https://gateway.storjshare.io',
    'bucket_name': 'safeclause-ai'
}


def upload_to_storj(file_path: str) -> Tuple[bool, str]:
    """
    Upload a file to Storj S3 bucket.
    
    Args:
        file_path: Local file path to upload
        
    Returns:
        Tuple of (success: bool, url_or_error: str)
    """
    try:
        # Extract filename
        object_key = os.path.basename(file_path)
        
        # Read file
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Create signed request
        auth = AWS4Auth(
            STORJ_CONFIG['access_key'],
            STORJ_CONFIG['secret_key'],
            'us-east-1',
            's3'
        )
        
        url = f"{STORJ_CONFIG['endpoint_url']}/{STORJ_CONFIG['bucket_name']}/{object_key}"
        
        headers = {
            'Content-Length': str(len(file_data)),
            'Content-Type': 'application/octet-stream'
        }
        
        # Upload
        response = requests.put(url, data=file_data, auth=auth, headers=headers)
        
        if response.status_code == 200:
            logger.info(f"Uploaded successfully: {object_key}")
            return True, url
        else:
            error_msg = f"Upload failed with status {response.status_code}: {response.text}"
            logger.error(error_msg)
            return False, error_msg
    
    except Exception as e:
        error_msg = f"Error uploading file: {str(e)}"
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