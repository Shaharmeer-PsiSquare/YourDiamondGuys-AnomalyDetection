import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import cv2
import base64
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import numpy as np
import logging
from time import time

# Configure logger for utils module
logger = logging.getLogger(__name__)

# Initialize S3 client with optimized configuration
logger.debug("Initializing S3 client with optimized configuration")
try:
    # S3 config with connection pooling and retries
    s3_config = Config(
        max_pool_connections=50,  # Connection pool size
        retries={'max_attempts': 3, 'mode': 'adaptive'}  # Retry strategy
    )
    s3 = boto3.client('s3', config=s3_config)
    logger.debug("S3 client initialized successfully with connection pooling")
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {str(e)}", exc_info=True)
    raise

# HTTP session for API calls (reused for better performance)
_api_session = None

def get_api_session():
    """Get or create HTTP session for API calls (singleton)"""
    global _api_session
    if _api_session is None:
        _api_session = requests.Session()
        # Retry strategy for API calls
        retry_strategy = Retry(
            total=2,  # Retry once on failure
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        _api_session.mount("http://", adapter)
        _api_session.mount("https://", adapter)
        logger.debug("Created HTTP session for API calls with retry strategy")
    return _api_session

def encodeimage(image):
    encode_start_time = time()
    logger.debug("Encoding image to base64 format")
    
    if image is None:
        logger.error("Input image is None - cannot encode")
        return None
    
    try:
        logger.debug(f"Input image shape: {image.shape if hasattr(image, 'shape') else 'N/A'}, dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
        logger.debug("Calling cv2.imencode('.jpg', image)")
        _, buffer = cv2.imencode('.jpg', image)
        
        if buffer is None:
            logger.error("Failed to encode image - cv2.imencode returned None")
            return None
        
        buffer_size = len(buffer)
        logger.debug(f"Image encoded to buffer. Buffer size: {buffer_size} bytes ({buffer_size / 1024:.2f} KB)")
        
        logger.debug("Encoding buffer to base64")
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        encoded_size = len(encoded_image)
        encode_duration = time() - encode_start_time
        logger.debug(f"Image base64 encoded. Encoded size: {encoded_size} bytes ({encoded_size / 1024:.2f} KB), duration: {encode_duration:.3f}s")
        
        return encoded_image
    except cv2.error as e:
        logger.error(f"OpenCV error encoding image: {str(e)}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}", exc_info=True)
        return None

def get_predict_from_model(image_array):
    """
    Get prediction from AI model API.
    Matches legacy implementation - returns raw API response.
    """
    api_start_time = time()
    url = "http://3.225.210.139:8009/"
    logger.info(f"Sending image to AI model API: {url}")
    
    if image_array is None:
        logger.error("Input image_array is None - cannot process")
        return None
    
    logger.debug(f"Input image array shape: {image_array.shape if image_array is not None else 'None'}")
    logger.debug(f"Input image array dtype: {image_array.dtype if hasattr(image_array, 'dtype') else 'N/A'}")
    
    # Encode image array back to base64 for API (matching legacy implementation)
    logger.debug("Encoding image for API request")
    encode_start = time()
    image_data = encodeimage(image_array)
    encode_duration = time() - encode_start
    
    if image_data is None:
        logger.error("Failed to encode image. Cannot proceed with API request")
        return None
    
    logger.debug(f"Image encoding completed in {encode_duration:.3f}s")
    
    # Create request payload matching legacy format
    files = {'encode': image_data}
    payload_size = len(image_data)
    logger.debug(f"Prepared API request payload. Encoded image size: {payload_size} bytes ({payload_size / 1024:.2f} KB)")
    
    try:
        logger.debug("Sending POST request to AI model API using session")
        request_start = time()
        # Use session for connection reuse
        session = get_api_session()
        response = session.post(url, data=files, timeout=30)
        request_duration = time() - request_start
        logger.debug(f"API response received. Status code: {response.status_code}, duration: {request_duration:.2f}s")
        
        # Parse JSON response (matching legacy - no status code check before parsing)
        try:
            logger.debug("Parsing JSON response")
            parse_start = time()
            result = response.json()
            parse_duration = time() - parse_start
            logger.debug(f"Successfully parsed JSON response in {parse_duration:.3f}s")
            
            # Log response structure for debugging
            if isinstance(result, dict):
                logger.debug(f"Response keys: {list(result.keys())}")
                # Log full response if it's unexpected
                if 'encode_img' not in result:
                    logger.warning(f"Response missing expected 'encode_img' key. Full response: {result}")
                    logger.debug(f"Response structure: {[(k, type(v).__name__) for k, v in result.items()]}")
            else:
                logger.warning(f"Response is not a dictionary. Type: {type(result)}, Value: {result}")
            
            total_duration = time() - api_start_time
            logger.info(f"AI model API call completed in {total_duration:.2f}s")
            
            # Return result directly (matching legacy behavior)
            return result
            
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response content type: {response.headers.get('Content-Type', 'Unknown')}")
            logger.debug(f"Response content (first 1000 chars): {response.text[:1000]}")
            return None
            
    except requests.exceptions.Timeout:
        total_duration = time() - api_start_time
        logger.error(f"Timeout while calling AI model API (30s timeout exceeded). Total time: {total_duration:.2f}s")
        return None
    except requests.exceptions.ConnectionError as e:
        total_duration = time() - api_start_time
        logger.error(f"Connection error while calling AI model API: {str(e)}. Total time: {total_duration:.2f}s")
        return None
    except requests.exceptions.RequestException as e:
        total_duration = time() - api_start_time
        logger.error(f"Request exception while calling AI model API: {str(e)}. Total time: {total_duration:.2f}s")
        return None
    except Exception as e:
        total_duration = time() - api_start_time
        logger.error(f"Unexpected error while calling AI model API: {str(e)}. Total time: {total_duration:.2f}s", exc_info=True)
        return None

def create_s3_folder(bucket_name, folder_name):
    folder_start_time = time()
    logger.debug(f"Creating S3 folder: bucket='{bucket_name}', folder='{folder_name}'")
    
    if not bucket_name:
        logger.error("Bucket name is empty or None")
        raise ValueError("Bucket name cannot be empty")
    
    if not folder_name:
        logger.error("Folder name is empty or None")
        raise ValueError("Folder name cannot be empty")
    
    original_folder_name = folder_name
    if not folder_name.endswith('/'):
        folder_name += '/'
        logger.debug(f"Appended '/' to folder name: '{original_folder_name}' -> '{folder_name}'")
    
    try:
        # Check if folder already exists to avoid unnecessary API calls
        try:
            s3.head_object(Bucket=bucket_name, Key=folder_name)
            logger.debug(f"S3 folder already exists: s3://{bucket_name}/{folder_name}")
            return  # Folder exists, no need to create
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                # Folder doesn't exist, create it
                logger.debug(f"Folder doesn't exist, creating: s3://{bucket_name}/{folder_name}")
            else:
                # Other error, log and continue to try creating
                logger.debug(f"S3 head_object returned: {error_code}, will attempt to create")
        
        logger.debug(f"Calling s3.put_object(Bucket='{bucket_name}', Key='{folder_name}')")
        s3.put_object(Bucket=bucket_name, Key=folder_name)
        folder_duration = time() - folder_start_time
        logger.debug(f"Successfully created S3 folder: s3://{bucket_name}/{folder_name} in {folder_duration:.3f}s")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', 'No message')
        # If folder already exists (409 Conflict), that's okay
        if error_code == 'BucketAlreadyOwnedByYou' or 'already exists' in error_message.lower():
            logger.debug(f"S3 folder already exists (non-critical): {folder_name}")
            return
        logger.error(f"S3 ClientError creating folder '{folder_name}' in bucket '{bucket_name}': {error_code} - {error_message}")
        logger.debug(f"Full error response: {e.response}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating S3 folder '{folder_name}' in bucket '{bucket_name}': {str(e)}", exc_info=True)
        raise

def upload_to_s3(file_stream, bucket, key):
    upload_start_time = time()
    logger.debug(f"Uploading file to S3: bucket='{bucket}', key='{key}'")
    
    if not bucket:
        logger.error("Bucket name is empty or None")
        raise ValueError("Bucket name cannot be empty")
    
    if not key:
        logger.error("S3 key is empty or None")
        raise ValueError("S3 key cannot be empty")
    
    try:
        # Get file size if possible
        logger.debug("Determining file size")
        current_pos = file_stream.tell()
        file_stream.seek(0, 2)  # Seek to end
        file_size = file_stream.tell()
        file_stream.seek(current_pos)  # Reset to original position
        
        logger.debug(f"File size: {file_size} bytes ({file_size / 1024:.2f} KB)")
        
        if file_size == 0:
            logger.warning(f"File size is 0 bytes - uploading empty file to s3://{bucket}/{key}")
        
        logger.debug(f"Calling s3.upload_fileobj() for s3://{bucket}/{key}")
        s3.upload_fileobj(file_stream, bucket, key)
        upload_duration = time() - upload_start_time
        upload_speed = (file_size / 1024) / upload_duration if upload_duration > 0 else 0
        logger.info(f"Successfully uploaded file to S3: s3://{bucket}/{key} ({file_size / 1024:.2f} KB) in {upload_duration:.2f}s ({upload_speed:.2f} KB/s)")
        
    except s3.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', 'No message')
        logger.error(f"S3 ClientError uploading file '{key}' to bucket '{bucket}': {error_code} - {error_message}")
        logger.debug(f"Full error response: {e.response}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading file '{key}' to bucket '{bucket}': {str(e)}", exc_info=True)
        raise