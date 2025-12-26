import re
import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from io import BytesIO
from time import sleep, time
import logging
from datetime import datetime
import os
from logging.handlers import RotatingFileHandler

# Configure detailed logging with both console and file output
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# File handler with rotation (10MB max, keep 5 backups)
file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'anomaly_detection.log'),
    maxBytes=10*1024*1024,
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

# Add handlers
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info("Logging system initialized - File: logs/anomaly_detection.log, Console: INFO level")

# Import from our new local modules
from database import (
    fetch_pending_diamonds, 
    insert_ai_data, 
    update_main_record, 
    update_ai_flags, 
    log_ai_timestamp,
    batch_update_ai_processing
)
from utils import (
    create_s3_folder, 
    get_predict_from_model, 
    upload_to_s3
)

# Create a reusable HTTP session with retry strategy
def create_http_session():
    """Create an HTTP session with retry strategy for better reliability"""
    session = requests.Session()
    
    # Retry strategy for transient failures
    retry_strategy = Retry(
        total=3,  # Total retries
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set default timeout
    session.timeout = 30
    
    return session

# Global HTTP session (reused across requests)
_http_session = None

def get_http_session():
    """Get or create HTTP session (singleton)"""
    global _http_session
    if _http_session is None:
        _http_session = create_http_session()
        logger.info("Created HTTP session with retry strategy")
    return _http_session

def process_batch():
    batch_start_time = time()
    logger.info("=" * 80)
    logger.info("Starting batch processing")
    logger.info("=" * 80)
    
    logger.info("Fetching pending records from database (limit: 1000)...")
    fetch_start_time = time()
    results = fetch_pending_diamonds(limit=1000)
    fetch_duration = time() - fetch_start_time
    logger.debug(f"Database fetch completed in {fetch_duration:.2f} seconds")
    
    if not results:
        logger.warning("No pending records found in database")
        return {'has_data': False, 'processed': 0, 'errors': 0, 'total': 0}

    logger.info(f"Successfully fetched {len(results)} pending records to process")

    processed_count = 0
    error_count = 0
    total_records = len(results)
    
    logger.info("=" * 80)
    logger.info(f"BATCH STATISTICS: Total Records: {total_records} | Successful: {processed_count} | Failed: {error_count} | Remaining: {total_records}")
    logger.info("=" * 80)
    
    # Progress update interval (every 10 records or 5%)
    progress_interval = max(10, total_records // 20)
    
    for idx, row in enumerate(results, 1):
        record_start_time = time()
        remaining = total_records - (processed_count + error_count)
        progress_pct = ((processed_count + error_count) / total_records * 100) if total_records > 0 else 0
        
        try:
            # Log progress every N records or at milestones
            if idx == 1 or idx % progress_interval == 0 or idx == total_records:
                logger.info(f"[{idx}/{total_records}] Processing diamond record ({progress_pct:.1f}% complete) | ✓ Successful: {processed_count} | ✗ Failed: {error_count} | ⏳ Remaining: {remaining}")
            else:
                logger.debug(f"[{idx}/{total_records}] Processing diamond record | ✓ Successful: {processed_count} | ✗ Failed: {error_count} | ⏳ Remaining: {remaining}")
            logger.debug(f"Record processing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            diamond_id, image_link, affiliate_id, web_page_link = row
            original_db_id = diamond_id  # Keep reference to DB key
            logger.debug(f"Extracted record fields - diamond_id: {diamond_id}, affiliate_id: {affiliate_id}")
            
            logger.debug(f"Raw record data - diamond_id: {diamond_id}, affiliate_id: {affiliate_id}, "
                        f"image_link: {image_link[:100]}..., web_page_link: {web_page_link[:100]}...")
            
            # --- 1. ID Normalization Logic (Regex) ---
            logger.debug("Starting ID normalization logic")
            if affiliate_id != 4:
                # Logic for Non-WhiteFlash
                logger.debug(f"Non-WhiteFlash affiliate detected (ID: {affiliate_id}), extracting diamond ID from image_link")
                match = re.search(r'diamond/(\d+)', image_link)
                if match:
                    diamond_id_new = match.group(1)
                    logger.debug(f"Extracted diamond ID from image_link: {diamond_id_new}")
                    if diamond_id_new != diamond_id.replace('b', ''):
                        logger.info(f"Updating diamond_id from '{diamond_id}' to '{diamond_id_new}' based on image_link")
                        diamond_id = diamond_id_new
                    else:
                        logger.debug(f"Diamond ID unchanged: {diamond_id}")
                else:
                    logger.warning(f"Could not extract diamond ID from image_link: {image_link[:100]}...")
            else:
                # Logic for WhiteFlash
                logger.debug("WhiteFlash affiliate detected (ID: 4), extracting diamond ID from web_page_link")
                pattern = r'sku-(\d+)\?'
                match = re.search(pattern, web_page_link)
                if match:
                    diamond_id = match.group(1)
                    logger.info(f"Extracted and updated diamond_id to '{diamond_id}' from WhiteFlash web_page_link")
                else:
                    logger.warning(f"Could not extract diamond ID from WhiteFlash web_page_link: {web_page_link[:100]}...")

            # Map affiliate ID to name
            affiliate_map = {
                1: 'james-allen',
                2: 'blue-nile',
                3: 'amazon',
                4: 'white-flash',
                5: 'rapnet'
            }
            affiliate_name = affiliate_map.get(affiliate_id, 'rapnet')
            
            if affiliate_id not in affiliate_map:
                logger.warning(f"Unknown affiliate_id: {affiliate_id}, defaulting to 'rapnet'")

            logger.info(f"Processing diamond: affiliate='{affiliate_name}', diamond_id='{diamond_id}', original_db_id='{original_db_id}'")

            # --- 2. S3 Folder Setup (optimized - create all at once) ---
            logger.info("Setting up S3 folder structure")
            base_folder = f'{affiliate_name}/{diamond_id}'
            logger.debug(f"Creating S3 folders for: {base_folder}")
            
            # Create all folders (S3 will handle existence checks)
            try:
                create_s3_folder('affiliated-partner', base_folder)
                create_s3_folder('affiliated-partner', f'{base_folder}/original')
                create_s3_folder('affiliated-partner', f'{base_folder}/result')
                logger.info("S3 folder structure created successfully")
            except Exception as e:
                # S3 folder creation errors are often non-critical (folder may already exist)
                logger.warning(f"S3 folder creation warning (may already exist): {str(e)}")
                # Continue processing as folders might already exist

            # --- 3. Image Download & AI Inference ---
            logger.info(f"Downloading image from: {image_link}")
            download_start_time = time()
            try:
                logger.debug(f"Making HTTP GET request using session")
                session = get_http_session()
                response = session.get(image_link, timeout=10)
                download_duration = time() - download_start_time
                logger.debug(f"Image download response status: {response.status_code}, duration: {download_duration:.2f}s")
                
                if response.status_code != 200:
                    error_count += 1
                    remaining = total_records - (processed_count + error_count)
                    logger.error(f"✗ [{idx}/{total_records}] FAILED - Image download error. Status: {response.status_code}, URL: {image_link[:100]}...")
                    logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                    continue

                image_data = response.content
                image_size = len(image_data)
                logger.info(f"Successfully downloaded image. Size: {image_size} bytes ({image_size / 1024:.2f} KB)")
                
                logger.debug("Decoding image data using OpenCV")
                image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
                
                if image_array is None:
                    error_count += 1
                    remaining = total_records - (processed_count + error_count)
                    logger.error(f"✗ [{idx}/{total_records}] FAILED - Image decode error. Image may be corrupted or unsupported format")
                    logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                    continue
                
                image_shape = image_array.shape
                logger.debug(f"Image decoded successfully. Shape: {image_shape}")
                
            except requests.exceptions.Timeout:
                error_count += 1
                remaining = total_records - (processed_count + error_count)
                logger.error(f"✗ [{idx}/{total_records}] FAILED - Timeout downloading image from: {image_link[:100]}...")
                logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                continue
            except requests.exceptions.RequestException as e:
                error_count += 1
                remaining = total_records - (processed_count + error_count)
                logger.error(f"✗ [{idx}/{total_records}] FAILED - Request exception downloading image: {str(e)}")
                logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                continue
            except Exception as e:
                error_count += 1
                remaining = total_records - (processed_count + error_count)
                logger.error(f"✗ [{idx}/{total_records}] FAILED - Unexpected error during image download/decode: {str(e)}")
                logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                continue
            
            # Call AI Model
            logger.info("Sending image to AI model for inference")
            ai_start_time = time()
            result_img = get_predict_from_model(image_array)
            ai_duration = time() - ai_start_time
            logger.debug(f"AI model inference completed in {ai_duration:.2f} seconds")
            
            if not result_img:
                error_count += 1
                remaining = total_records - (processed_count + error_count)
                logger.error(f"✗ [{idx}/{total_records}] FAILED - AI Model returned no data")
                logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                continue

            logger.info("AI model inference completed successfully")
            
            # Extract Data
            logger.debug("Extracting AI model results")
            try:
                # Log full response structure for debugging
                if isinstance(result_img, dict):
                    logger.debug(f"AI model response structure: {list(result_img.keys())}")
                    logger.debug(f"AI model response sample values: {[(k, str(v)[:50] if isinstance(v, str) else type(v).__name__) for k, v in list(result_img.items())[:5]]}")
                
                # Extract expected fields (matching legacy expected structure)
                # encode_img is a base64 string, decode it to get image bytes
                import base64
                result_image_data = base64.b64decode(result_img["encode_img"])
                flag_center = result_img["center_flag"]
                points = result_img["region_points"]
                acc = result_img["acuuracy"]
                anomaly_head = result_img["anomaly_head"]
                
                logger.info(f"AI Results extracted - center_flag: {flag_center}, accuracy: {acc}, "
                          f"anomaly_head: {anomaly_head}, points_count: {len(points) if points else 0}")
                logger.debug(f"Result image data size: {len(result_image_data)} bytes")
            except KeyError as e:
                error_count += 1
                remaining = total_records - (processed_count + error_count)
                logger.error(f"✗ [{idx}/{total_records}] FAILED - Missing key in AI model response: {str(e)}")
                logger.error(f"  Available keys: {list(result_img.keys()) if isinstance(result_img, dict) else 'N/A'}")
                logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                logger.debug(f"Full AI model response: {result_img}")
                continue
            except Exception as e:
                error_count += 1
                remaining = total_records - (processed_count + error_count)
                logger.error(f"✗ [{idx}/{total_records}] FAILED - Error extracting AI model results: {str(e)}")
                logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                continue

            # --- 4. Upload to S3 ---
            logger.info("Uploading images to S3")
            original_image_key = f'{affiliate_name}/{diamond_id}/original/{diamond_id}.png'
            result_image_key = f'{affiliate_name}/{diamond_id}/result/anomaly_detection_model_res_{diamond_id}.png'
            
            logger.debug(f"Uploading original image to S3: {original_image_key}")
            try:
                upload_to_s3(BytesIO(image_data), 'affiliated-partner', original_image_key)
                logger.info(f"Successfully uploaded original image to S3: {original_image_key}")
            except Exception as e:
                error_count += 1
                remaining = total_records - (processed_count + error_count)
                logger.error(f"✗ [{idx}/{total_records}] FAILED - Failed to upload original image to S3: {str(e)}")
                logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                continue
            
            logger.debug(f"Uploading result image to S3: {result_image_key}")
            try:
                upload_to_s3(BytesIO(result_image_data), 'affiliated-partner', result_image_key)
                logger.info(f"Successfully uploaded result image to S3: {result_image_key}")
            except Exception as e:
                error_count += 1
                remaining = total_records - (processed_count + error_count)
                logger.error(f"✗ [{idx}/{total_records}] FAILED - Failed to upload result image to S3: {str(e)}")
                logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                continue
            
            result_s3_link = f's3://affiliated-partner/{result_image_key}'
            logger.debug(f"Result S3 link: {result_s3_link}")

            # --- 5. Database Updates (Batched for efficiency) ---
            logger.info("Updating database with AI processing results (batched operation)")
            try:
                # Use batch update to combine all database operations into a single transaction
                ret_id = batch_update_ai_processing(
                    original_db_id, diamond_id, result_s3_link, flag_center, points, acc, anomaly_head
                )
                logger.info(f"Successfully completed batch database update. AI processing ID: {ret_id}")
            except Exception as e:
                error_count += 1
                remaining = total_records - (processed_count + error_count)
                logger.error(f"✗ [{idx}/{total_records}] FAILED - Database batch update error: {str(e)}")
                logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining")
                continue
            
            processed_count += 1
            record_duration = time() - record_start_time
            remaining = total_records - (processed_count + error_count)
            success_rate = (processed_count / (processed_count + error_count) * 100) if (processed_count + error_count) > 0 else 0
            logger.info(f"✓ [{idx}/{total_records}] SUCCESS - diamond_id: {diamond_id} (affiliate: {affiliate_name}) in {record_duration:.2f}s")
            logger.info(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining | Success Rate: {success_rate:.1f}%")

        except Exception as e:
            error_count += 1
            remaining = total_records - (processed_count + error_count)
            success_rate = (processed_count / (processed_count + error_count) * 100) if (processed_count + error_count) > 0 else 0
            logger.error(f"✗ [{idx}/{total_records}] FAILED - Unexpected error processing diamond: {str(e)}")
            logger.error(f"  Progress: {processed_count} successful, {error_count} failed, {remaining} remaining | Success Rate: {success_rate:.1f}%")
            logger.debug(f"Full error traceback:", exc_info=True)
            continue
    
    batch_duration = time() - batch_start_time
    total_processed_in_batch = processed_count + error_count
    success_rate = (processed_count / total_processed_in_batch * 100) if total_processed_in_batch > 0 else 0
    
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Records in Batch: {total_records}")
    logger.info(f"✓ Successfully Processed: {processed_count} ({success_rate:.1f}%)")
    logger.info(f"✗ Failed: {error_count} ({100-success_rate:.1f}%)")
    logger.info(f"⏸ Not Processed: {total_records - total_processed_in_batch}")
    logger.info(f"Batch Duration: {batch_duration:.2f} seconds ({batch_duration/60:.2f} minutes)")
    
    if processed_count > 0:
        avg_time_per_success = batch_duration / processed_count
        logger.info(f"Average time per successful record: {avg_time_per_success:.2f} seconds")
        throughput = processed_count / (batch_duration / 60) if batch_duration > 0 else 0
        logger.info(f"Throughput: {throughput:.2f} successful records/minute")
    
    if error_count > 0:
        avg_time_per_error = batch_duration / error_count
        logger.info(f"Average time per failed record: {avg_time_per_error:.2f} seconds")
    
    logger.info("=" * 80)
    
    # Return statistics
    return {
        'has_data': True,
        'processed': processed_count,
        'errors': error_count,
        'total': total_records
    }

if __name__ == "__main__":
    import signal
    import sys
    
    # Note: output.log file is managed by start.sh script
    # The shell script deletes it before starting and redirects output via nohup
    # We don't delete/recreate it here to avoid breaking the file descriptor connection
    
    app_start_time = time()
    logger.info("=" * 80)
    logger.info("Anomaly Detection System - Starting Application")
    logger.info(f"Application started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {__import__('sys').version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("=" * 80)
    logger.info("Running in continuous loop mode")
    
    # Graceful shutdown handler
    def signal_handler(sig, frame):
        logger.info("\n" + "=" * 80)
        logger.info("Shutdown signal received. Performing graceful shutdown...")
        logger.info("=" * 80)
        from database import close_connection_pool
        close_connection_pool()
        logger.info("Graceful shutdown completed")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    loop_count = 0
    total_processed_all_loops = 0
    total_errors_all_loops = 0
    
    # Continuous Loop Mode
    try:
        while True:
            loop_count += 1
            loop_start_time = time()
            logger.info(f"\n{'='*80}")
            logger.info(f"--- Loop iteration #{loop_count} ---")
            logger.info(f"Loop started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            batch_result = process_batch()
            loop_duration = time() - loop_start_time
            
            # Update global statistics
            if batch_result['has_data']:
                total_processed_all_loops += batch_result['processed']
                total_errors_all_loops += batch_result['errors']
            
            if not batch_result['has_data']:
                logger.info(f"No pending records found. Loop duration: {loop_duration:.2f}s")
                logger.info("Sleeping for 60 seconds before next check...")
                sleep(60)
            else:
                logger.info(f"Batch processing completed. Loop duration: {loop_duration:.2f}s")
                logger.info("Starting next batch immediately...")
            
            uptime = time() - app_start_time
            total_processed = total_processed_all_loops + total_errors_all_loops
            total_success_rate = (total_processed_all_loops / total_processed * 100) if total_processed > 0 else 0
            logger.info("=" * 80)
            logger.info(f"OVERALL STATISTICS (All Loops Combined)")
            logger.info(f"  Uptime: {uptime/3600:.2f} hours ({uptime:.0f} seconds)")
            logger.info(f"  Total Successful: {total_processed_all_loops}")
            logger.info(f"  Total Failed: {total_errors_all_loops}")
            logger.info(f"  Total Processed: {total_processed}")
            logger.info(f"  Overall Success Rate: {total_success_rate:.1f}%")
            logger.info("=" * 80)
            logger.debug(f"Application uptime: {uptime:.0f} seconds")
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {str(e)}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        from database import close_connection_pool
        close_connection_pool()
        logger.info("Application shutdown complete")