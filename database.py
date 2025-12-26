import psycopg2
from psycopg2 import pool
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz
import logging
from time import time
import threading

# Configure logger for database module
logger = logging.getLogger(__name__)

logger.debug("Loading environment variables from .env file")
load_dotenv()
logger.debug("Environment variables loaded")

# Connection pool configuration
_connection_pool = None
_pool_lock = threading.Lock()

def get_connection_pool():
    """Get or create a connection pool (thread-safe singleton)"""
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:  # Double-check locking
                try:
                    db_config = {
                        'database': os.environ.get('database_2'),
                        'user': os.environ.get('user2'),
                        'host': os.environ.get('host2'),
                        'port': os.environ.get('port'),
                        'password': os.environ.get('password2')
                    }
                    
                    # Validate required config
                    missing_config = [k for k, v in db_config.items() if k != 'password' and not v]
                    if missing_config:
                        logger.error(f"Missing database configuration: {missing_config}")
                        return None
                    
                    if not db_config['password']:
                        logger.error("Database password not found in environment variables")
                        return None
                    
                    logger.info("Creating database connection pool (min: 2, max: 10 connections)")
                    _connection_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=2,
                        maxconn=10,
                        database=db_config['database'],
                        user=db_config['user'],
                        password=db_config['password'],
                        host=db_config['host'],
                        port=db_config['port'],
                        connect_timeout=5  # 5 second connection timeout
                    )
                    logger.info("Database connection pool created successfully")
                except Exception as e:
                    logger.error(f"Failed to create connection pool: {str(e)}", exc_info=True)
                    return None
    
    return _connection_pool

def get_db_connection():
    """Get a connection from the pool with health check"""
    conn_start_time = time()
    logger.debug("Getting database connection from pool")
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            pool = get_connection_pool()
            if not pool:
                logger.error("Connection pool not available")
                return None
            
            connection = pool.getconn()
            if connection:
                # Check if connection is still alive
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                    conn_duration = time() - conn_start_time
                    logger.debug(f"Got healthy connection from pool in {conn_duration:.3f}s")
                    return connection
                except Exception as e:
                    # Connection is dead, close it and try again
                    logger.warning(f"Connection from pool is dead, closing and retrying: {str(e)}")
                    try:
                        connection.close()
                    except:
                        pass
                    if attempt < max_retries - 1:
                        continue
                    return None
            else:
                logger.error("Failed to get connection from pool")
                return None
            
        except Exception as e:
            conn_duration = time() - conn_start_time
            if attempt < max_retries - 1:
                logger.warning(f"Error getting connection from pool (attempt {attempt + 1}/{max_retries}): {str(e)}")
                continue
            logger.error(f"Error getting connection from pool after {conn_duration:.3f}s: {str(e)}", exc_info=True)
            return None
    
    return None

def return_db_connection(connection):
    """Return a connection to the pool"""
    if connection:
        try:
            # Check if connection is still valid before returning to pool
            try:
                if connection.closed == 0:  # Connection is open
                    pool = get_connection_pool()
                    if pool:
                        pool.putconn(connection)
                        logger.debug("Connection returned to pool")
                    else:
                        connection.close()
                else:
                    logger.debug("Connection was already closed, not returning to pool")
            except AttributeError:
                # Connection object doesn't have 'closed' attribute, try to return anyway
                pool = get_connection_pool()
                if pool:
                    pool.putconn(connection)
                    logger.debug("Connection returned to pool")
        except Exception as e:
            logger.error(f"Error returning connection to pool: {str(e)}")
            try:
                connection.close()
            except:
                pass

def close_connection_pool():
    """Close all connections in the pool (cleanup)"""
    global _connection_pool
    if _connection_pool:
        try:
            _connection_pool.closeall()
            logger.info("Connection pool closed")
            _connection_pool = None
        except Exception as e:
            logger.error(f"Error closing connection pool: {str(e)}")

def fetch_pending_diamonds(limit=1000):
    query_start_time = time()
    logger.info(f"Fetching pending diamonds from database (limit: {limit})")
    
    conn = get_db_connection()
    if not conn:
        logger.error("Cannot fetch pending diamonds - database connection failed")
        return []
    
    query = """
        SELECT diamond_id, image_link, affiliate_id, web_page_link 
        FROM "Affiliate_app_productinfo" 
        WHERE ai_processing_id IS NULL AND sell_status IS FALSE 
        LIMIT %s;
    """
    
    try:
        logger.debug(f"Executing query with limit: {limit}")
        logger.debug(f"Query: SELECT diamond_id, image_link, affiliate_id, web_page_link FROM \"Affiliate_app_productinfo\" WHERE ai_processing_id IS NULL AND sell_status IS FALSE LIMIT %s")
        
        execute_start = time()
        with conn.cursor() as cursor:
            cursor.execute(query, (limit,))
            execute_duration = time() - execute_start
            logger.debug(f"Query executed in {execute_duration:.3f}s")
            
            fetch_start = time()
            results = cursor.fetchall()
            fetch_duration = time() - fetch_start
            logger.debug(f"Results fetched in {fetch_duration:.3f}s")
            
            logger.debug(f"Query returned {len(results)} rows")
            if len(results) > 0:
                logger.debug(f"Sample first record: diamond_id={results[0][0] if len(results[0]) > 0 else 'N/A'}, affiliate_id={results[0][2] if len(results[0]) > 2 else 'N/A'}")
        
        return_db_connection(conn)
        total_duration = time() - query_start_time
        logger.info(f"Query executed successfully. Fetched {len(results)} pending diamond records in {total_duration:.3f}s")
        logger.debug("Database connection returned to pool")
        return results
        
    except psycopg2.Error as e:
        total_duration = time() - query_start_time
        logger.error(f"PostgreSQL error while fetching pending diamonds after {total_duration:.3f}s: {str(e)}")
        logger.debug(f"Error code: {e.pgcode if hasattr(e, 'pgcode') else 'N/A'}")
        if conn:
            return_db_connection(conn)
        return []
    except Exception as e:
        total_duration = time() - query_start_time
        logger.error(f"Unexpected error while fetching pending diamonds after {total_duration:.3f}s: {str(e)}", exc_info=True)
        if conn:
            return_db_connection(conn)
        return []

def insert_ai_data(response_link, diamond_id, center_flag, points, acc, anomaly_head):
    insert_start_time = time()
    logger.info(f"Inserting AI data for diamond_id: {diamond_id}")
    logger.debug(f"AI data - response_link: {response_link}, center_flag: {center_flag}, "
                f"accuracy: {acc}, anomaly_head: {anomaly_head}, points: {len(points) if points else 0}")
    
    conn = get_db_connection()
    if not conn:
        logger.error("Cannot insert AI data - database connection failed")
        raise Exception("Database connection failed")
    
    insert_query = """
        INSERT INTO public."Affiliate_app_aiprocessing" 
        (is_processed, ai_output_image, diamond_id, center_flag, polygon_points, ai_score, anomaly_head) 
        VALUES (true, %s, %s, %s, %s, %s, %s) 
        ON CONFLICT(diamond_id) DO UPDATE SET diamond_id = EXCLUDED.diamond_id 
        RETURNING id;
    """
    
    try:
        logger.debug("Executing INSERT query for AI processing data")
        logger.debug(f"Query parameters - diamond_id: {diamond_id}, center_flag: {center_flag}, accuracy: {acc}")
        
        execute_start = time()
        with conn.cursor() as cursor:
            cursor.execute(insert_query, (response_link, diamond_id, center_flag, points, acc, anomaly_head))
            execute_duration = time() - execute_start
            logger.debug(f"INSERT query executed in {execute_duration:.3f}s")
            
            ret_id = cursor.fetchone()[0]
            logger.debug(f"Insert query executed. Returned ID: {ret_id}")
        
        commit_start = time()
        conn.commit()
        commit_duration = time() - commit_start
        logger.debug(f"Transaction committed in {commit_duration:.3f}s")
        
        total_duration = time() - insert_start_time
        logger.info(f"Successfully inserted AI data. AI processing ID: {ret_id} in {total_duration:.3f}s")
        return_db_connection(conn)
        return ret_id
        
    except psycopg2.IntegrityError as e:
        total_duration = time() - insert_start_time
        logger.error(f"Database integrity error while inserting AI data after {total_duration:.3f}s: {str(e)}")
        logger.debug(f"Error code: {e.pgcode if hasattr(e, 'pgcode') else 'N/A'}")
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise
    except psycopg2.Error as e:
        total_duration = time() - insert_start_time
        logger.error(f"PostgreSQL error while inserting AI data after {total_duration:.3f}s: {str(e)}")
        logger.debug(f"Error code: {e.pgcode if hasattr(e, 'pgcode') else 'N/A'}")
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise
    except Exception as e:
        total_duration = time() - insert_start_time
        logger.error(f"Unexpected error while inserting AI data after {total_duration:.3f}s: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise

def update_main_record(diamond_id, ai_id):
    update_start_time = time()
    logger.info(f"Updating main product record: diamond_id={diamond_id}, ai_processing_id={ai_id}")
    
    conn = get_db_connection()
    if not conn:
        logger.error("Cannot update main record - database connection failed")
        raise Exception("Database connection failed")
    
    update_query = 'UPDATE public."Affiliate_app_productinfo" SET ai_processing_id=%s WHERE diamond_id = %s;'
    
    try:
        logger.debug(f"Executing UPDATE query for main product record")
        logger.debug(f"Query: UPDATE public.\"Affiliate_app_productinfo\" SET ai_processing_id=%s WHERE diamond_id = %s")
        logger.debug(f"Query parameters - ai_processing_id: {ai_id}, diamond_id: {diamond_id}")
        
        execute_start = time()
        with conn.cursor() as cursor:
            cursor.execute(update_query, (ai_id, diamond_id))
            execute_duration = time() - execute_start
            logger.debug(f"UPDATE query executed in {execute_duration:.3f}s")
            
            rows_affected = cursor.rowcount
            logger.debug(f"Update query executed. Rows affected: {rows_affected}")
            
            if rows_affected == 0:
                logger.warning(f"No rows updated for diamond_id={diamond_id}. Record may not exist.")
            elif rows_affected > 1:
                logger.warning(f"Multiple rows ({rows_affected}) updated for diamond_id={diamond_id}. Expected 1.")
            else:
                logger.debug(f"Exactly 1 row updated as expected")
        
        commit_start = time()
        conn.commit()
        commit_duration = time() - commit_start
        logger.debug(f"Transaction committed in {commit_duration:.3f}s")
        
        total_duration = time() - update_start_time
        logger.info(f"Successfully updated main product record in {total_duration:.3f}s")
        return_db_connection(conn)
        
    except psycopg2.Error as e:
        total_duration = time() - update_start_time
        logger.error(f"PostgreSQL error while updating main record after {total_duration:.3f}s: {str(e)}")
        logger.debug(f"Error code: {e.pgcode if hasattr(e, 'pgcode') else 'N/A'}")
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise
    except Exception as e:
        total_duration = time() - update_start_time
        logger.error(f"Unexpected error while updating main record after {total_duration:.3f}s: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise

def update_ai_flags(diamond_id, center_flag):
    update_start_time = time()
    logger.info(f"Updating AI flags: diamond_id={diamond_id}, center_flag={center_flag}")
    
    conn = get_db_connection()
    if not conn:
        logger.error("Cannot update AI flags - database connection failed")
        raise Exception("Database connection failed")
    
    update_query = 'UPDATE public."Affiliate_app_productcharacteristics" SET center_flag=%s WHERE diamond_id = %s;'
    
    try:
        logger.debug(f"Executing UPDATE query for AI flags")
        logger.debug(f"Query: UPDATE public.\"Affiliate_app_productcharacteristics\" SET center_flag=%s WHERE diamond_id = %s")
        logger.debug(f"Query parameters - center_flag: {center_flag}, diamond_id: {diamond_id}")
        
        execute_start = time()
        with conn.cursor() as cursor:
            cursor.execute(update_query, (center_flag, diamond_id))
            execute_duration = time() - execute_start
            logger.debug(f"UPDATE query executed in {execute_duration:.3f}s")
            
            rows_affected = cursor.rowcount
            logger.debug(f"Update query executed. Rows affected: {rows_affected}")
            
            if rows_affected == 0:
                logger.warning(f"No rows updated for diamond_id={diamond_id}. Record may not exist.")
            elif rows_affected > 1:
                logger.warning(f"Multiple rows ({rows_affected}) updated for diamond_id={diamond_id}. Expected 1.")
            else:
                logger.debug(f"Exactly 1 row updated as expected")
        
        commit_start = time()
        conn.commit()
        commit_duration = time() - commit_start
        logger.debug(f"Transaction committed in {commit_duration:.3f}s")
        
        total_duration = time() - update_start_time
        logger.info(f"Successfully updated AI flags in {total_duration:.3f}s")
        return_db_connection(conn)
        
    except psycopg2.Error as e:
        total_duration = time() - update_start_time
        logger.error(f"PostgreSQL error while updating AI flags after {total_duration:.3f}s: {str(e)}")
        logger.debug(f"Error code: {e.pgcode if hasattr(e, 'pgcode') else 'N/A'}")
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise
    except Exception as e:
        total_duration = time() - update_start_time
        logger.error(f"Unexpected error while updating AI flags after {total_duration:.3f}s: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise

def batch_update_ai_processing(original_db_id, diamond_id, result_s3_link, flag_center, points, acc, anomaly_head):
    """
    Batch update all AI processing related database operations in a single transaction.
    This is more efficient than making separate calls.
    """
    batch_start_time = time()
    logger.info(f"Batch updating AI processing data for diamond_id: {diamond_id}")
    
    conn = get_db_connection()
    if not conn:
        logger.error("Cannot batch update AI processing - database connection failed")
        raise Exception("Database connection failed")
    
    utc_time = datetime.now(pytz.utc)
    
    try:
        with conn.cursor() as cursor:
            # 1. Insert AI processing data
            insert_query = """
                INSERT INTO public."Affiliate_app_aiprocessing" 
                (is_processed, ai_output_image, diamond_id, center_flag, polygon_points, ai_score, anomaly_head) 
                VALUES (true, %s, %s, %s, %s, %s, %s) 
                ON CONFLICT(diamond_id) DO UPDATE SET diamond_id = EXCLUDED.diamond_id 
                RETURNING id;
            """
            cursor.execute(insert_query, (result_s3_link, diamond_id, flag_center, points, acc, anomaly_head))
            ret_id = cursor.fetchone()[0]
            logger.debug(f"Inserted AI data. AI processing ID: {ret_id}")
            
            # 2. Update main product record
            update_main_query = 'UPDATE public."Affiliate_app_productinfo" SET ai_processing_id=%s WHERE diamond_id = %s;'
            cursor.execute(update_main_query, (ret_id, original_db_id))
            rows_affected_main = cursor.rowcount
            logger.debug(f"Updated main product record. Rows affected: {rows_affected_main}")
            
            # 3. Update AI flags
            update_flags_query = 'UPDATE public."Affiliate_app_productcharacteristics" SET center_flag=%s WHERE diamond_id = %s;'
            cursor.execute(update_flags_query, (flag_center, diamond_id))
            rows_affected_flags = cursor.rowcount
            logger.debug(f"Updated AI flags. Rows affected: {rows_affected_flags}")
            
            # 4. Log timestamp (only if doesn't exist)
            check_timestamp_query = "SELECT 1 FROM authen_app_aidatetimerecord WHERE diamond_id = %s AND type = 'ai' LIMIT 1;"
            cursor.execute(check_timestamp_query, (str(diamond_id),))
            existing_timestamp = cursor.fetchone()
            
            if not existing_timestamp:
                insert_timestamp_query = "INSERT INTO authen_app_aidatetimerecord (diamond_id, type, created_at) VALUES (%s, 'ai', %s);"
                cursor.execute(insert_timestamp_query, (diamond_id, utc_time))
                logger.debug("Inserted AI timestamp record")
            else:
                logger.debug("AI timestamp record already exists, skipping")
        
        conn.commit()
        total_duration = time() - batch_start_time
        logger.info(f"Batch update completed successfully in {total_duration:.3f}s. AI processing ID: {ret_id}")
        return_db_connection(conn)
        return ret_id
        
    except psycopg2.Error as e:
        total_duration = time() - batch_start_time
        logger.error(f"PostgreSQL error during batch update after {total_duration:.3f}s: {str(e)}")
        logger.debug(f"Error code: {e.pgcode if hasattr(e, 'pgcode') else 'N/A'}")
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise
    except Exception as e:
        total_duration = time() - batch_start_time
        logger.error(f"Unexpected error during batch update after {total_duration:.3f}s: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise

def log_ai_timestamp(diamond_id):
    timestamp_start_time = time()
    logger.debug(f"Logging AI timestamp for diamond_id: {diamond_id}")
    
    conn = get_db_connection()
    if not conn:
        logger.error("Cannot log AI timestamp - database connection failed")
        raise Exception("Database connection failed")
    
    utc_time = datetime.now(pytz.utc)
    logger.debug(f"UTC timestamp: {utc_time} (ISO format: {utc_time.isoformat()})")
    
    check_query = "SELECT 1 FROM authen_app_aidatetimerecord WHERE diamond_id = %s AND type = 'ai' LIMIT 1;"
    insert_query = "INSERT INTO authen_app_aidatetimerecord (diamond_id, type, created_at) VALUES (%s, 'ai', %s);"
    
    try:
        with conn.cursor() as cursor:
            logger.debug("Checking if timestamp record already exists")
            logger.debug(f"Check query: SELECT 1 FROM authen_app_aidatetimerecord WHERE diamond_id = %s AND type = 'ai' LIMIT 1")
            
            check_start = time()
            cursor.execute(check_query, (str(diamond_id),))
            existing_record = cursor.fetchone()
            check_duration = time() - check_start
            logger.debug(f"Check query executed in {check_duration:.3f}s")
            
            if not existing_record:
                logger.debug("No existing timestamp record found. Inserting new record.")
                logger.debug(f"Insert query: INSERT INTO authen_app_aidatetimerecord (diamond_id, type, created_at) VALUES (%s, 'ai', %s)")
                
                insert_start = time()
                cursor.execute(insert_query, (diamond_id, utc_time))
                insert_duration = time() - insert_start
                logger.debug(f"INSERT query executed in {insert_duration:.3f}s")
                
                commit_start = time()
                conn.commit()
                commit_duration = time() - commit_start
                logger.debug(f"Transaction committed in {commit_duration:.3f}s")
                
                total_duration = time() - timestamp_start_time
                logger.info(f"Successfully logged AI timestamp for diamond_id: {diamond_id} in {total_duration:.3f}s")
            else:
                logger.debug(f"Timestamp record already exists for diamond_id: {diamond_id}. Skipping insert.")
        
        return_db_connection(conn)
        
    except psycopg2.Error as e:
        total_duration = time() - timestamp_start_time
        logger.error(f"PostgreSQL error while logging AI timestamp after {total_duration:.3f}s: {str(e)}")
        logger.debug(f"Error code: {e.pgcode if hasattr(e, 'pgcode') else 'N/A'}")
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise
    except Exception as e:
        total_duration = time() - timestamp_start_time
        logger.error(f"Unexpected error while logging AI timestamp after {total_duration:.3f}s: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
            logger.debug("Transaction rolled back")
            return_db_connection(conn)
        raise