import re
from app import celery, fetch_ids, database_update, db_connection_1, db_connection_2, run_update_model, update_record
import cv2
import boto3
import urllib.request
import requests
from tqdm import tqdm
from celery.utils.log import get_task_logger
import numpy as np
from io import BytesIO
from helper import create_folder, get_predict_from_model, digisation_result, get_predict_from_model2, \
    digisation_result_by_id
from celery.result import AsyncResult
import base64
import os
import uuid
import json
import traceback
from celery.beat import PersistentScheduler
import gc

req_shape_list = ["pear", "round", "oval", "princess", "emerald", "asscher", "marquise", "heart", "cushion", "radiant"]

# last_fetched_id = 0
last_process_id = 'cc22435d-1295-4ae1-8518-67eb202ca608'
name_Table = "worker_table_2"


# Define a custom scheduler class to expose the database connection
class CustomScheduler(PersistentScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._store = self.get_schedule()


CREATE_ROOMS_TABLE = (
        "create table IF NOT EXISTS " + name_Table + " (id text PRIMARY KEY, status text, affiliate_name text, response text,grading_lab text);")
INSERT_ROOM_RETURN_ID = "INSERT INTO worker_table_2 (id,status,affiliate_name,response,grading_lab) VALUES (%s,%s,%s,%s,%s) RETURNING id;"

logger = get_task_logger(__name__)

# url = 'http://44.217.3.244:8009/api/ProductInfoView/'

s3 = boto3.client('s3')


def checking_record(connection, diamond_id):
    gc.collect()
    # get_query = f"SELECT * FROM public.\"Affiliate_app_productinfo\" WHERE diamond_id::integer = {daimond_id};"
    column_name = 'status'
    get_query = f"SELECT {column_name} FROM public.\"Affiliate_app_productinfo\" WHERE diamond_id = %s ;"

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(get_query, (diamond_id,))
            rows = cursor.fetchall()
            # print("roooooow",rows)
            if rows:
                check_value = rows[0][0]
            else:
                check_value = None
        cursor.close()
    connection.commit()
    return check_value


def checking_record_digisation(connection, diamond_id):
    gc.collect()
    # get_query = f"SELECT * FROM public.\"Affiliate_app_productinfo\" WHERE diamond_id::integer = {daimond_id};"
    column_name = 'digitization'
    get_query = f"SELECT {column_name} FROM public.\"Affiliate_app_productinfo\" WHERE diamond_id = %s ;"

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(get_query, (diamond_id,))
            rows = cursor.fetchall()
            # print(rows)
            # if rows:
            check_value = rows[0][0]
        cursor.close()
    connection.commit()
    return check_value


def insert_ai_data(connection, status, response, diamond_id, center_flag, points, acc, anomaly_head):
    gc.collect()
    insert_query = f"INSERT INTO public.\"Affiliate_app_aiprocessing\" (is_processed, ai_output_image,diamond_id,center_flag,polygon_points,ai_score,anomaly_head) VALUES (true, '{response}','{diamond_id}','{center_flag}','{points}','{acc}','{anomaly_head}') ON CONFLICT(diamond_id) DO UPDATE SET diamond_id = EXCLUDED.diamond_id RETURNING id;"
    print(insert_query)
    cursor = connection.cursor()
    cursor.execute(insert_query)
    # print(cursor.fetchone()[0])
    res = cursor.fetchone()[0]  # print(type(worker_id))
    connection.commit()
    cursor.close()
    connection.close()
    return res


def insert_digitisation_data(connection, data, diamond_id):
    gc.collect()
    if "culet" in data:
        if data['culet'] == 'none':
            data['culet'] = 'None'
    if "shape" not in data:

        keys_to_check = ["culet", "depth", "fluorescence", "key_to_symbol", "measurement", "table_size", "crown_angle",
                         "crown_height", "girdle", "pavilion_height", "pavilion_angle", "clarity", "color",
                         "cut", "shape", "lower_half_length", "star_length", "symmetry",
                         "polish", "carat"]
        for key in keys_to_check:
            # Check if the key exists in the dictionary
            if key in list(data.keys()):
                pass
            else:
                # If the key is missing, you can set a default value or handle it accordingly
                data[key] = None
    if "round" in data["shape"].lower():
        keys_to_check = ["culet", "depth", "fluorescence", "key_to_symbol", "measurement", "table_size", "crown_angle",
                         "crown_height", "girdle", "pavilion_height", "pavilion_angle", "clarity", "color",
                         "cut", "shape", "lower_half_length", "star_length", "symmetry",
                         "polish", "carat"]
        for key in keys_to_check:
            # Check if the key exists in the dictionary
            if key in list(data.keys()):
                pass
            else:
                # If the key is missing, you can set a default value or handle it accordingly
                data[key] = None
        insert_query = f"""
            INSERT INTO public."Affiliate_app_digitization" (
                diamond_id, culet, depth, flouroscence, key_to_symbol, measurement, table_size, crown_angle, crown_height, girdle, pavilion_height, pavilion_angle, clarity, color, cut, shape, lower_half_length, star_length, symmetry, polish, carat,reprocess_status, value_check
            )
            VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s,%s,%s,%s, %s)
            ON CONFLICT (diamond_id) DO UPDATE SET diamond_id = EXCLUDED.diamond_id RETURNING id;;
        """
        value = (
            diamond_id, data["culet"], data["depth"], data["fluorescence"], data["key_to_symbol"], data["measurement"],
            data["table_size"], data["crown_angle"], data["crown_height"], data["girdle"], data["pavilion_height"],
            data["pavilion_angle"], data["clarity"], data["color"], data["cut"], data["shape"],
            data["lower_half_length"], data["star_length"], data["symmetry"], data["polish"], data["carat"], 'pending',
            'DF')
        cursor = connection.cursor()
        cursor.execute(insert_query, value)
    else:
        keys_to_check = ["culet", "depth", "fluorescence", "key_to_symbol", "measurement", "table_size", "girdle",
                         "clarity", "color", "shape_and_style", "symmetry", "polish", "carat"]
        for key in keys_to_check:
            # Check if the key exists in the dictionary
            if key in list(data.keys()):
                pass
            else:
                # If the key is missing, you can set a default value or handle it accordingly
                data[key] = None
        # print(data)
        insert_query = f"""
            INSERT INTO public."Affiliate_app_digitization" (
               diamond_id, culet, depth, flouroscence, key_to_symbol, measurement, table_size, girdle, clarity, color, shape, symmetry, polish, carat, reprocess_status, value_check
            )
            VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (diamond_id)DO UPDATE SET diamond_id = EXCLUDED.diamond_id RETURNING id;;
        """

        value = (
            diamond_id, data["culet"], data["depth"], data["fluorescence"], data["key_to_symbol"], data["measurement"],
            data["table_size"], data["girdle"], data["clarity"], data["color"], data["shape"], data["symmetry"],
            data["polish"], data["carat"], 'pending', 'DF')
        cursor = connection.cursor()

        cursor.execute(insert_query, value)

    worker_id = cursor.fetchone()[0]
    connection.commit()
    cursor.close()
    connection.close()
    return worker_id


def insert_digisation_score_data(connection, data, diamond_id):
    # print(data)
    if "cut_score" in list(data.keys()):

        keys_to_check = ["culet_score", "depth_score", "fluorescence_score", "girdle_score", "polish_score",
                         "symmetry_score", "table_size_score", "crown_angle_score", "cut_score", "measurement_score",
                         "pavilion_angle_score", "pavilion_height_score", "key_to_symbol_score", "digisation_score"]
        for key in keys_to_check:
            # Check if the key exists in the dictionary
            if key in list(data.keys()):
                pass
            else:
                # If the key is missing, you can set a default value or handle it accordingly
                data[key] = None

        insert_query = f"""INSERT INTO public.\"Affiliate_app_scoreinfo\" (diamond_id, culet_score ,depth_score ,flouroscence,gridle_score,polish_score ,symmetry_score ,table_size_score ,crown_angle_score ,cut_score ,measurement_score ,pavilion_angle_score ,pavilior_height_score ,key_to_symbol_score, digisation_score) 
        VALUES ('{diamond_id}', '{data["culet_score"]}','{data["depth_score"]}','{data["fluorescence_score"]}','{data["girdle_score"]}','{data["polish_score"]}','{data["symmetry_score"]}','{data["table_size_score"]}','{data['crown_angle_score']}','{data["cut_score"]}','{data["measurement_score"]}','{data["pavilion_angle_score"]}','{data["pavilion_height_score"]}','{data["key_to_symbol_score"]}','{data["digisation_score"].replace("%", "")}') ON CONFLICT (diamond_id) DO UPDATE SET diamond_id = EXCLUDED.diamond_id RETURNING id;"""

    else:
        keys_to_check = ["culet_score", "depth_score", "fluorescence_score", "girdle_score", "polish_score",
                         "symmetry_score", "table_size_score", "key_to_symbol_score", "digisation_score"]
        for key in keys_to_check:
            # Check if the key exists in the dictionary
            if key in list(data.keys()):
                pass
            else:
                # If the key is missing, you can set a default value or handle it accordingly
                data[key] = None
        insert_query = f"""
            INSERT INTO public.\"Affiliate_app_scoreinfo\" (
                 diamond_id, culet_score, depth_score, flouroscence, gridle_score, polish_score,
                symmetry_score, table_size_score, key_to_symbol_score,digisation_score
            )
            VALUES (
                '{diamond_id}','{data["culet_score"]}','{data["depth_score"]}','{data["fluorescence_score"]}',
                '{data["girdle_score"]}','{data["polish_score"]}','{data["symmetry_score"]}','{data["table_size_score"]}',
                '{data["key_to_symbol_score"]}','{data["digisation_score"].replace("%", "")}'
            )
            ON CONFLICT (diamond_id) DO UPDATE SET diamond_id = EXCLUDED.diamond_id RETURNING id;;
        """

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(insert_query)
            worker_id = cursor.fetchone()[0]
        cursor.close()
    connection.commit()
    connection.close()
    return worker_id


def update_product_info(connection, daimond_id, digisation_id, score_id):
    update_query = "UPDATE public.\"Affiliate_app_productinfo\" SET status=%s, digitization_id=%s ,score_info_id=%s WHERE diamond_id = %s;"
    values = (True, digisation_id, score_id, daimond_id)
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(update_query, values)
        cursor.close()
    connection.commit()
    connection.close()


def upadte_ProductCharacteristics(connection, digitisation_score, daimond_id, key_to_symbol_score):
    if digitisation_score == "None":
        digitisation_score = 0
    else:
        digitisation_score = digitisation_score.replace("%", "")
    update_query = "UPDATE public.\"Affiliate_app_productcharacteristics\" SET digitization_score=%s, key_to_symbol_score=%s WHERE diamond_id = %s;"
    values = (float(digitisation_score), float(key_to_symbol_score), daimond_id)
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(update_query, values)
        cursor.close()
    connection.commit()
    connection.close()

from datetime import datetime
import pytz
def ai_time_update(diamond_id,connection):
    utc_time = datetime.now(pytz.utc)
    print("working")

    cursor = connection.cursor()

    # Check if diamond_id already exists
    # check_query = "SELECT 1 FROM authen_app_aidatetimerecord WHERE diamond_id = %s LIMIT 1;"
    # cursor.execute(check_query, (str(diamond_id),))
    # exists = cursor.fetchone()
    
    record_type = 'digitization'

    # ✅ Check if diamond_id + type already exists
    check_query = """
        SELECT 1 
        FROM authen_app_aidatetimerecord 
        WHERE diamond_id = %s AND type = %s
        LIMIT 1;
    """
    cursor.execute(check_query, (str(diamond_id), record_type))
    exists = cursor.fetchone()

    if exists:
        print(f"Diamond ID {diamond_id} already exists. Skipping insert.")
    else:
        insert_query = """
            INSERT INTO authen_app_aidatetimerecord (diamond_id, type, created_at)
            VALUES (%s, %s, %s);
        """
        cursor.execute(insert_query, (diamond_id, 'digitization', utc_time))
        connection.commit()
        print(f"Diamond ID {diamond_id} inserted successfully.")
        
        
def ai_anomaly_time_update(diamond_id,connection):
    utc_time = datetime.now(pytz.utc)
    print("working")

    cursor = connection.cursor()

    # Check if diamond_id already exists
    # check_query = "SELECT 1 FROM authen_app_aidatetimerecord WHERE diamond_id = %s LIMIT 1;"
    # cursor.execute(check_query, (str(diamond_id),))
    # exists = cursor.fetchone()
    record_type = 'ai'

    # ✅ Check if diamond_id + type already exists
    check_query = """
        SELECT 1 
        FROM authen_app_aidatetimerecord 
        WHERE diamond_id = %s AND type = %s
        LIMIT 1;
    """
    cursor.execute(check_query, (str(diamond_id), record_type))
    exists = cursor.fetchone()

    if exists:
        print(f"Diamond ID {diamond_id} already exists. Skipping insert.")
    else:
        insert_query = """
            INSERT INTO authen_app_aidatetimerecord (diamond_id, type, created_at)
            VALUES (%s, %s, %s);
        """
        cursor.execute(insert_query, (diamond_id, 'ai', utc_time))
        connection.commit()
        print(f"Diamond ID {diamond_id} inserted successfully.")


# @celery.task(name='send_data')
def send_data(task_id, resp, affiliate_name, grading_lab):
    # connection = db_connection_1()
    # task_ids, response_list = fetch_ids(connection)

    # for task_id, resp in zip(task_ids, response_list):
    # print("Task is ready")
    # result = AsyncResult(task_id, app=celery)
    # print(result.ready())

    # print(task_id,resp)
    result_data = resp
    connection = db_connection_1()
    if result_data["cert"]["status"] is False:
        result_data["cert"] = digisation_result_by_id(result_data['diamond_id'])
        database_update(connection, task_id, result_data, affiliate_name, grading_lab)

    else:
        database_update(connection, task_id, resp, affiliate_name, grading_lab)

    # print(result_data)
    # img_res =
    if result_data["cert"] != None:
        cert_res = result_data["cert"]

        # id = os.path.splitext(img_res)[0]
        # imag_link = img_res

        connection2 = db_connection_2()
        diamond_id = result_data["diamond_id"]
        # checking_value = checking_record(connection2, diamond_id)
        # print(checking_value)
        # if checking_value is False or checking_value is None:
        if cert_res["status"] is True:
            # print("----------error------------",cert_res["error"])
            cert_res["digitized_data"]['shape'] = resp['shape']

            connection2 = db_connection_2()
            ret_digi_id = insert_digitisation_data(connection2, cert_res["digitized_data"], diamond_id)
            connection2 = db_connection_2()
            score_id = insert_digisation_score_data(connection2, cert_res["digitized_score_data"], diamond_id)
            connection2 = db_connection_2()
            ai_time_update(diamond_id,connection2)
            upadte_ProductCharacteristics(connection2, cert_res["digitized_score_data"]["digisation_score"],
                                          diamond_id, cert_res["digitized_score_data"]["key_to_symbol_score"])
            connection2 = db_connection_2()
            update_product_info(connection2, diamond_id, ret_digi_id, score_id)
        else:
            # digisation_result_by_id
            connection2 = db_connection_2()
            update_query = "UPDATE public.\"Affiliate_app_productinfo\" SET status=%s WHERE diamond_id = %s;"
            values = (None, diamond_id)
            with connection2:
                with connection2.cursor() as cursor:
                    cursor.execute(update_query, values)
                cursor.close()
            connection2.commit()
            connection2.close()


def encode2array(encoded_image):
    image_bytes = base64.b64decode(encoded_image)

    # Convert the bytes to a NumPy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the NumPy array using OpenCV
    image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
    return image


# @celery.task(name='process_list')
def process_list(diamond_id, certificate_link, affiliate_name):
    # Process the list (for example, return the sum)
    # result_data_list = []
    # result_cert = []
    # zips = zip(ids, image_link_ls, certificate_links)
    # for id, image_url,certificate_link in zips:
    #     print(id)

    # print("---------------------------------------------------",f'{affiliate_name}/{diamond_id}')
    # create_folder('affiliated-partners', f'{affiliate_name}/{diamond_id}')
    # create_folder('affiliated-partners', f'{affiliate_name}/{diamond_id}/original')
    # create_folder('affiliated-partners', f'{affiliate_name}/{diamond_id}/result')

    # Get the image from the URL using requests
    # response = requests.get(image_url)
    # image_data = response.content
    # status = is_image_or_pdf(certificate_link)
    # print(status)
    # if status == 'image':
    #     cert_response = requests.get(certificate_link)
    #     image_data_cert = cert_response.content

    #     # Read the image using OpenCV
    #     # image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    #     # result_img = get_predict_from_model(image_array)
    #     data_cert = digisation_result(cv2.imdecode(np.frombuffer(image_data_cert, np.uint8), -1),diamond_id)
    # else:
    sta, data_cert = digisation_result(certificate_link,diamond_id,image_flag=False, pdf_path=certificate_link)

    # return {"diamond_id": diamond_id, "cert": data_cert}
    return sta, data_cert


def is_image_or_pdf(link):
    with urllib.request.urlopen(link) as response:
        info = response.info()
        content_type = info.get_content_maintype()
    if 'image' in content_type:
        return "image"
    else:
        return "pdf"


def check_digtization_exsit_or_not(conection, diamond_id):
    check_query = 'SELECT id FROM "Affiliate_app_digitization" WHERE diamond_id = %s;'
    cursor = conection.cursor()
    cursor.execute(check_query, (diamond_id,))
    check_result = cursor.fetchone()
    return check_result


@celery.task(name='fetch_data')
def fetch_data():
    # global last_fetched_id
    if os.path.exists('fetch_id.json'):
        with open('fetch_id.json', 'r') as js:
            fetched_id = json.load(js)
            if fetched_id:
                last_fetched_id = fetched_id[0]
            else:
                last_fetched_id = 0
    else:
        last_fetched_id = 0
    if os.path.exists('output.json'):
        try:
            with open('output.json', 'r') as js:
                data_list1 = json.load(js)
        except:
            data_list1 = []
    else:
        data_list1 = []
    connection = db_connection_2()
    # query = 'SELECT id, diamond_id, certificate_link,grading_lab,shape,affiliate_id, image_link, web_page_link FROM "Affiliate_app_productinfo" app WHERE status IS False  AND id > %s AND sell_status IS False ORDER BY id   LIMIT 1000;'
    # query = 'SELECT id, diamond_id, certificate_link,grading_lab,shape,affiliate_id, image_link, web_page_link FROM "Affiliate_app_productinfo" app digitization_id IS NULL AND sell_status IS FALSE;'
    query = 'SELECT id, diamond_id, certificate_link,grading_lab,shape,affiliate_id, image_link, web_page_link FROM "Affiliate_app_productinfo" app WHERE digitization_id IS NULL AND sell_status IS FALSE LIMIT 1000;'
    cursor = connection.cursor()
    cursor.execute(query, (last_fetched_id,))
    results = cursor.fetchall()
    # cursor.close()
    # connection.close()
    total_count = 0
    crt_link = []
    crt_link_record = []

    for row in results:

        id, diamond_id, certificate_link, grading_lab, shape, affiliate_id, image_link, web_page_link = row
        print(diamond_id)
        new_shape = shape
        # print(certificate_link)
        if certificate_link:
            # try:
            # if diamond_id in data_list1:
            #     pass
            # else:
            check_result = check_digtization_exsit_or_not(connection, diamond_id)
            data_list1.append(diamond_id)
            if check_result:
                digitization_id = check_result[0]
                update_query = 'UPDATE "Affiliate_app_productinfo" SET digitization_id = %s WHERE diamond_id = %s;'
                cursor.execute(update_query, (digitization_id, diamond_id))
                connection.commit()
            else:
                last_fetched_id = id
                if affiliate_id != 4:
                    match = re.search(r'diamond/(\d+)', image_link)
                    if match:
                        diamond_id_new = match.group(1)
                        if diamond_id_new == diamond_id:
                            pass
                        else:
                            diamond_id = diamond_id_new
                else:
                    pattern = r'sku-(\d+)\?'
                    # Apply the pattern to the URL
                    match = re.search(pattern, web_page_link)
                    if match:
                        diamond_id_new = match.group(1)
                        if diamond_id_new == diamond_id:
                            pass
                        else:
                            diamond_id = diamond_id_new
                if shape.split()[0].lower() in req_shape_list:
                    if affiliate_id == 1:
                        affiliate_name = 'james-allen'
                    elif affiliate_id == 2:
                        affiliate_name = 'blue-nile'
                    elif affiliate_id == 3:
                        affiliate_name = 'amazon'
                    elif affiliate_id == 4:
                        affiliate_name = 'white-flash'
                    else:
                        affiliate_name = 'rapnet'
                    crt_link.append(certificate_link)
                    crt_link_record.append({'affiliate_name':affiliate_name,"diamond_id":diamond_id,"grading_lab":grading_lab, "new_shape":new_shape, "certificate_link":certificate_link})
                    total_count += 1
                    if total_count == 5:
                        # print("daimonds id ------",diamond_id,image_link,certificate_link)
                        new_status, all_data_cert = process_list(diamond_id, crt_link, affiliate_name)
                        crt_link = []
                        total_count = 0
                        if new_status:
                            for data_cert in all_data_cert:
                                result_data_list_id = uuid.uuid4()
                                # print(result_data_list.id)
                                certficate_link = data_cert['digitized_data']['image_url']
                                record = next((item for item in crt_link_record if item['certificate_link'] == certficate_link), None)
                                affiliate_name =  record['affiliate_name']
                                diamond_id =  record['diamond_id']
                                grading_lab =  record['grading_lab']
                                new_shape =  record['new_shape']
                                
                                
                                resp = {"diamond_id": diamond_id, "cert": data_cert}
                                resp['shape'] = new_shape
                                send_data(result_data_list_id, resp, affiliate_name, grading_lab)
                                
                                with open('output.json', 'w') as j:
                                    json.dump(data_list1, j)
                            
                else:
                    pass
        else:
            last_fetched_id = id

        #     print("Error", certificate_link)
        #     data_list1.append(diamond_id)
    with open("fetch_id.json", "w") as j_id:
        json.dump([last_fetched_id], j_id)


from selenium import webdriver
import cv2

@celery.task(name='AI_Process')
def AI_Process():
    # if os.path.exists("error1.json"):
    #     with open("error1.json", "r") as errr:
    #         error_ls = json.load(errr)
    # else:
    #     error_ls = []

    connection = db_connection_2()
    # query = 'SELECT diamond_id, image_link, affiliate_id, web_page_link FROM "Affiliate_app_productinfo" app WHERE  ai_processing_id IS NULL AND sell_status IS FALSE;'
    query = 'SELECT diamond_id, image_link, affiliate_id, web_page_link FROM "Affiliate_app_productinfo" app WHERE  ai_processing_id IS NULL AND sell_status IS FALSE  LIMIT 1000;'
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    for row in results:
        try:
            # print(row)
            diamond_id, image_link, affiliate_id, web_page_link = row
            dummy_id = diamond_id
            # print("dummy", dummy_id)
            if affiliate_id != 4:
                # if 'w' in diamond_id:

                match = re.search(r'diamond/(\d+)', image_link)
                if match:
                    diamond_id_new = match.group(1)
                    if diamond_id_new == diamond_id.replace('b', ''):
                        diamond_id = diamond_id_new
                        # pass
                    else:
                        diamond_id = diamond_id_new
            else:
                pattern = r'sku-(\d+)\?'
                # Apply the pattern to the URL
                match = re.search(pattern, web_page_link)
                if match:
                    diamond_id_new = match.group(1)
                    if diamond_id_new == diamond_id.replace('w', ''):
                        diamond_id = diamond_id_new
                    else:
                        diamond_id = diamond_id_new
            print("diamond_id:", diamond_id)
            print(image_link)
            if affiliate_id == 1:
                affiliate_name = 'james-allen'
            elif affiliate_id == 2:
                affiliate_name = 'blue-nile'
            elif affiliate_id == 3:
                affiliate_name = 'amazon'
            elif affiliate_id == 4:
                affiliate_name = 'white-flash'
            elif affiliate_id == 5:
                affiliate_name = 'rapnet'

            # Get the image from the URL using requests
            # try:
            create_folder('affiliated-partner', f'{affiliate_name}/{diamond_id}')
            create_folder('affiliated-partner', f'{affiliate_name}/{diamond_id}/original')
            create_folder('affiliated-partner', f'{affiliate_name}/{diamond_id}/result')
            # try:
            response = requests.get(image_link,timeout=2)
            image_data = response.content
            # if response.status_code != 200:
            #     result = 'output.png'
            #     options = webdriver.FirefoxOptions()
            #     options.add_argument("--headless")
            #     driver = webdriver.Firefox(options=options)
            #     print("m_image_link:",image_link)
            #     driver.get(image_link)
            #     driver.save_screenshot(result)
            #     image_array = cv2.imread(result)

            #     driver.close()
            # else:
            #     # Read the image using OpenCV
            image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
            result_img = get_predict_from_model(image_array)
            # except:
            #     result = 'output.png'
            #     options = webdriver.FirefoxOptions()
            #     options.add_argument("--headless")
            #     driver = webdriver.Firefox(options=options)
            #     print("m_image_link:",image_link)
            #     driver.get(image_link)
            #     driver.save_screenshot(result)
            #     image_array = cv2.imread(result)
            #     driver.close()
            # print(result_img)
            # result_cert.append(data_cert)
            print(result_img.keys())
            result_image_data = result_img["encode_img"].encode('utf-8')
            flag_center = result_img["center_flag"]
            points = result_img["region_points"]
            acc = result_img["acuuracy"]
            anomaly_head = result_img["anomaly_head"]
            # result_response = requests.get(result_img)
            # result_image_data = result_response.content
            # image_array = cv2.imdecode(np.frombuffer(result_image_data, np.uint8), -1)

            original_image_key = f'{affiliate_name}/{diamond_id}/original/{diamond_id}.png'
            result_image_key = f'{affiliate_name}/{diamond_id}/result/anomaly_detection_model_res_{diamond_id}.png'
            ori_image_stream = BytesIO(image_data)
            res_image_stream = BytesIO(result_image_data)
            s3.upload_fileobj(ori_image_stream, 'affiliated-partner', original_image_key)
            s3.upload_fileobj(res_image_stream, 'affiliated-partner', result_image_key)
            result_key_link = f's3://affiliated-partner/{affiliate_name}/{diamond_id}/result/anomaly_detection_model_res_{diamond_id}.png'
            # result_data_list.append(result_key_link)
            connection2 = db_connection_2()

            ret_id = insert_ai_data(connection2, "true", result_key_link, diamond_id, flag_center, points, acc,
                                    anomaly_head)
            # print("AI ID:", ret_id)
            connection = db_connection_2()
            ai_anomaly_time_update(diamond_id,connection)
            
            
            
            update_query = "UPDATE public.\"Affiliate_app_productinfo\" SET ai_processing_id=%s  WHERE diamond_id = %s ;"
            values = (ret_id, dummy_id,)
            cursor = connection.cursor()
            cursor.execute(update_query, values)
            connection.commit()
            cursor.close()
            connection.close()
            connection = db_connection_2()
            update_ai_flag_data(connection, diamond_id, flag_center)
        
        except Exception as e:
            print("error",str(e))
            


# def fetch_data(diamond_id):
#     "Affiliate_app_digitization"
#     qurey = f"SELECT table_size, girdle, clarity, color FROM \"Affiliate_app_digitization\"  WHERE diamond_id=%s;"
#     conn = db_connection_2()
#     cursor = conn.cursor()
#     cursor.execute(qurey, (diamond_id,))
#     # diamond_ids = [row[0] for row in cursor.fetchall()]
#     clarity,color,table_size,girdle = cursor.fetchall()[0]
#     cursor.close()
#     conn.close()
#     return clarity,color,table_size,girdle


# def update_product_info(connection, clarity,color,table_size,girdle,diamond_id):
#     update_query = "UPDATE public.\"Affiliate_app_digitization\" SET clarity=%s, color=%s ,table_size=%s ,girdle=%s WHERE diamond_id = %s;"
#     values = (clarity,color,table_size,girdle,diamond_id)
#     with connection:
#         with connection.cursor() as cursor:
#             cursor.execute(update_query, values)
#         cursor.close()
#     connection.commit()
#     connection.close()


@celery.task(name='process_error_fun')
def process_error_fun():
    with open('all_worker_data.json', 'r') as e:
        error_ls = json.load(e)
    with open("gia_done.json", 'r') as d:
        done_ls = json.load(d)
    rep_ls = []
    for err1 in error_ls[::-1]:
        for err in err1:

            data_id, resp = err

            res_data = json.loads(resp)
            try:
                id = res_data["diamond_id"]
                diamond_id = id
            except:
                id = os.path.splitext(res_data["image"])[0]
                diamond_id = id.split('_')[-1]
            print(diamond_id)

            if diamond_id not in done_ls and diamond_id not in rep_ls:
                connection = db_connection_2()
                query = 'SELECT certificate_link FROM "Affiliate_app_productinfo" app WHERE diamond_id=%s'
                cursor = connection.cursor()
                cursor.execute(query, (diamond_id,))
                results = cursor.fetchone()[0]
                print(results)
                cursor.close()
                connection.close()
                data_cert = process_list(diamond_id, results, None)
                if data_cert["cert"] != None:
                    cert_res = data_cert["cert"]

                    connection2 = db_connection_2()
                    diamond_id = data_cert["diamond_id"]
                    # checking_value = checking_record(connection2, diamond_id)
                    checking_value = False
                    # print(checking_value)
                    if checking_value is False or checking_value is None:
                        if cert_res["status"] is True:
                            # print("----------error------------",cert_res["error"])
                            connection2 = db_connection_2()
                            ret_digi_id = insert_digitisation_data(connection2, cert_res["digitized_data"], diamond_id)
                            connection2 = db_connection_2()
                            score_id = insert_digisation_score_data(connection2, cert_res["digitized_score_data"],
                                                                    diamond_id)
                            connection2 = db_connection_2()
                            upadte_ProductCharacteristics(connection2,
                                                          cert_res["digitized_score_data"]["digisation_score"],
                                                          diamond_id,
                                                          cert_res["digitized_score_data"]["key_to_symbol_score"])
                            connection2 = db_connection_2()
                            update_product_info(connection2, diamond_id, ret_digi_id, score_id)
                            update_record(data_cert, data_id)

            elif diamond_id in rep_ls:
                print("diamonds_rep:", diamond_id)
                del_qurey = f"DELETE FROM worker_table_2 WHERE id ='{str(data_id)}'"
                connection = db_connection_1()
                cursor = connection.cursor()
                cursor.execute(del_qurey)

                # Commit the changes
                connection.commit()

                # Close the cursor and connection
                cursor.close()
                connection.close()
            elif diamond_id in done_ls:
                rep_ls.append(diamond_id)
                print("diamonds_exist:", diamond_id)
                del_qurey = f"DELETE FROM worker_table_2 WHERE id ='{str(data_id)}'"
                connection = db_connection_1()
                cursor = connection.cursor()
                cursor.execute(del_qurey)

                # Commit the changes
                connection.commit()

                # Close the cursor and connection
                cursor.close()
                connection.close()


# @celery.task(name='process_error')
# @celery.task(name='process_error')
def process_error():
    if os.path.exists('ai_not_done.json'):
        with open('ai_not_done.json', 'r') as js:
            fetched_id_ls = json.load(js)
    else:
        fetched_id_ls = []

    for daimond_id in tqdm(fetched_id_ls):
        # print(daimond_id)
        try:
            connection = db_connection_2()
            query = f'SELECT id, diamond_id, image_link, affiliate_id, web_page_link FROM "Affiliate_app_productinfo" app WHERE diamond_id =%s'
            cursor = connection.cursor()
            cursor.execute(query, (str(daimond_id),))
            results = cursor.fetchall()
            # print(results)
            cursor.close()
            connection.close()
            # for row in results:
            # print(row)
            id, diamond_id, image_link, affiliate_id, web_page_link = results[0]
            # dummy_id = diamond_id
            last_fetched_id = id

            if affiliate_id != 4:

                match = re.search(r'diamond/(\d+)', image_link)
                if match:
                    diamond_id_new = match.group(1)
                    if diamond_id_new == diamond_id.replace('b', ''):
                        diamond_id = diamond_id_new
                        # pass
                    else:
                        diamond_id = diamond_id_new
            else:
                pattern = r'sku-(\d+)\?'
                # Apply the pattern to the URL
                match = re.search(pattern, web_page_link)
                if match:
                    diamond_id_new = match.group(1)
                    if diamond_id_new == diamond_id.replace('w', ''):
                        diamond_id = diamond_id_new
                    else:
                        diamond_id = diamond_id_new
            # print("diamond_id:", diamond_id)
            if affiliate_id == 1:
                affiliate_name = 'james-allen'
            elif affiliate_id == 2:
                affiliate_name = 'blue-nile'
            elif affiliate_id == 3:
                affiliate_name = 'amazon'
            elif affiliate_id == 4:
                affiliate_name = 'white-flash'
            elif affiliate_id == 5:
                affiliate_name = 'rapnet'

                # Get the image from the URL using requests
            response = requests.get(image_link)
            image_data = response.content
            # Read the image using OpenCV
            image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
            result_img = get_predict_from_model2(image_array)

            create_folder('affiliated-partner', f'{affiliate_name}/{diamond_id}')
            create_folder('affiliated-partner', f'{affiliate_name}/{diamond_id}/original')
            create_folder('affiliated-partner', f'{affiliate_name}/{diamond_id}/result')

            # result_cert.append(data_cert)
            result_image_data = result_img["encode_img"].encode('utf-8')
            flag_center = result_img["center_flag"]
            points = result_img["region_points"]
            acc = result_img["acuuracy"]
            anomaly_head = result_img["anomaly_head"]

            original_image_key = f'{affiliate_name}/{diamond_id}/original/{diamond_id}.png'
            result_image_key = f'{affiliate_name}/{diamond_id}/result/anomaly_detection_model_res_{diamond_id}.png'
            ori_image_stream = BytesIO(image_data)
            res_image_stream = BytesIO(result_image_data)
            s3.upload_fileobj(ori_image_stream, 'affiliated-partner', original_image_key)
            s3.upload_fileobj(res_image_stream, 'affiliated-partner', result_image_key)
            connection2 = db_connection_2()
            update_ai_processing_data(connection2, diamond_id, points, flag_center, acc, anomaly_head)
            connection2 = db_connection_2()
            update_ai_flag_data(connection2, diamond_id, flag_center)
        except:
            # output = traceback.format_exc()
            # print(output)
            print("error", daimond_id)
            pass


def update_ai_processing_data(connection, diamond_id, response, flag, acc, anomaly_head):
    update_qurey = "UPDATE public.\"Affiliate_app_aiprocessing\" SET polygon_points=%s,center_flag=%s, ai_score=%s, anomaly_head=%s WHERE diamond_id = %s;"
    values = (response, flag, acc, anomaly_head, diamond_id)
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(update_qurey, values)
        cursor.close()
    connection.commit()
    connection.close()


def update_ai_flag_data(connection, diamond_id, flag):
    update_qurey = "UPDATE public.\"Affiliate_app_productcharacteristics\" SET center_flag=%s WHERE diamond_id = %s;"
    values = (flag, diamond_id)
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(update_qurey, values)
        cursor.close()
    connection.commit()
    connection.close()


def add_ai_error(connection, result_data_list_id, response, diamond_id, image_link):
    qurey = "INSERT INTO anomaly_error (id,image_link,response,diamond_id) VALUES (%s,%s,%s,%s) RETURNING id;"
    values = (str(result_data_list_id), image_link, response, diamond_id)
    cursor = connection.cursor()
    cursor.execute(qurey, values)
    cursor.close()
    connection.commit()
    connection.close()


# start celery worker
# celery -A tasks  worker --loglevel=info

# start celery beat
# celery -A tasks.celery beat --loglevel=info


celery.conf.beat_schedule = {
    'Fetch data from db every 2 minutes': {
        'task': 'fetch_data',
        'schedule': 2700.0,
        'options': {'queue': 'fetch_data_queue'}
    },
    'AI_Process reminder in every 45 minutes': {
        'task': 'AI_Process',
        'schedule': 10.0,
        'options': {'queue': 'AI_Process_queue'}
    },

    # 'process_error reminder in every 1 hour': {
    #     'task': 'process_error',
    #     'schedule': 2700.0,
    #     'options': {'queue': 'process_error_queue'}
    # },
    'process_error_fun reminder': {
        'task': 'process_error_fun',
        'schedule': 2700.0,
        'options': {'queue': 'process_error_fun_queue'}
    }

}
celery.conf.scheduler_cls = CustomScheduler


def is_scheduled_task_active(task_name, app):
    # Check if the task is currently active
    print(app.conf.beat_schedule)
    return task_name in app.conf.beat_schedule


def ping_check():
    dict_data = {}
    print()
    active_status = celery.control.inspect().active()
    if active_status is not None:
        dict_data["active_flag"] = True
    else:
        dict_data["active_flag"] = False
    # process = fetch_data.apply_async()
    # states = process.state
    # process_AI = AI_Process.apply_async()
    # states_ai = process_AI.state
    # process_re = process_error.apply_async()
    # states_error = process_re.state
    # # if states == "PENDING" and states_ai == "PENDING" and states_error == "PENDING":
    # dict_data={"digitization_worker": states, "anomaly_detection": states_ai, "reprocess_worker": states_error}

    return dict_data