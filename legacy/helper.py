import boto3
import requests
import cv2
import base64
import os
import json
# import app as ap

s3 = boto3.client('s3')

def encodeimage(image):
    # Encode the image as a binary stream
    _, buffer = cv2.imencode('.jpg', image)

    # Convert the binary stream to base64
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image
def get_predict_from_model(image):
    url = "http://3.225.210.139:8009/"

    # Open the image file in binary mode for reading
    image_data=  encodeimage(image)
    # print(image)
        # Create a dictionary for the files to be sent in the request
    files = {'encode':  image_data}

    # Make the POST request with the image and additional data
    response = requests.post(url, data=files )
    # Checking the response
    # if response.status_code == 200:
    data = response.json()
    # print(data)
    return data


def get_predict_from_model2(image):
    url = "http://3.225.210.139:8009/"

    # Open the image file in binary mode for reading
    image_data=  encodeimage(image)
    # print(image)
        # Create a dictionary for the files to be sent in the request
    files = {'encode':  image_data}

    # Make the POST request with the image and additional data
    response = requests.post(url, data=files )
    # Checking the response
    # if response.status_code == 200:
    data = response.json()
    # print(data)
    return data

def create_folder(bucket_name, folder_name):
    # Ensure the folder name ends with a slash ("/")
    if not folder_name.endswith('/'):
        folder_name += '/'

    # Create an empty file as a placeholder for the folder
    s3.put_object(Bucket=bucket_name, Key=folder_name)


def get_object(bucket_name, folder_prefix):
    # List all objects in the specified folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)

    # Check if objects were found
    object_keys = [obj['Key'] for obj in response.get('Contents', [])]
    # print(object_keys)
    return object_keys
import numpy as np

def encode2array(encoded_image):
    image_bytes = base64.b64decode(encoded_image)

    # Convert the bytes to a NumPy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the NumPy array using OpenCV
    image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
    return image


def encodeimage(image):
    # Encode the image as a binary stream
    _, buffer = cv2.imencode('.jpg', image)
    # Convert the binary stream to base64
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

def digisation_result_by_id(diamond_id):
    url = "http://3.225.210.139:8000/GigitizedById/"
    files = {'diamond_id': diamond_id}
    response = requests.post(url, data=files, headers={
        "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.8) Gecko/20100722 Firefox/3.6.8 GTB7.1 (.NET CLR 3.5.30729)",
        "Referer": "http://example.com"})
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

def digisation_result(certificate_link,diamond_id,raw_text = None,image_flag=True,pdf_path=None):
    url = "http://3.225.210.139:8000/DigitizationOfCertificate/"
    # Open the image file in binary mode for reading
   
    
    payload = {
    "certificate_link": json.dumps(certificate_link)
    }
    print("payload: ",payload)

    # Send POST request
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        data = response.json()
        print("data:", data)
        st = data['status']
        if st:
            all_data = data['final_list']
            return True, all_data
        return False, None
    return False, None
    
    # if raw_text is not None:
    #     files = {'raw_text': raw_text,'digitzation_flag':True,'diamond_id': diamond_id}
    #     response = requests.post(url, data=files, headers={
    #         "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.8) Gecko/20100722 Firefox/3.6.8 GTB7.1 (.NET CLR 3.5.30729)",
    #         "Referer": "http://example.com"})
    #     # Checking the response
    #     if response.status_code == 200:
    #         data = response.json()
    #         # print(data)
    #         # print(data)
    #         return data
    #     else:
    #         return None
    # else:
    #     if image_flag:
    #         image_data = encodeimage(image)
    #         # print(image)
    #         # Create a dictionary for the files to be sent in the request
    #         files = {'encode_image': image_data,'digitzation_flag':True,'diamond_id':diamond_id}
    #         # Make the POST request with the image and additional data
    #         response = requests.post(url, data=files, headers={
    #             "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.8) Gecko/20100722 Firefox/3.6.8 GTB7.1 (.NET CLR 3.5.30729)",
    #             "Referer": "http://example.com"})
    #         # Checking the response
    #         if response.status_code == 200:
    #             data = response.json()
    #             # print(data)
    #             # print(data)
    #             return data
    #         else:
    #             return None
    #     else:
    #         # files = {'file': (os.path.basename(pdf_path), open(f'{pdf_path}', 'rb'), 'application/pdf')}
    #         data = {'digitzation_flag': True,'pdf_path':pdf_path,"diamond_id":diamond_id}

    #         response = requests.post(url,data=data,headers={
    #             "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.8) Gecko/20100722 Firefox/3.6.8 GTB7.1 (.NET CLR 3.5.30729)",
    #             "Referer": "http://example.com"})
    #         if response.status_code == 200:
    #             data = response.json()
    #             # print(data)
    #             # print(data)
    #             return data
    #         else:
    #             return None
