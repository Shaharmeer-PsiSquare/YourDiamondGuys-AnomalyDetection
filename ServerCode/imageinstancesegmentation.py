import cv2
import numpy as np
import base64
from yoloseg import YOLOSeg
import os
import torch
from shapely.geometry import Polygon, point
from scipy.spatial import distance

# from transformers import pipeline
# torch.cuda.empty_cache()
# pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

distance_threshold = 70

# Initialize YOLOv5 Instance Segmentator
# model_path = "static/models/updated_model.onnx"
model_path = "static/models/new_best.onnx"

yoloseg = YOLOSeg(model_path, conf_thres=0.2, iou_thres=0.45)

square_model  = YOLOSeg("static/models/det_square.onnx", conf_thres=0.20, iou_thres=0.45)

def encode2array(encoded_image):
    image_bytes = base64.b64decode(encoded_image)

    # Convert the bytes to a NumPy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the NumPy array using OpenCV
    image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
    return image


def convert_mask_polygon(binary_mask_ls):
    # Find contours in the mask
    polygon_points_list = []
    plygon_area_ls = [ ]
    for  binary_mask1 in binary_mask_ls:
        # cv2.imwrite(f'mask{i}.png', binary_mask1)
        binary_mask = np.array(binary_mask1, dtype=np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        # Calculate the area of the polygon
        polygon_area = cv2.contourArea(max_contour)
        plygon_area_ls.append(polygon_area)
        # Extract the polygon vertices from the max contour
        polygon = max_contour.reshape(-1, 2)
        polygon_points_list.append(polygon.tolist())
    return polygon_points_list,plygon_area_ls

def convert_mask_polygon_sq(binary_mask_ls):
    # Find contours in the mask
    polygon_points_list = []
    polygon_areas = []
    for  binary_mask1 in binary_mask_ls:
        binary_mask = np.array(binary_mask1, dtype=np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Convert contours to polygon points

        max_contour = max(contours, key=cv2.contourArea)

        polygon_area = cv2.contourArea(max_contour)
        polygon_areas.append(polygon_area)
        print(polygon_area)
        # Extract the polygon vertices from the max contour
        polygon = max_contour.reshape(-1, 2)

        polygon_points_list.append(polygon)

            # print(polygon_points)
    if len(polygon_areas) >=2:
        max_area = max(polygon_areas)
        idx = polygon_areas.index(max_area)
        polygon = polygon_points_list[idx]
        polygon_area = max_area
    else:
        polygon = polygon_points_list[-1]
        polygon_area = polygon_areas[-1]

    return polygon,polygon_area


def encodeimage(image):
    # Encode the image as a binary stream
    _, buffer = cv2.imencode('.jpg', image)

    # Convert the binary stream to base64
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image
def calculate_polygon_area(vertices):
    polygon = Polygon(vertices)
    return polygon.area

def calculate_centeroid(polygon_point):
    polygon1 = Polygon(polygon_point)
    centroid1 = polygon1.centroid
    return centroid1

def draw_polygons(image, polygons, square_polygon, defect_area_ls, square_area):
    center_flag = False
    height, width, _ = image.shape
    
    # Safely convert square_polygon
    try:
        square_polygon = np.array(square_polygon, dtype=np.int32)
    except Exception as e:
        print(f"Error converting square_polygon: {e}")
        return image, {"centre_region": [], "other_region": []}, False, "0.00", {}

    sq_centeroid = calculate_centeroid(square_polygon)
    square_polygon = square_polygon.reshape((-1, 1, 2))
    centre_region = []
    head_anomaly = {
        "head_points": square_polygon.tolist(), # Convert to list for JSON serialization safety
        "head_centeroid": (sq_centeroid.x, sq_centeroid.y),
        "defect_anomalies": []
    }

    other_region = []
    center_defect_area_list = []

    for id, polygon in enumerate(polygons):
        try:
            # --- FIX START: Sanitize the polygon data ---
            # Ensure polygon is a list of points where each point has exactly 2 coordinates [x, y]
            clean_polygon = []
            for point in polygon:
                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                    clean_polygon.append([point[0], point[1]])
            
            if not clean_polygon:
                continue # Skip empty or invalid polygons

            polygon_array = np.array(clean_polygon, dtype=np.int32)
            # --- FIX END ---

            # Use the clean array for vertex extraction
            vertex = (clean_polygon[0][0], clean_polygon[0][1])
            
            # Reshape for drawing
            polygon_reshaped = polygon_array.reshape((-1, 1, 2))

            # Check if the polygon is inside the square
            if cv2.pointPolygonTest(square_polygon, vertex, False) >= 0:
                cv2.polylines(image, [polygon_reshaped], isClosed=True, color=(0, 0, 255), thickness=2)
                centre_region.append(clean_polygon)
                
                try:
                    defect_centeroid = calculate_centeroid(polygon_array)
                    distance_between_centroids = distance.euclidean(
                        (defect_centeroid.x, defect_centeroid.y), 
                        (sq_centeroid.x, sq_centeroid.y)
                    )
                    
                    head_anomaly["defect_anomalies"].append({
                        "defect_cordinates": clean_polygon,
                        "defect_centeroid": (defect_centeroid.x, defect_centeroid.y),
                        "distance": distance_between_centroids
                    })

                    # Distance threshold logic
                    if (height == 500 and width == 500) and int(distance_between_centroids) <= 35:
                        center_flag = True
                    elif 'distance_threshold' in globals() and int(distance_between_centroids) <= distance_threshold:
                        center_flag = True
                    
                    if id < len(defect_area_ls):
                        center_defect_area_list.append(defect_area_ls[id])
                except Exception as inner_e:
                    print(f"Error calculating centroid/distance: {inner_e}")
                    pass

            else:
                # If outside, keep the color as cyan/yellow
                other_region.append(clean_polygon)
                cv2.polylines(image, [polygon_reshaped], isClosed=True, color=(0, 255, 255), thickness=2)

        except Exception as e:
            print(f"Skipping a bad polygon due to error: {e}")
            continue

    # Final calculations
    area_sum = sum(center_defect_area_list)
    if square_area > 0:
        percentage_defects = (area_sum / square_area) * 100
    else:
        percentage_defects = 0

    without_defect_per = float(100 - percentage_defects)
    percentage = f"{without_defect_per:.2f}"

    if center_flag is False:
        if float(without_defect_per) <= 97.0:
            center_flag = True
            
    # print(f"Percentage of defects: {percentage_defects:.2f}%")
    
    return image, {"centre_region": centre_region, "other_region": other_region}, center_flag, percentage, head_anomaly

def Draw_poly(image, points):
    clean_points = []
    for polygon in points:
        try:
            # --- FIX START: Sanitize ---
            clean_polygon = []
            if isinstance(polygon, (list, np.ndarray)):
                for point in polygon:
                    # Ensure we have at least [x, y]
                    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                        clean_polygon.append([point[0], point[1]])
            
            # Skip if empty
            if not clean_polygon: 
                continue

            # Convert safely
            polygon_np = np.array(clean_polygon, dtype=np.int32)
            polygon_reshaped = polygon_np.reshape((-1, 1, 2))
            
            # Draw
            cv2.polylines(image, [polygon_reshaped], isClosed=True, color=(255, 0, 0), thickness=2)
            clean_points.append(clean_polygon)
            # --- FIX END ---
            
        except Exception as e:
            print(f"Skipping bad polygon in Draw_poly: {e}")
            continue

    return image, {"other_region": clean_points}, None



def are_defects_inside_square(defect_polygons, square_polygon):
    square_polygon = square_polygon.reshape((-1, 1, 2))
    # Check if any two defect polygons are inside the square
    for i in range(len(defect_polygons)):
        for j in range(i + 1, len(defect_polygons)):
            is_inside_square_i = any(cv2.pointPolygonTest(square_polygon, tuple(vertex), False) >= 0 for vertex in defect_polygons[i])
            is_inside_square_j = any(cv2.pointPolygonTest(square_polygon, tuple(vertex), False) >= 0 for vertex in defect_polygons[j])

            if is_inside_square_i and is_inside_square_j:
                return True

    return False



def prediction(image_path,encode_flag,encode_image=None):
    if encode_flag:
        img = encode2array(encode_image)

        # cv2.imshow("asd", img)
        # cv2.waitKey()
    else:
        img = cv2.imread(image_path)

    # print(cv2.im)
    # cv2.imshow("asd",img)
    # cv2.waitKey()
    boxes, scores, class_ids, masks = yoloseg(img)
    sq_boxes, sq_scores, sq_class_ids, sq_masks = square_model(img)
    print(len(sq_masks))
    # print(sq_scores)
    # print(len(sq_masks))
    if len(sq_masks) >= 1:
        # print(len(sq_masks))
        # binary_mask = np.array(sq_masks[0], dtype=np.uint8)
        polygon_points , square_area= convert_mask_polygon_sq(sq_masks)
        defect_polygons , defect_area_ls = convert_mask_polygon(masks)


        combined_img, region_dict, center_falg, percentage, head_anomaly = draw_polygons(img, defect_polygons, polygon_points,defect_area_ls,square_area)
        print(center_falg)
        height, width, _ = combined_img.shape
        if height == 500 and width==500:
            combined_img = combined_img[:width - 63, :]
        # cv2.imwrite(f'static/rbmg_dir/test.png', combined_img)
        # pillow_image = pipe('static/rbmg_dir/test.png')
        # numpy_array = np.array(pillow_image)
        # Convert the color format from RGB to BGR (OpenCV uses BGR)
        # combined_img = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        # center_falg = are_defects_inside_square(defect_polygons,square_det)
        if encode_flag:
            encodeimg = encodeimage(combined_img)
            # cv2.imwrite("test.png", combined_img)
            # polygon_points = convert_mask_polygon(masks)

            return encodeimg, defect_polygons, True,center_falg,region_dict,percentage, head_anomaly
        else:

            cv2.imwrite(f'static/result/res8_{image_path.split("/")[-1]}', combined_img)
            return f'static/result/res8_{image_path.split("/")[-1]}', True, image_path,center_falg,region_dict,percentage,head_anomaly

    else:
        defect_polygons = convert_mask_polygon(masks)
        combined_img, region_dict, percentage = Draw_poly(img,defect_polygons)
        height, weight, _ = combined_img.shape
        if height == 500 and width==500:
            combined_img = combined_img[:width - 63, :]
        # cv2.imwrite(f'static/rbmg_dir/test.png', combined_img)
        # pillow_image = pipe('static/rbmg_dir/test.png')
        # numpy_array = np.array(pillow_image)
        # Convert the color format from RGB to BGR (OpenCV uses BGR)
        # combined_img = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        center_falg = False
        if encode_flag:
            encodeimg = encodeimage(combined_img)
            # polygon_points = convert_mask_polygon(masks)

            return encodeimg, defect_polygons, True,center_falg,region_dict,percentage,None
        else:
            cv2.imwrite(f'static/result/res8_{image_path.split("/")[-1]}', combined_img)
            return f'static/result/res8_{image_path.split("/")[-1]}', True, image_path,center_falg,region_dict,percentage, None


    # combined_img = yoloseg.draw_masks(img)

