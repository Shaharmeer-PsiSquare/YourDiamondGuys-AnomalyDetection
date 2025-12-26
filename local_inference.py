import cv2
import numpy as np
import base64
from yoloseg import YOLOSeg
import os
import logging
from shapely.geometry import Polygon
from scipy.spatial import distance

# Configure logger
logger = logging.getLogger(__name__)

distance_threshold = 70

# Model paths - using models directory in the project root
_model_dir = os.path.join(os.path.dirname(__file__), 'models')
_model_path = os.path.join(_model_dir, 'new_best.onnx')
_square_model_path = os.path.join(_model_dir, 'det_square.onnx')

# Initialize YOLOv5 Instance Segmentator models (lazy loading)
_yoloseg = None
_square_model = None

def _get_models():
    """Lazy load models on first use"""
    global _yoloseg, _square_model
    if _yoloseg is None:
        logger.info(f"Loading YOLOSeg model from: {_model_path}")
        _yoloseg = YOLOSeg(_model_path, conf_thres=0.2, iou_thres=0.45)
        logger.info("YOLOSeg model loaded successfully")
    
    if _square_model is None:
        logger.info(f"Loading square detection model from: {_square_model_path}")
        _square_model = YOLOSeg(_square_model_path, conf_thres=0.20, iou_thres=0.45)
        logger.info("Square detection model loaded successfully")
    
    return _yoloseg, _square_model

def encode2array(encoded_image):
    """Decode base64 encoded image to numpy array"""
    image_bytes = base64.b64decode(encoded_image)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
    return image

def encodeimage(image):
    """Encode image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

def convert_mask_polygon(binary_mask_ls):
    """Convert binary masks to polygon points - FIXED: handles empty contours"""
    polygon_points_list = []
    polygon_area_ls = []
    
    for binary_mask1 in binary_mask_ls:
        binary_mask = np.array(binary_mask1, dtype=np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # FIX: Check if contours is empty before calling max()
        if len(contours) == 0:
            logger.warning("No contours found in binary mask, skipping")
            continue
        
        max_contour = max(contours, key=cv2.contourArea)
        polygon_area = cv2.contourArea(max_contour)
        polygon_area_ls.append(polygon_area)
        polygon = max_contour.reshape(-1, 2)
        polygon_points_list.append(polygon.tolist())
    
    return polygon_points_list, polygon_area_ls

def convert_mask_polygon_sq(binary_mask_ls):
    """Convert square binary masks to polygon points - FIXED: handles empty contours"""
    polygon_points_list = []
    polygon_areas = []
    
    for binary_mask1 in binary_mask_ls:
        binary_mask = np.array(binary_mask1, dtype=np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # FIX: Check if contours is empty before calling max()
        if len(contours) == 0:
            logger.warning("No contours found in square binary mask, skipping")
            continue
        
        max_contour = max(contours, key=cv2.contourArea)
        polygon_area = cv2.contourArea(max_contour)
        polygon_areas.append(polygon_area)
        polygon = max_contour.reshape(-1, 2)
        polygon_points_list.append(polygon)
    
    if len(polygon_areas) >= 2:
        max_area = max(polygon_areas)
        idx = polygon_areas.index(max_area)
        polygon = polygon_points_list[idx]
        polygon_area = max_area
    elif len(polygon_areas) == 1:
        polygon = polygon_points_list[0]
        polygon_area = polygon_areas[0]
    else:
        # No valid polygons found
        logger.warning("No valid square polygons found")
        return None, 0
    
    return polygon, polygon_area

def calculate_centroid(polygon_point):
    """Calculate centroid of a polygon"""
    polygon1 = Polygon(polygon_point)
    centroid1 = polygon1.centroid
    return centroid1

def draw_polygons(image, polygons, square_polygon, defect_area_ls, square_area):
    """Draw polygons on image and calculate regions"""
    center_flag = False
    height, width, _ = image.shape
    
    # Safely convert square_polygon
    try:
        square_polygon = np.array(square_polygon, dtype=np.int32)
    except Exception as e:
        logger.error(f"Error converting square_polygon: {e}")
        return image, {"centre_region": [], "other_region": []}, False, "0.00", {}
    
    sq_centroid = calculate_centroid(square_polygon)
    # Reshape for drawing operations (needs to be (-1, 1, 2) for cv2.pointPolygonTest)
    square_polygon_reshaped = square_polygon.reshape((-1, 1, 2))
    # For JSON storage, flatten to 2D: [[x,y], [x,y], ...]
    square_polygon_flat = square_polygon.reshape(-1, 2)
    centre_region = []
    head_anomaly = {
        "head_points": square_polygon_flat.tolist(),  # 2D structure: [[x,y], [x,y], ...]
        "head_centeroid": (sq_centroid.x, sq_centroid.y),
        "defect_anomalies": []
    }
    
    other_region = []
    center_defect_area_list = []
    
    for id, polygon in enumerate(polygons):
        try:
            clean_polygon = []
            for point in polygon:
                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                    clean_polygon.append([point[0], point[1]])
            
            if not clean_polygon:
                continue
            
            polygon_array = np.array(clean_polygon, dtype=np.int32)
            vertex = (clean_polygon[0][0], clean_polygon[0][1])
            polygon_reshaped = polygon_array.reshape((-1, 1, 2))
            
            # Use reshaped version for cv2.pointPolygonTest (needs (-1, 1, 2) format)
            if cv2.pointPolygonTest(square_polygon_reshaped, vertex, False) >= 0:
                cv2.polylines(image, [polygon_reshaped], isClosed=True, color=(0, 0, 255), thickness=2)
                centre_region.append(clean_polygon)
                
                try:
                    defect_centroid = calculate_centroid(polygon_array)
                    distance_between_centroids = distance.euclidean(
                        (defect_centroid.x, defect_centroid.y),
                        (sq_centroid.x, sq_centroid.y)
                    )
                    
                    head_anomaly["defect_anomalies"].append({
                        "defect_cordinates": clean_polygon,
                        "defect_centeroid": (defect_centroid.x, defect_centroid.y),
                        "distance": distance_between_centroids
                    })
                    
                    if (height == 500 and width == 500) and int(distance_between_centroids) <= 35:
                        center_flag = True
                    elif int(distance_between_centroids) <= distance_threshold:
                        center_flag = True
                    
                    if id < len(defect_area_ls):
                        center_defect_area_list.append(defect_area_ls[id])
                except Exception as inner_e:
                    logger.warning(f"Error calculating centroid/distance: {inner_e}")
                    pass
            else:
                other_region.append(clean_polygon)
                cv2.polylines(image, [polygon_reshaped], isClosed=True, color=(0, 255, 255), thickness=2)
        except Exception as e:
            logger.warning(f"Skipping a bad polygon due to error: {e}")
            continue
    
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
    
    return image, {"centre_region": centre_region, "other_region": other_region}, center_flag, percentage, head_anomaly

def Draw_poly(image, points):
    """Draw polygons when no square is detected"""
    clean_points = []
    for polygon in points:
        try:
            clean_polygon = []
            if isinstance(polygon, (list, np.ndarray)):
                for point in polygon:
                    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                        clean_polygon.append([point[0], point[1]])
            
            if not clean_polygon:
                continue
            
            polygon_np = np.array(clean_polygon, dtype=np.int32)
            polygon_reshaped = polygon_np.reshape((-1, 1, 2))
            cv2.polylines(image, [polygon_reshaped], isClosed=True, color=(255, 0, 0), thickness=2)
            clean_points.append(clean_polygon)
        except Exception as e:
            logger.warning(f"Skipping bad polygon in Draw_poly: {e}")
            continue
    
    return image, {"other_region": clean_points}, None

def prediction(image_path, encode_flag, encode_image=None):
    """
    Perform local inference on an image.
    
    Args:
        image_path: Path to image file (ignored if encode_flag=True)
        encode_flag: If True, use encode_image instead of image_path
        encode_image: Base64 encoded image string
    
    Returns:
        tuple: (encodeimg, defect_polygons, flag, center_flag, region_dict, accuracy_per, head_anomaly)
    """
    try:
        # Load models
        yoloseg, square_model = _get_models()
        
        # Load image
        if encode_flag:
            img = encode2array(encode_image)
            if img is None:
                logger.error("Failed to decode base64 image")
                return None, [], False, False, {}, "0.00", None
        else:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image from path: {image_path}")
                return None, [], False, False, {}, "0.00", None
        
        # Run inference
        boxes, scores, class_ids, masks = yoloseg(img)
        sq_boxes, sq_scores, sq_class_ids, sq_masks = square_model(img)
        
        logger.debug(f"Detected {len(sq_masks)} square masks")
        
        if len(sq_masks) >= 1:
            polygon_points, square_area = convert_mask_polygon_sq(sq_masks)
            
            if polygon_points is None:
                # No valid square found, fall through to Draw_poly
                logger.warning("No valid square polygon found, using Draw_poly")
                # Match legacy behavior: only unpack first value
                defect_polygons = convert_mask_polygon(masks)
                combined_img, region_dict, percentage = Draw_poly(img, defect_polygons)
                height, width, _ = combined_img.shape
                if height == 500 and width == 500:
                    combined_img = combined_img[:width - 63, :]
                center_flag = False
                if encode_flag:
                    encodeimg = encodeimage(combined_img)
                    return encodeimg, defect_polygons, True, center_flag, region_dict, percentage, None
                else:
                    cv2.imwrite(f'static/result/res8_{image_path.split("/")[-1]}', combined_img)
                    return f'static/result/res8_{image_path.split("/")[-1]}', True, image_path, center_flag, region_dict, percentage, None
            
            defect_polygons, defect_area_ls = convert_mask_polygon(masks)
            combined_img, region_dict, center_flag, percentage, head_anomaly = draw_polygons(
                img, defect_polygons, polygon_points, defect_area_ls, square_area
            )
            
            height, width, _ = combined_img.shape
            if height == 500 and width == 500:
                combined_img = combined_img[:width - 63, :]
            
            if encode_flag:
                encodeimg = encodeimage(combined_img)
                return encodeimg, defect_polygons, True, center_flag, region_dict, percentage, head_anomaly
            else:
                cv2.imwrite(f'static/result/res8_{image_path.split("/")[-1]}', combined_img)
                return f'static/result/res8_{image_path.split("/")[-1]}', True, image_path, center_flag, region_dict, percentage, head_anomaly
        else:
            # Match legacy behavior: only unpack first value (even though convert_mask_polygon returns 2 values)
            # This matches the legacy code exactly, even though it's technically incorrect
            defect_polygons = convert_mask_polygon(masks)
            combined_img, region_dict, percentage = Draw_poly(img, defect_polygons)
            height, width, _ = combined_img.shape
            if height == 500 and width == 500:
                combined_img = combined_img[:width - 63, :]
            center_flag = False
            if encode_flag:
                encodeimg = encodeimage(combined_img)
                return encodeimg, defect_polygons, True, center_flag, region_dict, percentage, None
            else:
                cv2.imwrite(f'static/result/res8_{image_path.split("/")[-1]}', combined_img)
                return f'static/result/res8_{image_path.split("/")[-1]}', True, image_path, center_flag, region_dict, percentage, None
    
    except Exception as e:
        logger.error(f"Error in prediction function: {str(e)}", exc_info=True)
        return None, [], False, False, {}, "0.00", None

