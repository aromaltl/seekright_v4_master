import numpy as np
import cv2
import os
from uuid import uuid4
from typing import List, Dict, Optional
import cv2
import sys
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET
from datetime import datetime

from ultralytics import YOLO
# ultralytics.checks()

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def get_date_path():
    try:
        date_string = "2024-10-29"#input("Enter the date (yyyy-mm-dd):")
        # Attempt to parse the date string
        date = datetime.strptime(date_string, "%Y-%m-%d")
        # If parsing is successful, the date is valid
        print(f"Thank you, {date_string} is a valid date.")
        return f"{str(date.year).zfill(4)}/{str(date.month).zfill(2)}/{date.strftime('%Y-%m-%d')}"
    except ValueError:
        # If parsing fails, the date is invalid
        print(f"{date_string} is not a valid date.")
        get_date_path()
date_path = get_date_path()
version = input("Enter the version (Ex: 001, 002):")
#/run/user/1001/gvfs/afp-volume:host=Anton.local,user=ml_support,volume=labelstudio/data/old/fixed
data_path = os.path.join("/run/user/1000/gvfs/smb-share:server=enigma.local,share=labelstudio/data", date_path, version)

print(f"Reading data of path {os.path.join(date_path, version)}")
sys.path.append("..")

sam_checkpoint = "label-studio-ml-backend/label_studio_ml/examples/segment_anything_model/models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda:0"
print("Loading model ......")
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
print("Model loaded successfully")
# mask_generator = SamAutomaticMaskGenerator(sam)

def mask_to_polygons(mask: np.ndarray,h,w) -> List[np.ndarray]:
        """
        Converts a binary mask to a list of polygons.

        Parameters:
            mask (np.ndarray): A binary mask represented as a 2D NumPy array of
                shape `(H, W)`, where H and W are the height and width of
                the mask, respectively.

        Returns:
            List[np.ndarray]: A list of polygons, where each polygon is represented by a
                NumPy array of shape `(N, 2)`, containing the `x`, `y` coordinates
                of the points. Polygons with fewer points than `MIN_POLYGON_POINT_COUNT = 3`
                are excluded from the output.
        """
        MIN_POLYGON_POINT_COUNT = 2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_area = -1.0
        CONTROL_NUM_POINTS = 0.001 #less value more polygon points
        for contour in contours:
            current_contour_area = cv2.contourArea(contour)
            if contour_area <= current_contour_area:
                contour_area = current_contour_area
                if contour.shape[0] >= MIN_POLYGON_POINT_COUNT:
                    # RDP algorithm to reduce number of polygons
                    rdp_epsilon = CONTROL_NUM_POINTS*cv2.arcLength(contour,True)
                    contour = cv2.approxPolyDP(contour, epsilon = rdp_epsilon, closed=True)
                    c = np.squeeze(contour, axis=1).astype(np.float32)
                    # print(c)
                    c[:,0] = (c[:,0]/w)*100
                    c[:,1] = (c[:,1]/h)*100
                    result = [c.tolist()]
        return result

labelstudio_path = data_path[data_path.find("labelstudio") + len("labelstudio") + 1:]

def create_image_url(filepath):
    """
    Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
    if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
    Otherwise you can build links like /data/upload/filename.png to refer to the files
    """
    filename = os.path.basename(filepath)
    return f'/data/local-files/?d={labelstudio_path}/images/{filename}'

images_folder_path = os.path.join(data_path, "images")
labels_folder_path = os.path.join(data_path, "labels")

print(f"The path of images in import.json file will look like: /data/local-files/?d={labelstudio_path}/images/<image_name>'")
# mask_predictor = SamPredictor(sam)
json_data = []
empty_xml_count = 0
error_count = 0


network  = YOLO("/home/sultan/Desktop/seekright_v4_master/best.pt")
class_names =  network.names

for image_file_path in tqdm(sorted(os.listdir(images_folder_path))):
    try:
        image = cv2.imread(os.path.join(images_folder_path,image_file_path), flags=cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        image_basename = os.path.splitext(os.path.basename(image_file_path))[0]

        # mask_predictor.set_image(image)

        # tree = ET.parse(os.path.join(labels_folder_path, image_basename + ".xml"))
        # root = tree.getroot()
        score_sum = 0
        results = []
        # objects = root.iter('object')

        res = network(image)#,verbose=False
        res_box = res[0].boxes.cpu().numpy()
        masks = res[0].masks.data.detach().cpu().numpy()
        iterate = zip(res_box.cls.astype(int),res_box.conf,masks)

        for obj in iterate:
            region_id = str(uuid4())[:10]
            label_name = class_names[int(obj[0])]
            # if 'andat' in label_name:
            #     continue


            scores =0.71

            score_sum += 0.711

            # mask = masks[0, :, :].astype(np.uint8)*255
            # print(obj[2])
            mask = obj[2].astype(np.uint8)*255
            mask=cv2.resize(mask,(w,h))
            # cv2.imshow('out',mask[::3,::3])
            # cv2.waitKey(0)
            contours = mask_to_polygons(mask, h, w)
            print(contours)

            result = {
                    "id": region_id,
                    "from_name": "polygon",
                    "to_name": "image",
                    "original_width": w,
                    "original_height": h,
                    "image_rotation": 0,
                    "value": {
                        "points" : contours[0],
                        "polygonlabels" : [label_name],
                        "closed" : True
                    },
                    'score': float(scores),
                    "type": "polygonlabels",
                }
            results.append(result)
        if results:
            json_data.append({
                    'data': {
                        'image': create_image_url(image_file_path)
                    },
                    'predictions': [{
                        'model_version': "yolov8",
                        'score': float(0.71),
                        'result': results,
                    }]
            })
        else:
            json_data.append({
                    'data': {
                        'image': create_image_url(image_file_path)
                    }
            })
            empty_xml_count += 1
    except ET.ParseError:
        empty_xml_count += 1
        json_data.append({
                'data': {
                    'image': create_image_url(image_file_path)
                }
        })
    except Exception as e:
        print(e)
        json_data.append({
                'data': {
                    'image': create_image_url(image_file_path)
                }
        })
        error_count += 1

with open(os.path.join(data_path, "import.json"), "w") as json_file:
    json.dump(json_data, json_file, indent = 4)

print("Empty Xml Count: ", empty_xml_count)
print("Error Count: ", error_count)