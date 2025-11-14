import ultralytics
from ultralytics import YOLO
# ultralytics.checks()

import torch
import argparse
import copy
import os
import shutil
import signal
import sys
import time
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from queue import Queue
import cv2
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import pandas as pd
import json
import darknet
# from logger_helper import logger_obj
from slack_logger import SlackLogger
from meta_data_safecam import get_gps
# from position_utilities import PositionUtilities
import utilites
import config
#from asset_processing import data_pocessing
# from new_compare import generate_frames
# from socket_helper import SocketClass
import traceback
from datetime import datetime, timedelta
from scipy.spatial import distance
import geo
from datetime import date
import requests
noError = None
position_df = None
errorVideoFlag = False
arguments = config.getArgs()
from kalman import KalmanFilter
# from global_variables import variable_obj
# from rectification_linear import run_linear_rectification
import json
from global_variables import variable_obj
os.makedirs("csv_and_jsons",exist_ok=True)
#from processing import dataProcessing

def filter_weak_detections(data):
                # Count occurrences by asset_name + id combination
    asset_id_count = defaultdict(int)
    for frame, objects in data.items():
        if isinstance(objects, dict):
            for asset_name, detections in objects.items():
                for detection in detections:
                    obj_id = detection[0]
                    key = f"{asset_name}_{obj_id}"
                    
                    asset_id_count[key] += 1

    # Find asset+id combinations with < 3 occurrences
    # to_remove = []
    # for key, count in asset_id_count.items():

    to_remove = [(asset, obj_id) for key, count in asset_id_count.items() 
                if count < 3 for asset, obj_id in [key.rsplit('_', 1)]]
    
    if to_remove:
        for asset, obj_id in to_remove:
            print(f"  Removed {asset} ID '{obj_id}': {asset_id_count[f'{asset}_{obj_id}']} occurrences")

    # Remove detections with < 3 occurrences
    for frame in list(data.keys()):
        if isinstance(data[frame], dict):
            for asset_name in list(data[frame].keys()):
                data[frame][asset_name] = [det for det in data[frame][asset_name] 
                                            if (asset_name, det[0]) not in to_remove]
                if not data[frame][asset_name]:
                    del data[frame][asset_name]
            if not data[frame]:
                del data[frame]

class Detections:

    def __init__(self,cap):
        self.frame_queue = Queue(maxsize=128)
        self.darknet_image_queue = Queue(maxsize=128)
        self.detections_queue = Queue(maxsize=128)
        self.fps_queue = Queue(maxsize=128)
    

        self.arguments = config.getArgs()
        self.LeftGateThreshold = (-200, 360)
        self.RightGateThreshold = (450, 1000)
        self.frames_skipped = int(arguments["frames_skipped"])
        self.frame_count = 0
        self.network = network
        # print(network.__dict__['overrides']['imgsz'])
        # sys.exit()

        self.cap = cv2.VideoCapture(variable_obj.video_path)

        self.width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.video_name = video_name
        # self.width_show = 1280
        # self.height_show = 720
        self.ratio_x = self.width/1024
        self.ratio_y = self.height/576
        # print(self.width,self.ratio_x)
        self.accuracy_thresh = 0.25
        self.class_names = class_names

        self.detection_df = pd.DataFrame(columns=['Frame', 'Assets_LHS', 'Assets_RHS'])
        self.detection_dict = {}
        self.position_df = pd.DataFrame(columns=['Frame', 'Position', 'Speed'])
    
        # self.class_colors = colors
        self.objects_new = self.create_class_dict()
        self.tracker_flag = {}
        self.Left_Asset_Count = {}
        self.counter_flag = {}
        self.Right_Asset_Count = {}
        self.left_gate = {}
        self.right_gate = {}
        self.fixed_assets = set()
        self.create_counter()
        self.lineThickness = 3
        self.prev_percentage = 0
        self.totalFrames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.geo_detections = Queue()
        self.blank_image = np.ones((720, 800, 3), np.uint8)
        self.LEFT = []
        self.RIGHT = []
        self.FrameCount_list = Queue()
        self.quit = False
        # self.position_utilities = PositionUtilities()
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.t = time.time()
        self.show_count = arguments["show_count"]
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.linear_counter = {}
        self.linear_annotations = {}

        self.classes_queue = Queue()
        self.scores_queue = Queue()
        self.boxes_queue = Queue()
        self.masks_queue = Queue()

        self.all_detections = {}


    

    def create_class_dict(self):
        self.dets_new = {}
        self.active_trackers = {}
        self.age = {}
        self.asset_counter = {}
        self.reading = True
        self.infer = True
        # self.asset_list = {}
        self.tracker_list = {}
        self.json_out = {}
        for i in self.class_names:
            i=self.class_names[i]
            if i != '':
                self.dets_new[i] = []
                self.active_trackers[i] = {}
                self.age[i] = {}
                self.asset_counter[i] = 0
                # self.asset_list[i] = {}
                self.tracker_list[i] = {}  # creating trackers dictionary
        return self.dets_new

    def create_counter(self):
        self.min_area = {}
        self.class_accuracy = {}
        self.alpha = {}
        for key, value in arguments['HYPER_PARAMETERS'].items():
            # print(value, type(value))
            value = list(value.split(","))
            # value = ast.literal_eval(value)
            self.Left_Asset_Count[key] = 0  # Global count of left assets
            self.Right_Asset_Count[key] = 0  # global count for right assets
            self.left_gate[key] = 0
            self.right_gate[key] = 1024
            self.min_area[key] = float(value[0])
            self.class_accuracy[key] = float(value[1])
            self.fixed_assets.add(key)


    def count_image(self, fc):
        percentage_completed = (self.frame_count / self.totalFrames)
        blank_image = self.blank_image.copy()
        cv2.line(blank_image, (0, 716), (500, 716), (255, 255, 255), self.lineThickness)
        cv2.line(blank_image, (0, 716), (int(500 * percentage_completed), 716), (0, 255, 0), self.lineThickness)
        fps = self.fps_queue.get()
        cv2.putText(blank_image, 'fps : %s' % (str(round(fps, 2))), (150, 20), 0, 5e-3 * 150,
                    (255, 255, 255), 2)
        cv2.putText(blank_image, 'frame : %s' % (str(fc)), (270, 20), 0, 5e-3 * 150,
                    (255, 255, 255), 2)

        for i, (asset, count) in enumerate(self.asset_counter.items()):
            # obj = self.class_names[i]
            if len(self.tracker_list[asset]) > 0:
                # print(self.tracker_list[asset])
                cv2.putText(blank_image, asset + ":" + str(count), (25, 35 + 20 * i), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255, 255, 255), 1)
            else:
                cv2.putText(blank_image, asset + ":" + str(count), (25, 35 + 20 * i), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (i * 10, 255 - i * 10, 255), 1)
        return blank_image


    def set_saved_video(self, output_video):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        #video = cv2.VideoWriter(self.arguments['video_output_path'] + output_video, fourcc, fps, (1280, 720))

        # video = cv2.VideoWriter(self.arguments['video_output_path']+ output_video, fourcc, fps,(1280, 720))
        video = cv2.VideoWriter( save_folder+'/'+output_video, fourcc, fps,(1280, 720))
        return video

    def set_saved_video_count(self, output_video):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        video = cv2.VideoWriter(self.arguments['video_output_path'] + output_video, fourcc,
                                int(fps / self.frames_skipped), (1600, 448))
        return video

    # verification and validation of detections
    def check_detections(self, detections, fc):
        filtered_dets = defaultdict(list)  # $detections
        flag = copy.deepcopy(self.tracker_flag)
        counter_flag = copy.deepcopy(self.counter_flag)
        # print(detections,"@#@#@")
        for i in range(len(detections)):
            # print(detections)
            obj = detections[i][0]
            # print(det///ections[i][2][2] * detections[i][2][3],"@#@ss")
            # print(detections[i][1] , self.class_accuracy[obj],(detections[i][2][2] * detections[i][2][3]/14000)**4)
            if float(detections[i][1]) < self.class_accuracy[obj] or (
                    detections[i][2][2] * detections[i][2][3] < (self.min_area[obj])):
                flag[i] = 'i'
                counter_flag[i] = None
                continue
            filtered_dets[detections[i][0]].append([(i, fc), (
                int(detections[i][2][0]),#int(detections[i][2][0]-detections[i][2][2]/2),
                int(detections[i][2][1]-detections[i][2][3]/2))])  # (order of detection,frame_number, (x,y)coordinates)
        # print('detections',filtered_dets)
        # check if any obj has multiple bounding boxes
        # for example information board can have multiple bounding boxes
        if 'Signboard_Information_Board' in filtered_dets.keys():
            list_idx = []
            temp_list = []
            if len(filtered_dets['Signboard_Information_Board']) > 1:               
                filtered_dets['Signboard_Information_Board'].sort(key=lambda X: X[1][1])               
                for i in range(len(filtered_dets['Signboard_Information_Board']) - 1):
                    dist = distance.euclidean(filtered_dets['Signboard_Information_Board'][i][1],
                                              filtered_dets['Signboard_Information_Board'][i + 1][1])
                    if dist < 10:
                        flag_value = filtered_dets['Signboard_Information_Board'][i + 1][0][0]
                        flag[flag_value] = 'i'
                        counter_flag[flag_value] = None
                        list_idx.append(i + 1)

            for val in list_idx:
                if len(temp_list) < 1:
                    temp_list = filtered_dets['Signboard_Information_Board']
                del temp_list[val]
            if len(list_idx) > 0:
                filtered_dets.update({'Signboard_Information_Board': temp_list})

        return filtered_dets, flag, counter_flag

    def update_tracker(self):
        for asset, dict in self.age.items():
            if dict:
                old_trackers = []
                for trk, current_age in dict.items():
                    pred = self.tracker_list[asset][trk].predict()
                    # print('update',pred[0][0],pred)      
                    if not (0 <= (pred[0,0]) < 1030 and 0 <= pred[1,0] < 580):
                        old_trackers.append(trk)
                    if current_age > 18:
                        old_trackers.append(trk)
                    else:
                        self.age[asset][trk] += 1

                for trk in old_trackers:
                    try:
                        del self.tracker_list[asset][trk]
                        del self.age[asset][trk]
                    except Exception as err:
                        print(err,asset,trk,self.tracker_list[asset])
                        print(asset)
    


    def match_trackers(self, obj_list, asset, flag, counter_flag):
        # trackers = self.tracker_list[asset]
        temp_list = obj_list.copy()

        for id, tracker in self.tracker_list[asset].items():
            pred = tracker.predict()
            ##print('tracker predictions',pred)
            pred = pred.tolist()
            if int(pred[0][0]) not in range(0, 1024):
                ##print(pred)
                self.age[asset][id] = 60
                # print('made age greater than 60 **************')
                continue
            min1 = 10000
            if len(temp_list) < 1:
                # print("tracker with id - {0} for obj {1} doesn't match with any".format(str(id),asset))
                pass
            else:
                for point in temp_list:
                    x_dis = round(abs(point[1][0] - pred[0][0]), 2)
                    y_dis = round(abs(point[1][1] - pred[1][0]), 2)
                    distance = round((x_dis + y_dis), 2)
                    # print('distance',distance)
                    if distance < min1:
                        min1 = distance
                        #min2 = y_dis
                        pt = point
                        # print(min1,min2,pt)
                # print(min1,min2, pt)
                # remove that point from the object list
                # if min1 > 25:
                if min1 > 35:
                    # if len(self.tracker_list[asset]) == len(obj_list):
                    # check if the object is either at the left end or right end

                    if len(self.active_trackers[asset]) >= len(obj_list):
                        # last value of tracker
                        # print(len(self.active_trackers[asset]),min1)
                        x_val = self.active_trackers[asset][id][-1][0]
                        if x_val < 500:
                            # then it's a left asset
                            if pt[1][0] < x_val:
                                tracker.update(([pt[1][0]], [pt[1][1]]))
                                flag[pt[0][0]] = 'l'
                                counter_flag[pt[0][0]] = id
                                temp_list.remove(pt)
                                self.age[asset][id] = 1
                                self.active_trackers[asset][id].append((pt[1][0], pt[0][1]))
                            ##print('tracker {0} and point {1} matching with each other with a pixel distance of {2} for asset {3}'.format(str(id), str(pt), str(min1), asset))

                        if x_val > 500:
                            # then it's a right asset
                            if pt[1][0] > x_val:
                                tracker.update(([pt[1][0]], [pt[1][1]]))
                                flag[pt[0][0]] = 'r'
                                counter_flag[pt[0][0]] = id
                                temp_list.remove(pt)
                                self.age[asset][id] = 1
                                self.active_trackers[asset][id].append((pt[1][0], pt[0][1]))
                            ##print('tracker {0} and point {1} matching with each other with a pixel distance of {2} for asset {3}'.format(str(id), str(pt), str(min1), asset))
                    else:
                        self.age[asset][id] += 1
                        # to_remove.append(id)

                # tracker.update(([pt[1][0]],[pt[1][1]]))
                else:
                    ##print('tracker {0} and point {1} matching with each other with a pixel distance of {2} for asset {3}'.format(str(id),str(pt),str(min1),asset))
                    temp_list.remove(pt)
                    tracker.update(([pt[1][0]], [pt[1][1]]))
                    flag[pt[0][0]] = 'l'
                    counter_flag[pt[0][0]] = id
                    if pt[1][1] in range(325, 700):
                        self.age[asset][id] = -45
                    else:
                        self.age[asset][id] = 1
                    self.active_trackers[asset][id].append((pt[1][0], pt[0][1]))

        # unmatched items

        if len(temp_list) > 0:
            ##print('unmatched dets for the object {0}'.format(asset),temp_list)
            num = self.asset_counter[asset]
            # print(num)
            # print(self.tracker_list[asset])
            for x in temp_list:
                num += 1
                self.tracker_list[asset][num] = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
                val = ([x[1][0]], [x[1][1]])
                self.tracker_list[asset][num].update(val)
                # self.active_trackers[asset].append({num: [x[1]]})
                self.active_trackers[asset][num] = [x[1]]
                self.asset_counter[asset] += 1
                # left_new[obj][i].append(i+1)
                flag[x[0][0]] = 'c'
                self.age[asset][num] = 1
                counter_flag[x[0][0]] = num

        return flag, counter_flag


    def tracker_predict(self, left_new, obj, flag, counter_flag):
        if len(self.tracker_list[obj]) == 0:
            # create tracker
            #print("new {0} tracker is created".format(str(obj)))
            i = len(self.active_trackers[obj])
            for x in (left_new[obj]):
                self.tracker_list[obj][i + 1] = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
                val = ([x[1][0]], [x[1][1]])
                #print(val,x[1])
                self.tracker_list[obj][i + 1].update(val)
                # self.active_trackers[obj].append({i+1:[x[1]]})

                self.active_trackers[obj][i + 1] = [
                    (x[1][0], x[0][1])]  # created tuple with x_position and framecount (x_pos,fc)
                self.asset_counter[obj] += 1
                # left_new[obj][i].append(i+1)
                flag[x[0][0]] = 'c'
                if x[1][1] in range(325, 700):
                    self.age[obj][i + 1] = -10
                else:
                    self.age[obj][i + 1] = 1
                counter_flag[x[0][0]] = i + 1
                i+=1
   
        else:
            flag, counter_flag = self.match_trackers(left_new[obj], obj, flag, counter_flag)

        return flag, counter_flag

    def valid_plants(self, classes, boxes, image):
        try:
            image_half_width = image.shape[1] / 2
            plant_boxes = []
            # Separating all the plant bounding boxes
            for i in range(len(boxes)):
                if self.class_names[classes[i]] == "Plants":
                    # print(abs(boxes[i][0] - boxes[i][2]), image_half_width / 2)
                    if abs(boxes[i][0] - boxes[i][2]) < image_half_width / 2:
                        continue
                    else:
                        plant_boxes.append(boxes[i])
            # print(plant_boxes)
            if len(plant_boxes) == 0:
                return False
            if len(plant_boxes) == 1:
                return True

            # Sorting plant bounding boxes
            plant_boxes = sorted(plant_boxes, key=lambda x: x[0])

            distance_plants = False
            distance_threshold = image_half_width * 0.1  # 10% of half image
            for i in range(len(plant_boxes) - 1):
                if abs(plant_boxes[i][2] - plant_boxes[i][0]) > distance_threshold:
                    distance_plants = True

            return not distance_plants
        except:
            return True

    def calculate_area(self, mask_data, image):
        # [(y_start_pixel_range, y_end_pixel_range, area_per_pixel)]
        pixel_range = [(466, 533, 3.2, (255, 0, 0)), (533, 600, 2.0, (255, 255, 0)), (600, 666, 1.0, (255, 0, 255)),
                        (666, 720, 0.54, (0, 255, 255))]
        asset_pixels = np.nonzero(mask_data.cpu().detach().numpy())

        area_in_cm2 = 0
        for i in range(len(asset_pixels[0])):

            pixel_val = asset_pixels[0][i]
            for area_range in pixel_range:
                if area_range[0] <= pixel_val <= area_range[1]:
                    image = cv2.circle(image, (asset_pixels[1][i], asset_pixels[0][i]), 1, area_range[3], -1)
                    area_in_cm2 += area_range[2]
                    break

        return area_in_cm2, image


    def video_capture(self):
        global errorVideoFlag
        #self.cap.set(1,0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        unstable =0
        print("total frames: ",total)
        # total =min(20000,total)
        self.frame_count =-1
        break_after = 22
        try:
            while self.cap.isOpened():
                if unstable >1 and self.frame_count%30==2:
                    self.cap.set(1,self.frame_count+1)
                ret, frame = self.cap.read()
                self.frame_count+=1
                if self.frame_count >= total :
                    break
                if not ret or self.quit :
                    break_after-=1
                    if break_after <=0:
                        print("no ret for 22 frames")
                        break
                    if self.frame_count >= total :
                        break
                    print("Bad_Frame ",self.frame_count)
                    # print("Bad_Frame ",self.frame_count+1)
                    self.frame_count +=6
                    self.cap.set(1,self.frame_count+1)
                    unstable+=1
                    continue
                else:
                    break_after = 22

  
                if ret and  self.frame_count % self.frames_skipped == 0:
                    
                    # frame = cv2.resize(frame,(600,600))
                    self.darknet_image_queue.put(frame)
                    self.FrameCount_list.put(self.frame_count)
                # self.frame_count += 1
            # self.cap.release()
            print('done capturing')
        except Exception as e:
            # self.cap.release()
            try:
                error_message = "Error found: Video unable to play.."
                # socket_obj.addNotification(message=error_message, status=0)
            except Exception:
                utilites.PrintException()
            print("Exception in Video Capture", e)
            errorVideoFlag = True
            self.quit = True
            utilites.PrintException()

    

    def inference(self):
        time.sleep(2)
        global errorVideoFlag
        frame_no = -2
        try:
            while True:
                frame_no +=2
                start_time = time.time()
                if self.quit:
                    break
                try:
                    darknet_image = self.darknet_image_queue.get(timeout=6)
                    # print(darknet_image.shape)
                except Exception as err:
                    if self.darknet_image_queue.qsize() == 0:
                        print('Done inference')
                    else:
                        print('Exception in inference',err)
                    break
                # detections = darknet.detect_image(self.network, self.class_names, darknet_image,
                #                                 thresh=self.accuracy_thresh)
                darknet_image = cv2.cvtColor(darknet_image, cv2.COLOR_BGR2RGB)
                img = np.array(darknet_image)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(1024,576))
                results = self.network(img,verbose=False,iou =0.4,agnostic_nms=True)#,verbose=False
                res_box = results[0].boxes.cpu().numpy()
                masks = results[0].masks
                
                # if masks is not None:
                #     print(results[0].masks.data)
                
                #     while 1 :
                #         pass
                if masks is not None:
                    masks = torch.mean(masks.data.detach(),dim=(1,2)).cpu().numpy()
                else:
                    masks=[0]
                detections = []

                temp_box = []
                temp_score = []
                temp_cls = []
                temp_mask = []
                if masks is None:
                    masks=[]
                # print(res_box.cls.astype(int),res_box.conf,res_box.xywh.astype(int),masks,"@#@#@#@#$@$@")
                iterate = zip(res_box.cls.astype(int),res_box.conf,res_box.xywh.astype(int),res_box.xyxy.astype(int),masks)

                ############### LIGHTS LOGIC ###########

                def filterlights(iterate):
                    ignore_index =set()
                    l = None
                    r = None
                    for ind,(cls_name, boxxywh,boxxyxy) in enumerate(iterate):
                        # print(cls_name, boxxywh,boxxyxy)
                        cls_name = class_names[int(cls_name)]
                        if 'light' in cls_name.lower():
                            if boxxywh[0] < 1024//2:
                                if l is None:
                                    l = (boxxyxy[1],ind)
                                else:
                                    if l[0]>boxxyxy[1]:
                                        ignore_index.add(l[1])
                                        l=(boxxyxy[1],ind)
                                    else:
                                        ignore_index.add(ind)
                            else:
                                if r is None:
                                    r = (boxxyxy[1],ind)
                                else:
                                    if r[0]>boxxyxy[1]:
                                        ignore_index.add(r[1])
                                        r=(boxxyxy[1],ind)
                                    else:
                                        ignore_index.add(ind)
                    return ignore_index

                ###############
                igno=filterlights(zip(res_box.cls.astype(int),res_box.xywh.astype(int),res_box.xyxy.astype(int)))
                for ind,(cls_name, confidence, boxxywh,boxxyxy, mask) in enumerate(iterate):
                    # print(boxxywh,boxxyxy)
                    if ind in igno:
                        confidence =0.1
                    cls_name = class_names[int(cls_name)]
                    
                    # if str(frame_no) not in self.all_detections:
                        # self.all_detections[str(frame_no)]={}
                    # if cls_name not in self.all_detections[str(frame_no)]:
                        # self.all_detections[str(frame_no)][cls_name]=[]
                    xx = boxxyxy[::2]*self.ratio_x
                    yy = boxxyxy[1::2]*self.ratio_y
                    # self.all_detections[str(frame_no)][cls_name].append(["1",[xx[0],yy[0]],[xx[1],yy[1]]])
                    if cls_name in self.fixed_assets :
                        # continue
                        # print(boxxyxy)

                        if boxxyxy[0] < 120 and boxxyxy[1] < 60 and 'oard' in cls_name :
                            continue
                        if boxxyxy[0] <=0 or boxxyxy[1]<=0:
                            continue
                        if boxxyxy[0] >=1024 or boxxyxy[1]>=1024:
                            continue
                         
                        confidence = np.around(confidence*100,decimals=1)
                        detections.append((cls_name,confidence,boxxywh))

                    else:
                    
                        if boxxyxy[0] < 90  or 1024-boxxyxy[2] < 90  or boxxyxy[3] > 557: #

                            temp_box.append(boxxyxy)
                            temp_score.append(confidence)
                            temp_cls.append(cls_name)
                            temp_mask.append(mask)

                # for linear 
                self.classes_queue.put(temp_cls)
                self.boxes_queue.put(temp_box)
                self.scores_queue.put(temp_score)
                self.masks_queue.put(temp_mask)

                # for fixed
                # del masks
                self.detections_queue.put(detections)
                # print(darknet_image.shape)
                self.frame_queue.put(darknet_image)
                # print(detections)
                fps = int(1 / (time.time() - start_time))
                self.fps_queue.put(fps)             
                # darknet.free_image(darknet_image)   
                #       
            # self.all_detections
            # with open(save_folder+"/" + video_name + '_detections.json', 'w') as fp:
            #     json.dump(self.all_detections, fp)
        except Exception as e:
            print("Exception in Inference", e)
            errorVideoFlag = True
            self.quit = True
            utilites.PrintException()

    @staticmethod
    def put_text(image, text, org):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_color = (0, 255, 255)
        thickness = 1
        line_type = cv2.LINE_AA

        cv2.putText(image, text, org, font, font_scale,
                    font_color, thickness, line_type)

    def drawing(self):
        time.sleep(3)
        global errorVideoFlag, fromMetadata
        try:
            # video = self.set_saved_video("Output_"+self.video_name+".MP4")
            if arguments["show_count"]:
                video_count = self.set_saved_video("Output_count_" + str(video_name) + ".avi")
            fc = 0
            fc_list = []
            cv2.namedWindow('Inference', cv2.WINDOW_NORMAL)
            while True:
                # print("drawing")
                start_time = time.time()
                if self.quit:
                    break
                try:
                    detections = self.detections_queue.get(timeout=6)
                    frame = self.frame_queue.get(timeout=6)
                    fc =self.FrameCount_list.get(timeout=6)
                    classes = self.classes_queue.get(timeout=6)
                    boxes = self.boxes_queue.get(timeout=6)
                    scores = self.scores_queue.get(timeout=6)
                    masks = self.masks_queue.get(timeout=6)
                except Exception as ex:
                    # print(ex)
                    if self.frame_queue.qsize() != 0:
                        print('unable to fetch detection list')
                    else:
                        print('Done drawing')
                    break             
                
                speed_d = position_df.iloc[int(fc / self.frames_skipped)]["Speed"]
                location_d = position_df.iloc[int(fc / self.frames_skipped)]["Position"]
                location_data = str(speed_d) + "  " + str(location_d)     
                right_detections = []
                left_detections = []
                left_assets = []
                right_assets = []
                if arguments["plant_anomaly"]:
                    valid_plant_flag = self.valid_plants(classes, boxes, frame)
                else:
                    valid_plant_flag = False
                try:
                    # print(classes)
                    self.linear_annotations[fc]={}
                    for i in range(len(boxes)):
                        asset_name = classes[i]
                        if asset_name in arguments["LINEAR_HYPER_PARAMETERS"].keys():
                            if scores[i] < float(arguments["LINEAR_HYPER_PARAMETERS"][asset_name]):
                                continue
                        else:
                            if scores[i] < arguments["linear_accuracy_threshold"]:
                                continue
                        boxes[i][::2]=boxes[i][::2]*self.ratio_x
                        boxes[i][1::2]=boxes[i][1::2]*self.ratio_y
                        [x1, y1, x2, y2] = np.array(boxes[i],dtype=int)
                        pixel_mean = masks[i] ## it is already mean
                        if asset_name not in  self.linear_counter:
                            self.linear_counter[asset_name] = {"LEFT":[0,fc],"RIGHT":[1,fc],"c":1}

                        if asset_name not in self.linear_annotations[fc]:
                                self.linear_annotations[fc][asset_name]=[]

                        if (x1 + x2) / 2 > frame.shape[1] / 2 or asset_name in ("Cracks","Pothole"):  # Right assets
                        # print(asset_name, valid_plant_flag)

                            if fc-self.linear_counter[asset_name]["RIGHT"][1] > 50:
                                self.linear_counter[asset_name]["c"]+=1
                                self.linear_counter[asset_name]["RIGHT"][0]=self.linear_counter[asset_name]["c"]
                                

                            self.linear_counter[asset_name]["RIGHT"][1]=fc
 
                            self.linear_annotations[fc][asset_name].append([f"{self.linear_counter[asset_name]['RIGHT'][0]}",[int(x1),int(y1)],[int(x2),int(y2)]])


                            if asset_name == "Plants" and not valid_plant_flag:
                                right_detections.append((asset_name, boxes[i], scores[i],pixel_mean))
                            else:
                                right_detections.append((asset_name, boxes[i], scores[i],pixel_mean))
                                right_assets.append(asset_name)
                            cv2.putText(frame, f"{asset_name}:{self.linear_counter[asset_name]['RIGHT'][0]}", (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_TRIPLEX, 1,
                                    (0, 255, 0), 1, cv2.LINE_AA)

                        else:  # Left Assets
                            # if asset_name == "Bad_Kerbs" and scores[i] < 0.9:
                            #     continue

                            if fc-self.linear_counter[asset_name]["LEFT"][1] > 50:
                                self.linear_counter[asset_name]["c"]+=1
                                self.linear_counter[asset_name]["LEFT"][0]=self.linear_counter[asset_name]["c"]
                                

                            self.linear_counter[asset_name]["LEFT"][1]=fc

                            self.linear_annotations[fc][asset_name].append([f"{self.linear_counter[asset_name]['LEFT'][0]}",[int(x1),int(y1)],[int(x2),int(y2)]])

                            left_detections.append((asset_name, boxes[i], scores[i],pixel_mean))
                            left_assets.append(asset_name)
                            cv2.putText(frame, f"{asset_name}:{self.linear_counter[asset_name]['LEFT'][0]}", (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_TRIPLEX, 1,
                                    (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame, str(round(scores[i], 1)), (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_TRIPLEX,
                                1, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)



                    # self.json_out.update()
                    self.detection_df.loc[int(fc / self.frames_skipped)] = [fc, left_assets, right_assets]
                    self.detection_dict[int(fc)] = {"left": left_detections, "right": right_detections}



                except Exception as ex:
                    utilites.PrintException()
                    print("Error in linear: ",ex)

                # print("drawing2")
                try:
                    if int(speed_d)>0:
                        new_dets, flag, counter_flag = self.check_detections(detections, fc)
                        for obj in new_dets.keys():
                            flag, counter_flag = self.tracker_predict(new_dets, obj, flag, counter_flag)
                            # print('counter_flag',counter_flag)
                        self.update_tracker()
                        skip = False
                    else:
                        skip = True
                except Exception as err:
                    traceback.print_exc()
                    print("error in speed",err)
                    skip = True
                # print("drawing3")

                if frame is not None:# print('made age greater than 60 **************')
                    if not self.show_count:
                        #print(frame.shape)
                        if len(detections) > 0 and not skip:
                            frame, json_data = darknet.draw_boxes(fc, detections, frame, flag, counter_flag,
                                                                self.ratio_x, self.ratio_y)
                            self.json_out.update(json_data)
                            # print(json_data)
                        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # image = frame
                        self.put_text(image=image, text=location_data, org=(700, 70))
                        
                        text = "[meta]" if fromMetadata else "[ocr]"
                        
                        self.put_text(image, text, (1100, 70))
                        fps_d = int(1 / (time.time() - start_time))
                        image=cv2.resize(image,(1280, 720))
                        # video.write(image)

                    else:
                        # cv2.circle(frame, (int(self.active_trackers['Signboard_Caution_Board']), 100), 5, (0, 255, 0), -1)
                        # cv2.circle(frame, (int(self.right_gate['Signboard_Hazard_board']), 100), 5, (0, 255, 0), -1)
                        image = frame
                        image, json_data = darknet.draw_boxes(fc, detections, image, flag, counter_flag,
                                                            self.ratio_x, self.ratio_y)
                        # print('json_data',json_data)
                        self.json_out.update(json_data)
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        image = np.concatenate((image, self.count_image(fc)), axis=1)
                        if arguments['save_count_video']:
                            image=cv2.resize(image,(1280, 720))
                            video_count.write(image)

                    fps_d = f"fps_d: {fps_d}"
                    self.put_text(image, str(fc), (500, 70))
                    fps_i = self.fps_queue.get()
                    fps_i = f"fps_i: {fps_i}"
                    self.put_text(image, fps_i, (500, 90))

                    percentage_completed = fc / self.totalFrames
                    #print("percentage_completed", percentage_completed)
                    image[-7:-3, :(int(percentage_completed * image.shape[1])), 2] = 255
                    # Updating the progress
                    # if int(percentage_completed * 100) % 5 == 0:
                        # print(percentage_completed * 100)
                        #socket_obj.updateProgress(int(percentage_completed * 100))
                        # pass

                    cv2.imshow('Inference', image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.quit = True
                        break
                # del detections
                # fc += self.frames_skipped
                
                fc_list.append(fc)         

                
        except Exception as e:
            print("Exception in Drawing", e)
            errorVideoFlag = True
            self.quit = True
            traceback.print_exc()
            utilites.PrintException()
            cv2.destroyAllWindows()
            if arguments["show_count"]:
                video_count.release()
            # video.release()
        filter_weak_detections(self.json_out)
        if not self.quit:
            
            with open('csv_and_jsons/' + video_name + '.json', 'w') as fp:
                json.dump(self.active_trackers, fp)
            with open('csv_and_jsons/' + video_name + '_annotation_old.json', 'w') as fp:
                json.dump(self.json_out, fp)

            for x in self.linear_annotations:
                if x not in self.json_out:
                    self.json_out[x]={}
                self.json_out[x].update(self.linear_annotations[x])
            for x in self.linear_counter:
                self.json_out[x]=self.linear_counter[x]["c"]+1
            for x in self.active_trackers:
                try:
                    self.json_out[x] = max([int(i) for i in self.active_trackers[x].keys()])+1
                except:
                    self.json_out[x] = 9900
            for asset_name in arguments["LINEAR_HYPER_PARAMETERS"].keys():
                for side in ("_Start","_End"):
                    self.json_out[asset_name+side]=0
            with open(save_folder+"/" + video_name + '_annotation.json', 'w') as fp:
                json.dump(self.json_out, fp)


        # if arguments["show_count"]:
        #     video_count.release()
        cv2.destroyAllWindows()
        print('done drawing')
        # video.release()
        return self.active_trackers ,self.detection_df


if __name__ == '__main__':
    
    # db_conn = utilites.get_db_connection2() # connecting to database
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--video", required=True,
                    help="path to input video")

    ap.add_argument( "--vehicle", required=False,default="TEST",
                    help="vehicle name")

    ap.add_argument( "--vname", required=False, default="-",
                    help="video name")
   

    args = vars(ap.parse_args())
    arguments = config.getArgs()
    vehicle_name = args["vehicle"]
    
    global video_id, fromMetadata
    # video_id = args["video_id"]
    video_path = args["video"]
    TEST = True if vehicle_name == 'TEST' else False
    os.system(f"echo -n '{video_path},{str(datetime.now())}', >> log.csv")
    # vehicle_name = os.path.dirname(video_path).split("/")[-1]
    # if len(vehicle_name):
    #     vehicle_name = "vehicle_"+vehicle_name
    # else:
    #     vehicle_name = "client1"
    video_name = os.path.basename(video_path).split(".")[0]



    weights = arguments["weightPath"]

    variable_obj.algorithm = "fixed" 
    variable_obj.video_path = video_path
    variable_obj.video_id = -1

    # socket_obj = SocketClass(video_id)
    # socket_obj.connect_ui()
    slack_obj = SlackLogger()

    video_name = os.path.basename(video_path).split(".")[0]

    ####################### Timer for total processing ########################
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    data = {"Frame":[],"Position":[],"Speed":[],"Time":[],
        "video_name": args['vname'],
        "region": "riyadh",
        "date": str(date.today()),
        "road": "Expressway",
        "vehicle": vehicle_name,
        "distance_covered": 0,
        "distance_wasted": 0,
        "error_message": "" if noError else position_df
    }
    try:
        video_time = int(totalFrames / video_fps)
    except Exception as e:
        error_message = "Error in the video unable to play.."
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer d96a6eec4931f19a52b3d30574c364a873a9e662a674fef9c7964d8f3bacc795'
            }
        if not TEST:
            pass
            # response = requests.post('https://motl-api.seekright.com/api/video/', headers=headers, json=data)
            # print(response.content)
        # socket_obj.addNotification(message=error_message, status=0)
        # slack_obj.postMessage(video_name=video_name, algo_type=variable_obj.algorithm, text=error_message, status=0)
        # utilites.PrintException()
        # print("Exception ", e)
        # socket_obj.videoComplete()
        # try:
        #     src = video_path
        #     dst = arguments['error_video_path']
        #     shutil.copy2(src, dst)
        # except Exception:
        #     utilites.PrintException()
        # try:
        #     pass
        #     # socket_obj.disconnect_ui()
        # except:
        #     utilites.PrintException()
        os.system(f"echo  {str(datetime.now())},{data['error_message']} >> log.csv")
        cap.release()
        sys.exit()
    # exit()
    # signal.signal(signal.SIGALRM, socket_obj.video_process_timer)
    # signal.alarm(int(video_time * 5))  # seconds
    # print(int(video_time * 1.5))
    # signal.alarm(20)  # seconds
    # cap.release()
    ############################# Code Ends #################################

    global process_quit
    process_quit = False
    position_df, noError = get_gps(video_path)
    data["error_message"]= "" if noError else position_df
    print("GPS text noError", noError)
    # ##logger_obj.logger.info(f"GPS text noError {noError}")
    fromMetadata = True
    ##############################################################

    # [x for x in video_path.split('/') if 'ehicle' in x][0].replace("vehicle-","")
    # vehicle_name=""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer d96a6eec4931f19a52b3d30574c364a873a9e662a674fef9c7964d8f3bacc795'
    }


    latlngs =set(['N0.00000E0.00000'])
    # print(position_df)
    if noError:
        
        vname = os.path.basename(video_path).split(".")[0].replace("_","")
        vdate = datetime(int(vname[:4]), int(vname[4:6]), int(vname[6:8]), int(vname[8:10]), int(vname[10:12]),
            int(vname[12:14]))
        for i,j,k in zip(position_df["Frame"],position_df["Position"],position_df["Speed"]):
            if j not in latlngs:
                if len(latlngs)>1:
                    D=geo.calculateDistance(j,data["Position"][-1])
                    if D > 0.2:
                        data["distance_wasted"]+=D
                    elif D<0.6:
                        data["distance_covered"]+=D
                data["date"]=str(vdate.date())
                data["Frame"].append(i)
                data["Position"].append(j)
                data["Speed"].append(k)
                data["Time"].append(str(vdate+timedelta(seconds=i//video_fps)))
                latlngs.add(j)
    print("O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O")
    print("video_path : ",video_path)
    print("Total F    : ",totalFrames)
    if len(data["Position"]):
        print("Position   : ",data["Position"][-1],"-",data["Position"][-1],":",len(data["Position"]))
        print("Time       : ",data["Time"][-1],"-",data["Time"][-1],":",len(data["Time"]))
    else:
        print("<<<<<<<<<<<<<<<<<<<<<< Empty DataFrame >>>>>>>>>>>>>>>>>>>>>>>>>")
    print("covered    : ",data["distance_covered"])
    print("wasted     : ",data["distance_wasted"])
    print("vehicle    : ",data["vehicle"])
    print("error      : ",data["error_message"])
    print("O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O--+--O")
    if  not TEST:
        try:
            pass
            # response = requests.post('https://motl-api.seekright.com/api/video/', headers=headers, json=data)
            # print(response.content)
        except Exception as ex:
            print("####### cannot upload site statistics #######")
            print(ex)
            print("#############################################")

 
    ################################################################

    # exit()
    if noError:

        network  = YOLO(weights)
        class_names =  network.names
        if not os.path.exists("classes.json"):
            with open("classes.json","w") as f:
                f.write(json.dumps(class_names, indent=4))
        else:
            with open("classes.json","r") as f:
                class_names = {int(i):j for i,j in eval(f.read()).items()}
        # save_folder=os.path.join(os.path.dirname(video_path),video_name)
        save_folder = os.path.join(os.path.dirname(video_path),video_name)
        os.makedirs(save_folder,exist_ok=True)
        print(class_names)
        # exit()

        yolo = Detections(cap)
        yolo.network=network
        
        pool = ThreadPool(processes=3)
        t1 = pool.apply_async(yolo.video_capture)
        t2 = pool.apply_async(yolo.inference)
        t3 = pool.apply_async(yolo.drawing)

        
        try:
            detections_df,detections_df_lin = t3.get()
            pool.close()
            pool.join()
            
        except Exception as e:
            process_quit = True
            print("Exception in main threads",e)

   
    os.system(f"echo  {str(datetime.now())},{data['error_message']} >> log.csv")
    print(f"{video_name} is finished !!!")
    sys.exit()
        # signal.alarm(0)
        # exit()

    #     if not process_quit and socket_obj.save_flag:
    #         no_of_usecases = 0
    #         no_of_anomalies = 0
    #         try:
    #             Dfs, no_of_usecases = data_pocessing(detections_df, position_df, video_name)
    #             utilites.upload_to_dashboard(os.path.basename(video_path))


    #             Dfs_lin, no_of_anomalies_lin = dataProcessing(detections_df_lin, position_df, video_path, yolo.detection_dict,
    #                 yolo.class_names)
    #         except Exception:
    #             Dfs = []
    #             utilites.PrintException()
    #             print("Exception in data processing")

    #         test_dir_name = arguments['save_image_path'] + video_name
    #         if not os.path.exists(test_dir_name):
    #             os.mkdir(test_dir_name)
    #         master_dir_name = arguments['save_image_path'] + video_name + '_master'
    #         if not os.path.exists(master_dir_name):
    #             os.mkdir(master_dir_name)

    #         try:
    #             no_of_anomalies = generate_frames(video_name, Dfs, master_dir_name, test_dir_name,
    #                                               position_df.iloc[position_df.shape[0] - 1]['Frame'],
    #                                               skip_factor=arguments['frames_skipped'])

    #         except Exception as e:
    #             utilites.PrintException()
    #             print("error in compare",e)

    #         if not errorVideoFlag:
    #             try:
    #                 notification_message = "Video has been processed successfully and anomalies are generated"
    #                 status_code = 1
    #                 # socket_obj.addNotification(message="Video has been processed successfully and anomalies are generated", status=1)
    #             except Exception as e:
    #                 print("Exception Occurred in Notification", e)
    #                 utilites.PrintException()
    #                 # reconnect()
    #     else:

    #         notification_message = "Video has been processed successfully and 0 anomalies are generated"
    #         status_code = 1

    #         utilites.enter_logger("Quit command from UI with discard flag true")
    #         print("Total process Quit")
    # else:
    #     notification_message = "video has issue with GPS data"
    #     print("video has issue with position data")

    #     status_code = 0
    #     # slack_message = text_message
    #     error_message = "GPS Error"
    # try:
    #     # socket_obj.addNotification(message=notification_message, status=status_code)

    #     slack_obj.postMessage(video_name=video_name, algo_type=variable_obj.algorithm,
    #                           text=notification_message, status=status_code)
    # except Exception as e:
    #     print("Exception Occurred in Notification", e)
    #     utilites.PrintException()

    # # socket_obj.videoComplete()
    # if errorVideoFlag:
    #     try:
    #         src = video_path
    #         dst = arguments['error_video_path']
    #         shutil.copy2(src, dst)
    #     except Exception:
    #         utilites.PrintException()

    # # socket_obj.send_final_call()

    # # Removing Video
    # try:
    #     # utilites.remove_video(video_path)
    #     # socket_obj.refresh(100)
    #     time.sleep(1)
    #     # socket_obj.disconnect_ui()
    # except Exception as ex:
    #     utilites.PrintException()
    # 
