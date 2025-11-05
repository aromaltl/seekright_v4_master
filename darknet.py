#!/usr/bin/env python3

"""
Python 3 wrapper for identifying objects in images

Running the script requires opencv-python to be installed (`pip install opencv-python`)
Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)
Use pip3 instead of pip on some systems to be sure to install modules for python3
"""

# from ctypes import *
import math
import random
import os
import traceback
import cv2
import numpy as np
from colors import convert_color
import json




def bbox2points(bbox, ratio_x=1, ratio_y=1):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)) * ratio_x)
    xmax = int(round(x + (w / 2)) * ratio_x)
    ymin = int(round(y - (h / 2)) * ratio_y)
    ymax = int(round(y + (h / 2)) * ratio_y)
    return xmin, ymin, xmax, ymax

def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}



def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))







def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded

# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(detections, overlap_thresh):
    boxes = []
    for detection in detections:
        _, _, _, (x, y, w, h) = detection
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append(np.array([x1, y1, x2, y2]))
    boxes_array = np.array(boxes)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
    return [detections[i] for i in pick]

def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions


def remove_negatives_faster(detections, class_names, num):
    """
    Faster version of remove_negatives (very useful when using yolo9000)
    """
    predictions = []
    for j in range(num):
        if detections[j].best_class_idx == -1:
            continue
        name = class_names[detections[j].best_class_idx]
        bbox = detections[j].bbox
        bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
        predictions.append((name, detections[j].prob[detections[j].best_class_idx], bbox))
    return predictions

def draw_bounding_box(image,
                      box,
                      labels=[],  # xmin,ymin,xmax,ymax
                      color='red',
                      font_face=cv2.FONT_HERSHEY_TRIPLEX,
                      font_size=1.5,
                      font_weight=3,
                      font_color='white',
                      text_padding=3,
                      border_thickness=3
                      ):
    color = convert_color(color)
    font_color = convert_color(font_color)

    box = [int(x) for x in box]
    xmin, ymin, xmax, ymax = box

    # draw bounding box
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, border_thickness)

    text_dims = [cv2.getTextSize(x, font_face, font_size, font_weight) for x in labels]
    total_label_height = sum([x[0][1] for x in text_dims]) + sum([x[1] for x in text_dims]) + (
            text_padding * len(labels) * 1)

    border_offset = (border_thickness // 2) if border_thickness % 2 == 0 else (border_thickness // 2) + 1

    label_xmin = xmin - border_offset
    label_ymin = ymin - total_label_height - border_offset
    if label_ymin < 0:
        label_ymin = ymax + border_offset

    for i, label in enumerate(labels):
        text_width = text_dims[i][0][0]
        text_height = text_dims[i][0][1]
        text_baseline = text_dims[i][1]
        label_height = text_dims[i][0][1] + text_dims[i][1] + text_padding

        label_xmax = label_xmin + text_width + text_padding
        label_ymax = label_ymin + label_height

        # cv2.rectangle(image, (label_xmin, label_ymin), (label_xmax, label_ymax), color, cv2.FILLED)
        cv2.putText(image, label, (label_xmin + text_padding, label_ymin + text_height + text_padding), font_face,
                    font_size, color, font_weight, cv2.LINE_AA)

        label_ymin = label_ymax


def draw_boxes(frame_count,detections, image,flag,c_flag,ratio_x  ,ratio_y ):
    i  = 0
    dets = {}
    dets[frame_count] = {}
    # print(detections)
    # print(image.shape,"#@###")
    for label, confidence, bbox in (detections):
        box = bbox2points(bbox,ratio_x, ratio_y)
        # cv2.rectangle(image, (left, top), (right, bottom), , 1)
        # if label != "Water_Stagnation":
        #
        try:
            if len(dets[frame_count][label]) > 0:
                pass
        except Exception:
            dets[frame_count][label] = []
        if c_flag[i] == None:
            tracker_id = ''
        else:
            tracker_id = str(c_flag[i])
        if flag[i] == 'l':
            draw_bounding_box(image, box, labels=[label, tracker_id, str(float(confidence))], color='orange')
            i += 1
            dets[frame_count][label].append([tracker_id, [box[0], box[1]], [box[2], box[3]]])
        elif flag[i] == 'r':
            draw_bounding_box(image, box, labels=[label, tracker_id, str(float(confidence))], color='orange')
            i += 1
            dets[frame_count][label].append([tracker_id, [box[0], box[1]], [box[2], box[3]]])
        elif flag[i] == 'c':
            draw_bounding_box(image, box, labels=[label, tracker_id, str(float(confidence))], color='green')
            i += 1
            dets[frame_count][label].append([tracker_id, [box[0], box[1]], [box[2], box[3]]])
        elif flag[i] == 'i':
            draw_bounding_box(image, box, labels=[label, tracker_id, str(float(confidence))], color='blue')
            i += 1
            # dets[frame_count][label] = [tracker_id,[box[0],box[1]],[box[2],box[3]]]
    return image, dets

    # writing dictionary


def draw_boxes_night(detections, image, flag, ratio_x=1, ratio_y=1):
    from opencv_draw_annotation import draw_bounding_box
    i = 0
    for label, confidence, bbox in detections:
        box = bbox2points(bbox)
        # cv2.rectangle(image, (left, top), (right, bottom), , 1)
        if flag[i] == 'l':
            draw_bounding_box(image, box, labels=[label, str(float(confidence))], color='green')
            i += 1
        elif flag[i] == 'r':
            draw_bounding_box(image, box, labels=[label, str(float(confidence))], color='purple')
            i += 1
        elif flag[i] == 'c':
            draw_bounding_box(image, box, labels=[label, str(float(confidence))], color='orange')
            i += 1
        elif flag[i] == 'i':
            draw_bounding_box(image, box, labels=[label, str(float(confidence))], color='blue')
            i += 1
    return image



def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded


# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(detections, overlap_thresh=0.3):
    boxes = []
    for detection in detections:
        _, _, (x, y, w, h) = detection
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append(np.array([x1, y1, x2, y2]))
    boxes_array = np.array(boxes)
    # print("++++++++++detections")
    # print(detections)
    # print("boxes array")
    # print(boxes_array)
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    # print("++++++++++x1+++++++++")
    # print(x1)
    # raise "stop"
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
    return [detections[i] for i in pick]


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], bbox))
    return predictions


def remove_negatives_faster(detections, class_names, num):
    """
    Faster version of remove_negatives (very useful when using yolo9000)
    """
    predictions = []
    for j in range(num):
        if detections[j].best_class_idx == -1:
            continue
        name = class_names[detections[j].best_class_idx]
        bbox = detections[j].bbox
        bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
        predictions.append((name, detections[j].prob[detections[j].best_class_idx], bbox))
    return predictions


def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
        Returns a list with highest confidence class and their bbox
    """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])


