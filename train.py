# # from ultralytics import YOLO

# # # Load a model
# # model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# # # Train the model
# # results = model.train(data='crack-seg.yaml', epochs=1, imgsz=640)
# from ultralytics import YOLO

# # Load a model
# model = YOLO('/home/groot/aromal/yolov8seg/runs/segment/first4/weights/best.pt')  # pretrained YOLOv8n model

# # Run batched inference on a list of images
# results = model(['/home/groot/aromal/yolov8seg/datasets/seekright/test/5117.jpeg'])  # return a list of Results objects

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk



from ultralytics import YOLO

from ultralytics import settings

# Update a setting
settings.update({'mlflow': True})

# Reset settings to default values
settings.reset()
model = YOLO('yolov8l-seg.pt')  # load a pretrained model (recommended for training)

args={
    'data':'/home/sultan/Desktop/training/diversed/train_test_data.yml', 
    'epochs':45, 
    'imgsz':1024,
    # 'imgsz':320,
    'save_dir':'save',
    'project':"saudi",
    'name':'2nd',
    'scale': 0.1,
    # 'name':'mlflow_trial',
    'save_period': 10,
    'degrees': 8,
    'shear': 0.4,
    'batch': 8,
    'fliplr': 0.0,
    'perspective':0.00023,
    'resume':False,
    'mosaic':0.25,
    'device':0


}



# Train the model
# results = model.train(data='crack-seg.yaml', epochs=1, imgsz=640)
results = model.train(**args)

# task: segment
# mode: train
# model: yolov8n-seg.pt
# data: /home/groot/aromal/project-14-at-2024-03-20-08-43-daee8a40/train_demo.yaml
# epochs: 10
# time: null
# patience: 100
# batch: 16
# imgsz: 640
# save: true
# save_period: -1
# cache: false
# device: null
# workers: 8
# project: null
# name: train2
# exist_ok: false
# pretrained: true
# optimizer: auto
# verbose: true
# seed: 0
# deterministic: true
# single_cls: false
# rect: false
# cos_lr: false
# close_mosaic: 10
# resume: false
# amp: true
# fraction: 1.0
# profile: false
# freeze: null
# multi_scale: false
# overlap_mask: true
# mask_ratio: 4
# dropout: 0.0
# val: true
# split: val
# save_json: false
# save_hybrid: false
# conf: null
# iou: 0.7
# max_det: 300
# half: false
# dnn: false
# plots: true
# source: null
# vid_stride: 1
# stream_buffer: false
# visualize: false
# augment: false
# agnostic_nms: false
# classes: null
# retina_masks: false
# embed: null
# show: false
# save_frames: false
# save_txt: false
# save_conf: false
# save_crop: false
# show_labels: true
# show_conf: true
# show_boxes: true
# line_width: null
# format: torchscript
# keras: false
# optimize: false
# int8: false
# dynamic: false
# simplify: false
# opset: null
# workspace: 4
# nms: false
# lr0: 0.01
# lrf: 0.01
# momentum: 0.937
# weight_decay: 0.0005
# warmup_epochs: 3.0
# warmup_momentum: 0.8
# warmup_bias_lr: 0.1
# box: 7.5
# cls: 0.5
# dfl: 1.5
# pose: 12.0
# kobj: 1.0
# label_smoothing: 0.0
# nbs: 64
# hsv_h: 0.015
# hsv_s: 0.7
# hsv_v: 0.4
# degrees: 0.0
# translate: 0.1
# scale: 0.5
# shear: 0.0
# perspective: 0.0
# flipud: 0.0
# fliplr: 0.5
# mosaic: 1.0
# mixup: 0.0
# copy_paste: 0.0
# auto_augment: randaugment
# erasing: 0.4
# crop_fraction: 1.0
# cfg: null
# tracker: botsort.yaml
# save_dir: runs/segment/train2