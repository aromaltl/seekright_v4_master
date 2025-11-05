import cv2
import glob
import numpy as np
import os
import yaml

def draw_bounding_boxes(image_path, yolo_txt_path, output_image_path):
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Read YOLO txt file
    with open(yolo_txt_path, 'r') as file:
        lines = file.readlines()
    show = False
    # Loop through each line in the YOLO txt file
    for line in lines:
        vv=line.split(" ")[1:]
        class_id = int(line.split(" ")[0])
        color = (0, 255, 255)
        # if "andatory" in classes[class_id] :
        #     # continue
        #     color = (0, 255, 0)

        show=True

        X=np.array(vv[::2],dtype=np.float16)
        Y=np.array(vv[1::2],dtype=np.float16)
        
        # Convert YOLO format to pixel values
        x_min = int(np.min(X) * width)
        y_min = int(np.min(Y) * height)
        x_max = int(np.max(X) * width)
        y_max = int(np.max(Y) * height)

        # Draw bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Put class label
        cv2.putText(image, classes[int(class_id)], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save the image with bounding boxes
    # cv2.imwrite(output_image_path+os.path.basename(image_path), image)
    if show:
        cv2.imshow('out',image[::2,::2])
        cv2.waitKey(0)

# Example usage
path ="/home/sultan/Desktop/training/diversed"#"/home/sultan/Desktop/training/diversed"
imgs=glob.glob(os.path.join(path,'**','*.jpeg'),recursive=True)
yamlf =  [x for x in glob.glob(os.path.join(path,'**','*.yml'),recursive=True) if 'train_test_data.yml' in x][0]
with open(yamlf, 'r') as file:
    classes = yaml.safe_load(file)["names"]
print(classes)
for x in imgs[::-1]:
    print(x)
    # if 'train/' not in x:
    #     continue

    draw_bounding_boxes(x, x.replace("/images/","/labels/").replace(".jpeg",".txt"), "verify/")
