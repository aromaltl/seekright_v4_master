# # Load a model
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# # Train the model
# # results = model.train(data='crack-seg.yaml', epochs=1, imgsz=640)
# from ultralytics import YOLO

# # Load a model
# model = YOLO('/home/groot/aromal/yolov8seg/runs/segment/first4/weights/best.pt')  # pretrained YOLOv8n model

# # Run batched inference on a list of images
# results = model('/home/groot/Downloads/2024_0517_172348_F.MP4')  # return a list of Results objects

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/home/groot/aromal/yolov8seg/saudi/demo_sa5/weights/best.pt")

# Open the video file
video_path = "/home/groot/Downloads/20240517174348_000000.MP4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame,verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame[::2,::2])

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()