import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False, action='store_true')
parser.add_argument('--play_video', help="True/False", default=False, action='store_true')
parser.add_argument('--image', help="True/False", default=False, action='store_true')
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True, action='store_true')
args = parser.parse_args()

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return None, None, None, None
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    if img is not None and img.shape:
        height, width, channels = img.shape
        return img, height, width, channels
    else:
        return None, None, None, None


def start_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def start_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

def load_yolo_v8(model_path):
    model = YOLO(model_path)
    return model

def detect_objects_yolo_v8(img, model):
    results = model(img)
    boxes = []
    confs = []
    class_ids = []
    class_names = model.names

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Correct way to access coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1
            conf = float(box.conf)  # Correct way to access confidence
            class_id = int(box.cls)  # Correct way to access class ID
            boxes.append([x, y, w, h])
            confs.append(conf)
            class_ids.append(class_id)

    return boxes, confs, class_ids, class_names  # Return class names




def draw_labels_yolo_v8(boxes, confs, class_ids, class_names, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    car_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            label = str(class_names[class_id])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            if label == "car":
                car_count += 1

    cv2.putText(img, f"Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #cv2.imshow("Image", img)
    return car_count  # Return car count


def image_detect_yolo_v8(img_path, model):
    img, height, width, channels = load_image(img_path)
    if img is None:
        return 0

    boxes, confs, class_ids, class_names = detect_objects_yolo_v8(img, model)
    car_count = draw_labels_yolo_v8(boxes, confs, class_ids, class_names, img)

    return car_count


def webcam_detect_yolo_v8(model):
    cap = start_webcam()
    car_counts =  [] # List to store car counts for each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam frame not available")
            break

        boxes, confs, class_ids, class_names = detect_objects_yolo_v8(frame, model)
        car_count = draw_labels_yolo_v8(boxes, confs, class_ids, class_names, frame)
        car_counts.append(car_count)  # Append to list
        
    cap.release()
    return car_counts  # Return list of car counts


def start_video_yolo_v8(video_path, model):
    cap = start_video(video_path)
    car_counts = [] # List to store car counts for each frame
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return car_counts  # Return empty list

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confs, class_ids, class_names = detect_objects_yolo_v8(frame, model)
        car_count = draw_labels_yolo_v8(boxes, confs, class_ids, class_names, frame)
        car_counts.append(car_count)  # Append to list

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
    cap.release()
    return car_counts  # Return list of car counts


if __name__ == '__main__':
    #... (command-line argument parsing remains the same)

    car_counts_all = {} # Initialize list to store car counts

    image_count = 0  # Initialize image count

    # --- Process command-line arguments FIRST ---
    if args.webcam:
        #... (webcam processing)
        pass
    if args.play_video:
        #... (video processing)
        pass
    if args.image:
        #... (single image processing)p
        pass

    # --- THEN process the directory ---
    directory = '/Users/groy/Downloads/Project/images/10k/train'  # Your directory path

    if os.path.exists(directory) and os.path.isdir(directory):
        print(f"Processing images in directory: {directory}")  # Indicate directory processing
        iter = 0
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            iter +=1
            if os.path.isfile(f) and (iter<2001):
                print(f"Processing image: {f}")  # Print the filename being processed
                print(f"Processing image number {image_count} ...")
                image_count += 1
                model = load_yolo_v8('yolov8n.pt')  # Load the model *inside* the loop
                img, _, _, _ = load_image(f)
                if img is not None:
                    car_count = image_detect_yolo_v8(f, model)  # Pass the full path
                    # car_counts_all.append(car_count)
                    car_counts_all[filename] = car_count
                    # cv2.waitKey(0)  # Wait after each image (optional)
                   # cv2.destroyAllWindows()  # Close windows after each image
            else: 
                break
        # print(car_counts_all)
        df = pd.DataFrame(list(car_counts_all.items()), columns=['name', 'Value'])

        df.to_csv('Car_Count.csv')

    else:
        print(f"Error: Directory '{directory}' not found or is not a directory.")

    print("Car Counts (All):", car_counts_all)  # Print all car counts