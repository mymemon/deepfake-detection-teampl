# deepfake-detection-teampl
import cv2
import os

def extract_frames(video_path, output_folder, frame_count=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // frame_count, 1)

    frame_number = 0
    saved_frames = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened() and saved_frames < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % interval == 0:
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{saved_frames}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        frame_number += 1

    cap.release()
    print(f"Extracted {saved_frames} frames from {video_path}.")
    
video_path = 'FaceForensics++/original_sequences/youtube/video.mp4'  
output_folder = 'FaceForensics++/frames/original_sequences/youtube' 
extract_frames(video_path, output_folder, frame_count=5)


extract_frames(video_path, output_folder, frame_count=5)

from PIL import Image

def resize_image(image_path, output_size=(64, 64)):
    with Image.open(image_path) as img:
        img_resized = img.resize(output_size)
        img_resized.save(image_path)

frame_folder = 'FaceForensics++/frames/original_sequences/youtube/video1'
frame_paths = get_frame_paths(frame_folder)

resize_image(frame_path)

from PIL import Image

def resize_image(image_path, output_size=(64, 64)):
    with Image.open(image_path) as img:
        img_resized = img.resize(output_size)
        img_resized.save(image_path)

frame_path = 'path/to/your/frame.jpg' 
resize_image(frame_path)

import shutil
import random
import os

def split_data(input_folder, train_folder, val_folder, split_ratio=0.8):
    files = os.listdir(input_folder)
    random.shuffle(files)

    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for file in train_files:
        shutil.move(os.path.join(input_folder, file), train_folder)

    for file in val_files:
        shutil.move(os.path.join(input_folder, file), val_folder)


input_folder = 'path/to/your/frame.jpg'     
train_folder = "C:\Users\user\Downloads\models.py"       
val_folder = "C:\Users\user\Downloads\transform.py"         
split_data(input_folder, train_folder, val_folder)
