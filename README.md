# deepfake-detection-teampl
import cv2
import os
import shutil
import random

# Paths and parameters
video_path = '/mnt/data/aaqaifqrwn.mp4'  # Uploaded video path
output_folder = '/mnt/data/extracted_frames'  # Folder to save frames
frame_count = 5  # Number of frames to extract
train_folder = '/mnt/data/train_frames'  # Folder for training data
val_folder = '/mnt/data/val_frames'  # Folder for validation data
split_ratio = 0.8  # 80% train, 20% validation

# Function to extract frames from video
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
    return saved_frames

# Function to split data into training and validation folders
def split_data(input_folder, train_folder, val_folder, split_ratio=0.8):
    files = os.listdir(input_folder)
    random.shuffle(files)

    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    os.makedirs(train_folder, exist_ok=
    train_files = files[:split_index]
    val_files = files[split_index:]

    os.makedirs
True)
    os.makedirs(val_folder, exist_ok=
    os.makedirs(val
True)

    


for file in train_files:
        shutil.copy(os.path.join(input_folder, file), train_folder)

    
        shutil.copy(os.path.join(input_folder, file),

        shutil.copy(os.path
for file in val_files:
        shutil.copy(os.path.join(input_folder, file), val_folder)


        shutil.copy(os.path.join(input_folder, file),

        shutil.copy(os.path.join(input_folder,

        shutil.copy(os.path.join


split_data(output_folder, train_folder, val_folder, split

split_data(output_folder, train


len(os.listdir(train_folder)), len(os.listdir(val_folder))
