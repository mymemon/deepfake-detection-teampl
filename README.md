import cv2
import os
import shutil
import random
from PIL import Image

# 설정된 경로 및 파라미터
video_path = '/mnt/data/aaqaifqrwn.mp4'  
output_folder = '/mnt/data/extracted_frames'  
train_folder = '/mnt/data/train_frames'  
val_folder = '/mnt/data/val_frames'  
frame_count = 5 
split_ratio = 0.8 

# 비디오에서 프레임을 추출하는 함수
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

# 프레임 이미지를 리사이즈하는 함수
def resize_image(image_path, output_size=(64, 64)):
    with Image.open(image_path) as img:
        img_resized = img.resize(output_size)
        img_resized.save(image_path)

# 학습 및 검증 데이터를 분할하는 함수
def split_data(input_folder, train_folder, val_folder, split_ratio=0.8):
    files = os.listdir(input_folder)
    random.shuffle(files)

    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(input_folder, file), train_folder)

    for file in val_files:
        shutil.copy(os.path.join(input_folder, file), val_folder)

# 프레임 추출 및 리사이즈 수행
extracted_frame_count = extract_frames(video_path, output_folder, frame_count)

# 리사이즈 작업 수행
for frame_file in os.listdir(output_folder):
    resize_image(os.path.join(output_folder, frame_file), output_size=(64, 64))

# 학습 및 검증 데이터 분할
split_data(output_folder, train_folder, val_folder, split_ratio)

# 결과 출력
print(f"Extracted frames: {extracted_frame_count}")
print(f"Training frames: {len(os.listdir(train_folder))}")
print(f"Validation frames: {len(os.listdir(val_folder))}")
