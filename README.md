
import os
import cv2
import random
import shutil
import zipfile

# ------------------------------
# 1. 설정
# ------------------------------
zip_file_path = '/content/drive/MyDrive/dfdc_train_part_00.zip'  # 압축 파일 경로
data_folder = 'deepfake_dataset'  # 압축 해제 후 데이터 폴더
frames_folder = 'processed_frames'  # 추출된 프레임 저장 폴더
train_folder = 'dataset/train'  # 학습 데이터 저장 폴더
val_folder = 'dataset/val'  # 검증 데이터 저장 폴더
frame_count = 10  # 각 영상에서 추출할 프레임 수
split_ratio = 0.8  # 학습:검증 데이터 비율
image_size = (64, 64)  # 리사이즈 크기

# ------------------------------
# 2. 데이터 준비 함수
# ------------------------------
def extract_frames_from_videos(data_folder, output_folder, frame_count, image_size):
    os.makedirs(output_folder, exist_ok=True)
    for class_name in os.listdir(data_folder):  # 'real', 'fake' 순회
        class_path = os.path.join(data_folder, class_name)
        output_class_folder = os.path.join(output_folder, class_name)
        os.makedirs(output_class_folder, exist_ok=True)

        for video_file in os.listdir(class_path):  # 각 클래스의 영상 파일 처리
            video_path = os.path.join(class_path, video_file)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(total_frames // frame_count, 1)  # 프레임 간격 계산

            frame_number = 0
            saved_frames = 0
            video_name = os.path.splitext(video_file)[0]

            while cap.isOpened() and saved_frames < frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_number % interval == 0:  # 간격에 따라 프레임 저장
                    frame_filename = os.path.join(output_class_folder, f"{video_name}_frame_{saved_frames}.jpg")
                    resized_frame = cv2.resize(frame, image_size)
                    cv2.imwrite(frame_filename, resized_frame)
                    saved_frames += 1
                frame_number += 1
            cap.release()

def split_dataset(input_folder, train_folder, val_folder, split_ratio):
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        train_class_folder = os.path.join(train_folder, class_name)
        val_class_folder = os.path.join(val_folder, class_name)

        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(val_class_folder, exist_ok=True)

        files = os.listdir(class_path)
        random.shuffle(files)

        split_index = int(len(files) * split_ratio)
        train_files = files[:split_index]
        val_files = files[split_index:]

        for file in train_files:
            shutil.copy(os.path.join(class_path, file), train_class_folder)

        for file in val_files:
            shutil.copy(os.path.join(class_path, file), val_class_folder)

# ------------------------------
# 3. 압축 해제 및 데이터 준비 실행
# ------------------------------
# 압축 해제
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(data_folder)

# 프레임 추출 및 데이터셋 분할
extract_frames_from_videos(data_folder, frames_folder, frame_count, image_size)
split_dataset(frames_folder, train_folder, val_folder, split_ratio)

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# ------------------------------
# 1. 설정
# ------------------------------
batch_size = 16  # 배치 크기
num_epochs = 10  # 학습 에폭 수
learning_rate = 0.001  # 학습률
image_size = (64, 64)  # 이미지 크기

# ------------------------------
# 2. 데이터 로드
# ------------------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(train_folder, transform=transform)
val_data = datasets.ImageFolder(val_folder, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# ------------------------------
# 3. 모델 정의
# ------------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------------------
# 4. 학습 함수
# ------------------------------
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# ------------------------------
# 5. 검증 함수
# ------------------------------
def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

# ------------------------------
# 6. 학습 및 검증 실행
# ------------------------------
train_model(model, train_loader, criterion, optimizer, num_epochs)
validate_model(model, val_loader, criterion)
