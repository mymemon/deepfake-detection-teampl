import cv2
import os
import shutil
import random
from PIL import Image

# 설정된 경로 및 파라미터.
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



----------------------------------------------------------------------------------------------------------

import os
import cv2
import torch
import random
import shutil
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import shutil

# 설정된 경로 및 파라미터
video_path = 'aaqaifqrwn.mp4'  
output_folder = 'extracted_frames'
train_folder = 'train_frames'
val_folder = 'val_frames'
frame_count = 10
split_ratio = 0.8

# 비디오에서 프레임을 추출하는 함수
def extract_frames(video_path, output_folder, frame_count=10):
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

# Inference Function
def infer(model, image_path):
    model.eval()  # Set model to evaluation mode
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), model.classes[predicted.item()]

# 프레임 추출, 리사이즈, 데이터 분할 수행
extracted_frame_count = extract_frames(video_path, output_folder, frame_count)
for frame_file in os.listdir(output_folder):
    resize_image(os.path.join(output_folder, frame_file), output_size=(64, 64))
split_data(output_folder, train_folder, val_folder, split_ratio)

# 데이터셋 준비
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(train_folder, transform=train_transform)
val_data = datasets.ImageFolder(val_folder, transform=train_transform)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

# 간단한 모델
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
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

# 검증 함수
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

# 모델 학습 및 검증 실행
train_model(model, train_loader, criterion, optimizer, num_epochs=5)
validate_model(model, val_loader, criterion)

# 학습 및 검증 데이터 분할
split_data(output_folder, train_folder, val_folder, split_ratio)

# 클래스별 데이터 분류 함수
def organize_data(input_folder, train_folder, val_folder, classes):
    for cls in classes:
        os.makedirs(os.path.join(train_folder, cls), exist_ok=True)
        os.makedirs(os.path.join(val_folder, cls), exist_ok=True)

        # 클래스에 따라 파일 이동
        class_files = [f for f in os.listdir(input_folder) if cls in f]
        split_index = int(len(class_files) * split_ratio)
        train_files = class_files[:split_index]
        val_files = class_files[split_index:]

        for file in train_files:
            shutil.move(os.path.join(input_folder, file), os.path.join(train_folder, cls))

        for file in val_files:
            shutil.move(os.path.join(input_folder, file), os.path.join(val_folder, cls))
            
# 클래스 이름 정의
classes = ['real', 'fake']

# 데이터 정리 실행
organize_data(output_folder, train_folder, val_folder, classes)

train_data = datasets.ImageFolder(train_folder, transform=train_transform)
val_data = datasets.ImageFolder(val_folder, transform=train_transform)


# 결과 출력
extracted_frame_count, len(os.listdir(train_folder)), len(os.listdir(val_folder))
