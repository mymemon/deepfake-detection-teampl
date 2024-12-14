```
import cv2
import os
import shutil
import random
from PIL import Image
```
# 설정된 경로 및 파라미터.
```
video_path = '/mnt/data/aaqaifqrwn.mp4'  # 입력 비디오 파일 경로
output_folder = '/mnt/data/extracted_frames'   # 추출된 프레임 저장 폴더
train_folder = '/mnt/data/train_frames'   # 학습 데이터 저장 폴더
val_folder = '/mnt/data/val_frames'   # 검증 데이터 저장 폴더
frame_count = 5   # 추출할 프레임 개수
split_ratio = 0.8   # 학습/검증 데이터 분할 비율
```
# 비디오에서 프레임을 추출하는 함수
```
def extract_frames(video_path, output_folder, frame_count=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 총 프레임 수
    interval = max(total_frames // frame_count, 1)   # 프레임 간격 계산

    frame_number = 0
    saved_frames = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_folder, exist_ok=True)   # 출력 폴더 생성

    while cap.isOpened() and saved_frames < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % interval == 0:   # 설정된 간격에 따라 프레임 저장
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{saved_frames}.jpg")
            cv2.imwrite(frame_filename, frame)  # 프레임 이미지를 파일로 저장
            saved_frames += 1

        frame_number += 1

    cap.release()
    return saved_frames   # 저장된 프레임 개수 반환
```
# 프레임 이미지를 리사이즈하는 함수
```
def resize_image(image_path, output_size=(64, 64)):
    with Image.open(image_path) as img:
        img_resized = img.resize(output_size)   # 리사이즈
        img_resized.save(image_path)
```
# 학습 및 검증 데이터를 분할하는 함수
```
def split_data(input_folder, train_folder, val_folder, split_ratio=0.8):
    files = os.listdir(input_folder)
    random.shuffle(files)   # 데이터 셔플

    split_index = int(len(files) * split_ratio)   # 분할 인덱스 계산
    train_files = files[:split_index]
    val_files = files[split_index:]

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for file in train_files:   # 학습 데이터 복사
        shutil.copy(os.path.join(input_folder, file), train_folder)

    for file in val_files:   # 검증 데이터 복사
        shutil.copy(os.path.join(input_folder, file), val_folder)
```
# 프레임 추출 및 리사이즈 수행
```
extracted_frame_count = extract_frames(video_path, output_folder, frame_count)
```
# 리사이즈 작업 수행
```
for frame_file in os.listdir(output_folder):
    resize_image(os.path.join(output_folder, frame_file), output_size=(64, 64))
```


----------------------------------------------------------------------------------------------------------
```
import os
import cv2
import random
import shutil
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from PIL import Image
```

# ------------------------------
# 1. 설정
# ------------------------------
data_folder = 'deepfake_dataset'  # 데이터셋이 있는 기본 폴더
train_folder = 'dataset/train'   # 학습 데이터 저장 폴더
val_folder = 'dataset/val'       # 검증 데이터 저장 폴더
frame_count = 10                 # 각 영상에서 추출할 프레임 개수
split_ratio = 0.8                # 학습:검증 데이터 비율
image_size = (64, 64)            # 리사이즈 이미지 크기
batch_size = 16                  # 배치 크기
num_epochs = 10                  # 학습 에폭 수
learning_rate = 0.001            # 학습률

# ------------------------------
# 2. 데이터 준비 함수
# ------------------------------
def extract_frames_from_videos(data_folder, output_folder, frame_count, image_size):
    """
    각 클래스 폴더('real', 'fake')에서 영상 파일을 읽고 일정 수의 프레임을 추출.
    추출된 프레임은 클래스별 폴더에 저장.
    """
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
    """
    추출된 데이터를 학습 및 검증 데이터로 분할.
    """
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
# 3. 모델 정의 및 학습
# ------------------------------
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    모델 학습 루프.
    """
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

def validate_model(model, val_loader, criterion):
    """
    검증 데이터 평가.
    """
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
# 4. 실행
# ------------------------------
# 데이터 준비
frames_folder = 'processed_frames'
extract_frames_from_videos(data_folder, frames_folder, frame_count, image_size)
split_dataset(frames_folder, train_folder, val_folder, split_ratio)

# 데이터셋 로드
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(train_folder, transform=transform)
val_data = datasets.ImageFolder(val_folder, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# 모델 정의
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 및 검증
train_model(model, train_loader, criterion, optimizer, num_epochs)
validate_model(model, val_loader, criterion)
