
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
```
data_folder = 'deepfake_dataset'  # 원본 영상 데이터셋이 저장된 폴더
train_folder = 'dataset/train'   # 학습 데이터가 저장될 폴더
val_folder = 'dataset/val'       # 검증 데이터가 저장될 폴더
frame_count = 10                 # 각 영상에서 추출할 프레임 개수
split_ratio = 0.8                # 학습 데이터와 검증 데이터 비율 (80:20)
image_size = (64, 64)            # 프레임 리사이즈 크기 (64x64 픽셀)
batch_size = 16                  # 학습 배치 크기
num_epochs = 10                  # 학습 반복 횟수 (Epochs)
learning_rate = 0.001            # 옵티마이저 학습률
```

# ------------------------------
# 2. 데이터 준비 함수
# ------------------------------
```
def extract_frames_from_videos(data_folder, output_folder, frame_count, image_size):
    """
    각 클래스 폴더('real', 'fake')에서 영상 파일을 읽고 일정 수의 프레임을 추출.
    추출된 프레임은 클래스별 폴더에 저장.
    """
    os.makedirs(output_folder, exist_ok=True)   # 결과 저장 폴더 생성
    
    for class_name in os.listdir(data_folder):  # 'real', 'fake' 순회
        class_path = os.path.join(data_folder, class_name)
        output_class_folder = os.path.join(output_folder, class_name)   # 클래스 별 저장 폴더 생성
        os.makedirs(output_class_folder, exist_ok=True)
        
        for video_file in os.listdir(class_path):  # 각 클래스의 영상 파일 처리
            video_path = os.path.join(class_path, video_file)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(total_frames // frame_count, 1)  # 프레임 간격 계산

            frame_number = 0
            saved_frames = 0
            video_name = os.path.splitext(video_file)[0]

# 비디오에서 지정된 개수의 프레임을 추출
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
    for class_name in os.listdir(input_folder):   # 클래스 순회 ('real', 'fake')
        class_path = os.path.join(input_folder, class_name)
        train_class_folder = os.path.join(train_folder, class_name)  # 학습 데이터 저장 폴더
        val_class_folder = os.path.join(val_folder, class_name)  # 검증 데이터 저장 폴더

        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(val_class_folder, exist_ok=True)

        files = os.listdir(class_path)
        random.shuffle(files)

        split_index = int(len(files) * split_ratio)
        train_files = files[:split_index]
        val_files = files[split_index:]

# 학습 데이터 복사
        for file in train_files:
            shutil.copy(os.path.join(class_path, file), train_class_folder)

# 검증 데이터 복사
        for file in val_files:
            shutil.copy(os.path.join(class_path, file), val_class_folder)
```

# ------------------------------
# 3. 모델 정의 및 학습
# ------------------------------
```
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    모델 학습 루프.
    """
    model.train()   # 모델 학습 모드 설정
    for epoch in range(num_epochs):  
        running_loss = 0.0
        for images, labels in train_loader:   # 배치 단위로 데이터 로드
            optimizer.zero_grad()  # 기울기 초기화
            outputs = model(images)  # 모델에 입력 데이터 전달
            loss = criterion(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파로 기울기 계산
            optimizer.step()  # 파라미터 업데이트
            running_loss += loss.item() * images.size(0)  # 손실 누적

          # 에폭 별 평균 손실 출력
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

def validate_model(model, val_loader, criterion):
    """
    검증 데이터 평가.
    """
    model.eval()   # 모델 평가 모드 설정
    running_loss = 0.0
    correct = 0
    with torch.no_grad():   # 평가 시에는 기울기 계산하지 않음
        for images, labels in val_loader:  
            outputs = model(images)    # 모델에 데이터 전달
            loss = criterion(outputs, labels)   # 손실 계산
            running_loss += loss.item() * images.size(0)   # 손실 누적
            _, preds = torch.max(outputs, 1)   # 예측값 도출
            correct += (preds == labels).sum().item()   # 정확도 계산

# 검증 손실 및 정확도 출력
    val_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
```

# ------------------------------
# 4. 실행
# ------------------------------
## 데이터 준비
```
frames_folder = 'processed_frames'   # 추출된 프레임 저장 폴더
extract_frames_from_videos(data_folder, frames_folder, frame_count, image_size)   # 프레임 추출
split_dataset(frames_folder, train_folder, val_folder, split_ratio)   # 데이터셋 분할

# 데이터셋 로드
transform = transforms.Compose([
    transforms.Resize(image_size),   # 이미지 크기 조정
    transforms.ToTensor(),   # 텐서 변환
])

# 학습 및 검증 데이터셋 생성
train_data = datasets.ImageFolder(train_folder, transform=transform)
val_data = datasets.ImageFolder(val_folder, transform=transform)

# 데이터 로더 생성
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# 모델 정의
model = models.resnet18(pretrained=True)   # 사전 학습된 ResNet18 모델 로드
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))  # 출력 레이어 수정 (클래스 수에 맞게)

criterion = nn.CrossEntropyLoss()   # 손실 함수 정의 (교차 엔트로피)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)   # Adam 옵티마이저 정의

# 학습 및 검증
train_model(model, train_loader, criterion, optimizer, num_epochs)   # 모델 학습
validate_model(model, val_loader, criterion)   # 모델 검증
```
