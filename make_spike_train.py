import os
import torch
import snntorch.spikegen as spikegen
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from lib_Audio_MNIST import Audio_MNIST_func as audio_mnist

########################################
# AudioDataset 로드 (train, test)
########################################
train_ds = audio_mnist.AudioDataset(
    '/home/yongjin/PycharmProjects/sinabs-dynapcnn/datasets/Audio-MNIST',
    train=True
)
test_ds = audio_mnist.AudioDataset(
    '/home/yongjin/PycharmProjects/sinabs-dynapcnn/datasets/Audio-MNIST',
    train=False
)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

########################################
# 이벤트 데이터 저장 경로
########################################
SAVE_DIR = "/home/yongjin/PycharmProjects/sinabs-dynapcnn/datasets/Audio-MNIST-Spike_train_100"
os.makedirs(SAVE_DIR, exist_ok=True)
########################################
# Min-Max 정규화 함수
########################################
def min_max_normalize(tensor):
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max == t_min:
        return torch.zeros_like(tensor)
    return (tensor - t_min) / (t_max - t_min)


########################################
# Spike Train을 이벤트 데이터 (x, y, t, p)로 변환하는 함수
########################################
def spike_train_to_events(spike_train):
    """
    spike_train: (num_steps, 32, 32) 형태의 텐서
    반환: numpy structured array with fields (t, x, y, p)
    """
    # 텐서를 numpy array로 변환
    spike_train_np = spike_train.numpy()  # shape: (num_steps, 32, 32)
    # np.where는 (t, y, x) 인덱스를 반환함
    t_indices, y_indices, x_indices = np.where(spike_train_np > 0)

    # polarity는 기본적으로 1로 설정 (Audio-MNIST는 polarity 정보가 없으므로)
    p_values = np.ones_like(t_indices, dtype=np.int8)

    # structured array 생성 (필드: t, x, y, p)
    events_dtype = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int32), ("p", np.int8)])
    events = np.zeros(len(t_indices), dtype=events_dtype)
    events["x"] = x_indices
    events["y"] = y_indices
    events["t"] = t_indices
    events["p"] = p_values

    return events


########################################
# (Spike Train 이벤트, Label) 튜플로 저장하는 함수
########################################
def save_spike_train(data_tensor, label, idx, num_steps=300):
    """
    data_tensor: (32, 32) 형태의 정규화된 텐서 (0~1 범위)
    label: 정답 라벨 (0~9)
    idx: 샘플 인덱스
    num_steps: Spike Train 생성할 시간 스텝
    """
    # Poisson Spike Train 생성 → (num_steps, 32, 32)
    spike_train = spikegen.rate(data_tensor, num_steps=num_steps)

    # Spike Train을 이벤트 데이터 (x, y, t, p)로 변환
    events = spike_train_to_events(spike_train)

    # 라벨별 폴더 생성
    label_dir = os.path.join(SAVE_DIR, str(label))
    os.makedirs(label_dir, exist_ok=True)

    # 이벤트 데이터를 함께 저장 (.npy 파일)
    save_path = os.path.join(label_dir, f"spike_{idx}.npy")
    np.save(save_path, events)


########################################
# 학습 데이터셋 변환 후 저장
########################################
print("===== Converting Train Dataset =====")
with tqdm(total=len(train_loader), desc="Processing Train Dataset") as pbar:
    for i, (sample_data, label, file_path) in enumerate(train_loader):
        sample_data = sample_data.squeeze(0)  # (32, 32)
        normalized_data = min_max_normalize(sample_data)
        save_spike_train(normalized_data, label.item(), i)
        pbar.update(1)

print("===== Train Dataset Conversion Done =====")

########################################
# 테스트 데이터셋 변환 후 저장
########################################
print("===== Converting Test Dataset =====")
with tqdm(total=len(test_loader), desc="Processing Test Dataset") as pbar:
    for i, (sample_data, label, file_path) in enumerate(test_loader):
        sample_data = sample_data.squeeze(0)  # (32, 32)
        normalized_data = min_max_normalize(sample_data)
        # 학습 데이터셋과 겹치지 않도록 인덱스 조정
        save_spike_train(normalized_data, label.item(), i + len(train_ds))
        pbar.update(1)

print("===== Test Dataset Conversion Done =====")
