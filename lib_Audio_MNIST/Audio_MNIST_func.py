import os
import librosa
import librosa.display
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def wav2melSpec(AUDIO_PATH):
    audio, sr = librosa.load(AUDIO_PATH)
    return librosa.feature.melspectrogram(y=audio, sr=sr)

def imgSpec(ms_feature):
    fig, ax = plt.subplots()
    ms_dB = librosa.power_to_db(ms_feature, ref=np.max)
    print(ms_feature.shape)
    img = librosa.display.specshow(ms_dB, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()

def get_audio_info(path, show_melspec=False, label=None):
    spec = wav2melSpec(path)
    if label is not None:
        print("Label:", label)
    if show_melspec is not False:
        imgSpec(spec)

class AudioDataset(Dataset):
    """
    PyTorch Dataset 클래스: 오디오 데이터를 로드하여 학습/테스트용 Mel Spectrogram 변환 데이터를 제공하는 데이터셋.

    Args:
        path (str): 오디오 파일이 저장된 폴더 경로
        feature_transform (callable, optional): 오디오 데이터를 변환하는 함수 (예: Mel Spectrogram 변환)
        label_transform (callable, optional): 라벨 데이터를 변환하는 함수
        train (bool): True이면 학습 데이터, False이면 테스트 데이터 로드
        train_size (float): 학습 데이터의 비율 (기본값: 80%)
    """

    def __init__(self, path, train=True, train_size=0.80):
        self.path = path  # 오디오 파일이 저장된 폴더 경로
        self.file_list = []  # 오디오 파일 경로 리스트
        self.label_list = []  # 각 파일의 라벨 리스트

        # 디렉토리 탐색하여 파일 목록 및 라벨 리스트 생성
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                if filename[-3:] != "txt":  # `.txt` 파일 제외 (오디오 파일만 사용)
                    self.file_list.append(os.path.join(dirname, filename))  # 파일 경로 저장
                    self.label_list.append(int(filename[0]))  # 파일명의 첫 글자를 정수로 변환하여 라벨로 사용

        total_len = len(self.file_list)  # 전체 데이터 개수

        # 데이터셋을 학습(train)과 테스트(test)로 분할
        if train:
            # 학습 데이터 (train_size 비율만큼 사용, 기본값: 80%)
            self.file_list, self.label_list = self.file_list[:int(train_size * total_len)], self.label_list[:int(
                train_size * total_len)]
        else:
            # 테스트 데이터 (나머지 20%)
            self.file_list, self.label_list = self.file_list[int(train_size * total_len):], self.label_list[
                                                                                            int(train_size * total_len):]

    def __getitem__(self, idx):
        """
        데이터셋에서 `idx`에 해당하는 샘플을 가져오는 메서드.

        Args:
            idx (int): 가져올 데이터의 인덱스

        Returns:
            spec (Tensor): Mel Spectrogram 변환된 오디오 데이터
            label (int): 해당 오디오 데이터의 라벨
            file_path (str): 해당 오디오 파일의 경로
        """
        try:
            spec = wav2melSpec(self.file_list[idx])  # 오디오 파일을 Mel Spectrogram으로 변환
            spec_db = librosa.power_to_db(spec, ref=np.max)  # dB 변환
            mel_spec = torch.tensor(spec_db, dtype=torch.float32)
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
            mel_spec = F.avg_pool2d(mel_spec, kernel_size=(4,1))
            mel_spec = F.interpolate(mel_spec, size=(32,32), mode="bilinear")
            mel_spec = mel_spec.squeeze(0).squeeze(0)
            label = self.label_list[idx]  # 해당 데이터의 라벨
            return mel_spec, label, self.file_list[idx]  # 변환된 오디오 데이터, 라벨, 파일 경로 반환

        except:
            # 오류 발생 시 첫 번째 데이터 샘플을 대신 반환
            spec = wav2melSpec(self.file_list[0])
            mel_spec = torch.tensor(spec, dtype=torch.float32)
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
            mel_spec = F.avg_pool2d(mel_spec, kernel_size=(4, 1))
            mel_spec = F.interpolate(mel_spec, size=(32, 32), mode="bilinear")
            mel_spec = mel_spec.squeeze(0).squeeze(0)
            label = self.label_list[idx]  # 해당 데이터의 라벨
            return mel_spec, label, self.file_list[idx]  # 변환된 오디오 데이터, 라벨, 파일 경로 반환

    def __len__(self):
        """


        데이터셋의 전체 샘플 개수를 반환하는 메서드.

        Returns:
            int: 데이터셋의 크기
        """
        return len(self.file_list)

class AudioMNISTSpikeTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Audio-MNIST-Spike_train 데이터셋이 저장된 최상위 경로.
        transform: 이벤트 데이터를 변환하는 함수 (예: ToFrame)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # (파일 경로, label) 리스트

        # 라벨 폴더 탐색 (0~9)
        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                # 해당 라벨 폴더 내 모든 .npy 파일 추가
                for filename in os.listdir(label_path):
                    if filename.endswith(".npy"):
                        self.samples.append((os.path.join(label_path, filename), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        events = np.load(file_path, allow_pickle=True)  # 이벤트 데이터 로드

        # transform이 존재하면 적용
        if self.transform:
            events = self.transform(events)

        return events, label