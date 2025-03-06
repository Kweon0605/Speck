import os
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import os
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 오디오 → Mel Spectrogram 변환 함수
def wav2melSpec(AUDIO_PATH):
    audio, sr = librosa.load(AUDIO_PATH)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)  # (128, T)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # dB 변환
    return mel_spec_db, sr  # 샘플링 레이트도 반환

# 데이터 변환 함수 (128×T → 32×32)
def transform_mel_spec(mel_spec):
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32)  # NumPy → Tensor 변환
    mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # (128, T) → (1,1,128,T)

    # 주파수(128 → 32) 줄이기
    mel_spec = F.avg_pool2d(mel_spec, kernel_size=(4,1))

    # 시간(T → 32) 변환 (T값 그대로 보간)
    mel_spec = F.interpolate(mel_spec, size=(32, 32), mode="bilinear")

    mel_spec = mel_spec.squeeze(0).squeeze(0)  # (1,1,32,32) → (32, 32)
    return mel_spec.numpy()

# 오디오 파일 경로
audio_path = "./datasets/Audio-MNIST/02/2_02_0.wav"

# 🔹 원본 멜 스펙트로그램 변환
original_spec, sr = wav2melSpec(audio_path)  # (128, T) 데이터
transformed_spec = transform_mel_spec(original_spec)  # (32, 32) 데이터

# 🔹 원본 Mel-Spectrogram (128, T) 시각화
fig1, ax1 = plt.subplots()
img1 = librosa.display.specshow(original_spec, sr=sr, x_axis="time", y_axis="mel", ax=ax1, vmin=-80, vmax=0)
ax1.set_title("Original Mel-Spectrogram (128xT)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Frequency (Mel)")
fig1.colorbar(img1, ax=ax1, format='%+2.0f dB')

# 🔹 변형된 Mel-Spectrogram (32, 32) 시각화
fig2, ax2 = plt.subplots()
img2 = librosa.display.specshow(transformed_spec, x_axis="time", y_axis="mel", ax=ax2, vmin=-80, vmax=0)
ax2.set_title("Transformed Mel-Spectrogram (32x32)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Frequency (Mel)")
fig2.colorbar(img2, ax=ax2, format='%+2.0f dB')

plt.show()


# import os
# import torch
# import librosa
# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset
# import torch.nn.functional as F
#
# # 오디오 → Mel Spectrogram 변환 함수
# def wav2melSpec(AUDIO_PATH):
#     audio, sr = librosa.load(AUDIO_PATH)
#     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)  # (128, T)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # dB 변환
#     return mel_spec_db
#
# # 데이터 변환 함수
# def transform_mel_spec(mel_spec):
#     mel_spec = torch.tensor(mel_spec, dtype=torch.float32)  # NumPy → Tensor 변환
#     mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # (128, 21) → (1,1,128,21)
#
#     # 주파수(128 → 32)로 줄이기
#     mel_spec = F.avg_pool2d(mel_spec, kernel_size=(4,1))
#
#     # 시간(21 → 32)으로 늘리기
#     mel_spec = F.interpolate(mel_spec, size=(32, 32), mode="bilinear")
#
#     mel_spec = mel_spec.squeeze(0).squeeze(0)  # (1,1,32,32) → (32, 32)
#     return mel_spec.numpy()
#
# # 오디오 파일 경로
# audio_path = "./datasets/Audio-MNIST/01/2_01_0.wav"
#
# # 변형 전 멜 스펙트로그램
# original_spec = wav2melSpec(audio_path)  # (128, T) 데이터
# transformed_spec = transform_mel_spec(original_spec)  # (32, 32) 데이터
#
# # 🔹 변형 전후 시각화
# fig, ax = plt.subplots()
# fig_1, ax_1 = plt.subplots()
#
# # 원본 Mel-Spectrogram (128, 21)
# librosa.display.specshow(original_spec, x_axis='time', y_axis='mel', ax=ax)
# ax.set_title("Original Mel-Spectrogram (128x21)")
# ax.set_xlabel("Time")
# ax.set_ylabel("Frequency (Mel)")
#
# # 변형된 Mel-Spectrogram (32, 32)
# librosa.display.specshow(transformed_spec, x_axis='time', y_axis='mel', ax=ax_1)
# ax_1.set_title("Transformed Mel-Spectrogram (32x32)")
# ax_1.set_xlabel("Time")
# ax_1.set_ylabel("Frequency (Mel)")
#
# plt.colorbar(librosa.display.specshow(original_spec, ax=ax), ax=ax)
# plt.colorbar(librosa.display.specshow(transformed_spec, ax=ax_1), ax=ax_1)
#
# plt.show()
#
# def wav2melSpec(AUDIO_PATH):
#     audio, sr = librosa.load(AUDIO_PATH)
#     return librosa.feature.melspectrogram(y=audio, sr=sr)
#
# def imgSpec(ms_feature):
#     fig, ax = plt.subplots()
#     ms_dB = librosa.power_to_db(ms_feature, ref=np.max)
#     print(ms_dB)
#     img = librosa.display.specshow(ms_dB, x_axis='time', y_axis='mel', ax=ax)
#     fig.colorbar(img, ax=ax, format='%+2.0f dB')
#     ax.set(title='Mel-frequency spectrogram')
#     plt.show()
#
# def get_audio_info(path, show_melspec=False, label=None):
#     spec = wav2melSpec(path)
#     print(spec)
#     print(spec.shape)
#     if label is not None:
#         print("Label:", label)
#     if show_melspec is not False:
#         imgSpec(spec)
#
# get_audio_info('./datasets/Audio-MNIST/01/2_01_0.wav', show_melspec=True)