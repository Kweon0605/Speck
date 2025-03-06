import numpy as np
import pandas as pd

import os

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.ao.nn.quantized.functional import threshold
from torch.optim.lr_scheduler import CosineAnnealingLR

from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.activation.surrogate_gradient_fn import Gaussian
import sinabs.layers as sl
from sinabs.activation import MembraneSubtract, SingleSpike
from sinabs.backend.dynapcnn import DynapcnnNetwork

import torchvision
import torchvision.transforms as T

import random
import matplotlib.pyplot as plt

from lib_Audio_MNIST import Audio_MNIST_func as audio_mnist

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

### Audio-MNIST의 정보를 확인
# audio_mnist.get_audio_info('./datasets/Audio-MNIST/01/2_01_0.wav', show_melspec=True)


### Train dataset과 Test dataset을 구
root_dir = "/home/yongjin/PycharmProjects/sinabs-dynapcnn/Audio-MNIST-Spike_train"
audio_mnist_dataset = audio_mnist.AudioMNISTSpikeTrainDataset(root_dir)



train_ratio = 0.9
num_train = int(len(audio_mnist_dataset) * train_ratio)
num_test = len(audio_mnist_dataset) - num_train

# Train/Test 데이터셋 분할
train_ds, test_ds = random_split(audio_mnist_dataset, [num_train, num_test])

# 확인
print(f"Total Dataset Size: {len(audio_mnist_dataset)}")
print(f"Train Dataset Size: {len(train_ds)}")
print(f"Test Dataset Size: {len(test_ds)}")


### dataset 확인
sample_data, label = train_ds[1000]
print("🔹 Sample Data Info 🔹")
print(f"Type of data: {type(sample_data)}")  # 데이터 타입 확인
print(f"Shape of sample data: {sample_data.shape}")  # 데이터 형태 확인
print(f"Label: {label}")  # 정답 라벨
# sample_data = sample_data.squeeze(0).numpy()
print(sample_data)

# ############################# Define Train setting #############################
# n_time_steps = 4
#
# epochs = 200
# lr = 1e-3
# batch_size = 64
# num_workers = 8
# device = "cuda:0"
# # device = "cpu"
# shuffle = True
# ################################################################################
# ############################# Define SNN Model #############################
# vgg_speck = nn.Sequential(
#     # Layer 1 : [1, 32, 32] -> [32, 32, 32]
#     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
#     sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
#
#     # Layer 2 : [32, 32, 32] -> [32, 16, 16]
#     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
#     sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#
#     # Layer 3 : [32, 16, 16] -> [32, 16, 16]
#     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
#     sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
#
#     # Layer 4 : [32, 16, 16] -> [64, 8, 8]
#     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#     sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#
#     # Layer 5 : [64, 8, 8] -> [32, 8, 8]
#     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
#     sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
#
#     # Layer 6 : [32, 8, 8] -> [128, 4, 4]
#     nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
#     sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#
#     # Layer 7 : [128, 4, 4] -> [32, 4, 4]
#     nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
#     sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
#
#     # Layer 8 : [32, 4, 4] -> [32, 2, 2]
#     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
#     sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#
#     # [32, 2, 2] ->[128]
#     nn.Flatten(),
#
#     # Layer 9 : [128] -> [10]
#     nn.Linear(in_features=128, out_features=10),
# )
#
# # vgg_speck = nn.Sequential(
# #     # Layer 1 : [1, 32, 32] -> [32, 32, 32]
# #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
# #     sl.IAFSqueeze(spike_threshold=1.0, batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
# #
# #     # Layer 2 : [32, 32, 32] -> [32, 16, 16]
# #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
# #     sl.IAFSqueeze(spike_threshold=1.0, batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
# #     nn.AvgPool2d(kernel_size=2, stride=2),
# #
# #     # Layer 3 : [32, 16, 16] -> [32, 16, 16]
# #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
# #     sl.IAFSqueeze(spike_threshold=1.0, batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
# #
# #     # Layer 4 : [32, 16, 16] -> [64, 8, 8]
# #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
# #     sl.IAFSqueeze(spike_threshold=1.0, batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
# #     nn.AvgPool2d(kernel_size=2, stride=2),
# #
# #     # Layer 5 : [64, 8, 8] -> [32, 8, 8]
# #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
# #     sl.IAFSqueeze(spike_threshold=1.0, batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
# #
# #     # Layer 6 : [32, 8, 8] -> [128, 4, 4]
# #     nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
# #     sl.IAFSqueeze(spike_threshold=1.0, batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
# #     nn.AvgPool2d(kernel_size=2, stride=2),
# #
# #     # Layer 7 : [128, 4, 4] -> [32, 4, 4]
# #     nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
# #     sl.IAFSqueeze(spike_threshold=1.0, batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
# #
# #     # Layer 8 : [32, 4, 4] -> [32, 2, 2]
# #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
# #     sl.IAFSqueeze(spike_threshold=1.0, batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
# #     nn.AvgPool2d(kernel_size=2, stride=2),
# #
# #     # [32, 2, 2] ->[128]
# #     nn.Flatten(),
# #
# #     # Layer 9 : [128] -> [10]
# #     nn.Linear(in_features=128, out_features=10),
# # )
#
# # init the model weights
# for layer in vgg_speck.modules():
#     if isinstance(layer, (nn.Conv2d, nn.Linear)):
#         nn.init.xavier_normal_(layer.weight.data)
#
# # print(vgg_speck)
#
# # # 모델의 state_dict 가져오기
# # state_dict = vgg_speck.state_dict()
# #
# # # 각 매개변수의 이름과 텐서 크기 출력
# # for param_tensor in state_dict:
# #     print(f"{param_tensor}:\t{state_dict[param_tensor].size()}")
# ############################################################################
#
# #############################  Dataset preprocessing  #############################
# class ReplicatedDataset(Dataset):
#     """
#     정적인 이미지를 time_steps 개수만큼 복제하여 SNN 입력으로 변환하는 Dataset.
#     """
#
#     def __init__(self, static_dataset, time_steps=10):
#         self.static_dataset = static_dataset
#         self.time_steps = time_steps
#
#     def __getitem__(self, idx):
#         static_img, label, file_path = self.static_dataset[idx]
#         # static_img shape: [1, 32, 32]
#
#         # 동일한 이미지를 time_steps 번 반복 → shape: [time_steps, 1, 32, 32]
#         replicated_input = static_img.unsqueeze(0).repeat(self.time_steps, 1, 1, 1)
#
#         return replicated_input, label
#
#     def __len__(self):
#         return len(self.static_dataset)
#
# train_replicated_ds = ReplicatedDataset(train_ds, time_steps=n_time_steps)
# test_replicated_ds  = ReplicatedDataset(test_ds, time_steps=n_time_steps)
#
# train_loader = DataLoader(train_replicated_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
# test_loader  = DataLoader(test_replicated_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
# ####################################################################################
#
# #############################  Progress Train & Test  #############################
# vgg_speck = vgg_speck.to(device=device)
#
# optimizer = Adam(params=vgg_speck.parameters(), lr=lr)
# criterion = CrossEntropyLoss()
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
#
# for e in range(epochs):
#
#     # train
#     train_p_bar = tqdm(train_loader)
#     for data, label in train_p_bar:
#         # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
#         data = data.reshape(-1, 1, 32, 32).to(dtype=torch.float, device=device)
#         label = label.to(dtype=torch.long, device=device)
#         # forward
#         optimizer.zero_grad()
#         output = vgg_speck(data)
#         # reshape the output from [Batch*Time, num_classes] into [Batch, Time, num_classes]
#         output = output.reshape(batch_size, n_time_steps, -1)
#         # accumulate all time-steps output for final prediction
#         output = output.sum(dim=1)
#         loss = criterion(output, label)
#         # backward
#         loss.backward()
#         optimizer.step()
#
#         # detach the neuron states and activations from current computation graph(necessary)
#         for layer in vgg_speck.modules():
#             if isinstance(layer, sl.StatefulLayer):
#                 for name, buffer in layer.named_buffers():
#                     buffer.detach_()
#
#         # set progressing bar
#         train_p_bar.set_description(f"Epoch {e} - BPTT Training Loss: {round(loss.item(), 4)}")
#
#     # validate
#     correct_predictions = []
#     with torch.no_grad():
#         test_p_bar = tqdm(test_loader)
#         for data, label in test_p_bar:
#             # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
#             data = data.reshape(-1, 1, 32, 32).to(dtype=torch.float, device=device)
#             label = label.to(dtype=torch.long, device=device)
#             # forward
#             output = vgg_speck(data)
#             # reshape the output from [Batch*Time, num_classes] into [Batch, Time, num_classes]
#             output = output.reshape(batch_size, n_time_steps, -1)
#             # accumulate all time-steps output for final prediction
#             output = output.sum(dim=1)
#             # calculate accuracy
#             pred = output.argmax(dim=1, keepdim=True)
#             # compute the total correct predictions
#             correct_predictions.append(pred.eq(label.view_as(pred)))
#             # set progressing bar
#             test_p_bar.set_description(f"Epoch {e} - BPTT Testing Model...")
#
#         correct_predictions = torch.cat(correct_predictions)
#         print(f"Epoch {e} - BPTT accuracy: {correct_predictions.sum().item() / (len(correct_predictions)) * 100}%")
#
#     scheduler.step()
#     current_lr = scheduler.get_last_lr()[0]
# ############################################################################
#
# #############################  Save SNN Model  #############################
# base_save_path = "/home/yongjin/PycharmProjects/sinabs-dynapcnn/saved_models"
# model_name = "audio_snn_vgg_speck"
#
# os.makedirs(base_save_path, exist_ok=True)
#
# existing_files = os.listdir(base_save_path)
# counter = 0
#
# while f"{model_name}_{counter}.pth" in existing_files:
#     counter += 1
#
# model_save_path = os.path.join(base_save_path, f"{model_name}_{counter}.pth")
# torch.save(vgg_speck.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")
# ############################################################################