import time
from turtledemo.forest import start

import h5py
import os

import torch
import numpy as np

from collections import Counter

# from keras.src.backend import epsilon
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from sklearn.metrics import accuracy_score

from tonic.datasets.cifar10dvs import CIFAR10DVS
from tonic.transforms import ToFrame

import samna

from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
import sinabs.layers as sl
from sinabs.activation import MembraneSubtract, SingleSpike
from sinabs.backend.dynapcnn import DynapcnnNetwork
############################# Dataset preparation #############################
# Download dataset
root_dir = "/home/yongjin/PycharmProjects/sinabs-dynapcnn/datasets"
cifar10dvs = CIFAR10DVS(save_to=root_dir)

train_ratio = 0.9
num_train = int(len(cifar10dvs) * train_ratio)
num_test = len(cifar10dvs) - num_train

cifar10dvs_train_dataset, cifar10dvs_test_dataset = random_split(cifar10dvs, [num_train, num_test])
print(len(cifar10dvs_test_dataset))

sample_data, label = cifar10dvs_train_dataset[0]

print(f"Sample data type: {type(sample_data)}")
print(f"Sample data content: {sample_data}")
print(f"Label type: {type(label)}")
print(f"Label content: {label}")
###############################################################################


############################# Define Train setting #############################
n_time_steps = 4

epochs = 200
lr = 1e-3
batch_size = 16
num_workers = 8
device = "cuda:0"
shuffle = True
################################################################################

############################# Define SNN Model #############################
vgg_speck = nn.Sequential(
    # Layer 1 : [2, 32, 32] -> [32, 32, 32]
    nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
    sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),

    # Layer 2 : [32, 32, 32] -> [32, 16, 16]
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
    sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(kernel_size=2, stride=2),

    # Layer 3 : [32, 16, 16] -> [32, 16, 16]
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
    sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),

    # Layer 4 : [32, 16, 16] -> [64, 8, 8]
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(kernel_size=2, stride=2),

    # Layer 5 : [64, 8, 8] -> [32, 8, 8]
    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
    sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),

    # Layer 6 : [32, 8, 8] -> [128, 4, 4]
    nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
    sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(kernel_size=2, stride=2),

    # Layer 7 : [128, 4, 4] -> [32, 4, 4]
    nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
    sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),

    # Layer 8 : [32, 4, 4] -> [32, 2, 2]
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
    sl.LIFSqueeze(batch_size=batch_size, tau_mem=0.434, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(kernel_size=2, stride=2),

    # [32, 2, 2] ->[128]
    nn.Flatten(),

    # Layer 9 : [128] -> [10]
    nn.Linear(in_features=128, out_features=10),
)

# init the model weights
for layer in vgg_speck.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

# print(vgg_speck)

# # 모델의 state_dict 가져오기
# state_dict = vgg_speck.state_dict()
#
# # 각 매개변수의 이름과 텐서 크기 출력
# for param_tensor in state_dict:
#     print(f"{param_tensor}:\t{state_dict[param_tensor].size()}")
############################################################################

# #############################  Open learned model file  #############################
# # file path
# file_path = "./ep-0194.hdf5"
#
# # Name of the layer to be imported
# conv_kernel_layer_paths = [
#     "model_weights/conv1/conv1",
#     "model_weights/conv2/conv2",
#     "model_weights/conv3/conv3",
#     "model_weights/conv4/conv4",
#     "model_weights/conv5/conv5",
#     "model_weights/conv6/conv6",
#     "model_weights/conv7/conv7",
#     "model_weights/conv8/conv8",
# ]
#
# conv_bn_layer_paths = [
#     "model_weights/bn_conv1/bn_conv1",
#     "model_weights/bn_conv2/bn_conv2",
#     "model_weights/bn_conv3/bn_conv3",
#     "model_weights/bn_conv4/bn_conv4",
#     "model_weights/bn_conv5/bn_conv5",
#     "model_weights/bn_conv6/bn_conv6",
#     "model_weights/bn_conv7/bn_conv7",
#     "model_weights/bn_conv8/bn_conv8",
# ]
#
# prediction_kernel_layer_paths = [
#     "model_weights/predictions/predictions",
# ]
#
# # define conv_fusion function
# def conv_fusion(kernel, alpha, variance, epsilon=1.001e-5):
#     std = np.sqrt(variance + epsilon)
#     fused_kernel = kernel * (alpha / std).reshape(1, 1, 1, -1)
#     return fused_kernel
#
# # Opening HDF5 file and updating weights
# with h5py.File(file_path, "r") as f:
#     # Conv2D layer
#     for i, (conv_kernel_path, conv_bn_path) in enumerate(zip(conv_kernel_layer_paths, conv_bn_layer_paths)):
#         # load weight and bias
#         kernel_data = torch.tensor(f[f"{conv_kernel_path}/kernel:0"][:])
#         alpha_data = torch.tensor(f[f"{conv_bn_path}/gamma:0"][:])
#         variance = torch.tensor(f[f"{conv_bn_path}/moving_variance:0"][:])
#
#         fused_kernel = conv_fusion(kernel_data, alpha_data, variance)
#
#         # Convert to PyTorch format
#         fused_kernel = fused_kernel.permute(3, 2, 0, 1) # [H, W, in_channels, out_channels] -> [out_channels, in_channels, H, W]
#
#         if i <= 1 :
#             layer = vgg_speck[i * 2]
#         elif i <= 3 :
#             layer = vgg_speck[i * 2 + 1]
#         elif i <= 5 :
#             layer = vgg_speck[i * 2 + 2]
#         else :
#             layer = vgg_speck[i * 2 + 3]
#
#         # updates weight
#         with torch.no_grad():
#             if isinstance(layer, nn.Conv2d):
#                 layer.weight.copy_(fused_kernel)
#
#     # Prediction layer
#     prediction_kernel_data = torch.tensor(f[f"{prediction_kernel_layer_paths[0]}/kernel:0"][:])
#
#     #Convert to PyTorch format
#     prediction_kernel_data = prediction_kernel_data.T # [in_features, out_features] -> [out_features, in_features]
#
#     prediction_layer = vgg_speck[21]
#
#     # updates weight
#     with torch.no_grad():
#         if isinstance(prediction_layer, nn.Linear):
#             prediction_layer.weight.copy_(prediction_kernel_data)

# # #Check the Model
# for i, layer in enumerate(vgg_speck):
#     if isinstance(layer, (nn.Conv2d, nn.Linear)):
#         print(f"Layer {i}: weight shape = {layer.weight.shape}, bias shape = {layer.bias.shape}")
############################################################################

#############################  Dataset preprocessing  #############################
to_raster = ToFrame(sensor_size=CIFAR10DVS.sensor_size, n_time_bins=n_time_steps)

cifar10dvs = CIFAR10DVS(save_to=root_dir, transform=to_raster)

train_ratio = 0.9
num_train = int(len(cifar10dvs) * train_ratio)
num_test = len(cifar10dvs) - num_train

cifar10dvs_frame_train_dataset, cifar10dvs_frame_test_dataset = random_split(cifar10dvs, [num_train, num_test])

snn_train_dataloader = DataLoader(cifar10dvs_frame_train_dataset,batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True )
snn_test_dataloader = DataLoader(cifar10dvs_frame_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)
####################################################################################

#############################  Progress Train & Test  #############################
vgg_speck = vgg_speck.to(device=device)

optimizer = Adam(params=vgg_speck.parameters(), lr=lr)
criterion = CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

for e in range(epochs):

    # train
    train_p_bar = tqdm(snn_train_dataloader)
    for data, label in train_p_bar:
        # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
        data = data.reshape(-1, 2, 32, 32).to(dtype=torch.float, device=device)
        label = label.to(dtype=torch.long, device=device)
        # forward
        optimizer.zero_grad()
        output = vgg_speck(data)
        # reshape the output from [Batch*Time, num_classes] into [Batch, Time, num_classes]
        output = output.reshape(batch_size, n_time_steps, -1)
        # accumulate all time-steps output for final prediction
        output = output.sum(dim=1)
        loss = criterion(output, label)
        # backward
        loss.backward()
        optimizer.step()

        # detach the neuron states and activations from current computation graph(necessary)
        for layer in vgg_speck.modules():
            if isinstance(layer, sl.StatefulLayer):
                for name, buffer in layer.named_buffers():
                    buffer.detach_()

        # set progressing bar
        train_p_bar.set_description(f"Epoch {e} - BPTT Training Loss: {round(loss.item(), 4)}")

    # validate
    correct_predictions = []
    with torch.no_grad():
        test_p_bar = tqdm(snn_test_dataloader)
        for data, label in test_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            data = data.reshape(-1, 2, 32, 32).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)
            # forward
            output = vgg_speck(data)
            # reshape the output from [Batch*Time, num_classes] into [Batch, Time, num_classes]
            output = output.reshape(batch_size, n_time_steps, -1)
            # accumulate all time-steps output for final prediction
            output = output.sum(dim=1)
            # calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            # compute the total correct predictions
            correct_predictions.append(pred.eq(label.view_as(pred)))
            # set progressing bar
            test_p_bar.set_description(f"Epoch {e} - BPTT Testing Model...")

        correct_predictions = torch.cat(correct_predictions)
        print(f"Epoch {e} - BPTT accuracy: {correct_predictions.sum().item() / (len(correct_predictions)) * 100}%")

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
############################################################################

# #############################  Progress test  #############################
# vgg_speck = vgg_speck.to(device=device)
# for e in range(epochs):
#     # validate
#     correct_predictions = []
#     with torch.no_grad():
#         test_p_bar = tqdm(snn_test_dataloader)
#         for data, label in test_p_bar:
#             # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
#             data = data.reshape(-1, 2, 32, 32).to(dtype=torch.float, device=device)
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
# ############################################################################

#############################  Save SNN Model  #############################
base_save_path = "/home/yongjin/PycharmProjects/sinabs-dynapcnn/saved_models"
model_name = "tutorial_cifar10dvs_vgg_speck"

os.makedirs(base_save_path, exist_ok=True)

existing_files = os.listdir(base_save_path)
counter = 0

while f"{model_name}_{counter}.pth" in existing_files:
    counter += 1

model_save_path = os.path.join(base_save_path, f"{model_name}_{counter}.pth")
torch.save(vgg_speck.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
############################################################################

#############################  Depoly SNN To The Devkit  #############################
cpu_snn = vgg_speck.to(device="cpu")
dynapcnn = DynapcnnNetwork(snn=cpu_snn, input_shape=(2, 32, 32), discretize=True, dvs_input=False)
devkit_name = "speck2fdevkit"

# use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
print(f"The SNN is deployed on the core: {dynapcnn.chip_layers_ordering}")
######################################################################################
#
# #############################  Inference On The Devkit  #############################
# # Inference dataset
# subset_indices = list(range(0, len(cifar10dvs_test_dataset), 1000))
# cifar10dvs_test_dataset = Subset(cifar10dvs_test_dataset, subset_indices)
#
# #Inference
# inference_p_bar = tqdm(cifar10dvs_test_dataset)
#
# test_samples = 0
# correct_samples = 0
# total_output_spikes = 0
# total_inference_time = 0
#
# for events, label in inference_p_bar:
#
#     # create samna Spike events stream
#     samna_event_stream = []
#     for ev in events:
#         spk = samna.speck2f.event.Spike()
#         spk.x = ev['x']
#         spk.y = ev['y']
#         spk.timestamp = ev['t'] - events['t'][0]
#         spk.feature = ev['p']
#         # Spikes will be sent to layer/core #0, since the SNN is deployed on core: [0, 1, 2, 3]
#         spk.layer = 0
#         samna_event_stream.append(spk)
#
#     # inference on chip
#     # output_events is also a list of Spike, but each Spike.layer is 3, since layer#3 is the output layer
#     start_time = time.time()
#     output_events = dynapcnn(samna_event_stream)
#     end_time = time.time()
#
#     inference_time = end_time - start_time
#     total_inference_time += inference_time
#
#     total_output_spikes += len(output_events)
#
#     # use the most frequent output neuron index as the final prediction
#     neuron_index = [each.feature for each in output_events]
#     if len(neuron_index) != 0:
#         frequent_counter = Counter(neuron_index)
#         prediction = frequent_counter.most_common(1)[0][0]
#     else:
#         prediction = -1
#     inference_p_bar.set_description(f"label: {label}, prediction: {prediction}， output spikes num: {len(output_events)}")
#
#     if prediction == label:
#         correct_samples += 1
#
#     test_samples += 1
#
# print(f"Total output spikes: {total_output_spikes}")
# print(f"On chip inference accuracy: {correct_samples / test_samples}")
#
# # Stop to record inference time
# print(f"Total inference time on hareware: {total_inference_time :.4f} seconds")
# ####################################################################################