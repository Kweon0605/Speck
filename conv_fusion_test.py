import h5py
import torch
import numpy as np
from keras.src.backend import epsilon


# 파일 경로
input_file_path = "./ep-0194.hdf5"
output_file_path = "converted/ep-0001.hdf5"

# 변환할 conv kernel
conv_kernel_layer_paths = [
    "model_weights/conv1/conv1",
    "model_weights/conv2/conv2",
    "model_weights/conv3/conv3",
    "model_weights/conv4/conv4",
    "model_weights/conv5/conv5",
    "model_weights/conv6/conv6",
    "model_weights/conv7/conv7",
    "model_weights/conv8/conv8",
]

# batch norm 변수
conv_bn_layer_paths = [
    "model_weights/bn_conv1/bn_conv1",
    "model_weights/bn_conv2/bn_conv2",
    "model_weights/bn_conv3/bn_conv3",
    "model_weights/bn_conv4/bn_conv4",
    "model_weights/bn_conv5/bn_conv5",
    "model_weights/bn_conv6/bn_conv6",
    "model_weights/bn_conv7/bn_conv7",
    "model_weights/bn_conv8/bn_conv8",
]

# prediction kernel 경로
prediction_kernel_layer_paths = [
    "model_weights/predictions/predictions",
]

# conv_fusion 함수
def conv_fusion(kernel, gamma, variance, epsilon=np.float64(1e-3)):
    kernel = kernel.numpy().astype(np.float64)
    gamma = gamma.numpy().astype(np.float64)
    variance = variance.numpy().astype(np.float64)

    std = np.sqrt(variance + epsilon)
    fused_kernel = 0.5 * kernel * (gamma / std).reshape(1, 1, 1, -1)
    return fused_kernel

# HDF5 파일 열기
with h5py.File(input_file_path, 'r') as input_f, h5py.File(output_file_path, 'a') as output_f:
    # Conv2D layer
    for conv_kernel_path, conv_bn_path in zip(conv_kernel_layer_paths, conv_bn_layer_paths):
        kernel_data = torch.tensor(input_f[f"{conv_kernel_path}/kernel:0"][:])
        gamma_data = torch.tensor(input_f[f"{conv_bn_path}/gamma:0"][:])
        variance = torch.tensor(input_f[f"{conv_bn_path}/moving_variance:0"][:])

        fused_kernel = conv_fusion(kernel_data, gamma_data, variance)

        group = output_f.require_group(conv_kernel_path)

        if "kernel:0" in group:
            del group["kernel:0"]
        group.create_dataset("kernel:0", data=fused_kernel)

    prediction_kernel_data = torch.tensor(input_f[f"{prediction_kernel_layer_paths[0]}/kernel:0"][:])
    pred_group = output_f.require_group(prediction_kernel_layer_paths[0])
    if "kernel:0" in pred_group:
        del pred_group["kernel:0"]
    pred_group.create_dataset("kernel:0", data=prediction_kernel_data)