import datetime
import os
import time
from collections import Counter

import samna
#######################################################################################################
import torch
from sinabs.backend.dynapcnn import DynapcnnNetwork
from torch.utils.data import Subset
from torch.utils.data import random_split
from tqdm import tqdm
from torch import nn
import sinabs.layers as sl

from lib_Audio_MNIST import Audio_MNIST_func as audio_mnist
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential


# Depoly SNN To The Devkit
vgg_speck = torch.load("/home/yongjin/PycharmProjects/sinabs-dynapcnn/saved_models/audio_snn_vgg_speck_100_0.pth", weights_only=False)
print(vgg_speck)
#######################################################################################################
# cpu_snn = snn_convert.to(device="cpu")
root_dir = "/home/yongjin/PycharmProjects/sinabs-dynapcnn/datasets/Audio-MNIST-Spike_train_100"
audio_mnist_dataset = audio_mnist.AudioMNISTSpikeTrainDataset(root_dir)

train_ratio = 0.9
num_train = int(len(audio_mnist_dataset) * train_ratio)
num_test = len(audio_mnist_dataset) - num_train

# Train/Test 데이터셋 분할
train_ds, test_ds = random_split(audio_mnist_dataset, [num_train, num_test])



# cpu_snn = snn_convert.to(device="cpu")
cpu_snn = vgg_speck.to(device="cpu")
dynapcnn = DynapcnnNetwork(snn=cpu_snn, input_shape=(1, 32, 32), discretize=True, dvs_input=False)
devkit_name = "speck2fdevkit"

# use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
dynapcnn.to(device=devkit_name, chip_layers_ordering="auto", monitor_layers=[-1])
print(f"The SNN is deployed on the core: {dynapcnn.chip_layers_ordering}")
#######################################################################################################
#devkit_cfg = dynapcnn.make_config(device=devkit_name, monitor_layers=["dvs"])
devices = samna.device.get_all_devices()
device_names = [each.device_type_name for each in devices]
print(device_names)
devkit = samna.device.open_device("Speck2fDevKit:0")

power_monitor = devkit.get_power_monitor()
power_source_node = power_monitor.get_source_node()
power_buffer_node = samna.BasicSinkNode_unifirm_modules_events_measurement()

samna_graph = samna.graph.EventFilterGraph()
samna_graph.sequential([power_source_node, power_buffer_node])
samna_graph.start()
power_monitor.start_auto_power_measurement(100) # 100 Hz sample rate
#######################################################################################################
# Inference On The Devkit
snn_test_dataset =  test_ds
# for time-saving, we only select a subset for on-chip infernce， here we select 1/100 for an example run
subset_indices = list(range(0, len(snn_test_dataset), 1))
#subset_indices = list(range(len(snn_test_dataset))) # all test data
snn_test_dataset = Subset(snn_test_dataset, subset_indices)

inference_p_bar = tqdm(snn_test_dataset)

test_samples = 0
correct_samples = 0
total_input_spikes = 0
total_output_spikes = 0

# Start to record inference time
start_time = time.time()

# for events, label in inference_p_bar:
for events, label in inference_p_bar:

    # create samna Spike events stream
    samna_event_stream = []
    for ev in events:
        spk = samna.speck2f.event.Spike()
        spk.x = ev['x']
        spk.y = ev['y']
        spk.timestamp = ev['t'] - events['t'][0]
        spk.feature = ev['p']
        # Spikes will be sent to layer/core #0, since the SNN is deployed on core: [0, 1, 2, 3]
        spk.layer = 0
        samna_event_stream.append(spk)

    # inference on chip
    # output_events is also a list of Spike, but each Spike.layer is 3, since layer#3 is the output layer
    output_events = dynapcnn(samna_event_stream)
    total_input_spikes += len(samna_event_stream)
    total_output_spikes += len(output_events)

    # use the most frequent output neruon index as the final prediction
    neuron_index = [each.feature for each in output_events]
    if len(neuron_index) != 0:
        frequent_counter = Counter(neuron_index)
        prediction = frequent_counter.most_common(1)[0][0]
    else:
        prediction = -1
    inference_p_bar.set_description(f"label: {label}, prediction: {prediction}， output spikes num: {len(output_events)}")

    if prediction == label:
        correct_samples += 1

    test_samples += 1

print(f"Total input spikes: {total_input_spikes}")
print(f"Total output spikes: {total_output_spikes}")
print(f"On chip inference accuracy: {correct_samples / test_samples:.4f}")

# Stop to record inference time
end_time = time.time()
# Calculate total inference time
total_inference_time = end_time - start_time
print(f"Total inference time on hareware: {total_inference_time} seconds")
#######################################################################################################
power_monitor.stop_auto_power_measurement()
samna_graph.stop()
power_events = power_buffer_node.get_events()

power_each_track = {}
event_count_each_track = {}
for evt in power_events:
    track_id = evt.channel
    power_value = evt.value
    power_each_track[track_id] = power_each_track.get(track_id, 0) + power_value
    event_count_each_track[track_id] = event_count_each_track.get(track_id, 0) + 1

print("Dynamic Power Measurements During Inference:")
for track_id in range(5):
    avg_power = (power_each_track[track_id] / event_count_each_track[track_id]) * 1000
    print(f"Track {track_id}: Average Power = {avg_power:.3f} mW")

samna.device.close_device(devkit)