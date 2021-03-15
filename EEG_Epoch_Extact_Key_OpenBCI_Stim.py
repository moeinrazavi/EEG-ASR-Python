import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import pyxdf
import glob
import h5py
import CleanFlatlines, CleanDrifts, CleanASR
    
time_begin = -1
time_end = 2

time_baseline_begin = -0.8
time_baseline_end = -0.6

ch_names = ["FP1", "FP2", "C3", "C4", "Fz", "Cz", "Pz", "T6", "F7", "F8", "F3", "F4", "T3", "T4", "P3", "P4"]

ch_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]
sfreq = 125
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

streams_to_load = [{'name': 'OpenBCIEEG'}, {'name': 'Psychopy_key'}, {'name': 'Psychopy_stim_type'}]

path_key = "/Volumes/GoogleDrive/Shared drives/lab/Moein/Project/Flanker_OpenBCI/Datasets/Key/XDF/Renamed"

key_sub_counter = 0
for path in glob.glob(path_key + '/*.xdf'):
    
    if (path.split("/")[-1].split(".")[0].split("_")[-1] in ["9"]):
        continue
    trials = []
    
    key_sub_counter +=1
    
    print(path)

    logging.basicConfig(level=logging.DEBUG)  # Use logging.INFO to reduce output.
    fname = path  #Address to XDF file
    streams, fileheader = pyxdf.load_xdf(fname, select_streams=streams_to_load, verbose=0)
    
    # print("Found {} streams:".format(len(streams)))
    # for ix, stream in enumerate(streams):
    #     print("Stream {}: {} - type {} - uid {} - shape {} at {} Hz (effective {} Hz)".format(
    #         ix + 1, stream['info']['name'][0],
    #        	  stream['info']['type'][0],
    #        	  stream['info']['uid'][0],
    #         	 (int(stream['info']['channel_count'][0]), len(stream['time_stamps'])),
    #          	stream['info']['nominal_srate'][0],
    #         	 stream['info']['effective_srate'])
    #     	 )
    #     if any(stream['time_stamps']):
    #       	   print("\tDuration: {} s".format(stream['time_stamps'][-1] - stream['time_stamps'][0]))
    # print("Done.")
        

    for j in range(len(streams)):
    
        if streams[j]['info']["name"][0] == "OpenBCIEEG":
            EEG_index = j
        elif streams[j]['info']["name"][0] == "Psychopy_key":
            Key_index = j
        elif streams[j]['info']["name"][0] == "Psychopy_stim_type":
            Stim_index = j
    
    EEG_channels = streams[EEG_index]["time_series"][:,:len(ch_names)].T
    EEG_channels = EEG_channels.astype("float")
    EEG_time = streams[EEG_index]["time_stamps"]
    EEG_time_zero_ref = streams[EEG_index]["time_stamps"] - streams[EEG_index]["time_stamps"][0]
    
    
    Key_marker = streams[Key_index]["time_series"]
    Key_time = streams[Key_index]["time_stamps"]
    Key_time_zero_ref = streams[Key_index]["time_stamps"] - streams[EEG_index]["time_stamps"][0]
    
    Stim_marker = streams[Stim_index]["time_series"]
    Stim_time = streams[Stim_index]["time_stamps"]
    Stim_time_zero_ref = streams[Stim_index]["time_stamps"] - streams[EEG_index]["time_stamps"][0]
    
    EEG_channels = EEG_channels - np.mean(EEG_channels, 0)

    filtered_EEG = mne.filter.filter_data(EEG_channels, sfreq=sfreq, l_freq=0.5, h_freq=40, method="fir")

    # l_trans_bandwidth = min(max(l_freq * 0.25, 2), l_freq)   --> Default Low Freq Bandwidth
    # h_trans_bandwidth = min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq) --> Default High Freq Bandwidth
    
    t_begin = Stim_time + time_begin
    t_baseline_begin = Stim_time + time_baseline_begin
    
    trial_len = int((time_end-time_begin)*sfreq)
    baseline_len = int((time_baseline_end-time_baseline_begin)*sfreq)
    
    
    for i in range(len(t_begin)):
        if np.size(np.where(EEG_time >= t_begin[i])) == 0:
            break
        trial_begin=np.where(EEG_time >= t_begin[i])[0][0]
        trial_end= trial_begin + trial_len
        
        baseline_begin = np.where(EEG_time >= t_baseline_begin[i])[0][0]
        baseline_end = baseline_begin + baseline_len
        
        baseline = np.reshape(np.mean(filtered_EEG[:,baseline_begin:baseline_end], axis = 1), (-1,1))
        trials.append(filtered_EEG[:,trial_begin:trial_end] - baseline)
        
    trials = np.array(trials)
    trials = trials.swapaxes(0,1)
    shape_before_flatten = trials.shape
    trials = trials.reshape(shape_before_flatten[0],-1)
    
    trials = CleanDrifts.clean_drifts(trials, sfreq)
    trials = CleanASR.clean_asr(trials, sfreq)
    trials = trials.reshape(shape_before_flatten)

    
    sub_num_with_extension = path.split("/")[-1]

    sub_num = sub_num_with_extension.split(".")[0]
    
    if trials.shape[1] != len(Stim_marker):
        print("Failed to Extract Epochs for ", sub_num)
        continue
    
    save_path = "/Volumes/GoogleDrive/Shared drives/lab/Moein/Project/Flanker_OpenBCI/Analysis/Preprocessing/1_Extracting_Epochs/Python/Key/Stim/" + sub_num + ".npy"

    np.save(save_path, trials)
    print("Sub ", sub_num, " Saved.")