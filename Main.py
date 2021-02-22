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

L_FREQ = 0.5; H_FREQ = 40  # Set the low and high edges of the bandpass filter  

###############################################
######## Define Channel Name and Types ########
###############################################
ch_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "FCz", "PO8"]
ch_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]
###############################################

###############################################################################
#### Give the name of EEG, Response and Stimulus Streams from the XDF File ####
###############################################################################
EEG_Name = "Unicorn"
Resp_Stream_Name = "Psychopy_key"
Stimulus_Stream_Name = "Psychopy_stim_type"
###############################################################################


NUM_EEG_Channels = len(ch_names)


sfreq = 250     # The Sampling Frequency of the Device
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


streams_to_load = [{'name': EEG_Name}, {'name': Resp_Stream_Name}, {'name': Stimulus_Stream_Name}]

# Address to .XDF Dataset
path_key = "/Volumes/GoogleDrive/Shared drives/lab/Moein/Project/Exp1_Flanker_Unicorn/Datasets/Key/XDF/Renamed"

key_sub_counter = 0
for path in glob.glob(path_key + '/*.xdf'):
    
    trials = []
    
    key_sub_counter +=1
    
    print(path)

    logging.basicConfig(level=logging.DEBUG)  # Use logging.INFO to reduce output.
    fname = path  #Address to XDF file
    streams, fileheader = pyxdf.load_xdf(fname, select_streams=streams_to_load, verbose=0)

    for j in range(len(streams)):
    
        if streams[j]['info']["name"][0] == EEG_Name:     # Name of the EEG Stream in the XDF file
            EEG_index = j
        elif streams[j]['info']["name"][0] == Resp_Stream_Name: # Name of the response event Stream in the XDF file
            Key_index = j
        elif streams[j]['info']["name"][0] == Stimulus_Stream_Name:  # Name of Stimulus event Stream in the XDF file
            Stim_index = j
    
    EEG_channels = streams[EEG_index]["time_series"][:,:NUM_EEG_Channels].T         # Selecting only EEG channels from the EEG Stream
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

    filtered_EEG = mne.filter.filter_data(EEG_channels, sfreq=sfreq, l_freq=L_FREQ, h_freq=H_FREQ, method="fir")    # Bandpass Filtering of the EEG Data
    
    
    ######################################################################
    ######## Epoch the Data (Here with respect to Stimulus events ########
    ######################################################################
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
    ######################################################################
        
    trials = np.array(trials)
    trials = trials.swapaxes(0,1)

    shape_before_flatten = trials.shape
    trials = trials.reshape(shape_before_flatten[0],-1)
    
    ################################################
    ######## Clean the Artifacts from the Data #####
    ################################################
    trials = CleanDrifts.clean_drifts(trials, sfreq)    # Clean Drifts
    trials = CleanASR.clean_asr(trials, sfreq)          # **** IMPORTANT **** Clean the muscle and movement artifacts using Artifact Subspace Reconstruction (ASR) 
    trials = trials.reshape(shape_before_flatten)

    
    sub_num_with_extension = path.split("/")[-1]

    sub_num = sub_num_with_extension.split(".")[0]
    
    if trials.shape[1] != len(Stim_marker):
        print("Failed to Extract Epochs for ", sub_num)
        continue
    
    save_path = "/Volumes/GoogleDrive/Shared drives/lab/Moein/Project/Exp1_Flanker_Unicorn/Analysis/Preprocessing/1_Extracting_Epochs/Python/Key/Stim/" + sub_num + ".npy"

    np.save(save_path, trials)
    print("Sub ", sub_num, " Saved.")