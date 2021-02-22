#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[64]:


def clean_flatlines(Signal,SRate,MaxFlatlineDuration=5,MaxAllowedJitter=20):
    
    # Remove (near-) flat-lined channels.
    # Signal = clean_flatlines(Signal,MaxFlatlineDuration,MaxAllowedJitter)
    #
    # This is an automated artifact rejection function which ensures that 
    # the data contains no flat-lined channels.
    #
    # In:
    #   Signal : continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
    #            with a 0.5Hz - 2.0Hz transition band)
    #
    #   MaxFlatlineDuration : Maximum tolerated flatline duration. In seconds. If a channel has a longer
    #                         flatline than this, it will be considered abnormal. Default: 5
    #
    #   MaxAllowedJitter : Maximum tolerated jitter during flatlines. As a multiple of epsilon.
    #                      Default: 20
    #
    # Out:
    #   Signal : data set with flat channels removed
    #
    # Examples:
    #   % use with defaults
    #   eeg = clean_flatlines(eeg);
    #
    #                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
    #                                2012-08-30


    # flag channels
    removed_channels = np.array([False for i in range(len(Signal))])
    
    for c in range(len(Signal)):
        
        zero_intervals = np.reshape(np.where(np.diff([False] + list(abs(np.diff(Signal[c,:]))<(MaxAllowedJitter*np.finfo(float).eps))+[False])), (2,-1)).T
        
        if (len(zero_intervals) > 0):
            if (np.max(zero_intervals[:,1] - zero_intervals[:,0]) > MaxFlatlineDuration*SRate):
                removed_channels[c] = True
    new_channels_inds = np.where(~removed_channels)
    # remove them
    if all(removed_channels):
        print('Warning: all channels have a flat-line portion; not removing anything.')
    elif any(removed_channels):
        print('Now removing flat-line channels...')
        
        Signal = Signal[new_channels_inds]
    return Signal, new_channels_inds[0]

