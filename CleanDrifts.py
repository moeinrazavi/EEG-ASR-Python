#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.signal import kaiserord, firwin, filtfilt


# In[49]:


def clean_drifts(Signal,SRate,Transition=[0.5, 1],Attenuation=80):
    
    # Removes drifts from the data using a forward-backward high-pass filter.
    # Signal = clean_drifts(Signal,Transition)
    #
    # This removes drifts from the data using a forward-backward (non-causal) filter.
    # NOTE: If you are doing directed information flow analysis, do no use this filter but some other one.
    #
    # In:
    #   Signal : the continuous data to filter
    #
    #   Transition : the transition band in Hz, i.e. lower and upper edge of the transition
    #                (default: [0.5 1])
    #
    #   Attenuation : stop-band attenuation, in db (default: 80)
    #
    # Out:
    #   Signal : the filtered signal
    #
    # Notes:
    #   This function requires the Signal Processing toolbox.
    #
    #                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
    #                                2012-09-01

    # Copyright (C) Christian Kothe, SCCN, 2012, ckothe@ucsd.edu
    #
    # This program is free software; you can redistribute it and/or modify it under the terms of the GNU
    # General Public License as published by the Free Software Foundation; either version 2 of the
    # License, or (at your option) any later version.
    #
    # This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
    # even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    # General Public License for more details.
    #
    # You should have received a copy of the GNU General Public License along with this program; if not,
    # write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
    # USA


    # design highpass FIR filter
    # Nyquist rate.
    nyq_rate = SRate / 2

    # Width of the roll-off region.
    width = 2*(Transition[1]-Transition[0]) / nyq_rate

    num_of_taps, beta = kaiserord(Attenuation, width)
    if num_of_taps % 2 == 0:
        num_of_taps = num_of_taps + 1

    # Estimate the filter coefficients.
    B = firwin(num_of_taps, Transition[1]/nyq_rate, window=('kaiser', beta), pass_zero=False)
    Signal = filtfilt(B,1, Signal,axis=-1)

    return Signal

