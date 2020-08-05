import os
import mne
from mne import io
from mne.report import Report
import numpy as np
from autoreject import AutoReject, compute_thresholds 
from autoreject import get_rejection_threshold 

# data description
raw = io.read_raw_fif(file_name, preload = True) 
print(raw)
raw.info

#filter
raw.notch_filter(np.arange(50,150,200), n_jobs=1, fir_design='firwin')
raw=raw.copy().filter(0.1, 40., fir_design='firwin')

#events
events = mne.find_events(raw, stim_channel='STI101',shortest_event=1)
print (events)
events_id = {'visual1': 37, 'visual2': 77, 'audio1':117,'audio2': 157}

#epochs 
epochs = mne.Epochs(raw, events , event_id=events_id, tmin=-0.2, tmax=0.5, baseline=(None, 0), reject_by_annotation = True , verbose=True, preload=True)
print(epochs)

# local rejection threshold
this_epoch = epochs['visual1']
picks = mne.pick_types(epochs.info, meg=True, eeg=False, stim=False, eog=False)

ar = AutoReject(picks=picks, random_state=42, n_jobs=1, verbose='tqdm')
epochs_ar, reject_log = ar.fit_transform(this_epoch, return_log=True)

reject_log.plot_epochs(this_epoch, scalings='auto')
epochs_ar.plot(scalings='auto')
