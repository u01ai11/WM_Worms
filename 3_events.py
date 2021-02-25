"""
3_events.py
Alex Anwyl-Irvine 2021

Consider only cleaned files:
    1. Create trial epochs
    2. Exclude p's with no evoked signal

"""

from os.path import join
from os import listdir
import os
import numpy as np
import mne
try:
    import constants
    from REDTools import preprocess
except:
    import sys
    sys.path.insert(0, '/home/ai05/WM_Worms')
    import constants
    from REDTools import preprocess

#%% initial processing
cleandir = join(constants.BASE_DIRECTORY, 'cleaned') # dire
clean = [f for f in listdir(cleandir) if 'no' not in f]
ids = list(set([i.split('_')[0] for i in clean ]))

i = 0
part = ids[i]
part_files = [i for i in clean if part in i]

# load the first file
raw = mne.io.read_raw_fif(join(cleandir, part_files[0]), preload=True)

#%% get events
events = mne.find_events(raw)

timings = [(-0.1, 1), (-0.1, 1), (-0.1, 5), (-0.1, 5), (-0.1, 0.8)]
tnames = ['STIM', 'POSTCUE', 'L_PROBE', 'R_PROBE', 'RESP']
epochs_stim = mne.Epochs(raw, events, event_id=constants.TRIGGERS['STIM'], tmin=-0.1, tmax=1)
epochs_postcue = mne.Epochs(raw, events, event_id=constants.TRIGGERS['POSTCUE'], tmin=-0.1, tmax=1)
epochs_lprobe = mne.Epochs(raw, events, event_id=constants.TRIGGERS['L_PROBE'], tmin=-0.1, tmax=5)
epochs_rprobe = mne.Epochs(raw, events, event_id=constants.TRIGGERS['R_PROBE'], tmin=-0.1, tmax=5)
epochs_resp = mne.Epochs(raw, events, event_id=constants.TRIGGERS['RESP'], tmin=-0.1, tmax=0.8)

epo_list = [mne.Epochs(raw, events, event_id=constants.TRIGGERS[n], tmin=t[0], tmax=t[1]) for n, t in zip(tnames, timings)]

epochs_stim.load_data()
epochs_stim.resample(200., npad='auto')

#evokeds = [i.average() for i in[epochs_lprobe, epochs_rprobe]]
evokeds = [epochs_stim.average()]
mne.viz.plot_evoked_topo(evokeds)