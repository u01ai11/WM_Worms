"""
3_epochs.py
Alex Anwyl-Irvine 2021

Consider only cleaned files:
    1. Create trial epochs
    2. Exclude p's with no evoked signal

"""

from os.path import join
from os import listdir
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


#%%

def epoch_participant(part_files, event_dict,time_dict, indir, outdir, file_id):
    epochs = [epoch(f, event_dict, time_dict, indir) for f in part_files]
    if len(epochs) > 1:
        epochs = mne.concatenate_epochs(epochs)
    else:
        epochs = epochs[0]

    p_id = part_files[0].split('_')[0]
    #epochs.save(join(outdir, f'{p_id}_{file_id}-epo.fif'))
    return epochs

def epoch(file, event_dict, time_dict, indir):
    # load the first file
    raw = mne.io.read_raw_fif(join(indir, file), preload=True)
    raw._first_samps[0] = 0
    # Apply band-pass filter
    raw.filter(1, 200., fir_design='firwin', skip_by_annotation='edge')
    # detect blinks
    eog_events = mne.preprocessing.find_eog_events(raw)
    n_blinks = len(eog_events)
    # Center to cover the whole blink with full duration of 0.5s:
    onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_blinks)
    raw.set_annotations(mne.Annotations(onset, duration, ['bad blink'] * n_blinks,orig_time=raw.info['meas_date']))
    events = mne.find_events(raw, min_duration=0.002)
    epochs = mne.Epochs(raw, events, event_dict, tmin=time_dict['tmin'],
                        tmax=time_dict['tmax'], baseline=time_dict['baseline'],
                          preload=True, proj=True)
    epochs.pick_types(meg=True, exclude='bads')
    epochs.resample(600., npad='auto')
    return epochs

#%% initial processing
cleandir = join(constants.BASE_DIRECTORY, 'cleaned') # dire
clean = [f for f in listdir(cleandir) if 'no' not in f]
ids = list(set([i.split('_')[0] for i in clean]))
ids.sort()
i = 1
part = ids[i]
part_files = [i for i in clean if part in i]

event_dict = {
    'L_CUE': 250,
    'R_CUE': 251,
    'N_CUE': 252,
}

time_dict = {
    'tmin': 0.4,
    'tmax': 1.5,
    'baseline': (0.4,0.5)
}


epochs = epoch_participant(part_files=part_files,
                           event_dict=event_dict,
                           time_dict=time_dict,
                           indir=cleandir,
                           outdir=None,
                           file_id=None)