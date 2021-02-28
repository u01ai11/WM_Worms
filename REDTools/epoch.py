from os.path import join
from os import listdir
import os
import numpy as np
import mne
import joblib

def epoch_participant(part_files, event_dict,time_dict, indir, outdir, file_id):
    epochs = [epoch(f, event_dict, time_dict, indir) for f in part_files]
    if len(epochs) > 1:
        epochs = mne.concatenate_epochs(epochs)
    else:
        epochs = epochs[0]

    p_id = part_files[0].split('_')[0]
    epochs.save(join(outdir, f'{p_id}_{file_id}-epo.fif'))
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

def epoch_multiple(ids, event_dict,time_dict, indir, outdir, file_id, cluster, scriptdir, pythonpath):
    all_files = [f for f in listdir(indir) if 'no' not in f]
    # if we are not on the cluster run with joblib
    if cluster == False:
        part_files_all = [[i for i in all_files if p in i] for p in ids]
        joblib.Parallel(n_jobs=10)(
            joblib.delayed(epoch_participant)(part_files, event_dict,time_dict, indir, outdir, file_id) for part_files in part_files_all)
    # if we are on the cluster
    else:
        for _id in ids:
            #construct file string for the cluster command
            id_files = [i for i in all_files if _id in i]
            pycom = f"""
import sys
sys.path.insert(0, '/home/ai05/WM_Worms')
from REDTools import epoch

epoch.epoch_participant({id_files}, {event_dict},{time_dict}, '{indir}', '{outdir}', '{file_id}')
            """
        # save to file
        print(pycom, file=open(join(scriptdir, f'{_id}_epoch.py'), 'w'))
        # construct csh file
        tcshf = f"""#!/bin/tcsh
            {pythonpath} {join(scriptdir, f'{_id}_epoch.py')}
                    """
        # save to directory
        print(tcshf, file=open(join(scriptdir, f'{_id}_epoch.csh'), 'w'))
        # execute this on the cluster
        os.system(f"sbatch --job-name=epoch_{_id} --mincpus=4 -t 0-3:00 {join(scriptdir, f'{_id}_epoch.csh')}")