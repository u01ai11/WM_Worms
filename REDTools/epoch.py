from os.path import join
from os import listdir
import os
import numpy as np
import mne
import joblib
import pandas as pd

def epoch_participant(part_files, event_dict,time_dict, indir, outdir, file_id):
    """
    Epochs a single participant's files and combines them if there is more
    Filters and downsamples as wll

    :param part_files: list of files for single participant
    :param event_dict: dictionary of events for the epoch
    :param time_dict: times for surrounding the events
    :param indir: directory with files
    :param outdir: directory you want saved
    :param file_id: unique id to add to the output file name
    :return:
    """
    epochs = [epoch(f, event_dict, time_dict, indir) for f in part_files]
    if len(epochs) > 1:
        epochs = mne.concatenate_epochs(epochs)
    else:
        epochs = epochs[0]

    p_id = part_files[0].split('_')[0]
    epochs.save(join(outdir, f'{p_id}_{file_id}-epo.fif'))
    return epochs

def epoch(file, event_dict, time_dict, indir):
    """
    Epochs a single file
    filters and downsamples, removes blinks etc
    returns epochs object

    :param file: file to read
    :param event_dict: the dictionary of events
    :param time_dict: times to form epochs
    :param indir: filepath
    :return:
    """
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
    """
    Takes a list of participant IDs and a folder containing preprocessed MEG scans, then extracts all epochs
    Can work on a SLURM cluster or using joblib on seperate CPUs (njobs default is 10)


    :param ids: list of ids
    :param event_dict: dictionary with event names and codes
    :param time_dict: times for epochs
    :param indir: directory containing raw files
    :param outdir: output directory
    :param file_id: unique identifier to append to output filenames
    :param cluster: boolean to use SLURM or not
    :param scriptdir: directory to save scripts for each job
    :param pythonpath: path to python exec. for the SLURM jobs
    :return:
    """
    all_files = [f for f in listdir(indir) if 'no' not in f]
    # if we are not on the cluster run with joblib
    if cluster == False:
        part_files_all = [[i for i in all_files if p in i] for p in ids]
        joblib.Parallel(n_jobs=10)(
            joblib.delayed(epoch_participant)(part_files, event_dict,time_dict, indir, outdir, file_id) for part_files in part_files_all)
    # if we are on the cluster
    else:
        for _id in ids:
            print(_id)
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


def epoch_multiple_meta(ids, event_dict,time_dict, indir, outdir, file_id, njobs, all_trials):
    """
    Epoch all ids provided, and attatch the meta dat also
    Save with the id name and custom file_id

    It cascades through four functions:
        1. this one
        2. epoch_partcipant_meta - calculates unfiltered epochs, then combines them, calculates the correct raw file for 3
        3. epoch_no_filter - calculates unfiltered epocs
        4. epoch_meta - combines events using the raw file, this is the first one if more than one exists

    :param ids: a list of ids for file names
    :param event_dict: events for the epochs
    :param time_dict: timing around the events
    :param indir: where the raw data is stored
    :param outdir: where you want the epochs to be saved
    :param file_id: custom file ID
    :param njobs: how many parallel jobs to run
    :param all_trials: DatFrame containing all the meta data for all participants
    :return:
    """

    all_files = [f for f in listdir(indir) if 'no' not in f]
    part_files_all = [[i for i in all_files if p in i] for p in ids]

    all_es, missed = joblib.Parallel(n_jobs=njobs)(
        joblib.delayed(epoch_participant_meta)(part_files, event_dict,time_dict, indir, outdir, file_id, all_trials, _id) for part_files, _id in zip(part_files_all, ids)
    )

    return all_es, missed


def epoch_participant_meta(part_files, event_dict,time_dict, indir, outdir, file_id, all_trials, _id):
    """
    Epochs files from single participant and includes meta data

    :param part_files: List of files for this participant
    :param event_dict: dictionary of events to be extracted
    :param time_dict: dictionary of times for epochs
    :param indir: location of raw files
    :param outdir: output directory for epochs
    :param file_id: unique identifier for output trials
    :param all_trials: pandas dataframe containing Worms output data
    :return:
    """
    p_id = part_files[0].split('_')[0]
    if os.path.isfile(join(outdir, f'{p_id}_{file_id}-epo.fif')):
        return [], []


    epochs = [epoch_no_filter(f, event_dict, time_dict, indir, _id) for f in part_files] # get epochs

    # concatenate if needed
    if len(epochs) > 1:
        epochs = mne.concatenate_epochs(epochs) # concat
        # work out which raw file was first using timestamps
        s_times = []
        for f in part_files:
            raw = mne.io.read_raw_fif(join(indir, f))
            s_times.append((f, raw.info['meas_date'].timestamp()))
        file = sorted(s_times,key=lambda row: row[1])[0][0]

    else:
        epochs = epochs[0]
        file = part_files[0]

    ep_meta = epoch_meta(file, epochs, indir, all_trials, _id)
    epochs = [i[0] for i in ep_meta]
    missed_epochs = [i[1] for i in ep_meta]

    p_id = part_files[0].split('_')[0]
    epochs.save(join(outdir, f'{p_id}_{file_id}-epo.fif'), overwrite=True)
    return epochs, missed_epochs

def epoch_no_filter(file, event_dict, time_dict, indir, _id):
    """
    epoch from raw but without a filter

    :param file: raw file name
    :param event_dict: dictiopnary for events
    :param time_dict: dictionary for epoch cuts
    :param indir: were we are finding the raw file
    :param _id: participant ID
    :return:
    """
    raw = mne.io.read_raw_fif(join(indir, file), preload=True) # read file
    # detect blinks
    eog_events = mne.preprocessing.find_eog_events(raw)
    n_blinks = len(eog_events)
    # Center to cover the whole blink with full duration of 0.5s:
    onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_blinks)
    raw.set_annotations(mne.Annotations(onset, duration, ['bad blink'] * n_blinks,orig_time=raw.info['meas_date']))

    events = mne.find_events(raw, shortest_event=1) # read any events from one sanple
    epochs = mne.Epochs(raw, events, event_dict, tmin=time_dict['tmin'],
                        tmax=time_dict['tmax'], baseline=time_dict['baseline'],
                        preload=True, proj=True) # extract epochs
    return epochs

def epoch_meta(file, epochs, indir, all_trials, _id):
    """
    adds meta data to epochs
    :param file:
    :param epochs:
    :param event_dict:
    :param time_dict:
    :param indir:
    :param all_trials:
    :param _id:
    :return:
    """

    this_trials = all_trials[all_trials.id == _id]  # get trials for this ID
    this_trials = this_trials[this_trials.prac != 1]  # ignore practice trials
    raw = mne.io.read_raw_fif(join(indir, file), preload=True) # read file
    events = mne.find_events(raw, shortest_event=1) # read any events from one sanple
    #match to behavioural data
    # find first event for behavioural logging zero time
    b_zero = this_trials.iloc[0].iti_onset

    #match to behavioural data
    # find first event for behavioural logging zero time
    b_zero = this_trials.iloc[0].iti_onset

    # sometimes the first event doesn't come through, so we need to adjust our zero-time for this!
    acceptable_events = [201,202,203,250,251,252,205,240,241] # first events we will accept
    e_names = ['ITI', 'STIM', 'DELAY', 'LEFT_C', 'R_CUE', 'N_CUE', 'POSTCUE', 'PROBE_L', 'PROBE_R'] # names of events
    first_ITI = this_trials.iloc[0].iti # ITI of the first trial
    adjustments =[0, first_ITI, first_ITI+1000, first_ITI+2000, first_ITI+2000, first_ITI+2000, first_ITI+2500,
                  first_ITI+3500,first_ITI+3500,] # adjustments for these events
    # same for MEG events
    # now find first acceptable event index
    first = [(i, val) for i, val in enumerate(events[:,2]) if val in acceptable_events][0]
    first_t = events[first[0],0] # get time stamp
    t_zero = first_t - (adjustments[acceptable_events.index(first[1])]) # adjust according to adjustments

    # for every trial work out an encapsulating time range zero'd to the first ITI
    this_trials['z_start'] = this_trials['iti_onset'] - b_zero
    this_trials['z_end'] = this_trials['probe_onset'] - b_zero
    this_trials['z_end'] = this_trials['z_end'] + 5000
    # loop through epochs and find matching meta data
    meta_df = pd.DataFrame(columns=this_trials.columns)
    unfound_ind = []
    for ind in range(len(epochs)):
        this_ztime = epochs[ind].events[0,0] - t_zero
        if this_ztime < 0:
            t_zero = epochs[ind].events[0,0]
            this_ztime = epochs[ind].events[0, 0] - t_zero
        mask = (this_trials['z_start'] <= this_ztime) & (this_trials['z_end'] >= this_ztime)
        if mask.sum() == 0:
            unfound_ind.append(ind)
        meta_df = meta_df.append(this_trials[mask], ignore_index=True)
    try:
        epochs.metadata = meta_df
    except:
        print(f'Issue with {_id} {file}; Only {len(meta_df)} trials matched out of {len(epochs)} epochs!')
    epochs.filter(1, 200., fir_design='firwin', skip_by_annotation='edge')
    epochs.resample(600., npad='auto')
    return epochs, unfound_ind