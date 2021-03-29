from os.path import join
from os import listdir
import os
import numpy as np
import mne
import joblib
import pandas as pd
import sys
import traceback

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

    output = joblib.Parallel(n_jobs=njobs)(
        joblib.delayed(epoch_participant_meta)(part_files, event_dict,time_dict, indir, outdir, file_id, all_trials, _id) for part_files, _id in zip(part_files_all, ids)
    )


    return output


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
        #epochs = mne.read_epochs(join(outdir, f'{p_id}_{file_id}-epo.fif'), preload=False)
        epochs = f'{p_id}_{file_id}-epo.fif'
        return epochs, [], False

    try:
        # First we read in epochs from all passed in participant files
        datelist = []
        epolist = []
        for file in part_files:
            raw = mne.io.read_raw_fif(join(indir, file), preload=True)
            datelist.append(raw.info['meas_date'])
            #events = find_events_CBU(raw)
            events = mne.find_events(raw, shortest_event=1)
            epochs = mne.Epochs(raw, events, event_dict, tmin=time_dict['tmin'],
                                tmax=time_dict['tmax'], baseline=time_dict['baseline'],
                                preload=True, proj=True)
            epolist.append(epochs)



        # if we had more than one, then order them by date and calculate seconds difference for offsetting
        if len(part_files) > 1:
            epotime = [(d.timestamp(), e) for d, e in zip(datelist, epolist)] # tuples of time and epochs
            epotime = sorted(epotime, key=lambda row: row[0]) # sort epochs
            epolist = [i[1] for i in epotime]
        if len(part_files) > 1:
            epotime = [(d.timestamp(), e) for d, e in zip(datelist, epolist)] # tuples of time and epochs
            epotime = sorted(epotime, key=lambda row: row[0]) # sort epochs
            time_diffs = [epotime[i+1][0] - epotime[0][0] for i in range(len(epotime)-1)] # workout offset times for all in lists relative to first scan
            for off_ep in range(len(time_diffs)): # now adjust all these epochs (other than first one) to account for offset
                for i in range(len(epolist[off_ep+1])): # loop through epoch events
                    epolist[off_ep+1].events[i, 0] = epolist[off_ep+1].events[i, 0] + (time_diffs[off_ep] * 1000) # add this offset
            #make sure nchan the same
            nchans = [i.info['nchan'] for i in epolist]
            if len(set(nchans)) > 1:
                for i in range(len(epolist)):
                    epolist[i].pick_types(meg=True, chpi=False)
            epo = mne.concatenate_epochs(epolist, add_offset=False) # concatenate, disabling MNEs default offsetting
        else:
            epo = epolist[0]

        ep_meta = epoch_meta(epo, indir, all_trials, _id)
        epochs = ep_meta[0]
        missed_epochs = ep_meta[1]

        p_id = part_files[0].split('_')[0]
        epochs.save(join(outdir, f'{p_id}_{file_id}-epo.fif'), overwrite=True)
        error = False
    except:
        error = traceback.format_exc()
        epochs = []
        missed_epochs = []
    return epochs, missed_epochs, error


def epoch_meta(epo, indir, all_trials, _id):
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
    # calculate start and ends of epochs in time since start of recording
    this_trials['end'] = this_trials['probe_onset'] + this_trials['resp_onset'] + this_trials['resp_duration']
    this_trials['start'] = this_trials['iti_onset']

    # settings for iterating
    trial_adjust = 0
    iterations = 0
    itercrement = 250
    iterlim = 10000
    revs = 0

    # optimisation for speed (i.e. don't perform operations on pandas DF)
    time_array = this_trials[['start', 'end']].to_numpy()  # array is faster
    epo_ev = epo.events.copy() # copy to prevent issues with original object
    aligned = False # flag for alignment
    while not aligned:
        print(iterations)
        unfound_ind = []
        trial_inds = []
        for ind in range(len(epo)):
            epo_s = epo_ev[ind, 0]  # this epoch's time in ms
            mask = (time_array[:, 0] - trial_adjust <= epo_s) & (time_array[:, 1] - trial_adjust >= epo_s)
            if mask.sum() == 0:
                unfound_ind.append(ind)
            else:
                trial_inds.append(np.where(mask == True)[0][0])
        if len(unfound_ind) == 0:
            meta_df = this_trials.iloc[trial_inds]
            epo.metadata = meta_df
            aligned = True
        else:
            trial_adjust = trial_adjust + itercrement
            iterations += 1
            if iterations > iterlim:
                # reverse if we haven't already
                if itercrement > 0:
                    iterations = 0
                    trial_adjust = 0
                    itercrement = 0 - itercrement
                else:
                    # break if we have already
                    raise RuntimeError(f'Failed to find alignments. {len(unfound_ind)} out of {len(epo)} epochs misallinged')
                    break
    # we can filter somewhere else (commented out for speed)
    # epo.filter(1, 200., fir_design='firwin', skip_by_annotation='edge')
    # epo.resample(600., npad='auto')
    return epo, unfound_ind

def find_events_CBU(raw):
    """
    Find real events by reconstructing STIM from 8bit channel

    :param raw:
    :return: events_v the real events
    """
    stim_chs = ('STI001', 'STI002', 'STI003', 'STI004', 'STI005', 'STI006', 'STI007', 'STI008')
    stim_data = (raw
                 .copy()
                 .load_data()
                 .pick_channels(stim_chs)
                 .get_data())
    stim_data /= 5  # Signal is always +5V

    # First channel codes for bit 1, second for bit 2, etc.
    for stim_ch_idx, _ in enumerate(stim_chs):
        stim_data[stim_ch_idx, :] *= 2 ** stim_ch_idx

    # Create a virtual channel which is the sum of the individual channels.
    stim_data = stim_data.sum(axis=0, keepdims=True)
    info = mne.create_info(['STI_VIRTUAL'], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_data, info, first_samp=raw.first_samp)

    events_v = mne.find_events(stim_raw, stim_channel='STI_VIRTUAL',
                               shortest_event=1)
    return events_v


def epoch_downsample_cluster(files, epodir, high, low, rate, savgol, scriptdir, pythonpath):

    _tracker = 0
    for file in files:
        _tracker+=1
        pycom = f"""
import sys
sys.path.insert(0, '/home/ai05/WM_Worms')
from REDTools import epoch
epoch.epoch_downsample('{file}','{epodir}', {high}, {low}, {rate}, {savgol})
                    """
        # save to file
        print(pycom, file=open(join(scriptdir, f'{_tracker}_epoch.py'), 'w'))
        # construct csh file
        tcshf = f"""#!/bin/tcsh
                        {pythonpath} {join(scriptdir, f'{_tracker}_epoch.py')}
                                """
        # save to directory
        print(tcshf, file=open(join(scriptdir, f'{_tracker}_epoch.csh'), 'w'))
        # execute this on the cluster
        os.system(f"sbatch --job-name=epoch_{_tracker} --mincpus=4 -t 0-3:00 {join(scriptdir, f'{_tracker}_epoch.csh')}")


def epoch_downsample(eponame,epodir, high, low, rate, savgol):
    """
    Take an epoch and downsample it to 200Hz
    :param epoch:
    :return:

    """
    epo = mne.epochs.read_epochs(join(epodir, eponame))
    # if already filtered, don't do it again
    if (epo.info['sfreq'] == rate) & (epo.info['lowpass'] == low):
        return eponame
    # we can filter somewhere else (commented out for speed)
    if savgol:
        epo.savgol_filter(h_freq=high)
    else:
        epo.filter(high, low, fir_design='firwin', skip_by_annotation='edge')
    epo.resample(rate, npad='auto')
    epo.save(join(epodir, eponame),overwrite=True)

    return eponame

