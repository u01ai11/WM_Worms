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
    from REDTools import epoch
except:
    import sys
    sys.path.insert(0, '/home/ai05/WM_Worms')
    import constants
    from REDTools import epoch

#%% initial processing - postcue
cleandir = join(constants.BASE_DIRECTORY, 'cleaned') # dire
clean = [f for f in listdir(cleandir) if 'no' not in f]
ids = list(set([i.split('_')[0] for i in clean]))
ids.sort()

event_dict = {'L_CUE': 250,'R_CUE': 251,'N_CUE': 252,}

time_dict = {'tmin': 0.5,'tmax': 1.5,'baseline': None}

epochs = epoch.epoch_multiple(ids=ids,
                                 event_dict=event_dict,
                                 time_dict= time_dict,
                                 indir=cleandir,
                                 outdir=join(constants.BASE_DIRECTORY, 'epoched'),
                                 file_id='postcue',
                                 cluster=True,
                                 scriptdir=join(constants.BASE_DIRECTORY, 'b_scripts'),
                                 pythonpath='/home/ai05/.conda/envs/mne_2/bin/python')

#%% initial processing probe

event_dict =  {'L_PROBE': 240, 'R_PROBE': 241}

time_dict = {'tmin': 0.0,'tmax': 2,'baseline': (0.0,0.1)}

epochs = epoch.epoch_multiple(ids=ids,
                                 event_dict=event_dict,
                                 time_dict= time_dict,
                                 indir=cleandir,
                                 outdir=join(constants.BASE_DIRECTORY, 'epoched'),
                                 file_id='probe',
                                 cluster=True,
                                 scriptdir=join(constants.BASE_DIRECTORY, 'b_scripts'),
                                 pythonpath='/home/ai05/.conda/envs/mne_2/bin/python')

#%% initial processing whole cue

#%% initial processing - postcue
cleandir = join(constants.BASE_DIRECTORY, 'cleaned') # dire
clean = [f for f in listdir(cleandir) if 'no' not in f]
ids = list(set([i.split('_')[0] for i in clean]))
ids.sort()

event_dict = {'L_CUE': 250,'R_CUE': 251,'N_CUE': 252,}

time_dict = {'tmin': -0.5,'tmax': 1.5,'baseline': (None,0)}

epochs = epoch.epoch_multiple(ids=ids,
                                 event_dict=event_dict,
                                 time_dict= time_dict,
                                 indir=cleandir,
                                 outdir=join(constants.BASE_DIRECTORY, 'epoched'),
                                 file_id='wholecue',
                                 cluster=True,
                                 scriptdir=join(constants.BASE_DIRECTORY, 'b_scripts'),
                                 pythonpath='/home/ai05/.conda/envs/mne_2/bin/python')

#%% load in behavioural data
datadir = join(constants.BASE_DIRECTORY, 'behav')
data_f = listdir(datadir)
part_f = [[ii for ii in data_f if i in ii] for i in ids]