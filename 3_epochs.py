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
import pandas as pd

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
#%%
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

#%%
"""
EPOCHS WITH METADATA

"""
#%% load in behavioural data
datadir = join(constants.BASE_DIRECTORY, 'behav')
meta = pd.read_csv(join(constants.BASE_DIRECTORY, 'Combined3.csv'))
data_files = listdir(datadir)
data_trials = [i for i in data_files if 'trials' in i]
df = pd.read_csv(join(datadir, data_trials[1]), delimiter="\t")
all_trials = pd.DataFrame(columns=list(df.columns) + ['id'])
good_ids = []
for _id in ids:
    _file = [f for f in data_trials if _id in f]
    if len(_file) == 0:
        continue
    try:
        _df = pd.read_csv(join(datadir, _file[0]), delimiter="\t")
    except:
        continue
    _idcol = pd.DataFrame({'id': [_id]*len(_df)})
    _df = _df.join(_idcol)
    all_trials = all_trials.append(_df, ignore_index=True)
    good_ids.append(_id)

#%% Try with postcue
event_dict = {'L_CUE': 250,'R_CUE': 251,'N_CUE': 252,}
time_dict = {'tmin': -0.5,'tmax': 1.5,'baseline': (None,0)}
result = epoch_multiple_meta(
    ids=good_ids,
                                     event_dict = event_dict,
                                     time_dict=time_dict,
                                     indir=cleandir,
                                     outdir=join(constants.BASE_DIRECTORY, 'epoched'),
                                     file_id='metapostcue',
                                     njobs=10,
                                     all_trials=all_trials)

#%% mop up and manually align error trials
epodir = join(constants.BASE_DIRECTORY, 'epoched')
check = [i for i in listdir(epodir) if 'metapostcue' in i]

checklist = []
for file in check:
    epo = mne.epochs.read_epochs(join(epodir, file))
    try:
        checklist.append([(epo[i]._name, int(epo[i].metadata.cue_dir)) for i in range(len(epo))])
    except:
        checklist.append([])
