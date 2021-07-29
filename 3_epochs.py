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
import joblib
import copy
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

#%% save the all files for later
all_trials.to_csv(join(constants.BASE_DIRECTORY, 'all_trials.csv'))
#%% whittle out poor performers
trials = [(i, len(all_trials[all_trials.id == i])) for i in good_ids]
good_ids = [i[0] for i in trials if i[1] > 34]


#%% Try with postcue
event_dict = {'L_CUE': 250,'R_CUE': 251,'N_CUE': 252}
time_dict = {'tmin': -0.5,'tmax': 1.5,'baseline': (None,0)}
result = epoch_multiple_meta(
    ids=good_ids,
                                     event_dict = event_dict,
                                     time_dict=time_dict,
                                     indir=cleandir,
                                     outdir=join(constants.BASE_DIRECTORY, 'epoched'),
                                     file_id='metapostcue',
                                     njobs=5,
                                     all_trials=all_trials)

#%% Try with stim
event_dict = {'STIM': 202}
time_dict = {'tmin': -0.5,'tmax': 4.5,'baseline': None}
result = epoch_multiple_meta(
    ids=good_ids,
                                     event_dict = event_dict,
                                     time_dict=time_dict,
                                     indir=cleandir,
                                     outdir=join(constants.BASE_DIRECTORY, 'new_epochs_2'),
                                     file_id='metastim',
                                     njobs=11,
                                     all_trials=all_trials)

"""
Alex's note to self
- nchans not matching on some files during merge - fix
- small number of epochs -- check all files present
- manually fix some 
"""



#%%
result_e = [(ind, i[2]) for ind, i in enumerate(result) if i[2] != False]
error_ids = [good_ids[i[0]] for i in result_e]
#%% mop up and manually align error trials
epodir = join(constants.BASE_DIRECTORY, 'new_epochs_2')
check = [i for i in listdir(epodir) if 'metastim' in i]

checklength= []
for file in check:
    epo = mne.epochs.read_epochs(join(epodir, file), preload=False)
    tlen = len(all_trials[all_trials.id == file.split('_')[0]]) -4
    elen = len(epo)
    checklength.append((tlen,elen))

#%% filter and downsample the above
# uncomment to do locally
# output = joblib.Parallel(n_jobs=15)(
#         joblib.delayed(epoch_downsample)(file,epodir, 0, 45, 200, False) for files in check
# )

scriptdir = join(constants.BASE_DIRECTORY, 'b_scripts')
pythonpath = '/home/ai05/.conda/envs/mne_2/bin/python'
epoch.epoch_downsample_cluster(check, epodir, 1, 80, 250, False, scriptdir, pythonpath)



#%% Read in and visually inspect the visual evoked response


epodir = join(constants.BASE_DIRECTORY, 'new_epochs')
files = [i for i in listdir(epodir) if 'metastim' in i]
ids = [i.split('_')[0] for i in files]

#%% just look at group average with visual evoked

# First read in the files
def get_epochs(file,dir):
    #load in epochs
    _id = file.split('_')[0]
    epochs = mne.epochs.read_epochs(join(dir, file))
    epochs.pick_types(meg=True)
    epochs.crop(tmin=-0.5, tmax=1)
    epochs.apply_baseline(baseline=(None, 0))
    #evoked = epochs.average()
    return epochs

epochs_list = joblib.Parallel(n_jobs=15)(
    joblib.delayed(get_epochs)(file,epodir) for file in files)

#%%


#%% Loop through and look at the files to determine which are sufficient to proceed with
include = []
for i in range(len(epochs_list)):
    e = copy.deepcopy(epochs_list[i])
    evoked = e.average()
    evoked.plot_joint()
    _id = files[i].split('_')[0]
    print("keep (1) or chuck (anything)...?")
    resp = sys.stdin.read()
    if resp.split('\n')[0] == '1':
        include.append(i)
    else:
        continue

file_includes = np.array(files)[include]
np.save(join(constants.BASE_DIRECTORY, 'good_visual_evoked.npy'), file_includes)