"""
2_preprocessing.py
Alex Anwyl-Irvine 2021

This is fairly standard:

2. Clean files
    a. Filter
    b. Downsample
    c. ICA denoising

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


##############################
#%% 2. Filtering & Denoising #
##############################
maxpath=join(constants.BASE_DIRECTORY, 'maxfilter_mne')
flist = [f for f in listdir(maxpath) if 'fif' in f]
indir = maxpath
outdir = join(constants.BASE_DIRECTORY, 'cleaned')
scriptpath = join(constants.BASE_DIRECTORY, 'b_scripts')
pythonpath = constants.PYTHON_PATH
overwrite = True
preprocess.preprocess_cluster(flist, indir, outdir, scriptpath, pythonpath ,overwrite, constants.REPO_PATH)

#%% check output
started = listdir(maxpath)
done = listdir(outdir)
extension_names =  ['noeog_noecg_clean_raw.fif','noecg_clean_raw.fif',
                    'noeog_clean_raw.fif','clean_raw.fif']
success = []
for file in started:
    poslist = [f'{file.split(".")[0]}_{i}' for i in extension_names]
    match = [i in done for i in poslist]
    success.append(np.sum(match))

count = [np.sum([i.split('.')[0] in ii for ii in done]) for i in started]
print(sum(success) / len(success))

dropped = np.array(started)[np.array(success) == 0]

#%% rename some files that didn't load
# This is because we can't concetenate them due to bad channels excluded by maxfilter
# so try again
for file in dropped:
    if '-' in file: # if we are second part, rename
        os.system(f'mv {join(maxpath, file)} {join(maxpath, file.replace("-", "_"))}')
overwrite = False
preprocess.preprocess_cluster(flist, indir, outdir, scriptpath, pythonpath ,overwrite)

#%% check number of files for each UID
uid = list(set([i.split('_')[0] for i in started]))
count = [len([i for i in done if ii in i]) for ii in uid]

#%%filenames for manual ica -- where components of blinks etc are not clear
"""
NOTE: This requires manual input from the user. You need to run the first block, 
inspect the saved out component figures, enter the index of components in ica.exclude, 
then inspect the raw2 plot 

You may need to change the start time if there is no data at t=120 seconds

"""
man_ica = [f for f in listdir(outdir) if 'no' in f]
i = 0

# CHANGE THIS TO WHEREVER YOUR PLOTS ARE BEING SAVED
plot_out = '/home/ai05/'
#%% RUN THIS BLOCK THE FIRST TIME
i = 11
f = man_ica[i]
raw = mne.io.read_raw_fif(f'{outdir}/{f}', preload=True)
raw.plot(start=120).savefig('/home/ai05/raw1.png')
#%%
ica = mne.preprocessing.ICA(n_components=25, method='fastica').fit(raw)
comps = ica.plot_components()
comps[0].savefig(f'{plot_out}comp1.png')
comps[1].savefig(f'{plot_out}comp2.png')
print(man_ica[i])
#%% REPEAT THE BLOCKS BELOW FOR EACH FILE
raw.save(f'{outdir}/{f.split("_")[0]}_{f.split("_")[1]}_clean_raw.fif', overwrite=True)
i +=1
print(f'{i+1} out of {len(man_ica)}')
f = man_ica[i]
raw = mne.io.read_raw_fif(f'{outdir}/{f}', preload=True)
ica = mne.preprocessing.ICA(n_components=25, method='fastica').fit(raw)
comps = ica.plot_components()
comps[0].savefig(f'{plot_out}comp1.png')
comps[1].savefig(f'{plot_out}comp2.png')
raw.plot(start=120).savefig(f'{plot_out}raw1.png')
print(man_ica[i])
#%% SELECT THE COMPONENTS HERE
# change inds and decide
ica.exclude =[1]
ica.apply(raw)
# if you need to plot the channels
# CHECK THE PLOT TO SEE IF YOU PICKED A GOOD INDEX
# NOTE: you may need to change the start time
raw.plot(start=120).savefig(f'{plot_out}raw2.png')

#%% Identify participants with valid clean files
clean = [f for f in listdir(outdir) if 'no' not in f]
left_p = list(set([i.split('_')[0] for i in clean ]))