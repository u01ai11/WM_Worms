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
maxpath=join(constants.BASE_DIRECTORY, 'maxfilter_3')
flist = [f for f in listdir(maxpath) if 'trans1stdef.fif' in f]
indir = maxpath
outdir = join(constants.BASE_DIRECTORY, 'cleaned_cbu')
scriptpath = join(constants.BASE_DIRECTORY, 'b_scripts')
pythonpath = constants.PYTHON_PATH
overwrite = False


exclude = ['128739138']
flist = [i for i in flist if not any([ii in i for ii in exclude])]

#%% Manual bad segment and bad file removal
# unfortunately we have large artefacts in this dataset, so we need to manually label some artefactual segments
# this is a manual process, as relying on the automated functions below doesn't always work.

# loop through and annotate bad segments in the fif files
for raw_f in flist:
    print(f'{flist.index(raw_f)} out of {len(flist)}')
    raw = mne.io.read_raw_fif(join(maxpath, raw_f), verbose=False)
    fig = raw.plot()
    fig.canvas.key_press_event('a')
    response = input("0 to accept or 1 to reject:")
    if int(response) == 0:
        raw.save(join(maxpath, f'{flist[0][0:-16]}_checked.fif'))
    else:
        print('toobad!')


#%%
preprocess.preprocess_cluster(flist, indir, outdir, scriptpath, pythonpath ,overwrite, constants.REPO_PATH)

#%% check output
started = [i for i in listdir(maxpath) if '.fif' in i]
done = listdir(outdir)
extension_names =  ['_noeog_noecg_clean_','_noeog_clean_',
                    '_noecg_clean','_clean_']
success = []
for file in started:
    poslist = [[f'{os.path.basename(file).split("_")[0]}{i}{ext}' for i in extension_names] for ext in ['raw.fif', 'raw-1.fif']]
    poslist = poslist[0] + poslist[1]
    #print(poslist)
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
i =0
f = man_ica[i]
raw = mne.io.read_raw_fif(f'{outdir}/{f}', preload=True)
raw.plot(start=120).savefig('/home/ai05/raw1.png')
#%%
#raw.filter(1,75)
ica = mne.preprocessing.ICA(method='picard', n_components=15).fit(raw,decim=25)

#ica = mne.preprocessing.ICA(n_components=25, method='fastica').fit(raw,decim=25)
#ica = mne.preprocessing.ICA(method='picard').fit(raw)
comps = ica.plot_components()
comps[0].savefig(f'{plot_out}comp1.png')
comps[1].savefig(f'{plot_out}comp2.png')
print(man_ica[i])
#%% REPEAT THE BLOCKS BELOW FOR EACH FILE
num = os.path.basename(f).split('_')[0]
if os.path.basename(f).split('raw')[1] != '.fif':
    append = os.path.basename(f).split('raw')[1].split('.fif')[0]
else:
    append = ''
append = 'raw' + append + '.fif'

if raw is not None:
    raw.save(f'{outdir}/{num}_clean_{append}', overwrite=True)
    os.remove(f'{outdir}/{f}')

i +=1
print(f'{i+1} out of {len(man_ica)}')
f = man_ica[i]
raw = mne.io.read_raw_fif(f'{outdir}/{f}', preload=True)
if len(raw)/1000 < 60:
    print('file to small, delete!')
    del(raw)
    os.remove(f'{outdir}/{f}')
else:
    raw.plot(start=120).savefig(f'{plot_out}raw1.png')
    ica = mne.preprocessing.ICA(n_components=25, method='picard').fit(raw,decim=25)
    comps = ica.plot_components()
    #comps[0].savefig(f'{plot_out}comp1.png')
    #comps[1].savefig(f'{plot_out}comp2.png')
    ica.plot_sources(raw,start=120, show_scrollbars=False)
    print(man_ica[i])
#%% SELECT THE COMPONENTS HERE
# change inds and decide
ica.exclude =[0,9,10]
raw = ica.apply(raw)
# if you need to plot the channels
# CHECK THE PLOT TO SEE IF YOU PICKED A GOOD INDEX
# NOTE: you may need to change the start time
raw.plot(start=120).savefig(f'{plot_out}raw2.png')

#%% Identify participants with valid clean files
clean = [f for f in listdir(outdir) if 'no' not in f]
left_p = list(set([i.split('_')[0] for i in clean ]))