"""
1_maxfiltering.py
Alex Anwyl-Irvine 2021

This is fairly standard:

1. MaxFilter files

"""
from os.path import join
from os import listdir
import numpy as np
import mne
import joblib
from scipy.io import loadmat
import pandas as pd
try:
    import constants
    from REDTools import preprocess
except:
    import sys
    sys.path.insert(0, '/home/ai05/WM_Worms/')
    import constants
    from REDTools import preprocess
import os
############################
#%% 1. MAXFILTER RAW FILES #
############################
splitdir = join(constants.BASE_DIRECTORY, 'raw_split')#
rawdir = join(constants.BASE_DIRECTORY, 'raw')
#%%
# First resplit all files to be maxfilter friendly

def split_raw(file):
    if '-1.fif' in file or '-2.fif' in file:
        if os.path.isfile(join(rawdir,f'{file.split("-")[0]}.fif')):
            print(f'{file} is second part')
            return
    raw = mne.io.read_raw_fif(join(rawdir, file))
    raw.save(join(splitdir, file), split_size='1.80GB')
    print(file)
    return
epochs_list = joblib.Parallel(n_jobs=15)(
    joblib.delayed(split_raw)(file) for file in listdir(rawdir))
#%%
trans2 = join(constants.BASE_DIRECTORY, 'raw_split', '99040_worms_raw-1.fif')

#lets load in the origin points according to the cusotom fit we ran before

origs = loadmat(join(constants.BASE_DIRECTORY, 'maxfilter_3', 'all_orig.mat'))['all_orig']
orig_files = loadmat(join(constants.BASE_DIRECTORY, 'maxfilter_3', 'all_files.mat'))
orig_files = [i[0][0][0] for i in orig_files['all_files']]
files_and_orig = [[a,b] for a,b in zip(origs, orig_files)]
#%%
for file in listdir(splitdir):
#for file in dropped:
    #get orig
    #get_bads
    badfile = f"{file.split('.fif')[0]}_bad.txt"
    print(os.path.exists(join(constants.BASE_DIRECTORY, 'maxfilter_3', badfile)))

#%%
all_files = listdir(splitdir)
baddolist = []
#for i, file in enumerate(all_files):
for file in dropped:
    #get orig
    index_external = orig_files.index(file)
    orig = " ".join([str(int(i)) for i in origs[index_external]])
    fpath = join(splitdir, file)

    #get_bads
    badfile = join(constants.BASE_DIRECTORY, 'maxfilter_3', f"{file.split('.fif')[0]}_bad.txt")
    if os.path.exists(badfile):
        with open(badfile, 'r') as f:
            bads = f.read()
        badrows = [i.split(" ") for i in bads.split('\n')][0:-1]
        #badrows = np.array([[int(ii) for ii in i] for i in badrows])
        badrows = sum(badrows, [])
        bad_ids, bad_counts = np.unique(badrows, return_counts=True)
        bad_perc = bad_counts / len(bads.split('\n'))
        bad_final = bad_ids[bad_perc>.05]
        badstring = " ".join([str(i) for i in bad_final])
        print(badstring)
    else:
        badstring = ""
        continue
    baddolist.append(badstring)

#%
    max_opts = dict(max_cmd = '/neuro/bin/util/maxfilter-2.2',
    f = join(fpath),
    o = join(constants.BASE_DIRECTORY, 'maxfilter_4', file),
    trans = trans2,
    frame = 'head',
    regularize = 'in',
    st = '10',
    cor = '0.98',
    orig = orig,
    inval = '8',
    outval = '3',
    movecomp = 'inter',
    bads_cmd = f'-bad {badstring}',
    lg = join(constants.BASE_DIRECTORY, 'b_logs', f'{file.split(".")[0]}.log'),
    hpi_g = '0.98',
    hpi_e = '5',
    hpi_step = '200')

    maxFilt(cluster=True, **max_opts)

#%% check progress
started = listdir(splitdir)
done = listdir(join(constants.BASE_DIRECTORY, 'maxfilter_4'))
dropped = [i for i in started if i not in done]

#%% check dropped files and then re-run with HPI error allowance
# load reason for each to be failed
reasons = []
retry = []
for file in dropped:
    log_f = join(constants.BASE_DIRECTORY, 'b_logs', f'{file.split(".")[0]}.log')
    fileHandle = open(log_f, "r")
    lineList = fileHandle.readlines()
    fileHandle.close()
    if len(lineList) > 0:
        reasons.append(lineList)
        if 'Unknown' in lineList[-1]:
            retry.append(file)
    else:
        reasons.append('no log file')
        retry.append(file)

#%%
np.save(join(constants.BASE_DIRECTORY, 'failed_movecomp.npy'), retry, allow_pickle=True)

retry = np.load(join(constants.BASE_DIRECTORY, 'failed_movecomp.npy'))

#%% check we can open raw files

for i, file in enumerate(done):
    fpath = join(constants.BASE_DIRECTORY, 'maxfilter_4',file)
    size = os.path.getsize(fpath) /100000
    if size < 10:
        continue
    else:
        raw = mne.io.read_raw_fif(fpath)

#%% check before and after size
sizes = []
for i, file in enumerate(done):
    bef = os.path.getsize(join(splitdir,file)) /100000
    after = os.path.getsize(join(constants.BASE_DIRECTORY, 'maxfilter_2',file)) /100000
    sizes.append((file, bef, after))

#%% retry but without movecomp
for file in retry:
    fpath = join(constants.BASE_DIRECTORY, 'raw', file)
    max_opts['f'] = fpath
    max_opts['o'] = join(constants.BASE_DIRECTORY, 'MaxFiltered', file)
    max_opts['lg'] = join(constants.BASE_DIRECTORY, 'b_logs', f'{file.split(".")[0]}.log'),
    #max_opts['movecomp'] = False
    preprocess.maxFilt(cluster=True, **max_opts)

#%% plot diagnostic plots from the logs
summary_data = []
logpath = join(constants.BASE_DIRECTORY, 'b_logs')
for log in listdir(logpath):
    tmp = preprocess.plot_MaxLog(logpath=join(logpath, log),
                outpath=join(constants.BASE_DIRECTORY, 'plots'), plot=True)
    summary_data.append(tmp)

#%%try_all using MNE maxfilter for comparison!
scriptpath = join(constants.BASE_DIRECTORY, 'b_scripts')
pythonpath = constants.PYTHON_PATH
maxfilt_mne_bulk(files=started,
                 indir=join(constants.BASE_DIRECTORY, 'raw'),
                 outdir=join(constants.BASE_DIRECTORY, 'maxfilter_mne'),
                 scriptpath=scriptpath, pythonpath=pythonpath)

#%% check the undone files
started = listdir(join(constants.BASE_DIRECTORY, 'raw'))
done = [i for i in listdir(join(constants.BASE_DIRECTORY, 'maxfilter_mne')) if '.npy' not in i]
dropped = [i for i in started if i not in done]

#%% retry to replicate errors

maxfilt_mne_bulk(files=dropped,
                 indir=join(constants.BASE_DIRECTORY, 'raw'),
                 outdir=join(constants.BASE_DIRECTORY, 'maxfilter_mne'),
                 scriptpath=scriptpath, pythonpath=pythonpath)

#%% debug options

#more than one
uids = list(set([i.split('_')[0] for i in started]))
uids_split = [i for i in uids if len([ii for ii in started if i in ii]) > 1]
i = 20
_id = uids_split[i]
first_file = sorted([i for i in started if _id in i])

fname = join(constants.BASE_DIRECTORY, 'raw', first_file[0])
#raw = mne.io.read_raw_fif(fname, allow_maxshield='yes').load_data()
raw = mne.io.read_raw_fif(fname, allow_maxshield=True)
raw.crop(tmin=200, tmax=400)
chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)
#%%
ctc = '/imaging/local/software/neuromag/databases/ctc/ct_sparse.fif'
ss_cal = '/imaging/local/software/neuromag/databases/sss/sss_cal.dat'
raw_check = raw.copy()
auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
    raw_check, cross_talk=ctc, calibration=ss_cal,
    return_scores=True, verbose=True)
bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
raw.info['bads'] = bads
raw_sss = mne.preprocessing.maxwell_filter(
    raw, cross_talk=ctc, calibration=ss_cal, verbose=True, head_pos=head_pos)
cbu_max = join(constants.BASE_DIRECTORY, 'MaxFiltered')
cbu_f = [i for i in listdir(cbu_max) if  first_file[1] in i][0]
raw_cbusss = mne.io.read_raw_fif(join(cbu_max, cbu_f))
raw_cbusss.crop(tmin=200, tmax=400)

#%%


raw_sss = mne.preprocessing.maxwell_filter(
    raw, cross_talk=ctc, calibration=ss_cal, verbose=True,
    head_pos=head_pos, destination=(0,0,0.04), coord_frame='head',
    regularize='in', st_duration=10, st_correlation=0.98
    )



inval = '8',
outval = '3',
movecomp = 'inter',
bads_cmd = '',
lg = join(constants.BASE_DIRECTORY, 'b_logs', f'{file.split(".")[0]}.log'),
hpi_g = '0.98',
hpi_e = '5')