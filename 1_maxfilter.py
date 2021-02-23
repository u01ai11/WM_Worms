"""
1_maxfiltering.py
Alex Anwyl-Irvine 2021

This is fairly standard:

1. MaxFilter files

"""
from os.path import join
from os import listdir
import numpy as np

try:
    import constants
    from REDTools import preprocess
except:
    import sys
    sys.path.insert(0, '/home/ai05/WM_Worms')
    import constants
    from REDTools import preprocess

############################
#%% 1. MAXFILTER RAW FILES #
############################

for file in listdir(join(constants.BASE_DIRECTORY, 'raw')):
    fpath = join(constants.BASE_DIRECTORY, 'raw', file)
    max_opts = dict(max_cmd = 'maxfilter_2.2.12',
        f = join(fpath),
        o = join(constants.BASE_DIRECTORY, 'MaxFiltered', file),
        trans = 'default',
        frame = 'head',
        regularize = 'in',
        st = '10',
        cor = '0.98',
        orig = '0 0 45',
        inval = '8',
        outval = '3',
        movecomp = 'inter',
        bads_cmd = '',
        lg = join(constants.BASE_DIRECTORY, 'b_logs', f'{file.split(".")[0]}.log'),
        hpi_g = '0.98',
        hpi_e = '5')

    preprocess.maxFilt(cluster=True, **max_opts)

#%% check progress
started = listdir(join(constants.BASE_DIRECTORY, 'raw'))
done = listdir(join(constants.BASE_DIRECTORY, 'MaxFiltered'))
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

#%% retry but without movecomp
for file in retry:
    fpath = join(constants.BASE_DIRECTORY, 'raw', file)
    max_opts['f'] = fpath
    max_opts['o'] = join(constants.BASE_DIRECTORY, 'MaxFiltered', file)
    max_opts['lg'] = join(constants.BASE_DIRECTORY, 'b_logs', f'{file.split(".")[0]}.log'),
    max_opts['movecomp'] = False
    preprocess.maxFilt(cluster=True, **max_opts)

#%% plot diagnostic plots from the logs
summary_data = []
logpath = join(constants.BASE_DIRECTORY, 'b_logs')
for log in listdir(logpath):
    tmp = preprocess.plot_MaxLog(logpath=join(logpath, log),
                outpath=join(constants.BASE_DIRECTORY, 'plots'), plot=True)
    summary_data.append(tmp)

