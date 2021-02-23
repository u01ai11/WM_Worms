"""
1_preprocessing.py
Alex Anwyl-Irvine 2021

This is fairly standard:

2. Clean files
    a. Filter
    b. Downsample
    c. ICA denoising

"""
from os.path import join
from os import listdir
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
maxpath=join(constants.BASE_DIRECTORY, 'MaxFiltered')
flist = [f for f in listdir(maxpath) if 'fif' in f]
indir = maxpath
outdir = join(constants.BASE_DIRECTORY, 'cleaned')
scriptpath = join(constants.BASE_DIRECTORY, 'b_scripts')
pythonpath = constants.PYTHON_PATH
overwrite = True
preprocess.preprocess_cluster(flist, indir, outdir, scriptpath, pythonpath ,overwrite)

#%% check output
started = listdir(maxpath)
done = listdir(outdir)
dropped = [i for i in started if i not in done]

dropped[0]