"""
1_preprocessing.py
Alex Anwyl-Irvine 2021

This is fairly standard:

1. MaxFilter files
2. Clean files
    a. Filter
    b. Downsample
    c. ICA denoising

"""
from os.path import join
from os import listdir
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

for file in listdir(join(constants.BASE_DIRECTORY)):
    max_opts = dict(maxf_cmd = 'maxfilter_2.2.12',
        f = join(constants.BASE_DIRECTORY, 'raw', file),
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
        lg = join(constants.BASE_DIRECTORY, 'b_logs', f'{file.split("raw")[0]}.log'))

    preprocess.maxFilt(cluster=True, **max_opts)

##############################
#%% 2. Filtering & Denoising #
##############################
