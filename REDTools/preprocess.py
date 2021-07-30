import mne
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from os.path import join
from time import sleep
import sys
def preprocess_multiple(flist, indir, outdir, overwrite, njobs):
    """ Takes a list of raw files and preprocesses them
    Parameters
    ----------
    :param flist:
        A list of files we want to read in and pre-process
    :param indir:
        Where we find the files
    :param outdir:
        where we want to save those files
    :param overwrite:
        truee or false. whether to overwrite the files if already exist
    :return saved_files:
        A list of files we have saved
    """

    # first check if indir and outdir exist
    # if not outdoor make it
    # if not indir raise error
    if not os.path.isdir(indir):
        raise Exception(f'path {indir} does not exist, edit and try again')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    saved_files = []

    if njobs == 1:
        for i in range(len(flist)):
            savedfile = __preprocess_individual(os.path.join(indir, flist[i]), outdir, overwrite=overwrite)
            saved_files.append(savedfile)
    if njobs > 1:

        saved_files = joblib.Parallel(n_jobs =njobs)(
            joblib.delayed(__preprocess_individual)(os.path.join(indir, thisF), outdir, overwrite) for thisF in flist)

    return saved_files


def preprocess_cluster(flist, indir, outdir, scriptpath, pythonpath ,overwrite, repopath):
    """ Takes a list of raw files and preprocesses them
    Parameters
    ----------
    :param flist:
        A list of files we want to read in and pre-process
    :param indir:
        Where we find the files
    :param outdir:
        where we want to save those files
    :param overwrite:
        truee or false. whether to overwrite the files if already exist
    :return saved_files:
        A list of files we have saved
    """

    # get preprocess individual function as text
    for i in range(len(flist)):
        file = flist[i]
        # if '-1.fif' in file or '-2.fif' in file:
        #     if os.path.isfile(join(indir, f'{file.split("-")[0]}.fif')):
        #         print(f'{file} is second part')
        #         continue
        pythonf = f"""
import sys 
sys.path.insert(0, '{repopath}')
import REDTools.preprocess as red_preprocess
red_preprocess.__preprocess_individual('{indir}/{flist[i]}', '{outdir}', {overwrite})
        """
        # save to file
        print(pythonf, file=open(f'{scriptpath}/preproc_{i}.py', 'w'))

        # construct csh file
        tcshf = f"""#!/bin/tcsh
        {pythonpath} {scriptpath}/preproc_{i}.py
                """
        # save to directory
        print(tcshf, file=open(f'{scriptpath}/preproc_{i}.csh', 'w'))

        # execute this on the cluster
        os.system(f'sbatch --job-name=preproc_{i} --mincpus=5 -t 0-2:00 {scriptpath}/preproc_{i}.csh')


def __preprocess_individual(file, outdir, overwrite):
    """ Internal function for preprocessing raw MEG files
    :param file:
        input file along with path
    :param outdir:
        where we want to save this
    :return: save_file_path:
        a path to the saved and filtered file

    """
    save_file_path = ""

    num =  os.path.basename(file).split('_')[0]
    if os.path.basename(file).split('raw')[1] != '.fif':
        append =os.path.basename(file).split('raw')[1].split('.fif')[0]
    else:
        append = ''

    append = 'raw' + append + '.fif'

    base = os.path.basename(file).split('.')[0]
    # check if any of these files exists, if not overwrite then skip and return path
    # could be any of these names
    check_fnames = [f'{outdir}/{num}_noeog_noecg_clean_{append}',
                    f'{outdir}/{num}_noecg_clean_{append}',
                    f'{outdir}/{num}_noeog_clean_{append}',
                    f'{outdir}/{num}_clean_{append}']

    if np.any([os.path.isfile(f) for f in check_fnames]):
        index = np.where([os.path.isfile(f) for f in check_fnames])[0]
        if not overwrite:
            print(f'file for {os.path.basename(file)}  already exists, skipping to next')
            save_file_path = check_fnames[index[0]]
            return save_file_path

    # read file
    try:
        raw = mne.io.read_raw_fif(file, preload=True)
    except OSError:
        print('could not read ' + file)
        return ''

    raw.filter(0.1, None)
    raw.notch_filter(np.arange(50, 125, 50), filter_length='auto',
                     phase='zero')
    # Run ICA on raw data to find blinks and eog
    try:
        ica = mne.preprocessing.ICA(n_components=25, method='picard').fit(raw)
    except:
        raw.crop(1) # remove the first second due to NaNs
        ica = mne.preprocessing.ICA(n_components=25, method='picard').fit(raw)

    try:
        # look for and remove EOG
        eog_epochs = mne.preprocessing.create_eog_epochs(raw)  # get epochs of eog (if this exists)
        eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)  # try and find correlated components

        # define flags for tracking if we found components matching or not
        no_ecg_removed = False
        no_eog_removed = False

        # if we have identified something !
        if len(eog_inds) >0 :
            ica.exclude.extend(eog_inds)
        else:
            print(f'{os.path.basename(file)} cannot detect eog automatically manual ICA must be done')
            no_eog_removed = True

    except:
        print(f'{os.path.basename(file)} cannot detect eog automatically manual ICA must be done')
        no_eog_removed = True

    # now we do this with hearbeat
    try:
        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)  # get epochs of eog (if this exists)
        ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)  # try and find correlated components

        # if one component reaches above threshold then remove components automagically
        if len(ecg_inds) > 0:
            ica.exclude.extend(ecg_inds)  # exclude top 3 components

        else:  # flag for manual ICA inspection and removal
            print(f'{os.path.basename(file)} cannot detect ecg automatically manual ICA must be done')
            no_ecg_removed = True
    except:
        print(f'{os.path.basename(file)} cannot detect ecg automatically manual ICA must be done')
        no_ecg_removed = True

    # exclude all labelled components
    raw = ica.apply(inst=raw)
    # save the file
    if no_ecg_removed and no_eog_removed:
        outfname = f'{outdir}/{num}_noeog_noecg_clean_{append}'

    elif no_ecg_removed:
        outfname = f'{outdir}/{num}_noecg_clean_{append}'
    elif no_eog_removed:
        outfname = f'{outdir}/{num}_noeog_clean_{append}'
    else:
        outfname = f'{outdir}/{num}_clean_{append}'

    raw.save(outfname, overwrite=overwrite)
    save_file_path = outfname
    # return
    return save_file_path


def maxFilt(cluster=False, **kw):
    """
    A python wrapper for maxfilters command line functions
    :param cluster:
        Boolean value, if False it will run Maxfilter locally, if True it will submit it to the CBUs SLURM cluster
    :param kw:
        Dictionary containing keyword arguments, all are mandetory, but you can pass '' if you don't want to run that option
        maxf_cmd: The command that runs MaxFilter in your environemnt, this can be a command or address to the executable
        f: the input raw file
        o: the output raw file
        The other arguments are the standard options for MaxFilter
        trans, frame, regularize, st, cor (corr), orig, inval (in), outval(out), movecomp, the full bads_cmd, lg


    :return:
    """
    maxf_cmd = kw.get('max_cmd')
    f = kw.get('f')
    o = kw.get('o')
    trans = kw.get('trans')
    frame = kw.get('frame')
    regularize = kw.get('regularize')
    st = kw.get('st')
    cor = kw.get('cor')
    orig = kw.get('orig')
    inval = kw.get('inval')
    outval = kw.get('outval')
    movecomp = kw.get('movecomp')
    bads_cmd = kw.get('bads_cmd')
    lg = kw.get('lg')
    hpi_g = kw.get('hpi_g')
    hpi_e = kw.get('hpi_e')
    hpi_step = kw.get('hpi_step')

    if movecomp == False:
        max_cmd = f"{maxf_cmd} -f {f} -o {o} -trans {trans} -frame {frame} -regularize {regularize}" \
                  f" -st {st} -corr {cor} -origin {orig} -in {inval} -out {outval}" \
                  f" {bads_cmd}-autobad on -force -linefreq 50 -v -hpig {hpi_g} -hpie {hpi_e} | tee {lg}"
    else:
        max_cmd = f"{maxf_cmd} -f {f} -o {o} -trans {trans} -frame {frame} -regularize {regularize}" \
                  f" -st {st} -corr {cor} -origin {orig} -in {inval} -out {outval} -movecomp {movecomp} -hpistep {hpi_step}" \
                  f" {bads_cmd}-autobad on -force -linefreq 50 -v -hpig {hpi_g} -hpie {hpi_e} | tee {lg}"

    if cluster:
        # submit to cluster
        # make bash file
        tcshf = f"""#!/bin/tcsh
        {max_cmd}
        """
        # save in log directory
        tcpath = lg.split('.')[0] + '.tcsh'
        print(tcshf, file=open(tcpath, 'w'))
        # execute this on the cluster
        os.system(f'sbatch --job-name={os.path.basename(tcpath)} -t 0-4:00 {tcpath} --constraint="maxfilter"')
    else:  # run on current machine
        print(max_cmd)
        os.system(max_cmd)


def plot_MaxLog(logpath, outpath, plot=False):

    """
    This plots the different variables in MaxFilter log file for checking visually
    It also returns values for each of these people
    :param logpath:
        The path and filename of the .log file
    :param outpath:
        The directory you want the images saving in
    :return:
        It returns a summary of values from these files as well
        an array with tshape time x
            Fitting Error (cm)', 'Goodness of Fit', 'Translation (cm/s)', 'Rotation (Rads/s)', 'Drift (cm)
    """

    # read in file as list of strings
    with open(logpath, "r") as myfile:
        lines = myfile.readlines()

    #get just the lines starting with #t
    pos_only = [f for f in lines if f[0:2] == '#t']
    import re
    p = re.compile("\d+\.\d+")

    ts = [float(p.findall(f)[0]) for f in pos_only] # time (seconds)
    es = [float(p.findall(f)[1]) for f in pos_only]# fitting error (cm)
    gs = [float(p.findall(f)[2]) for f in pos_only] # goodness of fit
    vs = [float(p.findall(f)[3]) for f in pos_only] # translation (cm/s)
    rs = [float(p.findall(f)[4]) for f in pos_only] # rotation (rd/s)
    #ds = [float(p.findall(f)[5]) for f in pos_only] # drift (cm)

    # labels
    labels = ['Fitting Error (cm)', 'Goodness of Fit', 'Translation (cm/s)', 'Rotation (Rads/s)']
    if plot:
        plt.close('all')
        objs = plt.plot(ts,es,ts,gs,ts,vs,ts,rs)
        plt.legend(iter(objs), labels)
        plt.title(os.path.basename(logpath))
        plt.savefig(os.path.join(outpath,os.path.basename(logpath).split('.')[0]+'.png'))

    # return np.array([[np.mean(es), np.mean(gs), np.mean(vs), np.mean(rs), np.mean(ds)],
    #         [np.std(es), np.std(gs), np.std(vs), np.std(rs), np.std(ds)]
    #         ])
    summary = np.array([es,gs,vs,rs])
    return summary

def maxfilt_mne_bulk(files, indir, outdir, scriptpath, pythonpath):

    # get preprocess individual function as text
    for i in range(len(files)):
        pythonf = f"""
from os.path import join
from os import listdir
import numpy as np
import mne
from mne.preprocessing import find_bad_channels_maxwell

ctc = '/imaging/local/software/neuromag/databases/ctc/ct_sparse.fif'
ss_cal = '/imaging/local/software/neuromag/databases/sss/sss_cal.dat'

print('{files[i]}')
raw = mne.io.read_raw_fif(join('{indir}','{files[i]}'))

raw.info['bads'] = []
raw_check = raw.copy()
auto_noisy_chs, auto_flat_chs, auto_scores  = find_bad_channels_maxwell(
    raw_check, cross_talk=ctc, calibration=ss_cal,
    return_scores=True, verbose=False)
bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
raw.info['bads'] = bads
chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw, verbose=False)
chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes, verbose=False)
head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)
raw_sss = mne.preprocessing.maxwell_filter(
    raw, cross_talk=ctc, calibration=ss_cal, verbose=False,
    head_pos=head_pos, destination=(0,0,0.04), coord_frame='head',
    regularize='in', st_duration=10, st_correlation=0.98, bad_condition = 'ignore'
    )
raw_sss.save(join('{outdir}', '{files[i]}'), overwrite=True)
np.save(join('{outdir}', '{files[i].split('.')[0]+'_pos.npy'}'), head_pos)
            """
        # save to file
        print(pythonf, file=open(f'{scriptpath}/maxfiltmne_{i}.py', 'w'))

        # construct csh file
        tcshf = f"""#!/bin/tcsh
{pythonpath} {scriptpath}/maxfiltmne_{i}.py
                    """
        # save to directory
        print(tcshf, file=open(f'{scriptpath}/maxfiltmne_{i}.csh', 'w'))

        # execute this on the cluster
        os.system(f'sbatch --job-name=maxfilt3mne_{i} --mincpus=5 -t 0-12:00 {scriptpath}/maxfiltmne_{i}.csh')

