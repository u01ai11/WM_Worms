import mne
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn import plotting
from nilearn import  datasets
import pandas as pd
import sys
import time
import os
from os import listdir as listdir
from os.path import join
from os.path import isdir
from os.path import isfile
import numpy as np
import holoviews as hv
hv.extension('bokeh')
from holoviews import opts, dim
from matplotlib import patches

def plot_aparc_parcels(matrix, ax, fig, title):
    """
    :param matrix:
        The connectivity matrix (real or simulated) representing aparc parcels

    :param ax:
        axes to plot on
    :return:
    """

    # get labels and coordinates
    labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc', subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR'))
    labels = labels[0:-1]
    label_names = [label.name for label in labels]
    coords = []
    # TODO: Find a better way to get centre of parcel
    #get coords of centre of mass
    for i in range(len(labels)):
        if 'lh' in label_names[i]:
            hem = 1
        else:
            hem = 0
        coord = mne.vertex_to_mni(labels[i].center_of_mass(subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR')), subject='fsaverage_1',hemis=hem,subjects_dir=join(MAINDIR,'FS_SUBDIR'))
        coords.append(coord[0])

    plotting.plot_connectome_strength(matrix,
                                      node_coords=coords,
                                      title=title,
                                      figure=fig,
                                      axes=ax,
                                      cmap=plt.cm.YlOrRd)



def hv_chord(contrast, frequency, threshold, stats, re_order_ind, label_names, des, freq_vect):

    """
    Makes a holoview/boken chord diagram
    :param contrast:
        Which contrast in the glm data are we looking at
    :param frequency:
        The index of the frequency in the freq_vect we are looking at
    :param threshold:
        The percentile threshold for plotting (so the plot isn't messy)
    :param stats:
        The stats array we are plotting from
    :param re_order_ind:
        The indices for re-ordering the data into the Y-order of the parcels in the brain
    :param label_names:
        The names for each parcel
    :param des:
        The design matrix from the GLM
    :param freq_vect:
        The frequency vector containing the actual frequencies of each
    :return:
    """
    dtypes = np.dtype([
        ('source', int),
        ('target', int),
        ('value', int),
    ])

    data = np.empty(0, dtype=dtypes)
    links = pd.DataFrame(data)

    square = stats[contrast,:,:,frequency]


    #square = stats[contrast,:,:,:].sum(axis=2)
    thresh_mask = square > np.percentile(square, threshold)
    square[~thresh_mask] = 0

    #reorder
    X_sorted = np.copy(square)
    # Sort along first dim
    X_sorted = X_sorted[re_order_ind,:]
    # Sort along second dim
    X_sorted = X_sorted[:,re_order_ind]

    labels_sorted = np.array(label_names)[re_order_ind]
    # loop through Y axis of matrix
    counter = 0
    for i in range(X_sorted.shape[0]):
        for ii in range(X_sorted.shape[1]):
            links.loc[counter] = [i, ii, int(X_sorted[i,ii])]
            counter +=1

    # make label index
    dtypes = np.dtype([
        ('name', int),
        ('group', int),
    ])

    data = np.empty(0, dtype=dtypes)
    nodes = pd.DataFrame(data)

    for i in range(X_sorted.shape[0]):
        nodes.loc[i] = [labels_sorted[i], 1]

    graph = hv.Chord((links, hv.Dataset(nodes, 'index')))
    graph.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(),
                   labels='name', node_color=dim('index').str())
    )
    graph.relabel('Directed Graph').opts(directed=True)
    graph.opts(title=f'{des.contrast_names[contrast]} Partial Directed Coherence @ {int(freq_vect[frequency])}Hz')
    return graph

def get_labels():
    MAINDIR = join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS')
    labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc', subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR'))
    labels = labels[0:-1]
    label_names = [label.name for label in labels]
    coords = []
    # TODO: Find a better way to get centre of parcel
    #get coords of centre of mass
    for i in range(len(labels)):
        if 'lh' in label_names[i]:
            hem = 1
        else:
            hem = 0
        coord = mne.vertex_to_mni(labels[i].center_of_mass(subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR')),
                                  subject='fsaverage_1',hemis=hem,subjects_dir=join(MAINDIR,'FS_SUBDIR'))
        coords.append(coord[0])


    # First, we reorder the labels based on their location in the left hemi
    lh_labels = [name for name in label_names if name.endswith('lh')]
    rh_labels = [name for name in label_names if name.endswith('rh')]

    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)

    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

    # For the right hemi
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]

    # make a list with circular plot order
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    node_order = node_order[::-1] # reverse the whole thing

    # get a mapping to the original list
    re_order_ind = [label_names.index(x) for x in node_order]

    return labels, label_names, re_order_ind

def plot_roi(names):
    labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc', subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR'))
    #labels = labels[0:-1]
    label_names = [label.name for label in labels]

    #names = ['lingual-lh', 'parahippocampal-lh', 'parahippocampal-rh']

    ninds = [label_names.index(i) for i in names]


    # nilearn parcellation is a list representing each vertex, do left and right seperatl
    # MNE labels are a list representing the index of each vertex the parcel contains

    parcellation = np.zeros([10242])
    names_cmap_inds = []
    for i in range(len(names)):
        vinds = labels[ninds[i]].get_vertices_used()
        names_cmap_inds.append(i+1)
        for ii in vinds:
            parcellation[ii] = i+1



    fsaverage = datasets.fetch_surf_fsaverage()

    cmap = cm.get_cmap('gist_ncar')
    legend_elements = [patches.Patch(facecolor=cmap(x), edgecolor=cmap(x), label=y) for x,y in zip(names_cmap_inds, names)]

    fig, ax = plt.subplots(2,2, subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=parcellation,
                           hemi='left', view='medial',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, axes=ax[0,0], title= 'Left Hemisphere Medial',
                           cmap=cmap)

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=parcellation,
                           hemi='right', view='medial',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, axes=ax[0,1], title= 'Right Hemisphere Medial',
                           cmap=cmap)

    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=parcellation,
                           hemi='left', view='lateral',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, axes=ax[1,0], title= 'Left Hemisphere Lateral',
                           cmap=cmap)

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=parcellation,
                           hemi='right', view='lateral',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, axes=ax[1,1], title= 'Right Hemisphere Lateral',
                           cmap=cmap)

    fig.legend()
    plt.subplots_adjust(right=0.85)
    return [fig, ax]

def mne_connec(matrix, this_title, fig, *labels):
    MAINDIR = join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS')
    if labels is None:
        labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc',
                                            subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS', 'FS_SUBDIR'))
    labels = labels[0:-1]
    label_colors = [label.color for label in labels]

    label_names = [label.name for label in labels]
    coords = []
    # TODO: Find a better way to get centre of parcel
    # get coords of centre of mass
    for i in range(len(labels)):
        if 'lh' in label_names[i]:
            hem = 1
        else:
            hem = 0
        coord = mne.vertex_to_mni(
            labels[i].center_of_mass(subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS', 'FS_SUBDIR')),
            subject='fsaverage_1', hemis=hem, subjects_dir=join(MAINDIR, 'FS_SUBDIR'))
        coords.append(coord[0])

    # First, we reorder the labels based on their location in the left hemi
    lh_labels = [name for name in label_names if name.endswith('lh')]
    rh_labels = [name for name in label_names if name.endswith('rh')]

    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)

    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

    # For the right hemi
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]

    # make a list with circular plot order
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    node_order = node_order[::-1]  # reverse the whole thing

    # get a mapping to the original list
    re_order_ind = [label_names.index(x) for x in node_order]

    node_angles = mne.viz.circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=[0, len(label_names) / 2])

    fig2, ax = mne.viz.plot_connectivity_circle(matrix, label_names, n_lines=300,
                             node_angles=node_angles, node_colors=label_colors,
                             title=this_title, fig=fig,
                                                facecolor='white',
                                                textcolor='black')

    return fig2, ax

def plot_cluster(time_inds, space_inds, T_obs, evok):

    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = evok.times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True
    reg_masks.append(mask)
    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # plot average test statistic and mark significant sensors
    image, _ = mne.viz.plot_topomap(f_map, pos, mask=mask, axes=ax_topo, ch_type=sen,
                                    vmin=np.min, vmax=np.max, sphere=sphere, show=False)

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged T-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Retro-Cue Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    mne.viz.plot_compare_evokeds({'Left': L_E, 'Right': R_E}, title=title, picks=ch_inds, axes=ax_signals,
                                 colors=colors, show=False,
                                 split_legend=True, truncate_yaxis='auto', combine='mean')

    ax_signals.set_title(title, y=1.1)
    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)

    xmin, xmax = ax_signals.get_xlim()

    # add labels if needed
    for i, ev in enumerate(event_labels):
        if xmin <= event_onsets[i] <= xmax:
            ax_signals.axvline(event_onsets[i], linestyle='--')
            ax_signals.text(event_onsets[i], ymax, f'{ev} onset')
        if xmin <= event_offsets[i] <= xmax:
            ax_signals.axvline(event_offsets[i], linestyle='--')
            ax_signals.text(event_offsets[i], ymax, f'{ev} offset')

    ax_signals.set_ylabel('Field Strength (fT)')
    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)

    labels = [item.get_text() for item in ax_signals.get_xticklabels()]
    labels = [str(float(i) - plot_t_offset) for i in labels]
    ax_signals.set_xticklabels(labels)
    plt.show()
