import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import logomaker
from typing import List, Union
import os
import h5py
import numpy as np
import pandas as pd
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42


def _plot_weights(array, path, figsize=(10,3)):
    """Plot weights as a sequence logo and save to file."""

    if not os.path.isfile(path):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111) 

        df = pd.DataFrame(array, columns=['A', 'C', 'G', 'T'])
        df.index.name = 'pos'

        crp_logo = logomaker.Logo(df, ax=ax)
        crp_logo.style_spines(visible=False)
        plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())

        plt.savefig(path)
        plt.close()

    else:
        pass

def create_modisco_logos(modisco_h5py: os.PathLike, modisco_logo_dir, trim_threshold, pattern_groups: List[str]):
    """Open a modisco results file and create and write logos to file for each pattern."""
    modisco_results = h5py.File(modisco_h5py, 'r')

    tags = []

    for name in pattern_groups:
        if name not in modisco_results.keys():
            continue

        metacluster = modisco_results[name]
        # print(metacluster)
        key = lambda x: int(x[0].split("_")[-1])
        # key = lambda x: int(x[0].split("#")[-1])
        for pattern_name, pattern in sorted(metacluster.items(), key=key):
            tag = pattern_name
            tags.append(tag)

            cwm_fwd = np.array(pattern['contrib_scores'][:])
            cwm_rev = cwm_fwd[::-1, ::-1]

            score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
            score_rev = np.sum(np.abs(cwm_rev), axis=1)

            trim_thresh_fwd = np.max(score_fwd) * trim_threshold
            trim_thresh_rev = np.max(score_rev) * trim_threshold

            pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
            pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]

            start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(np.max(pass_inds_fwd) + 4 + 1, len(score_fwd) + 1)
            start_rev, end_rev = max(np.min(pass_inds_rev) - 4, 0), min(np.max(pass_inds_rev) + 4 + 1, len(score_rev) + 1)

            trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
            trimmed_cwm_rev = cwm_rev[start_rev:end_rev]

            _plot_weights(trimmed_cwm_fwd, path='{}/{}.cwm.fwd.png'.format(modisco_logo_dir, tag))
            _plot_weights(trimmed_cwm_rev, path='{}/{}.cwm.rev.png'.format(modisco_logo_dir, tag))

    modisco_results.close()
    return tags

def create_selin_logos(modisco_h5py: os.PathLike, modisco_logo_dir, trim_threshold, pattern_groups: List[str]):
    """Open a modisco results file and create and write logos to file for each pattern."""
    modisco_results = h5py.File(modisco_h5py, 'r')

    tags = []

    for name in pattern_groups:
        if name not in modisco_results.keys():
            continue

        metacluster = modisco_results[name]
        for pattern_name, pattern in metacluster.items():
            tag = pattern_name.replace('/', '-').replace("#", "-")
            tags.append(tag)

            cwm_fwd = np.array(pattern['contrib_scores'][:])
            cwm_rev = cwm_fwd[::-1, ::-1]

            score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
            score_rev = np.sum(np.abs(cwm_rev), axis=1)

            trim_thresh_fwd = np.max(score_fwd) * trim_threshold
            trim_thresh_rev = np.max(score_rev) * trim_threshold

            pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
            pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]

            start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(np.max(pass_inds_fwd) + 4 + 1, len(score_fwd) + 1)
            start_rev, end_rev = max(np.min(pass_inds_rev) - 4, 0), min(np.max(pass_inds_rev) + 4 + 1, len(score_rev) + 1)

            trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
            trimmed_cwm_rev = cwm_rev[start_rev:end_rev]

            _plot_weights(trimmed_cwm_fwd, path='{}/{}.cwm.fwd.png'.format(modisco_logo_dir, tag))
            _plot_weights(trimmed_cwm_rev, path='{}/{}.cwm.rev.png'.format(modisco_logo_dir, tag))

    modisco_results.close()
    return tags

def read_meme(filename):
    motifs = {}

    with open(filename, "r") as infile:
        motif, width, i = None, None, 0

        for line in infile:
            if motif is None:
                if line[:5] == 'MOTIF':
                    motif = line.split()[1]
                else:
                    continue

            elif width is None:
                if line[:6] == 'letter':
                    width = int(line.split()[5])
                    pwm = np.zeros((width, 4))

            elif i < width:
                pwm[i] = list(map(float, line.split()))
                i += 1

            else:
                motifs[motif] = pwm
                motif, width, i = None, None, 0

    return motifs

def compute_per_position_ic(ppm, background, pseudocount):
    alphabet_len = len(background)
    ic = ((np.log((ppm+pseudocount)/(1 + pseudocount*alphabet_len))/np.log(2))
          *ppm - (np.log(background)*background/np.log(2))[None,:])
    return np.sum(ic,axis=1)

def make_logo(match, logo_dir, motifs):
    if match == 'NA':
        return

    background = np.array([0.25, 0.25, 0.25, 0.25])
    ppm = motifs[match]
    ic = compute_per_position_ic(ppm, background, 0.001)

    _plot_weights(ppm*ic[:, None], path='{}/{}.png'.format(logo_dir, match))

def path_to_image_link(path):
    return '=IMAGE("' + path + '#"&RANDBETWEEN(1111111,9999999), 4, 80, 240)'