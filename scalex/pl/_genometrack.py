# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import deepdish as dd
import logomaker
from .GenomeTrack import GenomeTrack

class ShapTrack(GenomeTrack):
    """
    A track that plots GWAS variants within a specified region.
    This version hardcodes the input file path and processes data inside the plot method.
    The init_data() step is removed and no file input is read from configuration.
    """

    SUPPORTED_ENDINGS = []
    TRACK_TYPE = 'shap'
    OPTIONS_TXT = """
    height = 3
    title =
    file_type = shap
    sample =
    fold =
    """

    # Default values for properties
    DEFAULTS_PROPERTIES = {
    }

    # With no file in configuration, we don't need it as necessary property
    NECESSARY_PROPERTIES = ['sample', 'fold']

    SYNONYMOUS_PROPERTIES = {}
    POSSIBLE_PROPERTIES = {}
    BOOLEAN_PROPERTIES = []
    STRING_PROPERTIES = ['title', 'file_type', 'sample', 'fold']
    FLOAT_PROPERTIES = {
        'height': [0, np.inf],
        'min_value': [-np.inf, np.inf],
        'max_value': [-np.inf, np.inf]
    }
    INTEGER_PROPERTIES = {
    }

    def plot(self, ax, chrom, region_start, region_end):
        print("Plotting SHAP ...")

        # Hardcoded file path
        variant_shap_dir = '/oak/stanford/groups/akundaje/projects/CRC_finemap/peak_shap/specific_peaks'
        sample = self.properties['sample']
        fold = self.properties['fold']

        shap_h5 = dd.io.load(variant_shap_dir + '/' + sample + '/' + fold + '/' + sample + '.' + fold + '.specific_peaks.counts_scores.counts_scores.h5')
        shap_peaks = pd.read_table(variant_shap_dir + '/' + sample + '/' + fold + '/' + sample + '.' + fold + '.specific_peaks.counts_scores.interpreted_regions.bed',
                                   header=None)

        shap_peaks_with_flanks = shap_peaks.copy()
        shap_peaks_with_flanks['start'] = (shap_peaks[1] + shap_peaks[9]) - (shap_h5['projected_shap']['seq'].shape[2] // 2)
        shap_peaks_with_flanks['end'] = (shap_peaks[1] + shap_peaks[9]) + (shap_h5['projected_shap']['seq'].shape[2] // 2)
        shap_peaks_with_flanks.sort_values(by=[0, 'start', 'end'], inplace=True)

        region_peaks = shap_peaks_with_flanks[(shap_peaks_with_flanks[0] == chrom) & (((shap_peaks_with_flanks[1] >= region_start) & (shap_peaks_with_flanks[2] <= region_end)) |
                                                                    ((shap_peaks_with_flanks[1] <= region_start) & (shap_peaks_with_flanks[2] > region_start)) |
                                                                    ((shap_peaks_with_flanks[1] < region_end) & (shap_peaks_with_flanks[2] >= region_end)) |
                                                                    ((shap_peaks_with_flanks[1] <= region_start) & (shap_peaks_with_flanks[2] >= region_end)))].copy()
        if region_peaks.empty:
            print("No shap peaks found in this region.")

        region_peaks = region_peaks.sort_values(by=1)
        shap_values = []
        last_end = region_start

        for index,row in region_peaks.iterrows():
            assert (row['end'] - row['start']) == shap_h5['projected_shap']['seq'].shape[2]

            print('peak_start:', row['start'])
            print('peak_end:', row['end'])
            print()

            if last_end < row['start']:
                if len(shap_values) == 0:
                    shap_values = np.zeros((row['start'] - last_end, shap_h5['projected_shap']['seq'].shape[1]))
                    last_end = row['start']
                    print("Added starting zeros")
                    print(shap_values.shape)
                    print(last_end)
                    print()
                else:
                    shap_values = np.concatenate([shap_values, np.zeros((row['start'] - last_end, shap_h5['projected_shap']['seq'].shape[1]))])
                    last_end = row['start']
                    print("Added middle zeros")
                    print(shap_values.shape)
                    print(last_end)
                    print()

            if last_end > row['end']:
                continue

            elif last_end > row['start'] and last_end + shap_h5['projected_shap']['seq'].shape[2] <= row['end']:
                if len(shap_values) == 0:
                    shap_values = shap_h5['projected_shap']['seq'][index][:,last_end - row['start']:].T
                    last_end = row['end']
                    print("Added start trimmed shap")
                    print(shap_values.shape)
                    print(last_end)
                    print()
                else:
                    shap_values = np.concatenate([shap_values, shap_h5['projected_shap']['seq'][index][:,last_end - row['start']:].T])
                    last_end = row['end']
                    print("Added start trimmed shap")
                    print(shap_values.shape)
                    print(last_end)
                    print()

            elif last_end > row['start'] and last_end + shap_h5['projected_shap']['seq'].shape[2] > region_end:
                if len(shap_values) == 0:
                    shap_values = shap_h5['projected_shap']['seq'][index][:,last_end - row['start']:(region_end - last_end) + (last_end - row['start'])].T
                    last_end = region_end
                    print("Added start and end trimmed shap")
                    print(shap_values.shape)
                    print(last_end)
                    print()
                else:
                    shap_values = np.concatenate([shap_values, shap_h5['projected_shap']['seq'][index][:,last_end - row['start']:(region_end - last_end) + (last_end - row['start'])].T])
                    last_end = region_end
                    print("Added start and end trimmed shap")
                    print(shap_values.shape)
                    print(last_end)
                    print

            elif last_end + shap_h5['projected_shap']['seq'].shape[2] <= region_end:
                shap_values = np.concatenate([shap_values, shap_h5['projected_shap']['seq'][index].T])
                last_end = row['end']
                print("Added shap")
                print(shap_values.shape)
                print(last_end)
                print()

            elif last_end + shap_h5['projected_shap']['seq'].shape[2] > region_end:
                shap_values = np.concatenate([shap_values, shap_h5['projected_shap']['seq'][index][:,:region_end - last_end].T])
                last_end = region_end
                print("Added end trimmed shap")
                print(shap_values.shape)
                print(last_end)
                print()

            else:
                print("ERROR: Peaks are not sorted.")

        if last_end < region_end:
            shap_values = np.concatenate([shap_values, np.zeros((region_end - last_end, shap_h5['projected_shap']['seq'].shape[1]))])
            print("Added ending zeros")
            print(shap_values.shape)
            print()

        logo1 = logomaker.Logo(pd.DataFrame(shap_values,
                                    columns=['A','C','G','T']), ax=ax)
        if 'min_value' in self.properties and 'max_value' in self.properties:
            ax.set_ylim(float(self.properties['min_value']), float(self.properties['max_value']))
        ax.set_ylabel(sample + ' Counts Shap')

        return ax
