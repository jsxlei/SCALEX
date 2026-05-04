import numpy as np
import pandas as pd
import deepdish as dd
import logomaker
from pygenometracks.tracks import GenomeTrack
import matplotlib.pyplot as plt

class SHAPTrack(GenomeTrack):
    """
    GenomeTrack for plotting SHAP scores as a logo track.
    Can be used in pyGenomeTracks .ini files after registration.
    """

    SUPPORTED_ENDINGS = []
    TRACK_TYPE = 'shap'
    OPTIONS_TXT = """
    height = 3
    title =
    file_type = shap
    dir =
    sample =
    fold =
    """

    # Default values for properties
    DEFAULTS_PROPERTIES = {
        "shap_h5_path": "none",
        "shap_bed_path": "none",
        "min_value": -0.025,
        "max_value": 0.1,
    }

    # With no file in configuration, we don't need it as necessary property
    NECESSARY_PROPERTIES = ['dir', 'sample']

    SYNONYMOUS_PROPERTIES = {}
    POSSIBLE_PROPERTIES = {}
    BOOLEAN_PROPERTIES = []
    STRING_PROPERTIES = ['title', 'file_type', 'dir', 'sample', 'shap_h5_path', 'shap_bed_path']
    FLOAT_PROPERTIES = {
        'height': [0, np.inf],
        'min_value': [-np.inf, np.inf],
        'max_value': [-np.inf, np.inf]
    }
    INTEGER_PROPERTIES = {
    }



    def load_shap_data(self):
        self.sample = self.properties['sample']
        self.dir = self.properties['dir']
        self.shap_h5_path = self.properties['shap_h5_path']
        self.shap_bed_path = self.properties['shap_bed_path']
        if self.shap_h5_path == "none":
            shap_h5_path = f"{self.dir}/{self.sample}/average.counts.h5"
        else:
            shap_h5_path = self.shap_h5_path
        self.shap_h5 = dd.io.load(shap_h5_path)

        if self.shap_bed_path == "none":
            shap_bed_path = f"{self.dir}/{self.sample}/fold_0.interpreted_regions.bed"
        else:
            shap_bed_path = self.shap_bed_path
        self.shap_peaks = pd.read_table(shap_bed_path, header=None)

        self.shap_peaks_flanked = self.add_flanks(self.shap_peaks, self.shap_h5)

    # ----------------- 2. Add flanks -----------------
    def add_flanks(self, shap_peaks, shap_h5):
        shap_peaks_flanked = shap_peaks.copy()
        flank_size = shap_h5['projected_shap']['seq'].shape[2] // 2
        shap_peaks_flanked['start'] = (shap_peaks[1] + shap_peaks[9]) - flank_size
        shap_peaks_flanked['end'] = (shap_peaks[1] + shap_peaks[9]) + flank_size
        shap_peaks_flanked.sort_values(by=[0, 'start', 'end'], inplace=True)
        return shap_peaks_flanked

    # ----------------- 3. Filter peaks in region -----------------
    def filter_region_peaks(self, chrom, region_start, region_end):
        region_peaks = self.shap_peaks_flanked[
            (self.shap_peaks_flanked[0] == chrom) &
            (
                ((self.shap_peaks_flanked['start'] >= region_start) & (self.shap_peaks_flanked['end'] <= region_end)) |
                ((self.shap_peaks_flanked['start'] <= region_start) & (self.shap_peaks_flanked['end'] > region_start)) |
                ((self.shap_peaks_flanked['start'] < region_end) & (self.shap_peaks_flanked['end'] >= region_end)) |
                ((self.shap_peaks_flanked['start'] <= region_start) & (self.shap_peaks_flanked['end'] >= region_end))
            )
        ].copy()
        region_peaks.sort_values(by='start', inplace=True)
        return region_peaks

    # ----------------- 4. Construct SHAP array -----------------
    def construct_shap_values(self, region_peaks, region_start, region_end):
        shap_h5 = self.shap_h5
        if region_peaks.empty:
            return np.zeros((region_end - region_start, shap_h5['projected_shap']['seq'].shape[1]))

        shap_values = []
        last_end = region_start
        peak_seq_len = shap_h5['projected_shap']['seq'].shape[2]

        for index, row in region_peaks.iterrows():
            assert (row['end'] - row['start']) == peak_seq_len

            def append_shap(values):
                nonlocal shap_values
                if len(shap_values) == 0:
                    shap_values = values
                else:
                    shap_values = np.concatenate([shap_values, values])

            if last_end < row['start']:
                append_shap(np.zeros((row['start'] - last_end, shap_h5['projected_shap']['seq'].shape[1])))
                last_end = row['start']

            start_idx = max(0, last_end - row['start'])
            end_idx = min(peak_seq_len, region_end - row['start'])
            append_shap(shap_h5['projected_shap']['seq'][index][:, start_idx:end_idx].T)
            last_end = row['start'] + end_idx

        if last_end < region_end:
            append_shap(np.zeros((region_end - last_end, shap_h5['projected_shap']['seq'].shape[1])))

        return shap_values

    # ----------------- 5. Plot -----------------
    def plot(self, ax, chrom, region_start, region_end):
        # chrom, region_start, region_end = to_coord(region)

        self.load_shap_data()
        region_peaks = self.filter_region_peaks(chrom, region_start, region_end)
        shap_values = self.construct_shap_values(region_peaks, region_start, region_end)
        df = pd.DataFrame(shap_values, columns=['A', 'C', 'G', 'T'])
        logomaker.Logo(df, ax=ax)

        if ax is None:
            ax = plt.gca()
        ax.set_ylim(float(self.properties['min_value']), float(self.properties['max_value']))
        ax.set_ylabel(f"{self.sample} SHAP")
        return ax
