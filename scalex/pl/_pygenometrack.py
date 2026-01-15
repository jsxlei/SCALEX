import os
from configparser import ConfigParser
genome_dir = os.path.join(os.path.expanduser("~"), '.cache', 'genome')
hg38_dir = os.path.join(genome_dir, 'hg38')
GTF_FILE =  os.path.join(hg38_dir, 'gencode.v41.annotation.gtf.gz')
TSS_FILE = os.path.join(hg38_dir, 'tss.tsv')

import pandas as pd
import pyranges as pr

def strand_specific_start_site(df):
    df = df.copy()
    if set(df["Strand"]) != set(["+", "-"]):
        raise ValueError("Not all features are strand specific!")

    pos_strand = df.query("Strand == '+'").index
    neg_strand = df.query("Strand == '-'").index
    df.loc[pos_strand, "End"] = df.loc[pos_strand, "Start"] + 1
    df.loc[neg_strand, "Start"] = df.loc[neg_strand, "End"] - 1
    return df


def parse_tss(tss_file=TSS_FILE, gtf_file=GTF_FILE, drop_duplicates: str='gene_name') -> pr.PyRanges:
    """
    The transcription start sites (TSS) for the genome.

    Returns
    -------
    DataFrame
    """
    if tss_file is None or not os.path.exists(tss_file):
        gtf = pr.read_gtf(gtf_file).df.drop_duplicates(subset=[drop_duplicates], keep='first')
        tss = strand_specific_start_site(gtf)[['Chromosome', 'Start', 'End', 'Strand', drop_duplicates]]
        if tss_file is not None:
            os.makedirs(os.path.dirname(tss_file), exist_ok=True)
            tss.to_csv(tss_file, sep='\t', index=False)
    else:
        tss = pd.read_csv(tss_file, sep='\t')
    return tss


class Track:
    def __init__(self, genome='hg38', fig_dir='./', cell_type_colors=None):
        self.config = ConfigParser()
        self.spacer_kws = {"file_type": "spacer", "height": 0.5, "title": ""}
        self.line_kws = {
            "line_width": 0.5,
            "line_style": "solid", 
            "file_type": "hlines", 
            "y_values": 0,
            "overlay_previous": "yes",
            "show_data_range": "no",
            # "height": 0.1,
            # "title": "Hlines"
        } 

        # self.config.add_section("spacer")

        self.link_kws = {
            "links_type": "arcs", "line_width": 0.5, "line_style": "solid",
            "compact_arcs_level": 2, "use_middle": False, "file_type": "links"
        }

        self.bigwig_kws = {
            "file_type": "bigwig",
            "min_value": 0,
            "max_value": 10, #'auto',
            "height": 1,
            "n_bins": 400,
            # "grid": "true",
        }

        self.tss = parse_tss()

        # import matplotlib.pyplot as plt
        self.cell_type_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',] if cell_type_colors is None else cell_type_colors

        self.fig_dir = fig_dir

    def empty_config(self):
        self.config = ConfigParser()
        # self.config.add_section("spacer")

    def add_link(self, links, name, color='darkgreen', height=1.2, **kwargs):
        self.link_kws.update(kwargs)

        if not os.path.isfile(links):
            link_file = os.path.join(self.fig_dir, f"{name}.links")
            links.to_csv(link_file, sep='\t', index=None, header=None)
        else:
            link_file = links
        self.config[name] = {
            "file": link_file, 
            "title": name, 
            "color": color, 
            "height": height, 
            # **link_kws
        }

    def add_bed(self, peaks, name, display="collapsed", border_color="none", labels=False, **kwargs):
        pass

    def add_bigwig(self, bigwigs, **kwargs):
        for i, (name, bigwig) in enumerate(bigwigs.items()):
            bigwig_kws = self.bigwig_kws.copy()
            bigwig_kws.update(kwargs)
            self.config[name] = {
                "file": bigwig,
                "title": name,
                "color": self.cell_type_colors[i],
                # "height": 1,
                **bigwig_kws
            }
            self.config[f'hline {name}'] = self.line_kws


    def add_gene_annotation(self, gene, **kwargs):
        # gene annotation
        self.config["Genes"] = {
            "file": GTF_FILE, 
            "title": "Genes", 
            "prefered_name": "gene_name", 
            "merge_transcripts": True,
            "fontsize": 4, 
            "height": 1, 
            "labels": True, 
            "max_labels": 100,
            "all_labels_inside": True, 
            "labels_in_margin": True,
            "style": "UCSC", 
            "file_type": "gtf",
            # "max_value": max_value
        }

    def add_shap(self, dir, sample, fold='average', shap_h5_path="none", shap_bed_path="none", min_value=-0.025, max_value=0.1, **kwargs):
        self.config["SHAP"] = {
            "dir": dir,
            "sample": sample,
            "shap_h5_path": shap_h5_path,
            "shap_bed_path": shap_bed_path,
            "title": f"SHAP {sample}",
            "height": 1,
            "file_type": "shap",
            "min_value": min_value,
            "max_value": max_value,
            **kwargs
        }

    def generate_track(self, region, up=0, down=0, dpi=200, width=8, fontsize=4, save_dir='./'):
        with open(f"{save_dir}/tracks.ini", "w") as f:
            self.config.write(f)


        if isinstance(region, tuple):
            Chromosome, Start, End = region
        elif isinstance(region, str):
            if ':' in region:
                Chromosome, Start = region.split(':')
                Start, End = Start.split('-')
                Start, End = int(Start), int(End)
            else:
                region = self.get_region_with_gene(region)
        Start -= up
        End += down
        cmd = f"pyGenomeTracks --tracks {save_dir}/tracks.ini --region {Chromosome}:{Start}-{End} \
                -t ' ' --dpi {dpi} --width {width} --fontSize {fontsize} \
                --outFileName {save_dir}/tracks.pdf" # 2> /dev/null"
        import subprocess
        subprocess.run(cmd, shell=True, check=True)

    def get_region_with_gene(self, gene):
        return self.tss.query(f"gene_name == '{gene}'")[['Chromosome', 'Start', 'End']].values[0]
    
    def plot_gene(self, gene, bigwigs, up=50_000, down=2000, dpi=200, width=12, fontsize=4, **bigwig_kws):
        save_dir = os.path.join(self.fig_dir, gene)
        os.makedirs(save_dir, exist_ok=True)
        self.empty_config()
        self.add_bigwig(bigwigs, **bigwig_kws)
        self.add_gene_annotation(gene)
        self.generate_track(gene, up, down, dpi, width, fontsize, save_dir=save_dir)
        from IPython.display import Image, IFrame
        # from wand.image import Image as WImage
        return IFrame(f"{save_dir}/tracks.pdf", width=800, height=600)



def plot_tracks(
        gene, region, 
        peaks=None, 
        bigwigs=None,
        links={}, 
        all_link=None, 
        all_link_kwargs={},
        meta_gr=None, 
        extend=100000, 
        dpi=200,
        width=12,
        fontsize=4,
        fig_dir='./',
        bigwig_max_value='auto'
    ):
    fig_dir = os.path.join(fig_dir, gene)
    os.makedirs(fig_dir, exist_ok=True)

    config = ConfigParser()
    spacer_kws = {"file_type": "spacer", "height": 0.5, "title": ""}
    config.add_section("spacer")
    
    # links
    link_kws = {
        "links_type": "arcs", "line_width": 0.5, "line_style": "solid",
        "compact_arcs_level": 2, "use_middle": False, "file_type": "links"
    }
    for name, link in links.items():
        name = name.replace(' ', '_')
        if not os.path.isfile(link):
            link_file = os.path.join(fig_dir, f"{name}.links")
            link.to_csv(link_file, sep='\t', index=None, header=None)
        else:
            link_file = link
        config[name] = {
            "file": link_file, 
            "title": name, 
            "color": "darkgreen", 
            "height": 1.2, 
            **link_kws
        }

    if all_link is not None:
        all_links = all_link.query(f"gene == '{gene}'").copy()

        for name, kwargs in all_link_kwargs.items():
            link = all_links.loc[:, [*all_link.columns[:6], name]]
            if 'cutoff' in kwargs:
                cutoff = kwargs['cutoff']
                link = link[(link[name] > cutoff) | (link[name] < -cutoff)]
            else:
                link = link[link[name] == True]

            name = name.replace(' ', '_')
            link.to_csv(os.path.join(fig_dir, f'{name}.links'), sep='\t', index=None, header=None)
            config[name] = {
                "file": os.path.join(fig_dir, f"{name}.links"), 
                "title": name, 
                "color": "darkgreen", 
                "height": 1.2, 
                **link_kws,
                **kwargs
            }                        

        

    # peak
    bed_kws = {
        "display": "collapsed", "border_color": "none",
        "labels": False, "file_type": "bed"
    }
    if peaks is not None:
        for name, peak in peaks.items():
            peak.to_csv(os.path.join(fig_dir, f"{name}.bed"), sep='\t', index=None, header=None)
            config[name] = {
                "file": os.path.join(fig_dir, f"{name}.bed"),
                "title": name, 
                **bed_kws
            }
    
    config["spacer1"] = spacer_kws


    # atac bigwig
    cell_type_kws = {
        "file_type": "bigwig"
    }

    # import matplotlib.pyplot as plt
    cell_type_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',]
    
    if meta_gr is not None:
        cell_types = meta_gr.df.columns[3:]
        chromsizes = get_chromsize()
        chroms = meta_gr.df['Chromosome'].unique()
        if len(set(chroms) - set(chromsizes.df['Chromosome'].values)) > 0:
            import pyranges as pr
            meta_gr = meta_gr.df[meta_gr.df['Chromosome'].isin(chromsizes.df['Chromosome'].values)]
            meta_gr = pr.PyRanges(meta_gr)
        os.makedirs(os.path.join(fig_dir ,'bw'), exist_ok=True)
        for i, c in enumerate(cell_types):
            d = c
            c = c.replace(' ', '_')
            if not os.path.isfile(os.path.join(fig_dir, 'bw', f"{c}.bw")):
                meta_gr.to_bigwig(os.path.join(fig_dir, 'bw', f'{c}.bw'), chromosome_sizes=chromsizes, value_col=d, rpm=False)

            config[f"meta {c}"] = {
                "file": os.path.join(fig_dir, 'bw', f"{c}.bw"),
                "title": f"{c}",
                "color": cell_type_colors[i],
                "height": 1,
                "min_value": 0, 
                # "max_value": meta_max_value, 
                **cell_type_kws
            }

    if bigwigs is not None:
        i = 0
        for name, bigwig in bigwigs.items():
            
            config[name] = {
                "file": bigwig,
                "title": name,
                "color": cell_type_colors[i+1],
                "height": 1,
                "min_value": 0, 
                "max_value": bigwig_max_value, 
                **cell_type_kws
            }
            i+=1

    config["spacer2"] = spacer_kws
    
    if not os.path.isfile(GTF_FILE): 
        if os.path.exists(os.path.dirname(GTF_FILE)):
            os.makedirs(os.path.dirname(GTF_FILE), exist_ok=True)
        import wget
        wget.download("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz", GTF_FILE)
    # gene annotation
    config["Genes"] = {
        "file": GTF_FILE, 
        "title": "Genes", 
        "prefered_name": "gene_name", 
        "merge_transcripts": True,
        "fontsize": fontsize, 
        "height": 5, 
        "labels": True, 
        "max_labels": 100,
        "all_labels_inside": True, 
        "labels_in_margin": True,
        "style": "UCSC", 
        "file_type": "gtf",
        # "max_value": max_value
    }

    config["x-axis"] = {"fontsize": fontsize}
    with open(f"{fig_dir}/tracks.ini", "w") as f:
        config.write(f)

    Chromosome, Start, End = region
    Start -= extend
    End += extend
    cmd = f"pyGenomeTracks --tracks {fig_dir}/tracks.ini --region {Chromosome}:{Start}-{End} \
        -t 'Target gene: {gene}' --dpi {dpi} --width {width} --fontSize {fontsize} \
        --outFileName {fig_dir}/tracks.png" # 2> /dev/null"
    import subprocess
    subprocess.run(cmd, shell=True, check=True)

    print(f"{fig_dir}/tracks.png")


import pandas as pd
import numpy as np
import deepdish as dd
import logomaker


def to_coord(peak):
    if ':' not in peak:
        chrom, start, end = peak.split('-')
    else:
        chrom, start_end = peak.split(':')
        start, end = start_end.split('-')
    return chrom, int(start), int(end)

# class ShapPlotter:
#     def __init__(self, dir, sample=None, fold=0, min_value=-0.025, max_value=0.1, shap_h5=None, shap_bed=None):
#         self.dir = dir
#         self.sample = sample
#         self.fold = fold
#         self.min_value = min_value
#         self.max_value = max_value
#         self.shap_h5 = shap_h5
#         self.shap_bed = shap_bed

#         self.load_shap_data()

#     # ----------------- 1. Load data -----------------
#     def load_shap_data(self):
#         if self.shap_h5 is None:
#             shap_h5_path = f"{self.dir}/{self.sample}/fold_{self.fold}.counts_scores.h5"
#         else:
#             shap_h5 = self.shap_h5
#         self.shap_h5 = dd.io.load(shap_h5_path)

#         if self.shap_bed is None:
#             shap_bed_path = f"{self.dir}/{self.sample}/fold_{self.fold}.interpreted_regions.bed"
#         else:
#             shap_bed_path = self.shap_bed
#         self.shap_peaks = pd.read_table(shap_bed_path, header=None)

#         self.shap_peaks_flanked = self.add_flanks(self.shap_peaks, self.shap_h5)
        

#     # ----------------- 2. Add flanks -----------------
#     def add_flanks(self, shap_peaks, shap_h5):
#         shap_peaks_flanked = shap_peaks.copy()
#         flank_size = shap_h5['projected_shap']['seq'].shape[2] // 2
#         shap_peaks_flanked['start'] = (shap_peaks[1] + shap_peaks[9]) - flank_size
#         shap_peaks_flanked['end'] = (shap_peaks[1] + shap_peaks[9]) + flank_size
#         shap_peaks_flanked.sort_values(by=[0, 'start', 'end'], inplace=True)
#         return shap_peaks_flanked

#     # ----------------- 3. Filter peaks in region -----------------
#     def filter_region_peaks(self, shap_peaks_flanked, chrom, region_start, region_end):
#         region_peaks = shap_peaks_flanked[
#             (shap_peaks_flanked[0] == chrom) &
#             (
#                 ((shap_peaks_flanked[1] >= region_start) & (shap_peaks_flanked[2] <= region_end)) |
#                 ((shap_peaks_flanked[1] <= region_start) & (shap_peaks_flanked[2] > region_start)) |
#                 ((shap_peaks_flanked[1] < region_end) & (shap_peaks_flanked[2] >= region_end)) |
#                 ((shap_peaks_flanked[1] <= region_start) & (shap_peaks_flanked[2] >= region_end))
#             )
#         ].copy()
#         region_peaks.sort_values(by=1, inplace=True)
#         return region_peaks

#     # ----------------- 4. Construct SHAP values array -----------------
#     def construct_shap_values(self, region_peaks, shap_h5, region_start, region_end):
#         if region_peaks.empty:
#             print("No shap peaks found in this region.")
#             return np.zeros((region_end - region_start, shap_h5['projected_shap']['seq'].shape[1]))

#         shap_values = []
#         last_end = region_start
#         peak_seq_len = shap_h5['projected_shap']['seq'].shape[2]

#         for index, row in region_peaks.iterrows():
#             assert (row['end'] - row['start']) == peak_seq_len

#             def append_shap(values):
#                 nonlocal shap_values
#                 if len(shap_values) == 0:
#                     shap_values = values
#                 else:
#                     shap_values = np.concatenate([shap_values, values])

#             # Add zeros before peak if necessary
#             if last_end < row['start']:
#                 append_shap(np.zeros((row['start'] - last_end, shap_h5['projected_shap']['seq'].shape[1])))
#                 last_end = row['start']

#             # Determine overlap and trim SHAP if needed
#             start_idx = max(0, last_end - row['start'])
#             end_idx = min(peak_seq_len, region_end - row['start'])
#             append_shap(shap_h5['projected_shap']['seq'][index][:, start_idx:end_idx].T)
#             last_end = row['start'] + end_idx

#         # Add zeros after last peak if necessary
#         if last_end < region_end:
#             append_shap(np.zeros((region_end - last_end, shap_h5['projected_shap']['seq'].shape[1])))

#         return shap_values

#     # ----------------- 5. Plot -----------------
#     def plot_logo(self, shap_values, ax=None):
#         logo1 = logomaker.Logo(pd.DataFrame(shap_values, columns=['A', 'C', 'G', 'T']), ax=ax)
#         if ax is None:
#             ax = logo1.ax
#         if self.min_value is not None and self.max_value is not None:
#             ax.set_ylim(float(self.min_value), float(self.max_value))
#         ax.set_ylabel(f"{self.sample} Counts Shap")
#         return ax

#     # ----------------- Main plot function -----------------
#     def plot(self, region, ax=None): # region: chr:start-end
#         chrom, region_start, region_end = to_coord(region)
#         # print("Plotting SHAP ...")
#         region_peaks = self.filter_region_peaks(self.shap_peaks_flanked, chrom, region_start, region_end)
#         shap_values = self.construct_shap_values(region_peaks, self.shap_h5, region_start, region_end)
#         ax = self.plot_logo(shap_values, ax=ax)
#         return ax

import numpy as np
import pandas as pd
import deepdish as dd
import logomaker
import matplotlib.pyplot as plt

# ----------------- utility -----------------
def to_coord(region):
    """Convert chr:start-end string to tuple (chrom, start, end)."""
    chrom, coords = region.split(":")
    start, end = coords.split("-")
    return chrom, int(start), int(end)


class ShapPlotter:
    def __init__(self, dir, samples=None, fold=0, min_value=-0.025, max_value=0.1, shap_h5=None, shap_bed=None):
        """
        samples : str or list of str
            Sample name(s). If list, will plot all stacked vertically.
        """
        self.dir = dir
        if isinstance(samples, str):
            samples = [samples]
        self.samples = samples
        self.fold = fold
        self.min_value = min_value
        self.max_value = max_value
        self.shap_h5 = shap_h5
        self.shap_bed = shap_bed

        # Load shap data for all samples
        self.data = {}
        if self.samples is None:
            self.samples = os.listdir(self.dir)
        print(self.samples)
        for sample in self.samples:
            print(sample)
            self.data[sample] = self.load_shap_data(sample)

    # ----------------- 1. Load data -----------------
    def load_shap_data(self, sample):
        if self.shap_h5 is None:
            shap_h5_path = f"{self.dir}/{sample}/fold_{self.fold}.counts_scores.h5"
        else:
            shap_h5_path = self.shap_h5
        shap_h5 = dd.io.load(shap_h5_path)

        if self.shap_bed is None:
            shap_bed_path = f"{self.dir}/{sample}/fold_{self.fold}.interpreted_regions.bed"
        else:
            shap_bed_path = self.shap_bed
        shap_peaks = pd.read_table(shap_bed_path, header=None)

        shap_peaks_flanked = self.add_flanks(shap_peaks, shap_h5)

        return {"h5": shap_h5, "peaks": shap_peaks, "peaks_flanked": shap_peaks_flanked}

    # ----------------- 2. Add flanks -----------------
    def add_flanks(self, shap_peaks, shap_h5):
        shap_peaks_flanked = shap_peaks.copy()
        flank_size = shap_h5['projected_shap']['seq'].shape[2] // 2
        shap_peaks_flanked['start'] = (shap_peaks[1] + shap_peaks[9]) - flank_size
        shap_peaks_flanked['end'] = (shap_peaks[1] + shap_peaks[9]) + flank_size
        shap_peaks_flanked.sort_values(by=[0, 'start', 'end'], inplace=True)
        return shap_peaks_flanked

    # ----------------- 3. Filter peaks in region -----------------
    def filter_region_peaks(self, shap_peaks_flanked, chrom, region_start, region_end):
        region_peaks = shap_peaks_flanked[
            (shap_peaks_flanked[0] == chrom) &
            (
                ((shap_peaks_flanked[1] >= region_start) & (shap_peaks_flanked[2] <= region_end)) |
                ((shap_peaks_flanked[1] <= region_start) & (shap_peaks_flanked[2] > region_start)) |
                ((shap_peaks_flanked[1] < region_end) & (shap_peaks_flanked[2] >= region_end)) |
                ((shap_peaks_flanked[1] <= region_start) & (shap_peaks_flanked[2] >= region_end))
            )
        ].copy()
        region_peaks.sort_values(by=1, inplace=True)
        return region_peaks

    # ----------------- 4. Construct SHAP values array -----------------
    def construct_shap_values(self, region_peaks, shap_h5, region_start, region_end):
        if region_peaks.empty:
            print("No shap peaks found in this region.")
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

            # Add zeros before peak if necessary
            if last_end < row['start']:
                append_shap(np.zeros((row['start'] - last_end, shap_h5['projected_shap']['seq'].shape[1])))
                last_end = row['start']

            # Determine overlap and trim SHAP if needed
            start_idx = max(0, last_end - row['start'])
            end_idx = min(peak_seq_len, region_end - row['start'])
            append_shap(shap_h5['projected_shap']['seq'][index][:, start_idx:end_idx].T)
            last_end = row['start'] + end_idx

        # Add zeros after last peak if necessary
        if last_end < region_end:
            append_shap(np.zeros((region_end - last_end, shap_h5['projected_shap']['seq'].shape[1])))

        return shap_values

    # ----------------- 5. Plot -----------------
    def plot_logo(self, shap_values, sample, ax=None):
        logo1 = logomaker.Logo(pd.DataFrame(shap_values, columns=['A', 'C', 'G', 'T']), ax=ax)
        if ax is None:
            ax = logo1.ax
        if self.min_value is not None and self.max_value is not None:
            ax.set_ylim(float(self.min_value), float(self.max_value))
        ax.set_ylabel(f"{sample} SHAP")
        return ax

    # ----------------- Main plot function -----------------
    def plot(self, region, figsize=(12, 2), crop_length=None):
        chrom, region_start, region_end = to_coord(region)
        print(region_start, region_end)
        if crop_length is not None:
            print(crop_length)
            length = region_end - region_start
            print(length)
            crop = (crop_length - length) // 2
            print(crop)
            region_start += crop
            region_end -= crop
        print(region_start, region_end)

        fig, axes = plt.subplots(
            len(self.samples), 1, figsize=(figsize[0], figsize[1] * len(self.samples)), sharex=True, gridspec_kw={'hspace': 0}
        )
        if len(self.samples) == 1:
            axes = [axes]

        for ax, sample in zip(axes, self.samples):
            data = self.data[sample]
            region_peaks = self.filter_region_peaks(data["peaks_flanked"], chrom, region_start, region_end)
            shap_values = self.construct_shap_values(region_peaks, data["h5"], region_start, region_end)
            self.plot_logo(shap_values, sample, ax=ax)
            ax.label_outer()

        plt.tight_layout(pad=0)
        return fig, axes
