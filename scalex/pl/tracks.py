"""Genomic track visualization using pyGenomeTracks."""

import os
import subprocess
from configparser import ConfigParser

import pandas as pd

genome_dir = os.path.join(os.path.expanduser("~"), '.cache', 'genome')
hg38_dir = os.path.join(genome_dir, 'hg38')
GTF_FILE = os.path.join(hg38_dir, 'gencode.v41.annotation.gtf.gz')
TSS_FILE = os.path.join(hg38_dir, 'tss.tsv')

_CELL_TYPE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def strand_specific_start_site(df):
    """Convert gene intervals to strand-specific transcription start sites.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``Chromosome``, ``Start``, ``End``, ``Strand``.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame where each row's interval spans exactly one base
        at the TSS.

    Raises
    ------
    ValueError
        If the DataFrame contains rows without strand information.
    """
    df = df.copy()
    if set(df["Strand"]) != {"+", "-"}:
        raise ValueError("Not all features are strand specific!")
    pos_strand = df.query("Strand == '+'").index
    neg_strand = df.query("Strand == '-'").index
    df.loc[pos_strand, "End"] = df.loc[pos_strand, "Start"] + 1
    df.loc[neg_strand, "Start"] = df.loc[neg_strand, "End"] - 1
    return df


def parse_tss(tss_file: str = TSS_FILE, gtf_file: str = GTF_FILE, drop_duplicates: str = 'gene_name'):
    """Load or derive transcription start sites for hg38.

    Reads from ``tss_file`` if it exists; otherwise parses the GTF and
    caches the result.

    Parameters
    ----------
    tss_file : str
        Path to a cached TSV of TSS coordinates.
    gtf_file : str
        Path to a GTF annotation file (used when ``tss_file`` does not exist).
    drop_duplicates : str, default 'gene_name'
        Column used to deduplicate genes in the GTF.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Chromosome``, ``Start``, ``End``,
        ``Strand``, and ``{drop_duplicates}``.
    """
    import pyranges as pr
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
    """Builder for pyGenomeTracks configuration files.

    Parameters
    ----------
    genome : str, default 'hg38'
        Genome assembly (informational).
    fig_dir : str, default './'
        Directory where track files and figures are written.
    """

    def __init__(self, genome: str = 'hg38', fig_dir: str = './'):
        self.config = ConfigParser()
        self.spacer_kws = {"file_type": "spacer", "height": 0.5, "title": ""}
        self.config.add_section("spacer")
        self.link_kws = {
            "links_type": "arcs", "line_width": 0.5, "line_style": "solid",
            "compact_arcs_level": 2, "use_middle": False, "file_type": "links",
        }
        self.bigwig_kws = {"file_type": "bigwig", "min_value": 0, "max_value": 10, "height": 1}
        self.tss = parse_tss()
        self.cell_type_colors = _CELL_TYPE_COLORS
        self.fig_dir = fig_dir

    def empty_config(self):
        """Reset the config to a blank state."""
        self.config = ConfigParser()
        self.config.add_section("spacer")

    def add_link(self, links, name: str, color: str = 'darkgreen', height: float = 1.2, **kwargs):
        """Add an arc-links track to the config.

        Parameters
        ----------
        links : str or pd.DataFrame
            File path or DataFrame of links (peak, gene pairs).
        name : str
            Track section name.
        color : str, default 'darkgreen'
            Arc colour.
        height : float, default 1.2
            Track height in inches.
        """
        link_kws = {**self.link_kws, **kwargs}
        if not os.path.isfile(links):
            link_file = os.path.join(self.fig_dir, f"{name}.links")
            links.to_csv(link_file, sep='\t', index=None, header=None)
        else:
            link_file = links
        self.config[name] = {"file": link_file, "title": name, "color": color, "height": height, **link_kws}

    def add_bigwig(self, bigwigs: dict, **kwargs):
        """Add bigWig signal tracks for each cell type.

        Parameters
        ----------
        bigwigs : dict
            Mapping of track name → bigWig file path.
        **kwargs
            Override default bigWig track settings.
        """
        for i, (name, bigwig) in enumerate(bigwigs.items()):
            kws = {**self.bigwig_kws, **kwargs}
            self.config[name] = {"file": bigwig, "title": name, "color": self.cell_type_colors[i], **kws}
        self.config["spacer2"] = self.spacer_kws

    def add_gene_annotation(self, gene: str, **kwargs):
        """Add a gene annotation (GTF) track.

        Parameters
        ----------
        gene : str
            Gene name (informational only; the full GTF is displayed).
        """
        self.config["Genes"] = {
            "file": GTF_FILE, "title": "Genes", "prefered_name": "gene_name",
            "merge_transcripts": True, "fontsize": 4, "height": 5, "labels": True,
            "max_labels": 100, "all_labels_inside": True, "labels_in_margin": True,
            "style": "UCSC", "file_type": "gtf",
        }

    def generate_track(self, region, extend: int = 50_000, dpi: int = 200, width: int = 12, fontsize: int = 4):
        """Write the config and run pyGenomeTracks.

        Parameters
        ----------
        region : str or tuple
            Gene name (str) or (Chromosome, Start, End) tuple.
        extend : int, default 50000
            Extend the region by this many bases on each side.
        dpi, width, fontsize : int
            pyGenomeTracks rendering parameters.
        """
        with open(f"{self.fig_dir}/tracks.ini", "w") as f:
            self.config.write(f)
        if isinstance(region, str):
            region = self.get_region_with_gene(region)
        Chromosome, Start, End = region
        Start -= extend
        End += extend
        cmd = (
            f"pyGenomeTracks --tracks {self.fig_dir}/tracks.ini "
            f"--region {Chromosome}:{Start}-{End} "
            f"-t ' ' --dpi {dpi} --width {width} --fontSize {fontsize} "
            f"--outFileName {self.fig_dir}/tracks.png"
        )
        subprocess.run(cmd, shell=True, check=True)

    def get_region_with_gene(self, gene: str):
        """Look up genomic coordinates for a gene name.

        Parameters
        ----------
        gene : str
            Gene name to look up in the TSS table.

        Returns
        -------
        tuple
            (Chromosome, Start, End)
        """
        return self.tss.query(f"gene_name == '{gene}'")[['Chromosome', 'Start', 'End']].values[0]

    def plot(self, gene: str, bigwigs: dict, extend: int = 50_000, dpi: int = 200, width: int = 12, fontsize: int = 4, **bigwig_kws):
        """Convenience method: reset config, add bigwigs, annotate gene, render.

        Parameters
        ----------
        gene : str
            Gene to visualise.
        bigwigs : dict
            Mapping of track name → bigWig file path.
        extend : int, default 50000
            Flanking region size.
        dpi, width, fontsize : int
            Rendering parameters.

        Returns
        -------
        IPython.display.Image
            Inline image of the rendered track.
        """
        self.empty_config()
        self.add_bigwig(bigwigs, **bigwig_kws)
        self.add_gene_annotation(gene)
        self.generate_track(gene, extend, dpi, width, fontsize)
        from IPython.display import Image
        return Image(filename=f"{self.fig_dir}/tracks.png")


def plot_tracks(
    gene: str,
    region,
    peaks=None,
    bigwigs=None,
    links=None,
    all_link=None,
    all_link_kwargs=None,
    meta_gr=None,
    extend: int = 100_000,
    dpi: int = 200,
    width: int = 12,
    fontsize: int = 4,
    fig_dir: str = './',
    bigwig_max_value='auto',
) -> None:
    """Render a genomic track plot for a given gene and region.

    Writes pyGenomeTracks config and images under ``{fig_dir}/{gene}/``.

    Parameters
    ----------
    gene : str
        Target gene name (used for folder naming and annotation query).
    region : tuple
        (Chromosome, Start, End) to display.
    peaks : dict | None
        Mapping of track name → peak DataFrame or BED file path.
    bigwigs : dict | None
        Mapping of track name → bigWig file path.
    links : dict | None
        Mapping of track name → links DataFrame or file path.
    all_link : pd.DataFrame | None
        DataFrame of all links with a ``gene`` column.
    all_link_kwargs : dict | None
        Track rendering options per link track name.
    meta_gr : PyRanges | None
        Aggregated ATAC signal per cell type (converted to bigWig).
    extend : int, default 100000
        Flanking bases added to each side of the region.
    dpi, width, fontsize : int
        pyGenomeTracks rendering parameters.
    fig_dir : str, default './'
        Output directory.
    bigwig_max_value : str or float, default 'auto'
        Maximum y-axis value for bigWig tracks.
    """
    if all_link_kwargs is None:
        all_link_kwargs = {}
    if links is None:
        links = {}

    fig_dir = os.path.join(fig_dir, gene)
    os.makedirs(fig_dir, exist_ok=True)

    config = ConfigParser()
    spacer_kws = {"file_type": "spacer", "height": 0.5, "title": ""}
    config.add_section("spacer")

    link_kws = {
        "links_type": "arcs", "line_width": 0.5, "line_style": "solid",
        "compact_arcs_level": 2, "use_middle": False, "file_type": "links",
    }
    for name, link in links.items():
        name = name.replace(' ', '_')
        if not os.path.isfile(link):
            link_file = os.path.join(fig_dir, f"{name}.links")
            link.to_csv(link_file, sep='\t', index=None, header=None)
        else:
            link_file = link
        config[name] = {"file": link_file, "title": name, "color": "darkgreen", "height": 1.2, **link_kws}

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
                "title": name, "color": "darkgreen", "height": 1.2, **link_kws, **kwargs,
            }

    bed_kws = {"display": "collapsed", "border_color": "none", "labels": False, "file_type": "bed"}
    if peaks is not None:
        for name, peak in peaks.items():
            peak.to_csv(os.path.join(fig_dir, f"{name}.bed"), sep='\t', index=None, header=None)
            config[name] = {"file": os.path.join(fig_dir, f"{name}.bed"), "title": name, **bed_kws}

    config["spacer1"] = spacer_kws

    if meta_gr is not None:
        cell_types = meta_gr.df.columns[3:]
        chroms = meta_gr.df['Chromosome'].unique()
        from scalex.atac.snapatac2._utils import get_chromsize
        chromsizes = get_chromsize()
        if len(set(chroms) - set(chromsizes.df['Chromosome'].values)) > 0:
            import pyranges as pr
            meta_gr = pr.PyRanges(meta_gr.df[meta_gr.df['Chromosome'].isin(chromsizes.df['Chromosome'].values)])
        bw_dir = os.path.join(fig_dir, 'bw')
        os.makedirs(bw_dir, exist_ok=True)
        for i, c in enumerate(cell_types):
            d = c
            c = c.replace(' ', '_')
            bw_path = os.path.join(bw_dir, f'{c}.bw')
            if not os.path.isfile(bw_path):
                meta_gr.to_bigwig(bw_path, chromosome_sizes=chromsizes, value_col=d, rpm=False)
            config[f"meta {c}"] = {
                "file": bw_path, "title": f"{c}", "color": _CELL_TYPE_COLORS[i],
                "height": 1, "min_value": 0, "file_type": "bigwig",
            }

    if bigwigs is not None:
        for i, (name, bigwig) in enumerate(bigwigs.items()):
            config[name] = {
                "file": bigwig, "title": name, "color": _CELL_TYPE_COLORS[i + 1],
                "height": 1, "min_value": 0, "max_value": bigwig_max_value, "file_type": "bigwig",
            }

    config["spacer2"] = spacer_kws

    if not os.path.isfile(GTF_FILE):
        os.makedirs(os.path.dirname(GTF_FILE), exist_ok=True)
        import wget
        wget.download(
            "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz",
            GTF_FILE,
        )

    config["Genes"] = {
        "file": GTF_FILE, "title": "Genes", "prefered_name": "gene_name",
        "merge_transcripts": True, "fontsize": fontsize, "height": 5, "labels": True,
        "max_labels": 100, "all_labels_inside": True, "labels_in_margin": True,
        "style": "UCSC", "file_type": "gtf",
    }
    config["x-axis"] = {"fontsize": fontsize}

    ini_path = os.path.join(fig_dir, "tracks.ini")
    with open(ini_path, "w") as f:
        config.write(f)

    Chromosome, Start, End = region
    Start -= extend
    End += extend
    cmd = (
        f"pyGenomeTracks --tracks {ini_path} "
        f"--region {Chromosome}:{Start}-{End} "
        f"-t 'Target gene: {gene}' --dpi {dpi} --width {width} --fontSize {fontsize} "
        f"--outFileName {os.path.join(fig_dir, 'tracks.png')}"
    )
    subprocess.run(cmd, shell=True, check=True)
    print(os.path.join(fig_dir, "tracks.png"))
