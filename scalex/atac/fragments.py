import muon as mu
from muon import atac as ac
from muon import MuData
from anndata import AnnData
import pandas as pd
import numpy as np
from scipy.sparse import csr, csr_matrix
import pysam
from typing import Union
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

import os


def import_fragments(fragment_file, features, obs=None, genome='human'):
    if not os.path.exists(fragment_file+'.tbi'):
        pysam.tabix_index(fragment_file, force=True, preset='bed')
    if obs is None:
        obs = np.unique(pd.read_csv(fragment_file, sep='\t', comment='#', usecols=[3], header=None).iloc[:, 0].values)
    adata = AnnData(obs=obs, var=features)
    ac.tl.locate_fragments(adata, fragment_file)


    adata = ac.tl.count_fragments_features(adata, features, extend_upstream=0, extend_downstream=0)
    adata.X = csr_matrix(adata.X)

    return adata


def sort_fragment(fragment_file, new_fragment_file):
    if fragment_file.endswith('.gz') and is_gz_file(fragment_file):
        cmd = 'gzip -d {}'.format(fragment_file)
        print(cmd, flush=True)
        os.system(cmd)

    new_fragment_file = new_fragment_file.replace('.gz', '')
    cmd = 'sort -k1,1 -k2,2n {} > {}'.format(fragment_file, new_fragment_file)
    print(cmd, flush=True)
    os.system(cmd)

    cmd = 'bgzip -@ 8 {}'.format(new_fragment_file)
    print(cmd, flush=True)
    os.system(cmd)

    # cmd = 'rm {}'.format(new_fragment_file)
    # print(cmd, flush=True)
    # os.system(cmd)

    cmd = 'tabix -p bed {}'.format(new_fragment_file+'.gz')
    print(cmd, flush=True)
    os.system(cmd)


import numpy as np
import pandas as pd
import pyranges as pr
import pysam
import scipy as sp

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import anndata as ad
import pyranges as pr
from tqdm import tqdm
from joblib import Parallel, delayed

def make_peak_matrix(
    fragment_file, peaks_df, barcode_list, output_h5ad=None, n_jobs=32, meta=None
):
    """
    Create a sparse barcode x peak matrix from fragment file and peak file using PyRanges,
    multi-threaded, and save as an h5ad file.

    Parameters:
    - fragment_file: path to fragment.tsv file
    - peak_file: path to BED file with peaks
    - barcode_list: list or set of barcodes to include
    - output_file: path to output .h5ad file
    - n_jobs: number of parallel jobs
    """
    # Load peaks
    # peaks_df = pd.read_csv(peak_file, sep="\t", header=None, names=["Chromosome", "Start", "End", "Peak"])
    peaks_pr = pr.PyRanges(peaks_df)
    peak_names = peaks_df.index.values
    peak_name_to_idx = {name: i for i, name in enumerate(peak_names)}

    # Map barcodes to row indices
    barcode_list = list(barcode_list)
    barcode_to_idx = {bc: i for i, bc in enumerate(barcode_list)}

    # Function to process a chunk of fragments
    def process_chunk(chunk_df):
        chunk_df = chunk_df[chunk_df["Barcode"].isin(barcode_to_idx)]
        if chunk_df.empty:
            return [], [], []

        frag_pr = pr.PyRanges(chunk_df)
        overlaps = frag_pr.join(peaks_pr)

        rows, cols, data = [], [], []
        for _, row_data in overlaps.df.iterrows():
            barcode_idx = barcode_to_idx[row_data["Barcode"]]
            peak_idx = peak_name_to_idx[row_data["Peak"]]
            rows.append(barcode_idx)
            cols.append(peak_idx)
            data.append(row_data["Count"])
        return rows, cols, data

    # Read fragment file in chunks
    col_names = ["Chromosome", "Start", "End", "Barcode", "Count"]
    chunk_size = 500000  # adjust as needed
    reader = pd.read_csv(fragment_file, sep="\t", names=col_names, chunksize=chunk_size)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk) for chunk in tqdm(reader, desc="Processing fragments")
    )

    # Combine results
    all_rows, all_cols, all_data = [], [], []
    for rows, cols, data in results:
        all_rows.extend(rows)
        all_cols.extend(cols)
        all_data.extend(data)

    if not all_data:
        print("Warning: No overlaps found.")
        X = coo_matrix((len(barcode_list), len(peak_names))).tocsr()
    else:
        # Build sparse matrix (barcodes x peaks)
        X = coo_matrix((all_data, (all_rows, all_cols)), shape=(len(barcode_list), len(peak_names))).tocsr()

    X = csr_matrix(X)
    # Create AnnData object
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=barcode_list),     # barcodes as rows
        var=pd.DataFrame(index=peak_names)        # peaks as columns
    )

    # Save to h5ad
    adata.write(output_h5ad)
    print(f"Saved peak matrix to {output_h5ad}")

    return adata


# def process_peak_chunk(chunk_peaks, fragment_file, cell_barcodes, cell_index, peak_index):
#     """Process a chunk of peaks in parallel"""
#     rows, cols, data = [], [], []
    
#     with pysam.TabixFile(fragment_file) as tbx:
#         for _, peak in chunk_peaks.iterrows():
#             try:
#                 for fragment in tbx.fetch(peak.Chromosome, peak.Start, peak.End):
#                     chrom, start, end, barcode, count = fragment.split('\t')
                    
#                     if barcode in cell_barcodes:
#                         row_idx = peak_index[(chrom, int(start), int(end))]
#                         col_idx = cell_index[barcode]
                        
#                         rows.append(row_idx)
#                         cols.append(col_idx)
#                         data.append(count)
#             except (ValueError, KeyError):
#                 continue
                
#     return rows, cols, data

# def make_peak_matrix(fragment_file, peaks, cell_barcodes=None, meta=None, output_h5ad=None, n_cores=None):
#     """
#     Create a sparse peak-by-cell matrix from a fragment file and save as AnnData.

#     Parameters:
#     - fragment_file (str): Path to the fragment file (bgzipped & indexed).
#     - peaks (pandas.DataFrame): Peaks as a DataFrame with columns ['chr', 'start', 'end'].
#     - cell_barcodes (set/list): Cell barcodes to include. If None, use all barcodes in fragment file.
#     - output_h5ad (str): Path to save the AnnData object.
#     - n_cores (int): Number of CPU cores to use. If None, uses all available cores.

#     Returns:
#     - ad.AnnData: The sparse peak matrix as an AnnData object.
#     """
#     # Index fragment file if needed
#     if not os.path.exists(fragment_file + '.tbi'):
#         pysam.tabix_index(fragment_file, force=True, preset='bed')

#     # Get all cell barcodes if not provided
#     if cell_barcodes is None:
#         cell_barcodes = set(pd.read_csv(fragment_file, sep='\t', comment='#', 
#                                       usecols=[3], header=None).iloc[:, 0].values)
#     else:
#         cell_barcodes = set(cell_barcodes)

#     # Convert peaks to pyranges
#     peak_ranges = pr.PyRanges(peaks.rename(columns={"chr": "Chromosome", 
#                                                    "start": "Start", 
#                                                    "end": "End"}))
    
#     # Create mappings
#     cell_list = sorted(cell_barcodes)
#     cell_index = {bc: idx for idx, bc in enumerate(cell_list)}
#     peak_index = {(row.Chromosome, row.Start, row.End): idx 
#                  for idx, row in enumerate(peak_ranges.df.itertuples())}

#     # Set up multiprocessing
#     if n_cores is None:
#         n_cores = mp.cpu_count()
    
#     # Split peaks into chunks for parallel processing
#     chunk_size = len(peak_ranges) // n_cores
#     if chunk_size == 0:
#         chunk_size = 1
#     peak_chunks = [peak_ranges.df.iloc[i:i + chunk_size] for i in range(0, len(peak_ranges), chunk_size)]
    
#     # Create partial function with fixed arguments
#     process_func = partial(process_peak_chunk,
#                          fragment_file=fragment_file,
#                          cell_barcodes=cell_barcodes,
#                          cell_index=cell_index,
#                          peak_index=peak_index)
    
#     # Process chunks in parallel
#     with mp.Pool(n_cores) as pool:
#         results = list(tqdm(pool.imap(process_func, peak_chunks),
#                           total=len(peak_chunks),
#                           desc="Processing peak chunks"))
    
#     # Combine results
#     rows, cols, data = [], [], []
#     for r, c, d in results:
#         rows.extend(r)
#         cols.extend(c)
#         data.extend(d)

#     # Create sparse matrix
#     matrix = sp.sparse.csr_matrix((data, (rows, cols)), 
#                                 shape=(len(peaks), len(cell_list)))

#     # Create peak index names
#     peak_names = [f"{chr}:{start}-{end}" for chr, start, end in peaks.values]
    
#     # Create AnnData object
#     adata = AnnData(X=matrix,
#                    obs=pd.DataFrame(index=peak_names),
#                    var=pd.DataFrame(index=cell_list)).T
#     adata.X = csr_matrix(adata.X)
    
#     adata.var = bed_to_df(adata.var.index)
#     if meta is not None:
#         adata.obs = meta.loc[adata.obs_names].copy()

#     if output_h5ad:
#         adata.write(output_h5ad)

#     return adata



import gzip
def is_gz_file(name):
    # if os.stat(name).ST_SIZE == 0:
    #     return False

    with gzip.open(name, 'rb') as f:
        try:
            file_content = f.read(1)
            return True
        except:
            return False


from scalex.atac.bedtools import intersect_bed, extend_bed, subtract_bed, edge_index_to_coo, bed_to_df



def reindex_peak(atac, new_coord, dist=0, power_law=False, blacklist=None):
    print(f'Reindex peaks from {atac.shape[1]} to {len(new_coord)}', flush=True)
    if power_law:
        link_index, edges = get_link_index(atac.var_names, new_coord, dist=25_000)
        link_index['distance'] = link_index.apply(lambda row: (row[1] + row[2]) / 2 - row[4], axis=1).astype(int)
        link_index['score'] = archr_powerlaw(link_index['distance'].abs())
        adj = edge_index_to_coo(edges, data=link_index['score'], 
                                shape=(atac.shape[0], len(new_coord.shape[0])))
    else:
        if blacklist is not None:
            new_coord = subtract_bed(new_coord, blacklist)
        adj = intersect_bed(atac.var_names, new_coord, out='coo', up=dist, down=dist)

    X = atac.X @ adj
    adata = AnnData(X, atac.obs, new_coord)
    return adata





def make_gene_matrix(atac, rna_var=None, gene_region='combined', power_law=False, up=2000, down=500, **kwargs):

    if rna_var is None:
        # rna_var = get_tss_from_10x()
        rna_var = get_tss_from_gtf(gr=kwargs.get('gtf', None), by='gene_name', genome=kwargs.get('genome', 'hg38'))
    if gene_region == 'combined':
        rna_var = extend_bed(rna_var, up=up)
    elif gene_region == 'promoter':
        tss = strand_specific_start_site(rna_var)
        rna_var = extend_bed(tss, up=up, down=down)
    
    return reindex_peak(atac, rna_var)


def strand_specific_start_site(df):
    df = df.copy()
    if set(df["Strand"]) != set(["+", "-"]):
        raise ValueError("Not all features are strand specific!")

    pos_strand = df.query("Strand == '+'").index
    neg_strand = df.query("Strand == '-'").index
    df.loc[pos_strand, "End"] = df.loc[pos_strand, "Start"] + 1
    df.loc[neg_strand, "Start"] = df.loc[neg_strand, "End"] - 1
    return df
    

def get_promoter_interval(genes, up=1000, down=0):
    tss = strand_specific_start_site(genes)
    promoter = extend_bed(tss, up=up, down=down)
    return promoter


## Powerlaw from ABC
def get_powerlaw_at_distance(distances, gamma=.87, min_distance=5000, scale=None):
    assert(gamma > 0)

    #The powerlaw is computed for distances > 5kb. We don't know what the contact freq looks like at < 5kb.
    #So just assume that everything at < 5kb is equal to 5kb.
    #TO DO: get more accurate powerlaw at < 5kb
    distances = np.clip(distances, min_distance, np.Inf)
    log_dists = np.log(distances + 1)

    #Determine scale parameter
    #A powerlaw distribution has two parameters: the exponent and the minimum domain value 
    #In our case, the minimum domain value is always constant (equal to 1 HiC bin) so there should only be 1 parameter
    #The current fitting approach does a linear regression in log-log space which produces both a slope (gamma) and a intercept (scale)
    #Empirically there is a linear relationship between these parameters (which makes sense since we expect only a single parameter distribution)
    #It should be possible to analytically solve for scale using gamma. But this doesn't quite work since the hic data does not actually follow a power-law
    #So could pass in the scale parameter explicity here. Or just kludge it as I'm doing now
    #TO DO: Eventually the pseudocount should be replaced with a more appropriate smoothing procedure.

    #4.80 and 11.63 come from a linear regression of scale on gamma across 20 hic cell types at 5kb resolution. Do the params change across resolutions?
    if scale is None:
        scale = -4.80 + 11.63 * gamma

    powerlaw_contact = np.exp(scale + -1*gamma * log_dists)

    return(powerlaw_contact)



def archr_powerlaw(distances, min_distance=5000, max_distance=25_000): # TO DO
    # distances[np.where(distances > -min_distance & distances < 0)[0]] = 0
    scores = np.exp(-abs(distances/min_distance)) + np.exp(-1)
    scores[abs(distances) > max_distance] = 0 
    return scores
    # receptive field <25k


TSS_FILE = os.path.expanduser('~/.scalex/10X_GENE_ANNOTATION.txt')
def get_tss_from_10x():
    tss = pd.read_csv(TSS_FILE, sep='\t', index_col=0)
    tss = tss[['chrome', 'start', 'end', 'strand', 'symbol']].copy()
    tss.columns = ['Chromosome', 'Start', 'End', 'Strand', 'gene_name']
    tss['Strand'] = [i if i in ['+', '-'] else '+' for i in tss['Strand']]
    return tss

GENOME_PATH = os.path.expanduser('~/.scalex/')
def get_gtf(genome='hg38', drop_by='gene_name'):
    if genome == 'hg38':
        gtf_file = os.path.join(GENOME_PATH, 'gencode.v38.annotation.gtf.gz')
    elif genome == 'hg19':
        gtf_file = os.path.join(GENOME_PATH, 'gencode.v19.annotation.gtf.gz')
    elif genome == 'mm10':
        gtf_file = os.path.join(GENOME_PATH, 'gencode.vM10.annotation.gtf.gz')
    if not os.path.exists(gtf_file):
        version = genome.split('hg')[-1]
        os.system(f'wget -O {gtf_file} https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/gencode.v{version}.annotation.gtf.gz')
        
    gtf = pr.read_gtf(gtf_file)
    gtf = gtf.df.drop_duplicates(subset=[drop_by], keep="first")
    return pr.PyRanges(gtf)


def get_tss_from_gtf(gr=None, by='gene_name', genome='hg38'):
    if gr is None:
        gr = get_gtf(genome=genome, drop_by='gene_name')
    if not isinstance(gr, pr.PyRanges):
        gr = pr.PyRanges(gr)

    df = strand_specific_start_site(gr.df)[['Chromosome', 'Start', 'End', 'Strand', by]]
    df.index = df[by]
    return df

def get_link_index(genes, peaks, **kwargs):
    edges = intersect_bed(genes, peaks, **kwargs)
    link_index = get_link(genes, peaks, edges)
    return link_index, edges


def get_link(genes, peaks, edges):
    """
    Concat genes and peaks into dataframe based-on edges
    """
    gene_list = genes.index
    peak_list = peaks.index

    peaks_ = peaks.iloc[edges[1]] #, :3]
    genes_ = genes.iloc[edges[0]] #, :3]
    # edges_ = 
    peaks_.index = np.arange(peaks_.shape[0])
    genes_.index = np.arange(genes_.shape[0])
    link = pd.concat([peaks_.iloc[:, :3], genes_.iloc[:, :3]], axis=1, ignore_index=True)
    

    link.columns = ["chr", "start", "end", "chrom2", "start2", "end2"]
    link = pd.concat([link, peaks_.iloc[:, 3:], genes_.iloc[:, 3:8]], axis=1)
    link.index = link['peak_name']+'_'+link['gene_name']
    link.insert(6, 'name', link.index.values) 
    
    return link


if __name__ == "__main__":
    import argparse
    import scanpy as sc

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    gene_score_parser = subparsers.add_parser('gene')
    gene_score_parser.add_argument('--atac', type=str)
    gene_score_parser.add_argument('--output', type=str)
    gene_score_parser.add_argument('--genome', type=str, default='hg38')

    peak_score_parser = subparsers.add_parser('peak')
    peak_score_parser.add_argument('--n_jobs', type=int, default=32)
    peak_score_parser.add_argument('--fragment', type=str)
    peak_score_parser.add_argument('--peaks', type=str)
    peak_score_parser.add_argument('--cell_id', type=str)
    peak_score_parser.add_argument('--cell_col', type=str, default=None)
    peak_score_parser.add_argument('--output', type=str)
    peak_score_parser.add_argument('--genome', type=str, default='hg38')
    peak_score_parser.add_argument('--method', type=str, default='pyranges')
    args = parser.parse_args()

    if args.command == 'gene':
        atac = sc.read(args.atac)
        gene_matrix = make_gene_matrix(atac, gene_region='combined', genome=args.genome)

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        gene_matrix.write(args.output)

    elif args.command == 'peak':
        peaks = pd.read_csv(args.peaks, sep='\t').iloc[:, :3]
        peaks.columns = ['Chromosome', 'Start', 'End'] #['chr', 'start', 'end']
        peaks.index = peaks['Chromosome']+':'+peaks['Start'].astype(str)+'-'+peaks['End'].astype(str)

        meta = pd.read_csv(args.cell_id, sep='\t', index_col=0)
        if args.cell_col is not None:
            meta.set_index(args.cell_col, inplace=True)
            barcode = meta.index
            # barcode = meta[args.cell_col]
        else:
            barcode = meta.index
        print(meta.head())
        
        if args.method == 'pyranges':
            peak_matrix = make_peak_matrix(args.fragment, peaks, barcode_list=barcode, meta=meta, output_h5ad=args.output, n_jobs=args.n_jobs)
        elif args.method == 'muon':
            peak_matrix = import_fragments(args.fragment, peaks, obs=meta)
            peak_matrix.write(args.output)