import muon as mu
from muon import atac as ac
from muon import MuData
from anndata import AnnData
import pandas as pd
import numpy as np
from scipy.sparse import csr, csr_matrix
import pysam
from typing import Union

import os


def import_fragments(fragment_file, features, obs=None, genome='human'):
    if not os.path.exists(fragment_file+'.tbi'):
        pysam.tabix_index(fragment_file, force=True, preset='bed')
    ac.tl.locate_fragments(adata, fragment_file)

    if obs is None:
        obs = np.unique(pd.read_csv(fragment_file, sep='\t', comment='#', usecols=[3], header=None).iloc[:, 0].values)
    adata = AnnData(obs=obs, var=features)

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

def make_peak_matrix(fragment_file, peaks, cell_barcodes=None):
    """
    Create a sparse peak-by-cell matrix from a fragment file and save as AnnData.

    Parameters:
    - fragment_file (str): Path to the fragment file (bgzipped & indexed).
    - peaks (pandas.DataFrame): Peaks as a DataFrame with columns ['chr', 'start', 'end'].
    - cell_barcodes (set): A set of valid cell barcodes to filter fragments.
    - output_h5ad (str): Path to save the AnnData object.

    Returns:
    - ad.AnnData: The sparse peak matrix as an AnnData object.
    """
    
    # Convert peaks to pyranges for efficient interval querying

    peak_ranges = pr.PyRanges(peaks.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"}))
    
    # Create barcode index mapping
    cell_list = sorted(cell_barcodes)
    cell_index = {barcode: idx for idx, barcode in enumerate(cell_list)}
    
    # Create peak index mapping
    peak_index = {tuple(row): idx for idx, row in enumerate(peaks.itertuples(index=False, name=None))}

    # Sparse matrix lists
    row_idx, col_idx, counts = [], [], []

    with pysam.TabixFile(fragment_file) as f:
        for peak in peak_ranges.itertuples():
            chr, start, end = peak.Chromosome, peak.Start, peak.End
            
            try:
                for fragment in f.fetch(chr, start, end):
                    frag_chr, frag_start, frag_end, barcode, _ = fragment.split("\t")
                    frag_start, frag_end = int(frag_start), int(frag_end)

                    if barcode in cell_barcodes:
                        row = peak_index[(chr, start, end)]
                        col = cell_index[barcode]

                        row_idx.append(row)
                        col_idx.append(col)
                        counts.append(1)  # Count each fragment once
            except ValueError:
                continue

    # Create a sparse matrix
    sparse_matrix = sp.csr_matrix((counts, (row_idx, col_idx)), shape=(len(peaks), len(cell_barcodes)))

    # Create AnnData object
    adata = ad.AnnData(
        X=sparse_matrix,
        obs=pd.DataFrame(index=[f"{chr}:{start}-{end}" for chr, start, end in peak_index.keys()]),  # Peaks
        var=pd.DataFrame(index=cell_list)  # Cell barcodes
    )

    # Save as h5ad file
    adata.write_h5ad(output_h5ad)
    return adata



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
        rna_var = get_tss_from_10x()

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
    parser.add_argument('--atac', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    atac = sc.read(args.atac)
    gene_matrix = make_gene_matrix(atac, gene_region='combined')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    gene_matrix.write(args.output)