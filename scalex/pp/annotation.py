import pyranges as pr
import os
from scalex.atac.bedtools import bed_to_df

GENE_COLUMNS = ['Chromosome', 'Start', 'End', 'Strand', 'gene_name', 'gene_ids']
import re
def ens_trim_version(x: str):
    return re.sub(r"\.[0-9_-]+$", "", x)

def annotate_genes(gene_var, gtf=None, by='gene_name'):
    COLUMNS = ['Chromosome', 'Start', 'End', 'Strand', 'gene_name', 'gene_ids']
    if isinstance(gtf, str):
        gtf = get_gtf(gtf, drop_by=by).df
    elif isinstance(gtf, pr.PyRanges):
        gtf = gtf.df

    if 'gene_ids' in gene_var.columns:
        gene_var.index = gene_var['gene_ids']
        by = 'gene_id'
        print("Use gene_id")
        
    if by == 'gene_id':
        gtf.index = gtf[by].apply(ens_trim_version)
    else:
        gtf.index = gtf[by]

    gtf['gene_ids'] = gtf['gene_id']
    gene_var = gtf.reindex(gene_var.index).loc[:, COLUMNS]

    return gene_var

def remove_genes_without_annotation(adata, by='gene_name', exclude_mt=True):
    adata.var_names_make_unique()
    genes = adata.var.dropna(subset=[by]).index
    adata = adata[:, genes]
    if exclude_mt:
        indices = [i for i, name in enumerate(adata.var.gene_name) 
                          if not str(name).startswith(tuple(['ERCC', 'MT-', 'mt-']))]

        adata = adata[:, indices] #.copy()

    adata.var.Start = adata.var.Start.astype(int)
    adata.var.End = adata.var.End.astype(int)
    return adata

def add_interval_to_gene_var(gene_var):
    # gene_var = gene_var.copy()
    gene_var['interval'] = df_to_bed(gene_var)
    gene_var = strand_specific_start_site(gene_var)
    gene_var['tss'] = gene_var['Start']
    # gene_var['promoter_interval'] = df_to_bed(promoter)
    return gene_var
     


def rna_var_to_promoter(var, up=1000, down=100):
    var = strand_specific_start_site(var)
    var = get_promoter_interval(var, up=up, down=down)
    return var


def format_rna(
    rna, 
    gtf=os.path.expanduser('~/.scalex/gencode.v38.annotation.gtf.gz'), 
    up=1000, 
    down=100, 
    force=False
):
    if set(GENE_COLUMNS).issubset(rna.var.columns) and not force:
        # print("Already formatted")
        return rna
    
    rna.var = annotate_genes(rna.var, gtf)
    rna = remove_genes_without_annotation(rna)
    rna.var = add_interval_to_gene_var(rna.var)
    return rna

def format_atac(atac):
    if set(["Chromosome", "Start", "End"]).issubset(atac.var.columns):
        print("Already formatted")
        return atac
    else:
        atac.var = bed_to_df(atac.var_names)
        return atac
    

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
    from .bedtools import extend_bed
    promoter = extend_bed(tss, up=up, down=down)
    return promoter

def df_to_bed(x):
    # return (x.iloc[:, 0]+':'+x.iloc[:, 1].astype(str)+'-'+x.iloc[:, 2].astype(str)).values
    return  x.apply(lambda row: row.iloc[0]+':'+str(row.iloc[1])+'-'+str(row.iloc[2]), axis=1).values
    
def get_gtf(gtf_file, genome='hg38', drop_by='gene_name'):
    # if genome == 'hg38':
    #     gtf_file = GENOME_PATH / 'gencode.v38.annotation.gtf.gz'
    # elif genome == 'hg19':
    #     gtf_file = GENOME_PATH / 'gencode.v19.annotation.gtf.gz'
    if not os.path.exists(gtf_file):
        version = genome.split('hg')[-1]
        os.system(f'wget -O {gtf_file} https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/gencode.v{version}.annotation.gtf.gz')
        
    gtf = pr.read_gtf(gtf_file)
    gtf = gtf.df.drop_duplicates(subset=[drop_by], keep="first")
    return pr.PyRanges(gtf)