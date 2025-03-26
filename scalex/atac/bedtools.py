#!/usr/bin/env python

from pybedtools import BedTool
import pandas as pd
import numpy as np

import torch
from scipy.sparse import coo_matrix
import numpy as np

def coo_from_pandas(df, source, target, values=None, shape=None):
    if values is None:
        df['values'] = 1
        data = df['values']
    else:
        data = df[values]

    coo = coo_matrix((data, (df[source], df[target])), shape=shape)
    return coo


def edge_index_to_coo(edge, data=None, shape=None):
    data = np.ones_like(edge[0]) if data is None else data
    shape = (edge[0].max()+1, edge[1].max()+1) if shape is None else shape
    return coo_matrix((data, (edge[0], edge[1])), shape=shape)


def coo_to_sparse_tensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


# =======================================================
#                       Bed
# =======================================================

def bed_to_df(x, keep=False):
    """
    Convert list of peaks(str) into data frame.

    Args:
       x (list of str): list of peak names

    Returns:
       pandas.dataframe: peak info as DataFrame

    Examples:
       >>> x = ['chr1_3094484_3095479', 'chr1_3113499_3113979', 'chr1_3119478_3121690']
       >>> list_peakstr_to_df(x)
                   chr	start	end
            0	chr1	3094484	3095479
            1	chr1	3113499	3113979
            2	chr1	3119478	3121690
    """
    df = np.array([to_coord(i) for i in x])
    df = pd.DataFrame(df, columns=["Chromosome", "Start", "End"])
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)
    df.index = x

    return df


def to_coord(peak):
    if ':' not in peak:
        chrom, start, end = peak.split('-')
    else:
        chrom, start_end = peak.split(':')
        start, end = start_end.split('-')
    return chrom, int(start), int(end)


def df_to_bed(x):
    # return (x.iloc[:, 0]+':'+x.iloc[:, 1].astype(str)+'-'+x.iloc[:, 2].astype(str)).values
    return  x.apply(lambda row: row.iloc[0]+':'+str(row.iloc[1])+'-'+str(row.iloc[2]), axis=1).values


def extend_bed(df, up=0, down=0, start='Start', end='End'):
    # assert 'Start' in df.columns and 'End' in df.columns, 'Start and End columns are required.'
    df = df.copy()
    if not isinstance(df, pd.DataFrame):
        df = bed_to_df(df)
    if 'Strand' not in df.columns:
        df['Strand'] = '+'
        
    if 'Start' not in df.columns or 'End' not in df.columns:
        df.columns = ["Chromosome", "Start", "End"] + list(df.columns[3:])

    # x = x.apply(lambda row: (row[0], max(0, int(row[1])-down), int(row[2])+up) if 'Strand' in row and (row['Strand'] == '-') 
    #     else (row[0], max(0, int(row[1])-up), int(row[2])+down), axis=1, result_type='expand')
    # x.columns =["Chromosome", "Start", "End"]
    # return x
    pos_strand = df.query("Strand == '+'").index
    neg_strand = df.query("Strand == '-'").index
    
    df.loc[pos_strand, end] = df.loc[pos_strand, end] + down
    df.loc[pos_strand, start] = df.loc[pos_strand, start] - up

    df.loc[neg_strand, end] = df.loc[neg_strand, end] + up
    df.loc[neg_strand, start] = df.loc[neg_strand, start] - down

    df.loc[:, start] = df.loc[:, start].clip(lower=0)
    return df


def process_bed(bed, sort=False, add_index=True):
    if not isinstance(bed, pd.DataFrame):
        bed = bed_to_df(bed)
    if add_index:
        bed = bed.iloc[:, :3]
        bed['index'] = np.arange(len(bed))
    bed = BedTool.from_dataframe(bed)
    if sort:
        bed = bed.sort()
    return bed



def decode_dist(dist):
    if isinstance(dist, int):
        dist = dist
    elif isinstance(dist, str):
        if 'M' in dist:
            dist = int(dist.replace('M', '')) * 1_000_000
        elif 'K' in dist:
            dist = int(dist.replace('K', '')) * 1_000
        else:
            dist = int(dist)
    return dist

    
def intersect_bed(
        query, ref, 
        up=0,
        down=0, 
        add_query_index=True, 
        add_ref_index=True, 
        index=[3, 7], 
        out='edge_index',
        add_distance=False,
    ):
    """
    Return intersection index of query and ref, 
    make sure the fourth column of query and last column of ref are index. 
    """
    up = decode_dist(up)
    if up > 0 or down > 0:
        query = extend_bed(query, up=up, down=down)

    query = process_bed(query, add_index=add_query_index)
    ref = process_bed(ref, add_index=add_ref_index)

    intersected = query.intersect(ref, wa=True, wb=True).to_dataframe() # nonamecheck
    if len(intersected) == 0:
        raise ValueError('No intersection found, please check the input bed file.')
    if index is None:
        return intersected
    else: 
        edges = intersected.iloc[:, index].values.T.astype(int)

    if add_distance:
        mid1 = intersected.iloc[:, [1, 2]].mean(axis=1)
        mid2 = intersected.iloc[:, [5, 6]].mean(axis=1)
        distance = np.abs(mid1 - mid2).astype(int).values
    else:
        distance = None

    if out == 'edge_index':
        if add_distance:
            edges = np.concatenate([edges, distance.reshape(1, -1)], axis=0)
        return edges
    elif out == 'coo':
        return edge_index_to_coo(edges, data=distance, shape=(len(query), len(ref)))


def subtract_bed(query, ref):
    """
    Return subtraction index of query and ref, 
    make sure the fourth column of query and last column of ref are index. 
    """
    query = BedTool.from_dataframe(query)
    ref = BedTool.from_dataframe(ref)
    return query.subtract(ref).to_dataframe()

        
def closest_bed(query, ref, k=1, D='a', t='first'):
    """
    Return two matrix, one is query ref pair matrix, the other is query ref distance matrix
        row is query index, column is ref index
    """
    query = process_bed(query, sort=True)
    ref = process_bed(ref, sort=True)
    
    intersected = query.closest(ref, k=k, D=D, t=t).to_dataframe() 
    intersected = intersected[intersected.iloc[:, -3]!=-1]

    out = intersected.iloc[:, [3, -2, -1]]
    out.columns = ['query', 'ref', 'distance']
    pair = pd.DataFrame(out.groupby('query')['ref'].apply(list).to_dict()).T
    distance = pd.DataFrame(out.groupby('query')['distance'].apply(list).to_dict()).T
    return pair, distance


import torch
def get_promoter_offset_for_embedding(promoter, peak_list):
    result = intersect_bed(promoter, peak_list)
    promoter_dict = result.groupby('name')['thickEnd'].apply(list).to_dict()
    inputs = []
    offsets = []
    offset = 0
    for i in range(len(promoter)):
        offsets.append(offset)
        if i in promoter_dict:
            v = promoter_dict[i]
            inputs+=v
            offset += len(v)
    inputs = torch.LongTensor(inputs)
    offsets = torch.LongTensor(offsets)
    return inputs, offsets


