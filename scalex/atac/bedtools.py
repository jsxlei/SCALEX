"""BED interval helpers and pybedtools-backed set operations.

Consolidated module: ``bed_to_df`` / ``df_to_bed`` / ``to_coord`` are the one
canonical implementation; previously these lived in three places.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pybedtools import BedTool
from scipy.sparse import coo_matrix


# ── Format I/O ────────────────────────────────────────────────────────────────

def to_coord(peak: str):
    """Parse ``"chr:start-end"`` or ``"chr-start-end"`` into ``(chrom, start, end)``."""
    if ':' not in peak:
        chrom, start, end = peak.split('-')
    else:
        chrom, start_end = peak.split(':')
        start, end = start_end.split('-')
    return chrom, int(start), int(end)


def bed_to_df(x, keep: bool = False) -> pd.DataFrame:
    """Convert a list of ``"chr:start-end"`` peak strings into a 3-col DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns ``Chromosome``, ``Start``, ``End``; index is the original peak strings.

    Examples
    --------
    >>> bed_to_df(['chr1:3094484-3095479', 'chr1:3113499-3113979'])
    """
    df = np.array([to_coord(i) for i in x])
    df = pd.DataFrame(df, columns=["Chromosome", "Start", "End"])
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)
    df.index = x
    return df


def df_to_bed(x: pd.DataFrame) -> np.ndarray:
    """Render the first 3 columns of *x* as ``"chr:start-end"`` strings."""
    return x.apply(lambda row: row.iloc[0] + ':' + str(row.iloc[1]) + '-' + str(row.iloc[2]), axis=1).values


# ── Interval helpers ──────────────────────────────────────────────────────────

def decode_dist(dist):
    """Convert ``"5K"`` / ``"1M"`` / ``int`` to an integer base-pair distance."""
    if isinstance(dist, int):
        return dist
    if 'M' in dist:
        return int(dist.replace('M', '')) * 1_000_000
    if 'K' in dist:
        return int(dist.replace('K', '')) * 1_000
    return int(dist)


def extend_bed(df, up: int = 0, down: int = 0, start: str = 'Start', end: str = 'End') -> pd.DataFrame:
    """Extend interval ends strand-aware. Negative coordinates are clipped to 0."""
    df = df.copy()
    if not isinstance(df, pd.DataFrame):
        df = bed_to_df(df)
    if 'Strand' not in df.columns:
        df['Strand'] = '+'
    if 'Start' not in df.columns or 'End' not in df.columns:
        df.columns = ["Chromosome", "Start", "End"] + list(df.columns[3:])

    df[start] = df[start].astype(int)
    df[end] = df[end].astype(int)

    pos_strand = df.query("Strand == '+'").index
    neg_strand = df.query("Strand == '-'").index

    df.loc[pos_strand, end] = df.loc[pos_strand, end] + down
    df.loc[pos_strand, start] = df.loc[pos_strand, start] - up

    df.loc[neg_strand, end] = df.loc[neg_strand, end] + up
    df.loc[neg_strand, start] = df.loc[neg_strand, start] - down

    df.loc[:, start] = df.loc[:, start].clip(lower=0)
    return df


def process_bed(bed, sort: bool = False, add_index: bool = True) -> "BedTool":
    if not isinstance(bed, pd.DataFrame):
        bed = bed_to_df(bed)
    if add_index:
        bed = bed.iloc[:, :3]
        bed['index'] = np.arange(len(bed))
    bed = BedTool.from_dataframe(bed)
    if sort:
        bed = bed.sort()
    return bed


# ── Sparse helpers ────────────────────────────────────────────────────────────

def edge_index_to_coo(edge, data=None, shape=None) -> coo_matrix:
    data = np.ones_like(edge[0]) if data is None else data
    shape = (edge[0].max() + 1, edge[1].max() + 1) if shape is None else shape
    return coo_matrix((data, (edge[0], edge[1])), shape=shape)


def coo_from_pandas(df, source, target, values=None, shape=None) -> coo_matrix:
    if values is None:
        df['values'] = 1
        data = df['values']
    else:
        data = df[values]
    return coo_matrix((data, (df[source], df[target])), shape=shape)


def coo_to_sparse_tensor(coo):
    """Convert a scipy COO matrix to a torch sparse tensor (lazy torch import)."""
    import torch
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))


# ── Bedtools operations ───────────────────────────────────────────────────────

def intersect_bed(
    query,
    ref,
    up=0,
    down=0,
    add_query_index: bool = True,
    add_ref_index: bool = True,
    index=(3, 7),
    out: str = "edge_index",
    add_distance: bool = False,
):
    """Intersect *query* and *ref*, optionally with a strand-aware ±window.

    Distances reported are between the **original** (un-extended) intervals.
    Accepts DataFrames, lists of ``"chr:start-end"`` strings, or pybedtools.BedTool.
    """
    if not isinstance(query, pd.DataFrame):
        query = bed_to_df(query)
    if not isinstance(ref, pd.DataFrame):
        ref = bed_to_df(ref)

    qdf = query.copy()
    rdf = ref.copy()

    up = decode_dist(up)
    down = decode_dist(down)

    if up > 0 or down > 0:
        query = extend_bed(query, up=up, down=down)

    query = process_bed(query, add_index=add_query_index)
    ref = process_bed(ref, add_index=add_ref_index)

    intersected = query.intersect(ref, wa=True, wb=True).to_dataframe()
    if len(intersected) == 0:
        raise ValueError("No intersection found, please check the input bed file.")

    if index is None:
        return intersected

    edges = intersected.iloc[:, list(index)].values.T.astype(int)

    distance = None
    if add_distance:
        q_idx = edges[0]
        r_idx = edges[1]
        q_mid = (qdf.iloc[q_idx, 1].to_numpy().astype(int) + qdf.iloc[q_idx, 2].to_numpy().astype(int)) / 2.0
        r_mid = (rdf.iloc[r_idx, 1].to_numpy().astype(int) + rdf.iloc[r_idx, 2].to_numpy().astype(int)) / 2.0
        distance = (q_mid - r_mid).astype(int)

    if out == "edge_index":
        if add_distance:
            edges = np.vstack([edges, distance.reshape(1, -1)])
        return edges
    elif out == "coo":
        return edge_index_to_coo(edges, data=distance, shape=(len(qdf), len(rdf)))
    else:
        raise ValueError(f"Unknown out={out!r}")


def subtract_bed(query, ref) -> pd.DataFrame:
    """Remove *ref* intervals from *query*."""
    query = BedTool.from_dataframe(query)
    ref = BedTool.from_dataframe(ref)
    return query.subtract(ref).to_dataframe()


def closest_bed(query, ref, k: int = 1, D: str = 'a', t: str = 'first', exclude_self: bool = False):
    """Find the k nearest *ref* features for each *query* interval.

    Returns
    -------
    np.ndarray
        Shape ``(3, N)``: ``[query_idx, ref_idx, distance]``.
    """
    query = process_bed(query, sort=True)
    ref = process_bed(ref, sort=True)

    intersected = query.closest(ref, k=k, D=D, t=t, io=exclude_self).to_dataframe()
    intersected = intersected[intersected.iloc[:, -3] != -1]

    out = intersected.iloc[:, [3, -2, -1]]
    out.columns = ['query', 'ref', 'distance']
    return out.T.values


def get_promoter_offset_for_embedding(promoter, peak_list):
    """Build flat (inputs, offsets) tensors for an EmbeddingBag over peak indices."""
    import torch
    result = intersect_bed(promoter, peak_list)
    promoter_dict = pd.DataFrame(result.T).groupby(0)[1].apply(list).to_dict()
    inputs = []
    offsets = []
    offset = 0
    for i in range(len(promoter)):
        offsets.append(offset)
        if i in promoter_dict:
            v = promoter_dict[i]
            inputs += v
            offset += len(v)
    return torch.LongTensor(inputs), torch.LongTensor(offsets)
