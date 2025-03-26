from __future__ import annotations

from typing import Literal, Optional, List, Tuple
import os
import numpy as np
from pathlib import Path
import logging

# from snapatac2._snapatac2 import PyDNAMotif
# from snapatac2._utils import fetch_seq
# from snapatac2.genome import Genome
from ._diff import _p_adjust_bh


import numpy as np
from scipy.stats import binom
from multiprocessing import Pool
from pathlib import Path
import math

from pyfaidx import Fasta

class DNAMotif:
    def __init__(self, id: str, matrix: np.ndarray):
        self.id = id
        self.name = None
        self.family = None
        self.probability = np.array(matrix, dtype=np.float64)  # Convert to NumPy array

    @classmethod
    def from_string(cls, content: str) -> 'DNAMotif':
        lines = content.strip().split("\n")
        if not lines[0].startswith("MOTIF"):
            raise ValueError("MOTIF not found")
        id = lines[0].split("MOTIF ")[1].strip()
        matrix = np.array([list(map(float, line.split())) for line in lines[2:]], dtype=np.float64)
        return cls(id, matrix)

    def info_content(self) -> float:
        log_probs = np.where(self.probability > 0, self.probability * np.log2(self.probability), 0)
        return 2.0 * self.probability.shape[0] + np.sum(log_probs)

    def add_pseudocount(self, pseudocount: float = 0.0001):
        self.probability = np.maximum(self.probability, pseudocount)
        self.probability /= self.probability.sum(axis=1, keepdims=True)

    def optimal_scores_suffix(self, bg: np.ndarray) -> np.ndarray:
        scores = np.log(self.probability / bg[np.newaxis, :]).max(axis=1).cumsum()
        return scores[-1] - scores

    def look_ahead_search(self, bg: np.ndarray, remain_best: np.ndarray, seq: bytes, start: int, thres: float):
        n = self.probability.shape[0]
        cur_match = 0.0
        prob_map = {b'A': 0, b'C': 1, b'G': 2, b'T': 3}

        for cur_pos in range(n):
            nucleotide = seq[start + cur_pos:start + cur_pos + 1]
            i = prob_map.get(nucleotide, None)
            sc = 0.0 if i is None else np.log(self.probability[cur_pos, i] / bg[i])
            cur_match += sc
            cur_best = cur_match + remain_best[cur_pos]

            if cur_best < thres:
                return None
        return start, cur_best

from multiprocessing import Pool

class DNAMotifScanner:
    def __init__(self, motif: DNAMotif, background_prob=np.array([0.25, 0.25, 0.25, 0.25])):
        self.motif = motif
        self.background_prob = np.array(background_prob, dtype=np.float64)
    
    def find(self, seq: str, pvalue=1e-5):
        return [(i, pvalue) for i in range(len(seq) - len(self.motif.probability))]
    
    def exist(self, seq: str, pvalue=1e-5, rc=True):
        return bool(self.find(seq, pvalue)) or (rc and bool(self.find(rev_compl(seq), pvalue)))
    
    def exists(self, seqs: list, pvalue=1e-5, rc=True):
        with Pool() as pool:
            return pool.starmap(self.exist, [(seq, pvalue, rc) for seq in seqs])
        
    def with_background(self, seqs: list, pvalue=1e-5):
        n = len(seqs)
        occurrence_background = sum(self.exist(seq, pvalue, True) for seq in seqs)
        return DNAMotifTest(self, pvalue, occurrence_background, n)


class PyDNAMotif:
    def __init__(self, id: str, matrix):
        pwm = np.array(matrix)
        self._motif = DNAMotif(id, pwm)

    @property
    def id(self) -> str:
        return self._motif.id

    @id.setter
    def id(self, value: str):
        self._motif.id = value

    @property
    def name(self) -> Optional[str]:
        return self._motif.name

    @name.setter
    def name(self, value: str):
        self._motif.name = value

    @property
    def family(self) -> Optional[str]:
        return self._motif.family

    @family.setter
    def family(self, value: str):
        self._motif.family = value

    def info_content(self) -> float:
        return self._motif.info_content()

    def with_nucl_prob(self, a: float = 0.25, c: float = 0.25, g: float = 0.25, t: float = 0.25):
        return DNAMotifScanner(self._motif, [a, c, g, t])
    

def rev_compl(dna: str) -> str:
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return dna.translate(complement)[::-1]

class DNAMotifTest:
    def __init__(self, scanner: DNAMotifScanner, pvalue: float, occurrence_background: int, total_background: int):
        self.scanner = scanner
        self.pvalue = pvalue
        self.occurrence_background = occurrence_background
        self.total_background = total_background
    
    def test(self, seqs: list):
        n = len(seqs)
        occurrence = sum(self.scanner.exist(seq, self.pvalue, True) for seq in seqs)
        p = self.occurrence_background / self.total_background
        log_fc = np.log2((occurrence / n) / p)
        pval = 1.0 - binom.cdf(occurrence, n, p) if log_fc >= 0 else binom.cdf(occurrence, n, p)
        return log_fc, pval


def parse_meme(content: str) -> List[DNAMotif]:
    motifs = []
    sections = content.split("MOTIF")[1:]
    for section in sections:
        lines = section.strip().split("\n")
        id = lines[0].strip()
        iter_lines = iter(lines)
        while not next(iter_lines, "").startswith("letter-probability matrix"):
            continue
        matrix_lines = []
        for line in iter_lines:
            if line.strip():
                matrix_lines.append([float(v) for v in line.split()])
        matrix = np.array(matrix_lines)
        motifs.append(DNAMotif(id, matrix))
    return motifs


def read_motifs(filename: str) -> List[PyDNAMotif]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Couldn't open file: {filename}")
    
    with open(filename, 'r') as file:
        content = file.read()
    
    return [PyDNAMotif(motif.id, motif.probability) for motif in parse_meme(content)]


def fetch_seq(fasta, region):
    chr, x = region.split(':')
    start, end = x.split('-')
    start = int(start)
    end = int(end)
    seq = fasta[chr][start:end].seq
    l1 = len(seq)
    l2 = end - start
    if l1 != l2:
        raise NameError(
            "sequence fetch error: expected length: {}, but got {}.".format(l2, l1)
        )
    else:
        return seq
    

def cis_bp(unique: bool = True, name='cisBP_human') -> list[PyDNAMotif]:
    """A list of transcription factor motifs curated by the CIS-BP database.

    This function returns motifs curated from the CIS-BP database [Weirauch14]_.
    The motifs can be used to scan the genome for potential binding sites and
    to perform motif enrichment analysis.

    Parameters
    ----------
    unique
        A transcription factor may have multiple motifs. If `unique=True`, 
        only the motifs with the highest information content will be selected.

    Returns
    -------
    list[PyDNAMotif]
        A list of motifs.

    See Also
    --------
    :func:`~snapatac2.tl.motif_enrichment`: compute motif enrichment.
    """
    # motifs = read_motifs(register_datasets().fetch("cisBP_human.meme"))
    motif_file = Path(f'~/.cache/motifs/{name}.meme').expanduser()
    motifs = read_motifs(motif_file)
    for motif in motifs:
        motif.name = motif.id.split('+')[0]
    if unique:
        unique_motifs = {}
        for motif in motifs:
            name = motif.name
            if (
                    name not in unique_motifs or 
                    unique_motifs[name].info_content() < motif.info_content()
               ):
               unique_motifs[name] = motif
        motifs = list(unique_motifs.values())
    return motifs

import polars as pl
from scipy.stats import binom, hypergeom
from math import log2
from tqdm import tqdm


def motif_enrichment(
    regions: dict[str, list[str]],
    motifs: list[PyDNAMotif] = cis_bp(),
    genome_fasta: Path = Path('~/.cache/genome/hg38/hg38.fa').expanduser(),
    background: list[str] | None = None,
    method: Literal['binomial', 'hypergeometric'] | None = None,
) -> dict[str, 'polars.DataFrame']:
    """
    Identify enriched transcription factor motifs.

    Parameters
    ----------
    motifs
        A list of transcription factor motifs.
    regions
        Groups of regions. Each group will be tested independently against the background.
    genome_fasta
        A fasta file containing the genome sequences or a Genome object.
    background
        A list of regions to be used as the background. If None, the union of elements
        in `regions` will be used as the background.
    method
        Statistical testing method: "binomial" or "hypergeometric".
        To use "hypergeometric", the testing regions must be a subset of
        background regions.

    Returns
    -------
    dict[str, pl.DataFrame]:
        Dataframes containing the enrichment analysis results for different groups.
    """
    from pyfaidx import Fasta
    from tqdm import tqdm
    from scipy.stats import binom, hypergeom
    from math import log2
    import polars as pl

    def count_occurrence(query, idx_map, bound):
        return sum(bound[idx_map[q]] for q in query)

    if method is None:
        method = "hypergeometric" if background is None else "binomial"

    all_regions = set(p for ps in regions.values() for p in ps)
    if background is not None:
        for p in background:
            all_regions.add(p)
    all_regions = list(all_regions)
    region_to_idx = dict(map(lambda x: (x[1], x[0]), enumerate(all_regions)))

    logging.info("Fetching {} sequences ...".format(len(all_regions)))
    genome = genome_fasta #.fasta if isinstance(genome_fasta, Genome) else str(genome_fasta)
    genome = Fasta(genome, one_based_attributes=False)
    sequences = [fetch_seq(genome, region) for region in all_regions]

    motif_id = []
    motif_name = []
    motif_family = []
    group_name = []
    fold_change = []
    n_fg = []
    N_fg = []
    n_bg = []
    N_bg = []
    logging.info("Computing enrichment ...")
    for motif in tqdm(motifs):
        bound = motif.with_nucl_prob().exists(sequences)
        if background is None:
            total_bg = len(bound)
            bound_bg = sum(bound)
        else:
            total_bg = len(background)
            bound_bg = count_occurrence(background, region_to_idx, bound)
        
        for key, val in regions.items():
            total_fg = len(val)
            bound_fg = count_occurrence(val, region_to_idx, bound)

            if bound_fg == 0:
                log_fc = 0 if bound_bg == 0 else float('-inf')
            else:
                log_fc = log2((bound_fg / total_fg) / (bound_bg / total_bg)) if bound_bg > 0 else float('inf')

            motif_id.append(motif.id)
            motif_name.append(motif.name)
            motif_family.append(motif.family)
            group_name.append(key)
            fold_change.append(log_fc)
            n_fg.append(bound_fg)
            N_fg.append(total_fg)
            n_bg.append(bound_bg)
            N_bg.append(total_bg)
          
    if method == "binomial":
        pval = binom.cdf(n_fg, N_fg, np.array(n_bg) / np.array(N_bg))
    elif method == "hypergeometric":
        pval = hypergeom.cdf(n_fg, N_bg, n_bg, N_fg)
    else:
        raise NameError("'method' needs to be 'binomial' or 'hypergeometric'")

    result = dict(
        (key, {'id': [], 'name': [], 'family': [], 'log2(fold change)': [], 'p-value': []}) for key in regions.keys()
    )
    for i, key in enumerate(group_name):
        log_fc = fold_change[i]
        p = (1 - pval[i]) if log_fc >= 0 else pval[i]
        result[key]['id'].append(motif_id[i])
        result[key]['name'].append(motif_name[i])
        result[key]['family'].append(motif_family[i])
        result[key]['log2(fold change)'].append(log_fc)
        result[key]['p-value'].append(float(p))

    for key in result.keys():
        result[key]['adjusted p-value'] = _p_adjust_bh(result[key]['p-value'])
        result[key] = pl.DataFrame(result[key])
    return result

