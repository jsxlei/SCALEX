import pandas as pd

def e2g_to_loop(e2g: str, threshold: float = 0.1, score_column: str = "ENCODE-rE2G.Score"):
    """
    """
    if isinstance(e2g, str):
        e2g = pd.read_csv(e2g, sep="\t")
    elif isinstance(e2g, pd.DataFrame):
        pass
    else:
        raise ValueError("e2g must be a string or a pandas DataFrame")
    
    # e2g = e2g.rename(columns={"gene_id": "gene", "e2g_id": "e2g"})
    # e2g = e2g[["gene", "e2g"]]
    e2g['Start'] = (e2g['start'] + e2g['end']) / 2
    e2g['End'] = e2g['TargetGeneTSS']
    e2g['Chromosome'] = e2g['chr']
    e2g['score'] = e2g[score_column]
    e2g = e2g[e2g['score'] > threshold]
    return e2g