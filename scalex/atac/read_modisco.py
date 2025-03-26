from bs4 import BeautifulSoup
import pandas as pd

def read_modisco_html(motif_html):
    """
    Read modisco html file and return a dataframe.
    """
    if motif_html.endswith(".html"):
        with open(motif_html, "r") as file:
            modisco_report = BeautifulSoup(file, "html.parser")

        table = modisco_report.find_all("table")
        df = pd.read_html(str(table))[0]
    else:
        df = pd.read_csv(motif_html, sep="\t")

    # if motif_meta:
    #     meta = pd.read_csv(motif_meta, sep="\t")
    #     # df = pd.merge(df, meta, left_on='match0', right_on='motif_id', how='left')
    #     motif_id_dict = pd.Series(meta['tf_name'].values, meta['motif_id'].values).to_dict()
    #     source_id_dict = pd.Series(meta['tf_name'].values, meta['source_id'].values).to_dict()
    # else:
    #     motif_id_dict = None
    #     source_id_dict = None
    #     return df['match0']

    # mapping_dict = {}
    # for i, row in df.iterrows():
    #     # print(row)
    #     k = row.loc['pattern']
    #     v = row.loc['match0']
    #     if v in motif_id_dict:
    #         mapping_dict[k] = motif_id_dict[v]            
    #     elif v in source_id_dict:
    #         mapping_dict[k] = source_id_dict[v]
    #     else:
    #         print(f"{k} {v} Not found")

    # df['match0'] = df['pattern'].map(mapping_dict)
    return df[['match0', 'num_seqlets']].groupby('match0', as_index=False)['num_seqlets'].sum().set_index('match0').sort_values('num_seqlets', ascending=False)


def read_mapping_meta(motif_meta):
    """
    Read motif meta file and return a mapping dict.
    """
    meta = pd.read_csv(motif_meta, sep="\t")
    # df = pd.merge(df, meta, left_on='match0', right_on='motif_id', how='left')
    motif_id_dict = pd.Series(meta['tf_name'].values, meta['motif_id'].values).to_dict()
    source_id_dict = pd.Series(meta['tf_name'].values, meta['source_id'].values).to_dict()

    more_dict = {i: source_id_dict[i] for i in source_id_dict if i not in motif_id_dict}
    motif_id_dict.update(more_dict)
    return motif_id_dict
    # mapping_dict = {}
    # for i, row in df.iterrows():
        # print(row)
        # k = row.loc['pattern']
        # v = row.loc['match0']
        # if v in motif_id_dict:
        #     mapping_dict[v] = motif_id_dict[v]            
        # elif v in source_id_dict:
        #     mapping_dict[v] = source_id_dict[v]
        # else:
        #     print(f"{v} Not found")
    
    # return mapping_dict
