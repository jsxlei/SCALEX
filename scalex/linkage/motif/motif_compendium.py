import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.motif as utils_motif
from MotifCompendium.utils.similarity import set_default_options
from IPython.display import display, HTML, Image
import pandas as pd
import numpy as np
import os
import h5py
set_default_options(max_chunk=1000, max_cpus=16, use_gpu=False)

import argparse
import sys
import logging
import warnings




def load_modisco_results(modisco_dir):
    """
    Load the modisco results from the specified directory.
    """
    modisco_dir = args.modisco_dir #'/users/leixiong/projects/imac_igvf/results/modisco/'
    modisco_dict = {}

    for group in os.listdir(modisco_dir):
        # print(group)
        group_dir = f"{modisco_dir}/{group}/"

        for cluster in os.listdir(group_dir):
            modisco_dict[group+'_'+cluster] = f"{group_dir}/{cluster}/counts_modisco.h5"
    return modisco_dict


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    warnings.filterwarnings("ignore")
    out_dir = os.path.join(args.out_dir, 'all_data') #'/users/leixiong/projects/imac_igvf/results/motif_compendium/all_data/'
    os.makedirs(out_dir, exist_ok=True)

    ## 1. Load the modisco results and save the motif compendium
    logging.info("1. Loading modisco results...")
    modisco_dict = load_modisco_results(args.modisco_dir)
    mc = MotifCompendium.build_from_modisco(modisco_dict)
    mc['dataset'] = mc['name'].apply(lambda x: x.split('_')[0])
    # print(mc.metadata)
    mc.save(os.path.join(out_dir, 'all_data.motif_compendium.mc')) 
    utils_analysis.plot_similarity_distribution(mc, os.path.join(out_dir, 'all_data.similarity_distribution.html'))

    ## 2. Clustering
    logging.info("2. Clustering...")
    leiden = args.leiden #0.96
    leiden_name = 'leiden_' + str(int(leiden * 100))#'leiden_96'
    cluster_path = os.path.join(out_dir, leiden_name)
    os.makedirs(cluster_path, exist_ok=True)
    mc.cluster(algorithm="leiden", similarity_threshold=leiden, save_name=leiden_name)

    ## 3. Cluster average
    logging.info("3. Calculating cluster averages...")
    mc_avg = mc.cluster_averages(
        leiden_name,
        aggregations=[
            ('name', 'unique', 'num_patterns'),
            ('num_seqlets', 'sum', 'num_seqlets'),
            ("model", "unique", "num_samples"),
            ("dataset", "unique", "num_datasets"),
            ("dataset", "concat", "datasets"),
            ("posneg", "concat", "posneg")
        ],
        max_chunk=1000, #use_gpu=True
    )

    ## 4. Seqlets annotation
    logging.info("4. Annotating seqlets...")
    hocomoco_meme_file = args.motif_dir + '/hocomoco_v12/H12CORE_meme_format.meme'
    vierstra_meme_file = args.motif_dir + '/vierstra/all.dbs.meme'
    vierstra_meta_file = args.motif_dir + '/vierstra/metadata.tsv'
    selin_mc_file = args.motif_dir + '/selin/selin_compendium.mc'
    selin_avg_mc_file = args.motif_dir + '/selin/selin_compendium.avg.mc'
    selin_h5_file = args.motif_dir + '/selin/selin.motif_compendium.avg.h5' #'/oak/stanford/groups/akundaje/soumyak/motifs/latest/selin/selin.motif_compendium.avg.h5'

    ### 4.1. Hocomoco
    logging.info("4.1. Hocomoco...")
    utils_analysis.label_from_pfms(mc_avg, hocomoco_meme_file, "hocomoco_similarity", "hocomoco_match",
                               max_chunk=1000,) # use_gpu=True)

    ### 4.2. Vierstra
    logging.info("4.2. Vierstra...")
    utils_analysis.label_from_pfms(mc_avg, vierstra_meme_file, "vierstra_similarity", "vierstra_match", max_chunk=1000) #, use_gpu=True)
    vierstra_metadata = pd.read_table(vierstra_meta_file)
    mc_avg.metadata = mc_avg.metadata.merge(vierstra_metadata[['motif_id', 'tf_name']],
                                        left_on='vierstra_match', right_on='motif_id',
                                        how='left')
    mc_avg.metadata['vierstra_match'] = mc_avg.metadata['tf_name'] + '_' + mc_avg.metadata['vierstra_match']
    mc_avg.metadata.drop(columns=['motif_id', 'tf_name'], inplace=True)

    ### 4.3 Selin
    logging.info("4.3. Selin annotation...")
    selin_mc = MotifCompendium.load(selin_mc_file, safe=False)
    mc_avg.assign_clusters_from_other(selin_mc, "annotation", "selin_similarity", "selin_match",
                                  max_chunk=1000) #, use_gpu=True)
    

    ### 4.4 Combine all annotations
    logging.info("4.4. Combining all annotations...")
    cluster_labels_dict = dict()
    annotations = []
    threshold = 0.8

    for index, row in mc_avg.metadata.iterrows():
        if row["selin_similarity"] > threshold:
            cluster_labels_dict[index] = f"{row['selin_match']}".split('#')[0] #.replace("/", "-").replace("#", "-")
            annotations.append(f"{row['selin_match']}".split('#')[0]) #.replace("/", "-").replace("#", "-"))
        elif row["vierstra_similarity"] > threshold:
            cluster_labels_dict[index] = f"{row['vierstra_match']}"
            annotations.append(f"{row['vierstra_match']}")
        elif row["hocomoco_similarity"] > threshold:
            cluster_labels_dict[index] = f"{row['hocomoco_match']}"
            annotations.append(f"{row['hocomoco_match']}")
        else:
            cluster_labels_dict[index] = f"Unknown"
            annotations.append(f"Unknown")

    mc_avg["annotation"] = annotations
    mc["annotation"] = mc[leiden_name].map(cluster_labels_dict)

    ### 4.5 Save the annotations
    logging.info("4.5 Saving the annotations...")
    mc_metadata = mc.metadata.copy()
    mc_metadata['name'] = mc_metadata["name"].str.replace(r".*-(pos|neg)\.pattern_", r"\1_patterns.pattern_", regex=True)
    mc_metadata['model'] = mc_metadata['model'].str.split('_', n=1).str[1]

    # mc_metadata.loc[:, ['dataset', 'model', 'name', 'annotation']].to_csv(
    #     '../results/motif_compendium/all_data/leiden_96/annotation_threshold_0.8.txt', sep='\t', index=None)
    
    ### 4.6 Rename merged patterns
    logging.info("4.6 Renaming merged patterns...")
    pos_sofar = 0
    neg_sofar = 0
    names = []

    for index, row in mc_avg.metadata.iterrows():
        if row['posneg'] == 'pos':
            names.append(f"pos_patterns.pattern_{pos_sofar}")
            pos_sofar += 1
        else:
            names.append(f"neg_patterns.pattern_{neg_sofar}")
            neg_sofar += 1

    mc_avg.metadata['name'] = names

    # mc_avg.metadata


    size_4 = mc_avg.motifs.shape[2] == 4
    motifs = mc_avg.motifs if size_4 else utils_motif.motif_8_to_4(mc_avg.motifs)
    pos_neg = np.sum(motifs, axis=(1, 2)) > 0
    pos_neg = ["pos" if x > 0 else "neg" for x in pos_neg]
    mc_avg.metadata['posneg'] = pos_neg
    mc_avg.metadata.reset_index(inplace=True)

    # mc_avg.metadata

    ### 4.7 Summarize the annotations
    logging.info("4.7 Summarizing the annotations...")
    utils_analysis.plot_unique_per_cluster(mc, "model", os.path.join(cluster_path, "unique_motif.html"))
    utils_analysis.cluster_grouping_upset_plot(mc, leiden_name, "model", os.path.join(cluster_path, "upset_plot_celltype_vs_cluster.png"), min_subset_size=3)

    ### Save and export
    mc_avg.summary_table_html(os.path.join(cluster_path, f"all_data.summary_table.{leiden_name}.html"), 
                          mc_avg.metadata.columns, 16)
    mc_avg.save(os.path.join(cluster_path, f'all_data.motif_compendium.avg.{leiden_name}.mc'))
    mc.save(os.path.join(cluster_path, f'all_data.motif_compendium.{leiden_name}.mc'))
    utils_analysis.export_modisco(mc_avg, 'name',
                              os.path.join(cluster_path, f'all_data.motif_compendium.avg.{leiden_name}.h5'))
    mc_avg.metadata.to_csv(os.path.join(cluster_path, f'all_data.motif_compendium.avg.metadata.{leiden_name}.tsv'),
                       sep='\t', index=False)

    mc_avg_annotated_only_metadata = mc_avg.metadata.loc[~mc_avg.metadata['annotation'].str.startswith('Unknown')][['name', 'annotation']]
    mc_avg_annotated_only_metadata.to_csv(os.path.join(cluster_path, f'all_data.motif_compendium.avg.metadata.annotated_only.{leiden_name}.tsv'),
                                      sep='\t', header=False, index=False)


    ## 5. Plot motif logo
    logging.info("5. Plotting motif logos...")
    from logo_utils import create_modisco_logos, create_selin_logos, make_logo, read_meme, path_to_image_link

    mc_h5_file = os.path.join(cluster_path, f'all_data.motif_compendium.avg.{leiden_name}.h5')
    metadata_file = os.path.join(cluster_path, f'all_data.motif_compendium.avg.metadata.{leiden_name}.tsv')
    logo_dir = os.path.join(out_dir, 'logos')
    os.makedirs(logo_dir, exist_ok=True)

    ### 5.3 meta data
    metadata = pd.read_table(metadata_file)
    metadata['selin_match'] = metadata['selin_match'].str.replace('/', '-')
    metadata['selin_match'] = metadata['selin_match'].str.replace("#", "-")

    print(metadata)
    ### 5.1 load h5 file
    # mc_h5 = h5py.File(mc_h5_file, 'r')
    # selin_h5 = h5py.File(selin_h5_file, 'r')


    ### 5.2 Create motif logos
    modisco_logo_dir = os.path.join(logo_dir, f'modisco/all_data/{leiden_name}')
    # if not os.path.isdir(modisco_logo_dir):
    os.makedirs(modisco_logo_dir, exist_ok=True)

    # pattern_groups = [group for group in mc_h5.keys()]
    # print(pattern_groups)

    trim_threshold = 0.1

    #### 5.2.1 Create modisco logos
    logging.info("5.2.1 Creating modisco logos...")
    create_modisco_logos(mc_h5_file, modisco_logo_dir, trim_threshold, ['pos_patterns'])
    create_modisco_logos(mc_h5_file, modisco_logo_dir, trim_threshold, ['neg_patterns'])

    #### 5.2.2 Create selin logos
    logging.info("5.2.2 Creating selin logos...")
    selin_logo_dir = os.path.join(logo_dir, 'selin')
    if not os.path.isdir(selin_logo_dir):
        os.mkdir(selin_logo_dir)

    # pattern_groups = [group for group in selin_h5.keys()]
    # print(pattern_groups)

    # trim_threshold = 0.1

    create_selin_logos(selin_h5_file, selin_logo_dir, trim_threshold, ['pos_patterns'])
    create_selin_logos(selin_h5_file, selin_logo_dir, trim_threshold, ['neg_patterns'])

    #### 5.2.3 Create hocomoco and vierstra logos
    logging.info("5.2.3 Creating hocomoco and vierstra logos...")
    hocomoco_meme = read_meme(hocomoco_meme_file)
    vierstra_meme = read_meme(vierstra_meme_file)

    # vierstra_metadata = pd.read_table('/oak/stanford/groups/akundaje/soumyak/motifs/latest/vierstra/metadata.tsv')
    vierstra_metadata = pd.read_table(vierstra_meta_file)
    vierstra_subset = vierstra_metadata.loc[vierstra_metadata.apply(lambda x: x['tf_name'] + '_' + x['motif_id'] in metadata['vierstra_match'].values,
                                                                axis=1)]

    vierstra_meme_subset = {}

    for index,row in vierstra_subset.iterrows():
        vierstra_meme_subset[row['tf_name'] + '_' + row['motif_id']] = vierstra_meme[row['motif_id']]

    hocomoco_meme_subset = {key: hocomoco_meme[key] for key in hocomoco_meme.keys() if key in metadata['hocomoco_match'].values}
    vierstra_logo_dir = os.path.join(logo_dir, 'vierstra')
    if not os.path.isdir(vierstra_logo_dir):
        os.mkdir(vierstra_logo_dir)

    for motif in vierstra_meme_subset.keys():
        make_logo(motif, vierstra_logo_dir, vierstra_meme_subset)

    hocomoco_logo_dir = os.path.join(logo_dir, 'hocomoco')
    if not os.path.isdir(hocomoco_logo_dir):
        os.mkdir(hocomoco_logo_dir)

    for motif in hocomoco_meme_subset.keys():
        make_logo(motif, hocomoco_logo_dir, hocomoco_meme_subset)




    ### 
    
    logo_link_base = args.url_dir.replace('/users/', 'https://mitra.stanford.edu/kundaje/oak/') 
    # logo_link_base = 'https://mitra.stanford.edu/kundaje/oak/leixiong/public/igvf_imac_results/motif_compendium/logos'

    metadata['modisco_fwd'] = path_to_image_link(logo_link_base + f'/modisco/all_data/{leiden_name}/' + metadata['name'] + '.cwm.fwd.png')
    metadata['modisco_rev'] = path_to_image_link(logo_link_base + f'/modisco/all_data/{leiden_name}/' + metadata['name'] + '.cwm.rev.png')
    metadata['selin_fwd'] = path_to_image_link(logo_link_base + '/selin/' + metadata['selin_match'] + '.cwm.fwd.png')
    metadata['selin_rev'] = path_to_image_link(logo_link_base + '/selin/' + metadata['selin_match'] + '.cwm.rev.png')
    metadata['vierstra_logo'] = path_to_image_link(logo_link_base + '/vierstra/' + metadata['vierstra_match'] + '.png')
    metadata['hocomoco_logo'] = path_to_image_link(logo_link_base + '/hocomoco/' + metadata['hocomoco_match'] + '.png')
    metadata.rename(columns={'name': 'modisco_pattern',
                            'annotation': 'auto_annotation'},
                            inplace=True)
    metadata['manual_annotation'] = ''

    metadata = metadata[['modisco_pattern', 'modisco_fwd', 'modisco_rev', 'manual_annotation', 'auto_annotation',
                        'selin_similarity', 'selin_match', 'selin_fwd', 'selin_rev',
                        'vierstra_similarity', 'vierstra_match', 'vierstra_logo',
                        'hocomoco_similarity', 'hocomoco_match', 'hocomoco_logo',
                        'num_patterns', 'num_seqlets', 'num_samples', 'num_datasets',
                        'datasets', 'posneg', 'index']]
    
    metadata.to_csv(os.path.join(cluster_path, f'all_data.motif_compendium.avg.metadata.{leiden_name}.logos.tsv'),
                sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motif Compendium")
    parser.add_argument('--modisco_dir', type=str, help='Directory containing modisco results')
    parser.add_argument('--out_dir', type=str, help='Directory to save the results')
    parser.add_argument('--url_dir', type=str)
    parser.add_argument('--leiden', type=float, default=0.96, help='Leiden clustering threshold')
    parser.add_argument('--motif_dir', type=str, default='/oak/stanford/groups/akundaje/soumyak/motifs/latest/', help='Directory containing motif files')
    # parser.add_argument('--hocomoco_meme', type=str, default="/oak/stanford/groups/akundaje/soumyak/motifs/latest/hocomoco_v12/H12CORE_meme_format.meme", help='Path to the hocomoco file')
    # parser.add_argument('--vierstra_meme', type=str, default="/oak/stanford/groups/akundaje/soumyak/motifs/latest/vierstra/all.dbs.meme", help='Path to the vierstra file')
    # parser.add_argument('--vierstra_meta', type=str, default='/oak/stanford/groups/akundaje/soumyak/motifs/latest/vierstra/metadata.tsv', help='Path to the vierstra metadata file')
    # parser.add_argument('--selin_mc', type=str, default="/oak/stanford/groups/akundaje/soumyak/motifs/latest/selin/selin_compendium.mc", help='Path to the selin file')
    # parser.add_argument('--selin_avg_mc', type=str, default="/oak/stanford/groups/akundaje/soumyak/motifs/latest/selin/selin_compendium.avg.mc", help='Path to the selin average file')

    args = parser.parse_args()

    main(args)
