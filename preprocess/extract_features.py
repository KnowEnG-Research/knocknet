""" module to pairing metadata contents"""
from argparse import ArgumentParser
import os
import yaml
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress import parse

def main_parse_args():
    """Processes command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('paired_metadata_infile', help='file containing lincs metadata')
    parser.add_argument('gctx_infile', help='file containing lincs level 3 features')
    parser.add_argument('probeset_infile', help='file containing probes to keep')
    parser.add_argument('output_dir', help='directory for outputs')
    parser.add_argument('-dl', '--dataset_label', default='file', help='output file basename')
    parser.add_argument('-os', '--output_suff', default='.data', help='output file suffix')
    parser.add_argument('-mc', '--maxchunks', default=10, type=int, help='maximum number of chunks')
    args = parser.parse_args()
    args_dict = vars(args)
    print("args_dict: " + str(args_dict) +'\n')
    return args_dict


def main():
    """Parse args and reads and write files for metadata pairing"""
    args_dict = main_parse_args()

    # get list of probes
    probeset_df = pd.read_table(args_dict['probeset_infile'], sep='\t')
    probeset = np.array(map(str, probeset_df['pr_gene_id'].values))

    # read in paired metadata
    paired_metadata_df = pd.read_table(args_dict['paired_metadata_infile'], sep='\t',
                                       header=0, index_col=0)
    chunksize = int(paired_metadata_df.shape[0] / args_dict['maxchunks'])

    # get info about gctx file
    col_metadata = parse.parse(args_dict['gctx_infile'], col_meta_only=True)
    geoexpset = set(col_metadata.index.values)

    if not os.path.exists(args_dict["output_dir"]):
        os.makedirs(args_dict["output_dir"])

    # info_yml
    info_dict = {}
    info_dict['num_files'] = args_dict['maxchunks']
    info_dict['num_metadata'] = paired_metadata_df.shape[1]-1
    info_dict['num_features'] = len(probeset)*2
    info_dict['class_column'] = len(probeset)*2+1

    num_examples = 0
    chunkstart = 0
    chunkend = min(paired_metadata_df.shape[0], chunksize)

    # fetch and write one chunk at a time
    for chunk in range(0, args_dict['maxchunks']):

        # get ids for one chunk
        # chunk_labels = paired_metadata_df.iloc[chunkstart:chunkend, 0].values
        chunk_s1_ids = paired_metadata_df.iloc[chunkstart:chunkend, 1].values
        chunk_s2_ids = paired_metadata_df.iloc[chunkstart:chunkend, 2].values

        # keep only ids in gctx file
        print("Filtering exp ids for chunk " + str(chunk) + "...")
        s1set = set(chunk_s1_ids) & geoexpset
        s2set = set(chunk_s2_ids) & geoexpset
        s1bool = np.isin(chunk_s1_ids, list(s1set))
        s2bool = np.isin(chunk_s2_ids, list(s2set))
        keepbool = np.logical_and(s1bool, s2bool)
        print("chunk " + str(chunk) + ": keep " + str(sum(keepbool)) + " of " +
              str(len(keepbool)))
        keepidxs = paired_metadata_df.index[chunkstart:chunkend][keepbool]
        keep_s1ids = paired_metadata_df.iloc[keepidxs, 1]
        keep_s2ids = paired_metadata_df.iloc[keepidxs, 2]
        validexp_ids = np.unique(np.append(keep_s1ids, keep_s2ids))
        meta_df = paired_metadata_df.iloc[keepidxs].T
        print("unique s1-" + str(len(np.unique(keep_s1ids))) +
              " s2-" + str(len(np.unique(keep_s2ids))))

        # fetch data from gctx
        print("Fetching chunk " + str(chunk) + "...")
        allexps_gct = parse.parse(args_dict['gctx_infile'], rid=probeset, cid=validexp_ids)
        s1_df = allexps_gct.data_df[keep_s1ids]
        s2_df = allexps_gct.data_df[keep_s2ids]
        s1_df.columns = keepidxs
        s2_df.columns = keepidxs

        # merge and write outfile
        print("Merging dfs chunk " + str(chunk) + "...")
        merge_df = pd.concat([s1_df, s2_df, meta_df], 0)
        print("shapes - s1, s2, labels, meta, merge: " + str([s1_df.shape, s2_df.shape,
                                                              meta_df.shape,
                                                              merge_df.shape]))
        outfile = os.path.join(args_dict['output_dir'], args_dict['dataset_label']
                               + str(chunk) + args_dict['output_suff'])
        print("Writing outfile: "+ outfile)
        merge_df.T.to_csv(outfile, sep='\t', header=False, index=False)

        # update chunk and end
        chunkstart = chunkend
        chunkend = min(paired_metadata_df.shape[0], chunkstart+chunksize)
        num_examples = num_examples + merge_df.shape[1]

    # print out info.yml
    info_dict['num_examples'] = num_examples
    param_yml = os.path.join(args_dict["output_dir"], 'info.yml')
    with open(param_yml, 'w') as ymlfile:
        yaml.dump(info_dict, ymlfile, default_flow_style=False)

if __name__ == "__main__":
    main()
