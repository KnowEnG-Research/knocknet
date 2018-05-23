""" module to partition metadata contents"""
from argparse import ArgumentParser
import numpy as np
import pandas as pd

def main_parse_args():
    """Processes command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('metadata_infile', help='file containing orig lincs metadata')
    parser.add_argument('outfile_prefix', help='prefix for outfile partitions')
    parser.add_argument('-pk', '--partition_key', default='pert_iname',
                        help='field values to distribute equally across partitions')
    parser.add_argument('-pp', '--partition_percentages', default='1',
                        help='comma separated list of partition percentages')
    parser.add_argument('-pn', '--partition_names', default='all',
                        help='comma separated list of partition names')
    parser.add_argument('-os', '--outfile_suffix', default='.metadata.txt',
                        help='suffix for output partition files')
    args = parser.parse_args()
    args_dict = vars(args)
    print("args_dict: " + str(args_dict) +'\n')
    return args_dict


def partition_metadata(expmeta_df, partition_key, percent_list):
    """Partitions metadata based on percentages for partition_key values."""
    print('\n' + 'original df_shape: ' + str(expmeta_df.shape))
    shuffled_df = expmeta_df.sample(frac=1, random_state=19850411)
    percent_floats = [float(i) for i in percent_list]
    cumul_sum = np.cumsum(percent_floats)/np.sum(percent_floats)
    index_collection = {}
    for i in range(0, len(cumul_sum)):
        index_collection[i] = []
    for val in shuffled_df[partition_key].unique():
        # val = shuffled_df[partition_key].unique()[0]
        bools = shuffled_df[partition_key].isin([val])
        inst_ids = shuffled_df.index[bools]
        last_idxs = (cumul_sum*len(inst_ids)).astype(int)
        print(val + ' last indexes: ' + str(last_idxs))
        start_idx = 0
        for i in range(0, len(cumul_sum)):
            # i = 0
            end_idx = last_idxs[i]
            if start_idx == end_idx:
                continue
            part_idxs = inst_ids[start_idx:end_idx]
            # print(str(i) + ' part_idxs shape: ' + str(part_idxs.shape))
            index_collection[i].extend(list(part_idxs))
            start_idx = end_idx
    return index_collection


def main():
    """Parse args and reads and write files for metadata partition"""
    args_dict = main_parse_args()
    expmeta_df = pd.read_table(args_dict['metadata_infile'], sep='\t', header=0, index_col=0)
    percent_list = args_dict['partition_percentages'].split(",")
    index_collection = partition_metadata(expmeta_df, args_dict['partition_key'], percent_list)
    names_list = args_dict['partition_names'].split(",")
    for idx, idx_list in index_collection.items():
        outfile_name = args_dict['outfile_prefix'] + names_list[idx] + args_dict['outfile_suffix']
        print(str(idx) + ' printing: ' + outfile_name)
        expmeta_df.loc[idx_list].to_csv(outfile_name, sep='\t', header=True, index=True)

if __name__ == "__main__":
    main()
