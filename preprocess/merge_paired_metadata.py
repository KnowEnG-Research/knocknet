""" module to merging paired metadata contents"""
from argparse import ArgumentParser
import pandas as pd


def main_parse_args():
    """Processes command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('paired_metadata1_infile', help='file containing paired lincs metadata')
    parser.add_argument('paired_metadata2_infile', help='file containing paired lincs metadata')
    parser.add_argument('output_file', help='output merged file')
    parser.add_argument('-sl1', '--sub_labels1', default=-1, type=int,
                        help='substitute labels from infile 1')
    parser.add_argument('-sl2', '--sub_labels2', default=-1, type=int,
                        help='substitute labels from infile 2')
    parser.add_argument('-oo', '--original_order', action='store_true', default=False,
                        help='print out paired_meta in the original selected order')
    args = parser.parse_args()
    args_dict = vars(args)
    print("args_dict: " + str(args_dict) +'\n')
    return args_dict


def main():
    """Parse args and reads and write files for metadata pair merging"""
    args_dict = main_parse_args()
    # read in infiles
    state1meta_df = pd.read_table(args_dict['paired_metadata1_infile'], sep='\t',
                                  header=0, index_col=0)
    state2meta_df = pd.read_table(args_dict['paired_metadata2_infile'], sep='\t',
                                  header=0, index_col=0)

    # substitute labels if necessary
    if args_dict['sub_labels1'] > -1:
        state1meta_df.iloc[:, 0] = args_dict['sub_labels1']
    if args_dict['sub_labels2'] > -1:
        state2meta_df.iloc[:, 0] = args_dict['sub_labels2']

    # merge dataframes
    merge_df = pd.concat([state1meta_df, state2meta_df], 0)
    merge_df = merge_df.reset_index(drop=True)

    # shuffle the paired_metadata df
    print('\n' + 'printing output_file: ' + args_dict['output_file'])
    if args_dict['original_order']:
        merge_df.to_csv(args_dict['output_file'], sep='\t', header=True, index=True)
    else: #default
        shuffled_df = merge_df.sample(frac=1)
        shuffled_df = shuffled_df.reset_index(drop=True)
        shuffled_df.to_csv(args_dict['output_file'], sep='\t', header=True, index=True)


if __name__ == "__main__":
    main()
