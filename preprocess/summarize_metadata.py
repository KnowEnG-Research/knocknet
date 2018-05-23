""" module to summarize metadata contents"""
from argparse import ArgumentParser
import pandas as pd

def main_parse_args():
    """Processes command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('metadata_infile', help='file containing lincs metadata')
    parser.add_argument('pert_iname_outfile', help='file containing pert_iname_counts')
    args = parser.parse_args()
    args_dict = vars(args)
    print("args_dict: " + str(args_dict) +'\n')
    return args_dict


def summarize_metadata(expmeta_df):
    """Warns contents of LINCS metdata dataframe and prints pert_iname counts."""
    print('\n' + 'df_shape: ' + str(expmeta_df.shape))
    dfcolnames = list(expmeta_df.columns)
    print('\n' + 'colnames: ' + str(dfcolnames))
    print('\n' + 'pert_type values')
    print(expmeta_df['pert_type'].value_counts())
    print('\n' + 'cell_id values')
    cell_id_counts = expmeta_df['cell_id'].value_counts()
    print(cell_id_counts)
    print('cell_id_counts_shape: ' + str(cell_id_counts.shape))
    print('\n' + ','.join(list(cell_id_counts.index)))
    pert_iname_counts = expmeta_df['pert_iname'].value_counts()
    ranks = list(range(0, pert_iname_counts.shape[0]))
    pert_iname_ranks = pd.Series(ranks, index=pert_iname_counts.index, name='rank')
    pert_iname_df = pd.concat([pert_iname_ranks, pert_iname_counts.rename('counts')], axis=1)
    print('\n' + 'pert_iname_counts_shape: ' + str(pert_iname_counts.shape))
    return pert_iname_df


def main():
    """Parse args and reads and write files for metadata summary"""
    args_dict = main_parse_args()
    expmeta_df = pd.read_table(args_dict['metadata_infile'], sep='\t', header=0, index_col=0)
    pert_iname_df = summarize_metadata(expmeta_df)
    print('\n' + 'printing pert_iname_outfile: ' + args_dict['pert_iname_outfile'])
    pert_iname_df.to_csv(args_dict['pert_iname_outfile'], sep='\t', header=False, index=True)

if __name__ == "__main__":
    main()
