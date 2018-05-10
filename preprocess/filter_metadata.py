""" module to filter metadata contents"""
from argparse import ArgumentParser
import pandas as pd

def main_parse_args():
    """Processes command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('metadata_infile', help='file containing orig lincs metadata')
    parser.add_argument('metadata_outfile', help='file containing final lincs metadata')
    parser.add_argument('-pt', '--keep_pert_types', default=None,
                        help='comma separated list of pert_type values to keep')
    parser.add_argument('-ci', '--keep_cell_ids', default=None,
                        help='comma separated list of cell_id values to keep')
    parser.add_argument('-pif', '--keep_pert_iname_file', default=None,
                        help='file with the pert_iname to keep in first column')
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


def filter_metadata(expmeta_df, pert_type_list, cell_id_list, pert_iname_list):
    """Applies filters of optional arguments."""
    print('\n' + 'original df_shape: ' + str(expmeta_df.shape))
    if pert_type_list is not None:
        print('\n' + 'selecting ' + str(len(pert_type_list)) + ' pert_types')
        expmeta_df = expmeta_df.loc[expmeta_df['pert_type'].isin(pert_type_list)]
        print('new df_shape: ' + str(expmeta_df.shape))
    if cell_id_list is not None:
        print('\n' + 'selecting ' + str(len(cell_id_list)) + ' cell_ids')
        expmeta_df = expmeta_df.loc[expmeta_df['cell_id'].isin(cell_id_list)]
        print('new df_shape: ' + str(expmeta_df.shape))
    if pert_iname_list is not None:
        print('\n' + 'selecting ' + str(len(pert_iname_list)) + ' pert_inames')
        expmeta_df = expmeta_df.loc[expmeta_df['pert_iname'].isin(pert_iname_list)]
        print('new df_shape: ' + str(expmeta_df.shape))
    return expmeta_df


def main():
    """Parse args and reads and write files for metadata filter"""
    args_dict = main_parse_args()
    expmeta_df = pd.read_table(args_dict['metadata_infile'], sep='\t', header=0, index_col=0)
    pert_type_list = None
    if args_dict['keep_pert_types'] is not None:
        pert_type_list = args_dict['keep_pert_types'].split(",")
    cell_id_list = None
    if args_dict['keep_cell_ids'] is not None:
        cell_id_list = args_dict['keep_cell_ids'].split(",")
    pert_iname_list = None
    if args_dict['keep_pert_iname_file'] is not None:
        pert_iname_df = pd.read_table(args_dict['keep_pert_iname_file'], sep='\t',
                                      header=None, index_col=0)
        pert_iname_list = list(pert_iname_df.index)
    filtered_df = filter_metadata(expmeta_df, pert_type_list, cell_id_list, pert_iname_list)
    print('\n' + 'printing metadata_outfile: ' + args_dict['metadata_outfile'])
    filtered_df.to_csv(args_dict['metadata_outfile'], sep='\t', header=True, index=True)

if __name__ == "__main__":
    main()
