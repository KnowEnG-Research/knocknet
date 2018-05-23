""" module to pairing metadata contents"""
from argparse import ArgumentParser
import numpy as np
import pandas as pd


def main_parse_args():
    """Processes command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('state1_metadata_infile', help='file containing lincs metadata')
    parser.add_argument('state2_metadata_infile', help='file containing lincs metadata')
    parser.add_argument('output_file', help='file containing matched pairs')
    parser.add_argument('-clf', '--class_label_file', default=None,
                        help='file with the label name and label id in first columns')
    parser.add_argument('-cs', '--class_state', default=2,
                        help='which infile contains the class labels')
    parser.add_argument('-mk', '--match_key', default='cell_id',
                        help='field values to match or unmatch on')
    parser.add_argument('-um', '--unmatch', action='store_true', default=False,
                        help='find paired experiments that do not match on match key')
    parser.add_argument('-npc', '--num_per_class', default=None,
                        help='total number of pairs to generate per class')
    parser.add_argument('-epc', '--exp_per_class', default=10,
                        help='total number of experiments to pair up per class')
    parser.add_argument('-npe', '--num_per_exp', default=1,
                        help='total number of paired examples per class experiment')
    parser.add_argument('-oo', '--original_order', action='store_true', default=False,
                        help='print out paired_meta in the original selected order')
    args = parser.parse_args()
    args_dict = vars(args)
    print("args_dict: " + str(args_dict) +'\n')
    return args_dict


def select_class_instances(class_meta_df, class_label_df,
                           num_per_class, exp_per_class, num_per_exp):
    """returns metadata for requested class instances sorted by class and inst_id"""
    class_key = 'pert_iname'
    print('\nusing ' + class_key + ' for class labels')
    print('\n' + 'class_meta_df: ' + str(class_meta_df.shape))
    # for each class label
    all_selected_class_insts = []
    for label_name in list(class_label_df.index):
        # label_name = class_label_df.index[0]
        inst_ids = list(class_meta_df.index)
        if label_name != 'NoLabel':
            bools = class_meta_df[class_key].isin([label_name])
            inst_ids = list(class_meta_df.index[bools])
        if len(inst_ids) == 0:
            continue;
        selected_insts = []
        if num_per_class is not None:
            selected_insts = balanced_sampler(inst_ids, int(num_per_class), True)
        else:
            # collect exp_per_class X num_per_exp examples
            selected_exps = balanced_sampler(inst_ids, int(exp_per_class), True)
            selected_insts = np.repeat(selected_exps, int(num_per_exp))
        print(label_name, len(inst_ids), len(selected_insts))
        all_selected_class_insts.extend(selected_insts)
    selected_class_meta = class_meta_df.loc[all_selected_class_insts]
    selected_class_meta.reset_index(inplace=True)
    print('\n' + 'selected_class_meta: ' + str(selected_class_meta.shape))
    return selected_class_meta


def select_matched_instances(match_meta_df, selected_class_meta, match_key, unmatch):
    """returns metadata for (un)matched instances in repeated shuffled order"""
    print('\n' + 'match_meta_df: ' + str(match_meta_df.shape))
    print('match_val, uniq_vals, vals_sampled')
    all_selected_match_insts_series = selected_class_meta[match_key].copy()
    # for each match_key value
    for match_val in sorted(list(selected_class_meta[match_key].unique())):
        # match_val = selected_class_meta[match_key].unique()[0]
        replace_bools = all_selected_match_insts_series.isin([match_val])
        nreplaces = sum(replace_bools)
        match_bools = match_meta_df[match_key].isin([match_val])
        match_inst_ids = list(match_meta_df.index[match_bools])
        if unmatch:
            match_inst_ids = list(match_meta_df.index[-match_bools])
        print([match_val, len(match_inst_ids), nreplaces])
        selected_insts = balanced_sampler(match_inst_ids, nreplaces)
        all_selected_match_insts_series[replace_bools] = selected_insts
    selected_match_meta = match_meta_df.loc[all_selected_match_insts_series]
    selected_match_meta.reset_index(inplace=True)
    print('\n' + 'selected_class_meta: ' + str(selected_class_meta.shape))
    return selected_match_meta


def merge_metadata(selected_class_labels, selected_state1_meta,
                   selected_state2_meta, match_key):
    """merge dataframes label, s1_id, s2_id, s1_meta, s2_meta and shuffle"""
    # extract state1 inst_ids that correspond to class instances
    selected_state1_insts = selected_state1_meta['inst_id'].copy()
    selected_state1_insts.name = 'state1'
    # extract state2 inst_ids that correspond to class instances
    selected_state2_insts = selected_state2_meta['inst_id'].copy()
    selected_state2_insts.name = 'state2'
    print('\n' + 'selected_class_labels: ' + str(selected_class_labels.shape))
    print('selected_state1_insts: ' + str(selected_state1_insts.shape))
    print('selected_state2_insts: ' + str(selected_state2_insts.shape))
    print('selected_state1_meta: ' + str(selected_state1_meta.shape))
    print('selected_state2_meta: ' + str(selected_state2_meta.shape))
    # merge dataframes label, s1_id, s2_id, s1_meta, s2_meta
    merge_df = pd.concat([selected_class_labels['label_id'],
                          selected_state1_insts,
                          selected_state2_insts,
                          selected_state1_meta,
                          selected_state2_meta], 1)
    print('')
    print('label', 'total_inst', 'uniq_inst', 'uniq_s1_ids', 'uniq_s1_clines',
          'uniq_s2_ids', 'uniq_s2_clines')
    # summarize per class
    for label in list(merge_df['label_id'].unique()):
        # label = list(merge_df['label_id'].unique())[0]
        bools = merge_df['label_id'].isin([label])
        idxs = list(merge_df.index[bools])

        print(label,
              merge_df.loc[idxs].shape[0],
              merge_df.loc[idxs].groupby(['state1', 'state2']).size().shape[0],
              len(merge_df.ix[idxs, 1].unique()),
              len(merge_df.loc[idxs, match_key].ix[:, 0].unique()),
              len(merge_df.ix[idxs, 2].unique()),
              len(merge_df.loc[idxs, match_key].ix[:, 1].unique())
             )
    return merge_df


def get_class_labels(class_label_df, selected_class_meta):
    """extract class labels that correspond to class instances."""
    class_key = 'pert_iname'
    selected_class_labels = class_label_df.loc[selected_class_meta[class_key], 1]
    selected_class_labels = selected_class_labels.reset_index()
    selected_class_labels.columns = ['label_name', 'label_id']
    return selected_class_labels


def balanced_sampler(object_list, nsamples, return_sorted=False):
    """Return as close to equal nsamples of objects shuffled"""
    nobjects = len(object_list)
    nreps = int(nsamples / nobjects)
    nextra = nsamples - nreps * nobjects
    # print([nsamples, nobjects, nreps, nextra])
    np.random.shuffle(object_list)
    outlist = object_list*nreps
    outlist.extend(object_list[0:nextra])
    #np.random.shuffle(outlist)
    # print(outlist)
    if return_sorted:
        return sorted(outlist)
    return outlist


def main():
    """Parse args and reads and write files for metadata pairing"""
    args_dict = main_parse_args()
    # read in infiles
    state1meta_df = pd.read_table(args_dict['state1_metadata_infile'], sep='\t',
                                  header=0, index_col=0)
    state2meta_df = pd.read_table(args_dict['state2_metadata_infile'], sep='\t',
                                  header=0, index_col=0)
    # create NoLabel class_label_df
    class_label_df = None
    tmpd = {1: [-1], 2: [-1]}
    class_label_df = pd.DataFrame(data=tmpd, index=['NoLabel'])
    # overwrite NoLabel class_label_df if file provide
    if args_dict['class_label_file'] is not None:
        class_label_df = pd.read_table(args_dict['class_label_file'], sep='\t',
                                       header=None, index_col=0)
    np.random.seed(19850411)
    # if pertubogen class is related to the first state
    paired_metadata_df = []
    if args_dict['class_state'] == '1':
        # select class instances and metadata
        selected_class_meta = select_class_instances(state1meta_df,
                                                     class_label_df,
                                                     args_dict['num_per_class'],
                                                     args_dict['exp_per_class'],
                                                     args_dict['num_per_exp'])
        # extract class labels that correspond to class instances
        selected_class_labels = get_class_labels(class_label_df,
                                                 selected_class_meta)
        # select (un)match instances and metadata
        selected_match_meta = select_matched_instances(state2meta_df,
                                                       selected_class_meta,
                                                       args_dict['match_key'],
                                                       args_dict['unmatch'])
        # merge dataframes label, s1_id, s2_id, s1_meta, s2_meta and shuffle
        paired_metadata_df = merge_metadata(selected_class_labels,
                                            selected_class_meta,
                                            selected_match_meta,
                                            args_dict['match_key'])
    # assume class state is second state (main mode)
    else:
        # select class instances and metadata
        selected_class_meta = select_class_instances(state2meta_df,
                                                     class_label_df,
                                                     args_dict['num_per_class'],
                                                     args_dict['exp_per_class'],
                                                     args_dict['num_per_exp'])
        # extract class labels that correspond to class instances
        selected_class_labels = get_class_labels(class_label_df,
                                                 selected_class_meta)
        # select (un)match instances and metadata
        selected_match_meta = select_matched_instances(state1meta_df,
                                                       selected_class_meta,
                                                       args_dict['match_key'],
                                                       args_dict['unmatch'])
        # merge dataframes label, s1_id, s2_id, s1_meta, s2_meta and shuffle
        paired_metadata_df = merge_metadata(selected_class_labels,
                                            selected_match_meta,
                                            selected_class_meta,
                                            args_dict['match_key'])
    # shuffle the paired_metadata df
    print('\n' + 'printing output_file: ' + args_dict['output_file'])
    if args_dict['original_order']:
        paired_metadata_df.to_csv(args_dict['output_file'], sep='\t', header=True, index=True)
    else: #default
        shuffled_df = paired_metadata_df.sample(frac=1)
        shuffled_df.to_csv(args_dict['output_file'], sep='\t', header=True, index=True)


if __name__ == "__main__":
    main()
