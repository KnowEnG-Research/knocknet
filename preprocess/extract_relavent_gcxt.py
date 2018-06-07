""" module to extract features for paired metadata"""
from argparse import ArgumentParser
import os
import yaml
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress import parse
from cmapPy.pandasGEXpress import write_gctx

def main_parse_args():
    """Processes command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('gctx_infile', help='file containing lincs level 3 features')
    parser.add_argument('expid_infile', help='file containing lincs metadata')
    parser.add_argument('probeset_infile', help='file containing probes to keep')
    parser.add_argument('output_file', help='directory for outputs')
    args = parser.parse_args()
    args_dict = vars(args)
    print("args_dict: " + str(args_dict) +'\n')
    return args_dict


def main():
"""Parse args and reads and write expression files for paired metadata"""
    args_dict = main_parse_args()
    
    # get list of probes
    probeset_df = pd.read_table(args_dict['probeset_infile'], sep='\t')
    probeset = np.array(map(str, probeset_df['pr_gene_id'].values))
    
    # get list of experiments
    expid_df = pd.read_table(args_dict['expid_infile'], sep='\t', header=None)
    myexpids = np.array(map(str, expid_df[0].values))
    
    # get info about gctx file
    col_metadata = parse.parse(args_dict['gctx_infile'], col_meta_only=True)
    geoexpset = set(col_metadata.index.values)
    
    
    # keep only ids in gctx file
    print("Filtering exp ids for chunk " + str(chunk) + "...")
    validexp_ids = np.array(list(set(myexpids) & geoexpset))
    
    # fetch data from gctx
    print("Fetching chunk " + str(chunk) + "...")
    allexps_gct = parse.parse(args_dict['gctx_infile'], rid=probeset, cid=validexp_ids)
    #returns rows and columns in different order
    print("Gene Ids Order: " + str(list(allexps_gct.data_df.index)))
    
    # merge and write outfile
    print("Writing outfile: "+ args_dict['outfile'])
    write_gctx.write(allexps_gct, args_dict['outfile'])

if __name__ == "__main__":
    main()
