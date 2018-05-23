""" module to pairing metadata contents"""
import argparse
import os
import yaml
import numpy as np
import tensorflow as tf


def main_parse_args():
    """Processes command line arguments."""
    parser = argparse.ArgumentParser(description="serialize a dataset")
    parser.add_argument("indir", help="path of the in-directory")
    parser.add_argument("outdir", help="path of the out-directory")
    parser.add_argument("--label_col", type=int, default=1,
                        help="column number (from 1) of the label")
    parser.add_argument("--insuff", help="the suffix of the infiles", default=".data")
    parser.add_argument("--outsuff", help="the suffix of the outfiles", default=".tfrecord")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args_dict = vars(args)
    print("args_dict: " + str(args_dict) +'\n')
    return args_dict


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def main():
    """Parse args and reads and write files for metadata pairing"""

    # parse arguments
    args_dict = main_parse_args()

    # create outdir
    os.makedirs(args_dict['outdir'], exist_ok=True)

    # find names of infiles
    data_files = []
    for name in os.listdir(args_dict['indir']):
        if name.endswith(args_dict['insuff']):
            data_files.append(name)

    # for each infile
    info_dict = {}
    num_examples = 0
    for i in range(0, len(data_files)):
        infile = os.path.join(args_dict['indir'], data_files[i])
        print('reading ' + infile)

        # create corresponding outfile
        dataset_name = data_files[i].replace(args_dict['insuff'], "")
        outfile = os.path.join(args_dict['outdir'], dataset_name + args_dict['outsuff'])
        print(outfile)
        writer = tf.python_io.TFRecordWriter(outfile)

        f_stream = open(infile)
        # for each infile line
        for line in f_stream:
            #fetch line
            line = line.strip().split('\t')
            line_end = len(line)
            # break line into three components
            example_features = np.array(line[0:args_dict['label_col']-1]).astype(float)
            example_label = np.array(line[args_dict['label_col']-1]).astype(int)
            example_meta = np.array(line[args_dict['label_col']:line_end]).astype(bytes)
            # create and write example object
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label' : _int64_feature(int(example_label)),
                        'feature_values' : _float_feature(example_features),
                        'meta_values' : _bytes_feature(example_meta)
                    }))
            serialized_example = example.SerializeToString()
            writer.write(serialized_example)
            num_metadata = len(example_meta)
            num_features = len(example_features)
            num_examples = num_examples+1

    info_dict['num_examples'] = num_examples
    info_dict['class_column'] = args_dict['label_col']
    info_dict['num_files'] = len(data_files)
    info_dict['num_features'] = num_features
    info_dict['num_metadata'] = num_metadata
    param_yml = os.path.join(args_dict["outdir"], 'info.yml')
    with open(param_yml, 'w') as ymlfile:
        yaml.dump(info_dict, ymlfile, default_flow_style=False)


if __name__ == "__main__":
    main()
