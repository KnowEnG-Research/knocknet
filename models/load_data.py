"""load_data module"""
from argparse import ArgumentParser
import os
import yaml
import params
import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_batch(param_dict=params.default_param_dict(), shuffled=True):
    """ uses data_batch_size, data_dir, data_mode, data_serialized """

    batch_size = param_dict["data_batch_size"]
    all_files = sorted(os.listdir(param_dict["data_dir"]))
    nthreads = 1

    # get data characteristics
    data_char_file = os.path.join(param_dict["data_dir"], "info.yml")
    with open(data_char_file) as infile:
        data_dict = yaml.safe_load(infile)
    print("### start load_data")
    print("data class_column: " + str(data_dict['class_column']))
    print("data num_metadata: " + str(data_dict['num_metadata']))
    print("data num_examples: " + str(data_dict['num_examples']))

    # get list of files
    filenames = []
    filesuffix = '.data'
    readtype = tf.TextLineReader
    if param_dict["data_serialized"]:
        filesuffix = '.tfrecord'
        readtype = tf.TFRecordReader
    for fname in all_files:
        if filesuffix in fname:
            filenames.extend([param_dict["data_dir"] + fname])
    nreaders = min(nthreads, len(filenames))
    print("data number of datafiles: " + str(len(filenames)))
    print("data example datafile: " + filenames[0])
    print("data batch_size: " + str(batch_size))
    print("data nreaders: " + str(nreaders))

    # read in example
    if shuffled:
        # Reads multiple records in parallel from data_sources using n readers.
        key, example = slim.parallel_reader.parallel_read(filenames,
                                                          readtype,
                                                          num_epochs=None,
                                                          num_readers=nreaders,
                                                          shuffle=True,
                                                          dtypes=None,
                                                          capacity=32*batch_size,
                                                          min_after_dequeue=16*batch_size,
                                                          seed=19850411,
                                                          scope=None)
    else:
        # Reads sequentially the data_sources using the reader, doing a single pass.
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
        reader = readtype()
        key, example = reader.read(filename_queue)

    #print("data raw example size: " + str(example))

    # decode example into features, label and metadata
    if param_dict["data_serialized"]:
        parse_example = tf.parse_single_example(example,
            features={
                'feature_values' : tf.FixedLenFeature([data_dict['class_column']-1], tf.float32),
                'label' : tf.FixedLenFeature([1], tf.int64),
                'meta_values' : tf.FixedLenFeature([data_dict['num_metadata']], tf.string)
            })
        features = tf.cast(parse_example['feature_values'], tf.float32)
        label = tf.cast(parse_example['label'], tf.int32)
        metadata = tf.cast(parse_example['meta_values'], tf.string)
    else:
        record_defaults = [[1.0] for dim in range(data_dict['class_column']-1)]
        record_defaults.extend([[1]])
        record_defaults.extend([['str'] for dim in range(data_dict['num_metadata'])])
        print("data record_defaults length: " + str(len(record_defaults)))
        reader = tf.decode_csv(records=example, record_defaults=record_defaults,
                               field_delim="\t")
        #print("data size of reader: " + str(reader))
        #tf.decode_csv() from slim.parallel_reader.parallel_read() returns tensors
        #with <unknown> shape.
        #This shape needs to be casted to () to be used with tf.train.batch()
        reshaped_reader = []
        for tensor in reader:
            reshaped_reader.append(tf.reshape(tensor, []))
        #print("data size of reshaped_reader: " + str(reshaped_reader))
        features = reshaped_reader[0:data_dict['class_column']-1]
        label = reshaped_reader[data_dict['class_column']-1:data_dict['class_column']]
        metadata = reshaped_reader[data_dict['class_column']:(data_dict['class_column']
                                                              +data_dict['num_metadata'])]
    #print("data size of features: " + str(features))
    #print("data size of label: " + str(label))
    #print("data size of metadata: " + str(metadata))

    # reformat example features
    label = tf.squeeze(label)
    input_size = data_dict['class_column']-1
    if param_dict['data_mode'] == 'diff':
        input_size = int((input_size)/2)
        features = (tf.slice(features, [input_size], [input_size])
                    - tf.slice(features, [0], [input_size]))
    elif param_dict['data_mode'] == 'exp_only':
        input_size = int((input_size)/2)
        features = tf.slice(features, [input_size], [input_size])
    #features.set_shape([input_size])
    param_dict['data_input_size'] = input_size
    print("data orig input_size: " + str(data_dict['class_column']-1))
    print("data final input_size: " + str(input_size))

    # create batch
    if shuffled:
        feat_b, label_b, meta_b = tf.train.shuffle_batch([features, label, metadata],
                                                         batch_size=batch_size,
                                                         num_threads=nthreads,
                                                         capacity=32*batch_size,
                                                         min_after_dequeue=16*batch_size,
                                                         seed=19850411,
                                                         allow_smaller_final_batch=True)
    else:
        feat_b, label_b, meta_b = tf.train.batch([features, label, metadata],
                                                 batch_size=batch_size,
                                                 num_threads=1,
                                                 capacity=batch_size,
                                                 allow_smaller_final_batch=True)

    return feat_b, label_b, meta_b, input_size, data_dict['num_metadata'], data_dict['num_examples']


def load_data_test():
    """test for load_data module"""
    parser = ArgumentParser()
    parser = params.add_trainer_args(parser)
    param_dict = vars(parser.parse_args())
    feat_b, label_b, meta_b, input_size, nummeta, numexamps = get_batch(param_dict, param_dict["training"])
    print("data final input_size: " + str(input_size))
    print("data nummeta: " + str(nummeta))
    with tf.Session() as sess:
        # initialize the variables
        sess.run(tf.initialize_all_variables())
        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("data feat_batch, label_batch, meta_batch: ")
        for step in range(param_dict["train_max_steps"]):
            print("data train_step: " + str(step))
            print(step)
            print(sess.run([feat_b, label_b, meta_b]))
        # We request our child threads to stop ...
        coord.request_stop()
        # ... and we wait for them to do so before releasing the main thread
        coord.join(threads)

if __name__ == '__main__':
    load_data_test()
