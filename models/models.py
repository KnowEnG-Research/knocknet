"""model module"""
from argparse import ArgumentParser
import math
import params
import load_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def construct_model(param_dict=params.default_param_dict, is_training=True, shuffle=True):
    """set up params, inputs, and model"""

    # quick sanity checks
    coherence_checks(param_dict)

    # get input data
    (feat_batch, true_labels, meta_batch,
     input_size, num_metadata, num_examples) = load_data.get_batch(param_dict, shuffle)
    param_dict["data_input_size"] = input_size
    param_dict["data_num_metadata"] = num_metadata
    param_dict["data_num_examples"] = num_examples
    tf.summary.histogram('batch_data/true_labels', true_labels) #B
    tf.summary.histogram('batch_data/orig_feat_vals', feat_batch) #BxI

    # input layer
    print("### start model construction")
    print("model true_labels: " + str(true_labels))
    print("model in_feats_orig: " + str(feat_batch))
    print("model meta_batch: " + str(meta_batch))
    # Assuming feat_batch inputs are in [batch_size, input_size]
    in_feats_mod = feat_batch
    if param_dict['data_batch_norm']:
        in_feats_mod = batch_normalize(feat_batch)
    print("model in_feats_mod: " + str(in_feats_mod))
    tf.summary.histogram('data/mod_feat_vals', in_feats_mod) #BxI

    # convolutional layer
    conv_outputs = build_conv_layer(is_training, in_feats_mod, param_dict)
    print("model conv_outputs: " + str(conv_outputs))

    # fully connected hidden layers
    fc_outputs, layers = build_fc_layers(is_training, conv_outputs, param_dict)
    print("model fc_outputs: " + str(fc_outputs))

    # output layer
    out_activation_fn = get_act_fn(param_dict['out_actv_str'])
    logits = slim.fully_connected(inputs=fc_outputs, num_outputs=param_dict['out_label_count'],
                                  activation_fn=out_activation_fn)
    print("model logits: " + str(logits))
    tf.summary.histogram('batch_model/logits', logits) #BxO
    layers.append(logits)

    return param_dict, true_labels, logits, layers, meta_batch


def build_conv_layer(is_training, in_feats_mod, param_dict):
    """ needs data_batch_size, conv_depth, conv_actv_str, conv_gene_pair, conv_batch_norm,
     and data_input_size, reg_do_keep_prob, is_training"""
    if param_dict['conv_depth'] > 0:
        #Assuming inputs are still in [batch_size, 2*gene_count]. Will first
        #change it to be [batch_size, gene_count, 2], which slim expects
#        in_feats_mod = tf.Print(in_feats_mod, [in_feats_mod], message="in_feats_mod: ", 
#                                summarize=param_dict['data_batch_size']*param_dict['data_input_size'])
        conv_inputs = tf.reshape(in_feats_mod,
                                 [param_dict['data_batch_size'], 2,
                                  int(int(in_feats_mod.shape[1])/2)])
#        conv_inputs = tf.Print(conv_inputs, [conv_inputs], message="conv_inputs1: ", 
#                               summarize=param_dict['data_batch_size']*param_dict['data_input_size'])
        conv_inputs = tf.transpose(conv_inputs, [0, 2, 1])
#        conv_inputs = tf.Print(conv_inputs, [conv_inputs], message="conv_inputs2: ", 
#                               summarize=param_dict['data_batch_size']*param_dict['data_input_size'])
        print("model conv reshape: " + str(conv_inputs))
        conv_actv_fn = get_act_fn(param_dict['conv_actv_str'])
        if param_dict['conv_gene_pair']:
            conv_outputs = gene_pair_convolution(conv_inputs,
                                                 param_dict['data_batch_size'],
                                                 [int(param_dict['data_input_size']/2), 2,
                                                  param_dict['conv_depth']],
                                                 conv_actv_fn)
        else:
            conv_outputs = slim.convolution(inputs=conv_inputs,
                                            num_outputs=param_dict['conv_depth'],
                                            kernel_size=1,
                                            stride=1,
                                            data_format='NWC',
                                            activation_fn=conv_actv_fn)
        conv_outputs = tf.contrib.layers.flatten(conv_outputs)
        if param_dict['conv_batch_norm']:
            conv_outputs = batch_normalize(conv_outputs)
        if param_dict['reg_do_keep_prob'] < 1:
            conv_outputs = slim.dropout(conv_outputs,
                                        keep_prob=param_dict['reg_do_keep_prob'],
                                        is_training=is_training)
    else:
        # Flattens the input while maintaining the batch_size
        conv_outputs = tf.contrib.layers.flatten(in_feats_mod)
    return conv_outputs

def build_fc_layers(is_training, conv_outputs, param_dict):
    """ needs fcs_dimension_str, fcs_actv_str, fcs_res_block_size, fcs_batch_norm,
     and reg_do_keep_prob, is_training """
    dimensions = [int(dim) for dim in param_dict['fcs_dimension_str'].split(',')]
    layers = []
    num_layer = 0
    block_idx = 0
    fc_outputs = conv_outputs
    fc_activation_fn = get_act_fn(param_dict['fcs_actv_str'])
    for dim in dimensions:
        if dim < 1:
            continue
        fc_outputs = slim.fully_connected(inputs=fc_outputs, num_outputs=dim,
                                          activation_fn=fc_activation_fn)
        # for layers that are multiples of the block size
        if (num_layer != 0 and param_dict['fcs_res_block_size'] != 0 and
                num_layer % param_dict['fcs_res_block_size'] == 0):
            # add first layer of the block to current layer
            fc_outputs += layers[block_idx]
            print("model adding layer index " + str(block_idx) +
                  " to fc_output index " + str(num_layer))
            block_idx = num_layer
        if param_dict['fcs_batch_norm']:
            fc_outputs = batch_normalize(fc_outputs)
        fc_outputs = slim.dropout(fc_outputs,
                                  keep_prob=param_dict['reg_do_keep_prob'],
                                  is_training=is_training,
                                  seed=19850411)
        print("model fc_layer: " + str(num_layer) + "- " + str(dim))
        print("model fc_outputs: " + str(fc_outputs))
        layers.append(fc_outputs)
        num_layer += 1
    return fc_outputs, layers

def model_summarize(true_labels, logits, out_label_count):
    """needs true_labels, logits, out_label_count"""
    pred_labels = tf.cast(tf.argmax(logits, 1), tf.int32)
    correct_bool = tf.equal(true_labels, pred_labels)
    prob_vec = tf.nn.softmax(logits, axis=-1)
    entropy = tf.reduce_sum(-1.0 * prob_vec * tf.log(prob_vec + 1e-10) /
                            math.log(out_label_count), axis=1)
    true_labels_one_hot = tf.one_hot(true_labels, depth=out_label_count,
                                     on_value=1, off_value=0)
    pred_labels_one_hot = tf.one_hot(pred_labels, depth=out_label_count,
                                     on_value=1, off_value=0)
    true_prob_score = tf.reduce_max(tf.multiply(prob_vec,
                                                tf.cast(true_labels_one_hot, tf.float32)),
                                    axis=1)
    pred_prob_score = tf.reduce_max(tf.multiply(prob_vec,
                                                tf.cast(pred_labels_one_hot, tf.float32)),
                                    axis=1)
    corr_prob_score = tf.boolean_mask(true_prob_score, correct_bool)
    corr_entropy = tf.boolean_mask(entropy, correct_bool)
    incorrect_true_scores = tf.boolean_mask(true_prob_score,
                                            tf.logical_not(correct_bool))
    incorrect_pred_scores = tf.boolean_mask(pred_prob_score,
                                            tf.logical_not(correct_bool))
    incorrect_entropy = tf.boolean_mask(entropy, tf.logical_not(correct_bool))
    # TODO: figure out perc_zero and non-zero-distrib
    # TODO: maybe figure out how to use ranks tf.top_k
    # add summary histograms to summaries
    tf.summary.histogram('batch_model/pred_labels', pred_labels) #B
    tf.summary.histogram('batch_scores/prob_vec', prob_vec) #BxO
    tf.summary.histogram('batch_scores/true_prob_score', true_prob_score) #B
    tf.summary.histogram('batch_scores/pred_prob_score', pred_prob_score) #B
    tf.summary.histogram('batch_scores/corr_prob_score', corr_prob_score) #B
    tf.summary.histogram('batch_scores/incorrect_true_scores', incorrect_true_scores) #B
    tf.summary.histogram('batch_scores/incorrect_pred_scores', incorrect_pred_scores) #B
    tf.summary.histogram('batch_entropy/entropy', entropy) #B
    tf.summary.histogram('batch_entropy/corr_entropy', corr_entropy) #B
    tf.summary.histogram('batch_entropy/incorrect_entropy', incorrect_entropy) #B
    tf.summary.scalar('batch_model/accuracy', (tf.reduce_sum(tf.cast(correct_bool, tf.float32)) /
                                               tf.cast(tf.shape(correct_bool)[0], tf.float32)))
    tf.summary.scalar('batch_model/num_correct', tf.reduce_sum(tf.cast(correct_bool, tf.float32)))
    tf.summary.scalar('batch_avg_scores/true_prob_score', tf.reduce_mean(true_prob_score))
    tf.summary.scalar('batch_avg_scores/pred_prob_score', tf.reduce_mean(pred_prob_score))
    tf.summary.scalar('batch_avg_scores/corr_prob_score', tf.reduce_mean(corr_prob_score))
    tf.summary.scalar('batch_avg_scores/incorrect_true_scores', tf.reduce_mean(incorrect_true_scores))
    tf.summary.scalar('batch_avg_scores/incorrect_pred_scores', tf.reduce_mean(incorrect_pred_scores))
    tf.summary.scalar('batch_avg_entropy/entropy', tf.reduce_mean(entropy))
    tf.summary.scalar('batch_avg_entropy/corr_entropy', tf.reduce_mean(corr_entropy))
    tf.summary.scalar('batch_avg_entropy/incorrect_entropy', tf.reduce_mean(incorrect_entropy))

    print("metrics pred_labels: " + str(pred_labels))
    print("metrics correct_bool: " + str(correct_bool))
    print("metrics prob_vec: " + str(prob_vec))
    print("metrics entropy: " + str(entropy))
    print("metrics true_prob_score: " + str(true_prob_score))
    print("metrics pred_prob_score: " + str(pred_prob_score))

    return pred_labels, correct_bool, prob_vec, entropy, true_prob_score, pred_prob_score


def coherence_checks(param_dict):
    """sanity checks for parameter combos"""
    if param_dict['conv_depth'] > 0 and param_dict['data_mode'] not in ['all']:
        print(param_dict['data_mode'] + " cannot be used with convolution")
        exit()
    dimensions = [int(dim) for dim in param_dict['fcs_dimension_str'].split(',')]
    if np.var(dimensions) > 0 and param_dict['fcs_res_block_size'] > 0:
        print("Residual blocks only work with constant dimension size: "
              + param_dict['fcs_dimension_str'])
        exit()
    return


def batch_normalize(orig_values):
    """normalize layer across batch"""
    [mean, var] = tf.nn.moments(orig_values, 0)
    norm_values = tf.nn.batch_normalization(orig_values, mean, var,
                                            offset=None,
                                            scale=None,
                                            variance_epsilon=1e-3)
    return norm_values


def get_act_fn(fn_str='sigmoid'):
    """ convert string to function"""
    if fn_str == "sigmoid":
        activation_fn = tf.nn.sigmoid
    elif fn_str == "relu":
        activation_fn = tf.nn.relu
    elif fn_str.lower() == "none":
        activation_fn = None
    else:
        print(fn_str + " activation function not supported!")
        exit()
    return activation_fn


def gene_pair_convolution(inputs, batch_size, kernel_size, activation_fn, name='Conv'):
    """ conv function per gene """
    num_output = kernel_size[2]
    input_list = [tf.squeeze(x) for x in tf.split(inputs, batch_size)]
    output_list = []
    with tf.name_scope(name) as scope:
        for i in range(0, num_output):
            # weights shape = [num_genes, 2] and bias shape = [num_genes]
            weights = slim.model_variable(scope + 'weights/kernel/' + str(i),
                                          shape=kernel_size[0:2])
            bias = slim.model_variable(scope + 'bias/kernel/' + str(i), shape=kernel_size[0])
            ytmp = []
            for xtmp in input_list:
                ztmp = tf.squeeze(tf.reduce_sum(tf.multiply(xtmp, weights), 1)) + bias
                ztmp = activation_fn(ztmp)
                ytmp.append(ztmp)
            # y is batch_size elements of [num_genes]
            # output is [batch_size, num_genes]
            output_list.append(tf.stack(ytmp))
        # output_list is num_output instances of [batch_size, num_genes]
        output = tf.stack(output_list)
        output = tf.transpose(output, perm=[1, 2, 0])
        # output is of shape [batch_size, num_genes, num_output]
        return output

def model_test():
    """test model module"""
    parser = ArgumentParser()
    parser = params.add_trainer_args(parser)
    param_dict = vars(parser.parse_args())
    # construct models
    [param_dict, true_labels, logits, layers,
     meta_batch] = construct_model(param_dict, is_training=True)
    # create model summaries stats
    [pred_labels, correct_bool, prob_vec, entropy, true_prob_score,
     pred_prob_score] = model_summarize(true_labels, logits, param_dict['out_label_count'])
    with tf.Session() as sess:
        # initialize the variables
        sess.run(tf.initialize_all_variables())
        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("model initial logits: ")
        print(sess.run([meta_batch, true_labels, pred_labels, correct_bool, entropy,
                        true_prob_score, pred_prob_score, prob_vec]))
        # print model parameter stats
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        print('model total_params: %d\n' % param_stats.total_parameters)
        # We request our child threads to stop ...
        coord.request_stop()
        # ... and we wait for them to do so before releasing the main thread
        coord.join(threads)


if __name__ == '__main__':
    model_test()
