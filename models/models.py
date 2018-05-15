"""model module"""
from argparse import ArgumentParser
import math
import params
import load_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Model:
    """store params, inputs, outputs, losses, and summaries"""
    def __init__(self, param_dict=params.default_param_dict, is_training=True):
        """set up params, inputs, and model"""
        self.param_dict = param_dict
        self.is_training = is_training

        # quick sanity checks
        coherence_checks(self.param_dict)

        # get input data
        (feat_batch, label_batch, meta_batch,
         input_size, num_metadata) = load_data.get_batch(self.param_dict, self.is_training)

        self.feat_batch = feat_batch
        self.true_labels = label_batch
        self.meta_batch = meta_batch
        self.param_dict["input_size"] = input_size
        self.param_dict["num_metadata"] = num_metadata

        # input layer
        print("### start model construction")
        print("model in_feats_orig: " + str(feat_batch))
        # Assuming feat_batch inputs are in [batch_size, input_size]
        in_feats_mod = feat_batch
        if self.param_dict['data_batch_norm']:
            in_feats_mod = batch_normalize(feat_batch)
        print("model in_feats_mod: " + str(in_feats_mod))
        self.in_feats_mod = in_feats_mod

        # convolutional layer
        conv_outputs = self.build_conv_layer(in_feats_mod, self.param_dict)
        print("model conv_outputs: " + str(conv_outputs))

        # fully connected hidden layers
        fc_outputs, layers = self.build_fc_layers(conv_outputs, self.param_dict)
        print("model fc_outputs: " + str(fc_outputs))

        # output layer
        out_activation_fn = get_act_fn(param_dict['out_actv_str'])
        logits = slim.fully_connected(inputs=fc_outputs, num_outputs=param_dict['out_label_count'],
                                      activation_fn=out_activation_fn)
        print("model logits: " + str(logits))
        self.logits = logits
        layers.append(logits)
        self.layers = layers

        self.pred_labels = tf.cast(tf.argmax(self.logits, 1), tf.int32)
        self.correct_bool = tf.equal(self.true_labels, self.pred_labels)
        self.prob_vec = tf.nn.softmax(self.logits, dim=-1)
        self.entropy = tf.reduce_sum(-1.0 * self.prob_vec * tf.log(self.prob_vec + 1e-10) /
                                     math.log(param_dict['out_label_count']), axis=1)
        self.true_labels_one_hot = tf.one_hot(self.true_labels, depth=self.param_dict['out_label_count'],
                                         on_value=1, off_value=0)
        pred_labels_one_hot = tf.one_hot(self.pred_labels, param_dict['out_label_count'],
                                         on_value=1, off_value=0)
        self.true_prob_score = tf.reduce_max(tf.multiply(self.prob_vec,
                                                         tf.cast(self.true_labels_one_hot, tf.float32)),
                                             axis=1)
        self.pred_prob_score = tf.reduce_max(tf.multiply(self.prob_vec,
                                                         tf.cast(pred_labels_one_hot, tf.float32)),
                                             axis=1)

    def build_conv_layer(self, in_feats_mod, param_dict):
        """ needs batch_size, conv_depth, conv_actv_str, conv_gene_pair, conv_batch_norm,
         and input_size, reg_do_keep_prob, self.is_training"""
        if param_dict['conv_depth'] > 0:
            #Assuming inputs are still in [batch_size, 2*gene_count]. Will first
            #change it to be [batch_size, gene_count, 2], which slim expects
            conv_inputs = tf.reshape(in_feats_mod,
                                     [param_dict['batch_size'], 2,
                                      int(int(in_feats_mod.shape[1])/2)])
            conv_inputs = tf.transpose(conv_inputs, [0, 2, 1])
            print("conv reshape: " + str(conv_inputs))
            conv_actv_fn = get_act_fn(param_dict['conv_actv_str'])
            if param_dict['conv_gene_pair']:
                conv_outputs = gene_pair_convolution(conv_inputs,
                                                     param_dict['batch_size'],
                                                     [int(param_dict['input_size']/2), 2,
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
                                            is_training=self.is_training)
        else:
            # Flattens the input while maintaining the batch_size
            conv_outputs = tf.contrib.layers.flatten(in_feats_mod)
        return conv_outputs

    def build_fc_layers(self, conv_outputs, param_dict):
        """ needs fcs_dimension_str, fcs_actv_str, fcs_res_block_size, fcs_batch_norm,
         and reg_do_keep_prob, self.is_training """
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
                print("adding layer index " + str(block_idx) +
                      " to fc_output index " + str(num_layer))
                block_idx = num_layer
            if param_dict['fcs_batch_norm']:
                fc_outputs = batch_normalize(fc_outputs)
            fc_outputs = slim.dropout(fc_outputs,
                                      keep_prob=param_dict['reg_do_keep_prob'],
                                      is_training=self.is_training,
                                      seed=19850411)
            print("model fc_layer: " + str(num_layer) + "- " + str(dim))
            print("model fc_outputs: " + str(fc_outputs))
            layers.append(fc_outputs)
            num_layer += 1
        return fc_outputs, layers

    def model_summarize(self):
        """needs true_labels, pred_labels, prob_vec, correct_bool, entropy"""
        corr_prob_score = tf.boolean_mask(self.true_prob_score, self.correct_bool)
        corr_entropy = tf.boolean_mask(self.entropy, self.correct_bool)
        incorrect_true_scores = tf.boolean_mask(self.true_prob_score,
                                                tf.logical_not(self.correct_bool))
        incorrect_pred_scores = tf.boolean_mask(self.pred_prob_score,
                                                tf.logical_not(self.correct_bool))
        incorrect_entropy = tf.boolean_mask(self.entropy, tf.logical_not(self.correct_bool))
        # TODO: figure out perc_zero and non-zero-distrib
        # TODO: maybe figure out how to use ranks tf.top_k

        # add summary histograms to summaries
        tf.summary.histogram('data/true_labels', self.true_labels) #B
        tf.summary.histogram('data/in_feats_orig', self.feat_batch) #BxI
        tf.summary.histogram('data/in_feats_mod', self.in_feats_mod) #BxI
        tf.summary.histogram('model/logits', self.logits) #BxO
        tf.summary.histogram('model/pred_labels', self.pred_labels) #B
        tf.summary.histogram('scores/prob_vec', self.prob_vec) #BxO
        tf.summary.histogram('scores/true_prob_score', self.true_prob_score) #B
        tf.summary.histogram('scores/pred_prob_score', self.pred_prob_score) #B
        tf.summary.histogram('scores/corr_prob_score', corr_prob_score) #B
        tf.summary.histogram('scores/incorrect_true_scores', incorrect_true_scores) #B
        tf.summary.histogram('scores/incorrect_pred_scores', incorrect_pred_scores) #B
        tf.summary.histogram('entropy/entropy', self.entropy) #B
        tf.summary.histogram('entropy/corr_entropy', corr_entropy) #B
        tf.summary.histogram('entropy/incorrect_entropy', incorrect_entropy) #B


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
    model = Model(param_dict, param_dict["training"])
    with tf.Session() as sess:
        # initialize the variables
        sess.run(tf.initialize_all_variables())
        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("model initial logits: ")
        print(sess.run([model.prob_vec, model.true_labels, model.pred_labels,
                        model.entropy, model.meta_batch]))
        # print model parameter stats
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        print('total_params: %d\n' % param_stats.total_parameters)
        # We request our child threads to stop ...
        coord.request_stop()
        # ... and we wait for them to do so before releasing the main thread
        coord.join(threads)


if __name__ == '__main__':
    model_test()
