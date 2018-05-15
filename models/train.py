"""train module"""

from argparse import ArgumentParser
import os
import math
import yaml
import params
from models import Model
import tensorflow as tf
import tensorflow.contrib.slim as slim

KL_SCALE = -1.0
KL_SPARCE = -1.0

def train():
    """run training"""

    # get parameter values
    parser = ArgumentParser()
    parser = params.add_trainer_args(parser)
    param_dict = vars(parser.parse_args())
    if not os.path.exists(param_dict["chkpt_dir"]):
        os.makedirs(param_dict["chkpt_dir"])
    param_yml = os.path.join(param_dict["chkpt_dir"], 'params.yml')
    with open(param_yml, 'w') as outfile:
        yaml.dump(param_dict, outfile, default_flow_style=False)

    # load batch and make model predictions
    model = Model(param_dict, is_training=True)
    param_dict = model.param_dict

    # calculated losses
    classification_loss = tf.losses.softmax_cross_entropy(model.true_labels_one_hot, model.logits)
    weights = tf.trainable_variables()
    # l1
    l1_reg = slim.l1_regularizer(float(param_dict['reg_l1_scale']))
    l1_loss = slim.regularizers.apply_regularization(l1_reg, weights_list=weights)
    # l2
    l2_reg = slim.l2_regularizer(float(param_dict['reg_l2_scale']))
    l2_loss = slim.regularizers.apply_regularization(l2_reg, weights_list=weights)
    # KL
    global KL_SCALE
    global KL_SPARCE
    KL_SCALE = param_dict['reg_kl_scale']
    KL_SPARCE = param_dict['reg_kl_sparsity']
    print("kl_params: " + str([KL_SCALE, KL_SPARCE]))
    kl_loss = slim.regularizers.apply_regularization(kl_regularizer, weights_list=model.layers)
    total_loss = tf.losses.get_total_loss()

    # create summaries stats
    model.model_summarize()
    # add summary scalars
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        'eval/accuracy' : slim.metrics.streaming_accuracy(model.pred_labels, model.true_labels),
        'eval/entropy' : slim.metrics.streaming_mean(model.entropy),
        'eval/avg_true_prob_score' : slim.metrics.streaming_mean(model.true_prob_score),
        'eval/avg_pred_prob_score' : slim.metrics.streaming_mean(model.pred_prob_score),
        'optimization/softmax_ce_loss' : slim.metrics.streaming_mean(classification_loss),
        'optimization/kl_loss' : slim.metrics.streaming_mean(kl_loss),
        'optimization/l1_loss' : slim.metrics.streaming_mean(l1_loss),
        'optimization/l2_loss' : slim.metrics.streaming_mean(l2_loss),
        'optimization/total_loss' : slim.metrics.streaming_mean(total_loss),
    })
    # write the metrics as summaries
    for metric_name, metric_value in metrics_to_values.items():
        tf.summary.scalar(metric_name, metric_value)

    # create optimizer
    if param_dict['train_optimizer_str'] == "Adam":
        optimizer = tf.train.AdamOptimizer(param_dict['train_learning_rate'])
    else:
        optimizer = tf.train.MomentumOptimizer(param_dict['train_learning_rate'])

    # create training op
    train_op = slim.learning.create_train_op(total_loss, optimizer=optimizer,
                                             summarize_gradients=False)

    # save training parameters
    with open(param_yml, 'w') as outfile:
        yaml.dump(model.param_dict, outfile, default_flow_style=False)

    # run training
    error = slim.learning.train(train_op,
                                param_dict['chkpt_dir'],
                                number_of_steps=param_dict['train_max_steps'],
                                save_summaries_secs=param_dict['eval_interval_secs'],
                                save_interval_secs=param_dict['train_save_ckpt_secs'],
                                session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                                              log_device_placement=False,
                                                              allow_soft_placement=True))
    print(error)


def kl_regularizer(layer):
    """Calculate kl divergence for layer average activation
    """
    with tf.name_scope('scope', 'kl_regularizer', [layer]):
        avg_activation = tf.reduce_mean(tf.cast(layer, tf.float64), 0)
        lt_ones_tensor = tf.constant(1-1e-8, shape=avg_activation.shape, dtype=tf.float64)
        gt_zeros_tensor = tf.constant(1e-8, shape=avg_activation.shape, dtype=tf.float64)
        avg_activation = tf.maximum(avg_activation, gt_zeros_tensor)
        avg_activation = tf.minimum(avg_activation, lt_ones_tensor)
        #avg_activation = tf.Print(avg_activation, [avg_activation], message="avg_activation: ")
        #print("kl_params_local: " + str([KL_SCALE, KL_SPARCE]))
        return tf.cast(KL_SCALE * kl_divergence(KL_SPARCE, avg_activation), tf.float32)


def kl_divergence(prob, p_hat):
    """Calculate kl divergence for tensor of probabilities"""
    ret = tf.reduce_sum(prob*(math.log(prob)-tf.log(p_hat)) +
                        (1-prob)*(math.log(1-prob)-tf.log(1-p_hat)))
    return ret

if __name__ == '__main__':
    train()
