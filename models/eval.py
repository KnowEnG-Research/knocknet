"""train module"""

from argparse import ArgumentParser
import os
import yaml
import params
import models
import tensorflow as tf
import tensorflow.contrib.slim as slim

def evaluate():
    """run evaluation"""

    # get train params from chkpt.yml
    parser = ArgumentParser()
    parser = params.add_evaluater_args(parser)
    tmp_dict = vars(parser.parse_args())
    ckpt_param_file = os.path.join(tmp_dict["train_chkpt_dir"], 'train_params.yml')
    print("eval ckpt_param_file: " + ckpt_param_file)
    if os.path.exists(ckpt_param_file):
        param_dict = yaml.load(open(ckpt_param_file))
    else:
        print("No chkpt param file " + ckpt_param_file)
        exit()

    # overwrite eval parameters
    for key, value in tmp_dict.items():
        param_dict[key] = value
    if not os.path.exists(param_dict["log_dir"]):
        os.makedirs(param_dict["log_dir"])

    # load batch and make model predictions
    shuffled = (param_dict['eval_interval_secs'] > 0)
    print("shuffled: " + str(shuffled))
    [param_dict, true_labels, logits, layers,
     meta_batch] = models.construct_model(param_dict, is_training=False, shuffle=shuffled)

    # create model summaries stats
    [pred_labels, correct_bool, prob_vec, entropy, true_prob_score,
     pred_prob_score] = models.model_summarize(true_labels, logits, param_dict['out_label_count'])

    # create streaming_ops
    # add scalars to tf.summary
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'streaming/accuracy' : slim.metrics.streaming_accuracy(pred_labels, true_labels),
        'streaming/avg_entropy' : slim.metrics.streaming_mean(entropy),
        'streaming/avg_entropy_corr' : slim.metrics.streaming_mean(tf.boolean_mask(entropy,
                                                                                   correct_bool)),
        'streaming/avg_entropy_incorr' : slim.metrics.streaming_mean(tf.boolean_mask(entropy,
                                                                                     tf.logical_not(correct_bool))),
        'streaming/avg_true_prob' : slim.metrics.streaming_mean(true_prob_score),
        'streaming/avg_true_prob_corr' : slim.metrics.streaming_mean(tf.boolean_mask(true_prob_score,
                                                                                     correct_bool)),
        'streaming/avg_true_prob_incorr' : slim.metrics.streaming_mean(tf.boolean_mask(true_prob_score,
                                                                                       tf.logical_not(correct_bool))),
        'streaming/avg_pred_prob' : slim.metrics.streaming_mean(pred_prob_score),
        'streaming/avg_pred_prob_incorr' : slim.metrics.streaming_mean(tf.boolean_mask(pred_prob_score,
                                                                                       tf.logical_not(correct_bool))),
    })
    # write the metrics as summaries
    for metric_name, metric_value in names_to_values.items():
        tf.summary.scalar(metric_name, metric_value)

    # run eval
    if param_dict['eval_interval_secs'] > 0: # get summary stats on shuffle batch
        slim.evaluation.evaluation_loop(master='',
                                        checkpoint_dir=param_dict['train_chkpt_dir'],
                                        logdir=param_dict['log_dir'],
                                        eval_op=list(names_to_updates.values()),
                                        eval_interval_secs=param_dict['eval_interval_secs'],
                                        session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                                                      log_device_placement=False,
                                                                      allow_soft_placement=True,
                                                                      device_count={'GPU' : 0}))
    else: # go through all examples once
        # get number of samples and number of evals
        my_eval_op = list(names_to_updates.values())
        max_evals = int(param_dict['data_num_examples']/param_dict['data_batch_size'])
        param_dict['eval_max_evals'] = max_evals
        print("eval data_num_examples: " + str(param_dict['data_num_examples']))
        print("eval max_evals: " + str(max_evals))
        outmatrix = tf.stack([tf.cast(true_labels, tf.float32),
                              tf.cast(pred_labels, tf.float32),
                              tf.cast(correct_bool, tf.float32),
                              true_prob_score,
                              pred_prob_score,
                              entropy], 1)
        pr_outmatrix = tf.Print(outmatrix, [outmatrix], message="outmatrix",
                                summarize=param_dict['data_batch_size']*6)
        pr_metabatch = tf.Print(pr_outmatrix, [meta_batch], message="meta_batch",
                                summarize=param_dict['data_batch_size']*param_dict['data_num_metadata'])
        pr_probs = tf.Print(pr_metabatch, [prob_vec], message="probabilities",
                            summarize=param_dict['data_batch_size']*param_dict['out_label_count'])
        if param_dict['eval_run_mode'] == 'preds':
            my_eval_op.append(pr_metabatch)
        if param_dict['eval_run_mode'] == 'probs':
            my_eval_op.append(pr_probs)
        final_evals = slim.evaluation.evaluate_once(master='',
                                                    checkpoint_path=tf.train.latest_checkpoint(param_dict['train_chkpt_dir']),
                                                    logdir=param_dict['log_dir'],
                                                    eval_op=my_eval_op,
                                                    final_op=list(names_to_values.values()),
                                                    num_evals=max_evals,
                                                    session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                                                                  log_device_placement=False,
                                                                                  allow_soft_placement=True,
                                                                                  device_count={'GPU' : 0}))
        print("eval metrics: " + str(list(names_to_values.keys())))
        print("eval final_evals: " + str(final_evals))

    # save eval parameters
    param_yml = os.path.join(param_dict["log_dir"], 'eval_params.yml')
    with open(param_yml, 'w') as outfile:
        yaml.dump(param_dict, outfile, default_flow_style=False)

if __name__ == '__main__':
    evaluate()
