"""define parameters"""
from argparse import ArgumentParser
import os
import yaml

DEFAULT_CHKPT_DIR = 'chkpts'
DEFAULT_LOG_DIR = 'logs'
DEFAULT_BATCH_SIZE = 128
def add_global_params(parser):
    """define global parameters"""
    parser.add_argument('--chkpt_dir', default=DEFAULT_CHKPT_DIR, type=str,
                        help='Directory for saved model')
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR, type=str,
                        help='Directory for summary and logs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Number of examples in a minibatch')
    return parser


DEFAULT_DATA_MODE = 'all'
DEFAULT_DATA_DIR = './'
def add_data_params(parser):
    """define data parameters"""
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR, type=str,
                        help='Directory storing input data')
    parser.add_argument('--data_serialized', dest='data_serialized', default=False,
                        action='store_true', help='Default .data files or serialized .tfrecords')
    parser.add_argument('--data_mode', type=str, default=DEFAULT_DATA_MODE,
                        help='Which part of the data to use: "all", "diff", "exp_only"')
    parser.add_argument('--data_batch_norm', dest='data_batch_norm', default=False,
                        action='store_true', help='Batch normalize outputs of in layer')
    return parser


DEFAULT_ACTV_STR = 'sigmoid'
DEFAULT_CONV_DEPTH = 0
def add_conv_params(parser):
    """define convolutional parameters"""
    parser.add_argument('--conv_depth', type=int, default=DEFAULT_CONV_DEPTH,
                        help='Depth of convolutional_layer')
    parser.add_argument('--conv_actv_str', type=str, default=DEFAULT_ACTV_STR,
                        help='Activiation function')
    parser.add_argument('--conv_batch_norm', dest='conv_batch_norm', default=False,
                        action='store_true', help='Batch normalize outputs of conv layer')
    parser.add_argument('--conv_gene_pair', default=False, action="store_true",
                        help='Do gene pair convolution')
    return parser


DEFAULT_DIM_STR = '5,5'
DEFAULT_RES_BLOCK_SIZE = 0
def add_fcs_params(parser):
    """define fully connected layer parameters"""
    parser.add_argument('--fcs_dimension_str', type=str, default=DEFAULT_DIM_STR,
                        help='Hidden layers of the model')
    parser.add_argument('--fcs_actv_str', type=str, default=DEFAULT_ACTV_STR,
                        help='Activiation function')
    parser.add_argument('--fcs_batch_norm', dest='fcs_batch_norm', default=False,
                        action='store_true', help='Batch normalize outputs of fc layer')
    parser.add_argument('--fcs_res_block_size', type=int, default=DEFAULT_RES_BLOCK_SIZE,
                        help='Number of layers per residual block')
    return parser


DEFAULT_NCLASSES = 50
def add_outlayer_params(parser):
    """define output parameters"""
    parser.add_argument('--out_actv_str', type=str, default=DEFAULT_ACTV_STR,
                        help='Activiation function')
    parser.add_argument('--out_label_count', type=int, default=DEFAULT_NCLASSES,
                        help='Number of classes for training')
    return parser


DEFAULT_KL_SPARCITY = 0.2
def add_regularization_params(parser):
    """define regularization parameters"""
    parser.add_argument('--reg_do_keep_prob', type=float, default=1.0,
                        help='Keep probability for training dropout')
    parser.add_argument('--reg_kl_sparsity', type=float, default=DEFAULT_KL_SPARCITY,
                        help='Probability for activation of neuron')
    parser.add_argument('--reg_kl_scale', type=float, default=0,
                        help='Regulariztion scalar for KL activation loss')
    parser.add_argument('--reg_l1_scale', type=float, default=0,
                        help='Regulariztion scalar for L1 weights loss')
    parser.add_argument('--reg_l2_scale', type=float, default=0,
                        help='Regulariztion scalar for L2 weights loss')
    return parser


DEFAULT_MAX_STEPS = 100000
DEFAULT_SECS_PER_SAVE = 600
DEFAULT_OPTIMIZER_STR = "Adam"
DEFAULT_LEARNING_RATE = "0.0001"
def add_train_params(parser):
    """define training parameters"""
    parser.add_argument('--training', dest='training', default=False,
                        action='store_true', help='Mark if training model')
    parser.add_argument('--train_max_steps', type=int, default=DEFAULT_MAX_STEPS,
                        help='Number of steps to run trainer')
    parser.add_argument('--train_save_ckpt_secs', type=int, default=DEFAULT_SECS_PER_SAVE,
                        help='Number of seconds to run trainer before saving ckpt')
    parser.add_argument('--train_save_summ_secs', type=int, default=DEFAULT_SECS_PER_SAVE,
                        help='Number of seconds to run trainer before saving summaries')
    parser.add_argument('--train_optimizer_str', type=str, default=DEFAULT_OPTIMIZER_STR,
                        help='Optimizer function')
    parser.add_argument('--train_learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Initial learning rate')
    return parser


DEFAULT_EVAL_MODE = 'stats'
DEFAULT_EVAL_INTERVAL = 0
def add_eval_params(parser):
    """define evaluation parameters"""
    parser.add_argument('--eval_run_mode', type=str, default=DEFAULT_EVAL_MODE,
                        help='Mode: "stats", "preds", "probs"')
    parser.add_argument('--eval_interval_secs', type=int, default=DEFAULT_EVAL_INTERVAL,
                        help='Time between sampling the evaluation metrics')
    return parser


def add_trainer_args(parser):
    """merge parameters for training"""
    group1 = parser.add_argument_group('global parameters')
    group1 = add_global_params(group1)
    group2 = parser.add_argument_group('data parameters')
    group2 = add_data_params(group2)
    group3 = parser.add_argument_group('training parameters')
    group3 = add_train_params(group3)
    group4 = parser.add_argument_group('regularization parameters')
    group4 = add_regularization_params(group4)
    group5 = parser.add_argument_group('convolutional_layer parameters')
    group5 = add_conv_params(group5)
    group6 = parser.add_argument_group('fully_connected_layer parameters')
    group6 = add_fcs_params(group6)
    group7 = parser.add_argument_group('output_layer parameters')
    group7 = add_outlayer_params(group7)
    group8 = parser.add_argument_group('evaluation and summary parameters')
    group8 = add_eval_params(group8)
    return parser


def add_evaluater_args(parser):
    """merge parameters for evaluation"""
    group1 = parser.add_argument_group('global parameters')
    group1 = add_global_params(group1)
    group2 = parser.add_argument_group('data parameters')
    group2 = add_data_params(group2)
    group3 = parser.add_argument_group('evaluation and summary parameters')
    group3 = add_eval_params(group3)
    return parser

def default_param_dict():
    """create param_dict with defaults"""
    parser = ArgumentParser()
    parser = add_trainer_args(parser)
    param_dict = vars(parser.parse_args())
    return param_dict


def param_dict_test():
    """test param module"""
    parser = ArgumentParser()
    parser = add_trainer_args(parser)
    param_dict = vars(parser.parse_args())
    param_yml = os.path.join(param_dict["log_dir"], 'params.yml')
    with open(param_yml, 'w') as outfile:
        yaml.dump(param_dict, outfile, default_flow_style=False)


if __name__ == '__main__':
    param_dict_test()
