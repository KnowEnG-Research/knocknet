# KnockNet - Modeling changes in transcriptomic state from a massive library of cellular signatures
Charles Blatti[blatti@illinois.edu], Casey Hanson[crhanso2@illinois.edu], Bryce Kille[kille2@illinois.edu], and Saurabh Sinha[sinhas@illinois.edu] 

KnowEnG BD2K Center of Excellence  
University of Illinois Urbana-Champaign  

## Table of Contents
1. [Motivation](#motivation)
2. [Installation](#installation)
3. [Tutorial](#tutorial)
    1. [Creating Gene Expression Datasets](#creating-gene-expression-datasets)
    2. [Example Training Runs](#example-training-runs)
    3. [Example Model Prediction](#example-model-prediction)
4. [KnockNet Resources](#knocknet-resources)
    1. [Parameters](#parameters)
    2. [Input File Formats](#input-file-formats)
    3. [Output File Formats](#output-file-formats)

## Motivation

Learning signatures of gene knockdowns from paired transcriptional states.

![Method Overview](images/DRaWR_method.small.png)

[Return to TOC](#table-of-contents)

## Installation

### Local copy of KnockNet repository

If you wish to use the sample files necessary to complete this tutorial or the datasets from the paper, first clone this repository from github:
```
git clone https://github.com/knoweng-research/knocknet.git
```

### Installing Tensorflow

Examples for creating the environment with tensorflow installed can be found [here](https://github.com/caseyrhanson/tensorflow_setup/blob/master/docker_setup.md).

[Return to TOC](#table-of-contents)

## Tutorial

This section of the README is meant to walk a user through a process of using Knocknet to find the related genes knockdown signature of their paired trancriptional datasets using the L1000 trained model or training a new model on an additional dataset.   

### Creating Gene Expression Datasets

Examples for pulling data from L1000 and formatting it into an appropriate dataset for training or testing can be found [here](README_data_preprocess.md).

### Example Training Runs

Now that we have our data set prepared, we are ready to train the knocknet model. Examples for training the model can be found [here](README_train_models.md).

### Example Model Prediction

Now that we have a trained model, we are read to use it to predict gene knockout signatures in pair tumor progression datasets as shown [here](README_train_models.md#evaluate-mcf10a-and-tcga)

[Return to TOC](#table-of-contents)

## KnockNet Resources

### Parameters
```
usage: train.py [-h] [--train_chkpt_dir TRAIN_CHKPT_DIR] [--log_dir LOG_DIR]
                [--data_dir DATA_DIR] [--data_batch_size DATA_BATCH_SIZE]
                [--data_serialized] [--data_mode DATA_MODE]
                [--data_batch_norm] [--training]
                [--train_max_steps TRAIN_MAX_STEPS]
                [--train_save_ckpt_secs TRAIN_SAVE_CKPT_SECS]
                [--train_save_summ_secs TRAIN_SAVE_SUMM_SECS]
                [--train_optimizer_str TRAIN_OPTIMIZER_STR]
                [--train_learning_rate TRAIN_LEARNING_RATE]
                [--reg_do_keep_prob REG_DO_KEEP_PROB]
                [--reg_kl_sparsity REG_KL_SPARSITY]
                [--reg_kl_scale REG_KL_SCALE] [--reg_l1_scale REG_L1_SCALE]
                [--reg_l2_scale REG_L2_SCALE] [--conv_depth CONV_DEPTH]
                [--conv_actv_str CONV_ACTV_STR] [--conv_batch_norm]
                [--conv_gene_pair] [--fcs_dimension_str FCS_DIMENSION_STR]
                [--fcs_actv_str FCS_ACTV_STR] [--fcs_batch_norm]
                [--fcs_res_block_size FCS_RES_BLOCK_SIZE]
                [--out_actv_str OUT_ACTV_STR]
                [--out_label_count OUT_LABEL_COUNT]
                [--eval_run_mode EVAL_RUN_MODE]
                [--eval_interval_secs EVAL_INTERVAL_SECS]

optional arguments:
  -h, --help            show this help message and exit

global parameters:
  --train_chkpt_dir TRAIN_CHKPT_DIR
                        Directory for saved model
  --log_dir LOG_DIR     Directory for summary and logs

data parameters:
  --data_dir DATA_DIR   Directory storing input data
  --data_batch_size DATA_BATCH_SIZE
                        Number of examples in a minibatch
  --data_serialized     Default .data files or serialized .tfrecords
  --data_mode DATA_MODE
                        Which part of the data to use: "all", "diff",
                        "exp_only"
  --data_batch_norm     Batch normalize outputs of in layer

training parameters:
  --training            Mark if training model
  --train_max_steps TRAIN_MAX_STEPS
                        Number of steps to run trainer
  --train_save_ckpt_secs TRAIN_SAVE_CKPT_SECS
                        Number of seconds to run trainer before saving ckpt
  --train_save_summ_secs TRAIN_SAVE_SUMM_SECS
                        Number of seconds to run trainer before saving
                        summaries
  --train_optimizer_str TRAIN_OPTIMIZER_STR
                        Optimizer function
  --train_learning_rate TRAIN_LEARNING_RATE
                        Initial learning rate

regularization parameters:
  --reg_do_keep_prob REG_DO_KEEP_PROB
                        Keep probability for training dropout
  --reg_kl_sparsity REG_KL_SPARSITY
                        Probability for activation of neuron
  --reg_kl_scale REG_KL_SCALE
                        Regulariztion scalar for KL activation loss
  --reg_l1_scale REG_L1_SCALE
                        Regulariztion scalar for L1 weights loss
  --reg_l2_scale REG_L2_SCALE
                        Regulariztion scalar for L2 weights loss

convolutional_layer parameters:
  --conv_depth CONV_DEPTH
                        Depth of convolutional_layer
  --conv_actv_str CONV_ACTV_STR
                        Activiation function
  --conv_batch_norm     Batch normalize outputs of conv layer
  --conv_gene_pair      Do gene pair convolution

fully_connected_layer parameters:
  --fcs_dimension_str FCS_DIMENSION_STR
                        Hidden layers of the model
  --fcs_actv_str FCS_ACTV_STR
                        Activiation function
  --fcs_batch_norm      Batch normalize outputs of fc layer
  --fcs_res_block_size FCS_RES_BLOCK_SIZE
                        Number of layers per residual block

output_layer parameters:
  --out_actv_str OUT_ACTV_STR
                        Activiation function
  --out_label_count OUT_LABEL_COUNT
                        Number of classes for training

evaluation and summary parameters:
  --eval_run_mode EVAL_RUN_MODE
                        Mode: "stats", "preds", "probs"
  --eval_interval_secs EVAL_INTERVAL_SECS
                        Time between sampling the evaluation metrics
```

#### Relating to input data files
| Parameter      | Type    | Default  | Description                                                                                                                                                                                                                                                                             |
|----------------|---------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data_dir     | string  | './' | Directory storing input data.                                                                                                                                                                                                                     |
| data_batch_size        | int  | 128 | Number of examples in a minibatch.                                                                                                                                                                                                                                                 |
| data_serialized    | bool  | False | Default .data files or serialized .tfrecords.                                                    |
| data_mode    | str  | 'all' | Which part of the data to use: "all", "diff", "exp_only".                                                                                                                                                                                                                                           |
| data_batch_norm    | bool  | False | Batch normalize outputs of in layer.                                                                                                                                                                                                                                           |

### Input File Formats
#### Expression .data format

An example input file in the correct data format can be found [here](tests/data/test_tsv/file1.data)

#### Metadata about data set

An example metadata file in the correct format can be found [here](tests/data/test_tsv/info.yml)


#### Serialize Format

Serialize examples are found [here](tests/data/test_tfrecord)

### Output File Formats

#### .preds Prediction File

| Columns         | Type  | Description                                                         |
|-----------------|-------|---------------------------------------------------------------------|
| true_labels     | int   | True class id of example                                            |
| pred_labels     | int   | Predicted class id of example                                       |
| correct_bool    | bool  | True if true class id matches predicted class id                    |
| true_prob_score | float | Predicted probability of the true class                             |
| pred_prob_score | float | Predicted probability of the predicted class                        |
| entropy         | float | Entropy of the example's probability distribution over class states |

#### .probs Probability File

Each row of this file contains the probability distribution over class states for an example

[Return to TOC](#table-of-contents)
