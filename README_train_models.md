# Train Model for Top 50 dataset

```
CODEDIR='/home/tfuser/code/models'
DATASET_NAME="ASC.ctl.top50"

DATA_DIR="/home/tfuser/data/serialized_examples/top50/$DATASET_NAME/"
DATA_BATCH_SIZE="4" #128
DATA_MODE="all"
DATA_SERIALIZED="--data_serialized"
DATA_BATCH_NORM="" #--data_batch_norm"

CONV_DEPTH="15"
CONV_ACTV_STR="relu"
CONV_BATCH_NORM="--conv_batch_norm"
CONV_GENE_PAIR=""

FCS_DIMENSION_STR="978,978,978,978,978"
FCS_ACTV_STR="relu"
FCS_BATCH_NORM="--fcs_batch_norm"
FCS_RES_BLOCK_SIZE="0"

OUT_LABEL_COUNT="50"
OUT_ACTV_STR="sigmoid"

REG_DO_KEEP_PROB="0.7"
REG_L1_SCALE="0"
REG_L2_SCALE="0"
REG_KL_SCALE="0"
REG_KL_SPARSITY="0.2"

TRAIN_DIR="/home/tfuser/models/$DATASET_NAME-dm_$DATA_MODE-cd_$CONV_DEPTH-dim_$FCS_DIMENSION_STR-do_$REG_DO_KEEP_PROB"
TRAIN_CHKPT_DIR="$TRAIN_DIR/chkpts/"
TRAIN_LEARNING_RATE="0.0001"
TRAIN_MAX_STEPS="1" #100000
TRAIN_OPTIMIZER_STR="Adam"
TRAIN_SAVE_CKPT_SECS="600"
TRAIN_SAVE_SUMM_SECS="15"

TRAINCMD="python3 $CODEDIR/train.py \
    --data_dir $DATA_DIR \
    --data_batch_size $DATA_BATCH_SIZE \
    --data_mode $DATA_MODE \
    $DATA_SERIALIZED \
    $DATA_BATCH_NORM \
    --conv_depth $CONV_DEPTH \
    --conv_actv_str $CONV_ACTV_STR \
    $CONV_BATCH_NORM \
    $CONV_GENE_PAIR \
    --fcs_dimension_str $FCS_DIMENSION_STR \
    --fcs_actv_str $FCS_ACTV_STR \
    $FCS_BATCH_NORM \
    --fcs_res_block_size $FCS_RES_BLOCK_SIZE \
    --out_label_count $OUT_LABEL_COUNT \
    --out_actv_str $OUT_ACTV_STR \
    --reg_do_keep_prob $REG_DO_KEEP_PROB \
    --reg_l1_scale $REG_L1_SCALE \
    --reg_l2_scale $REG_L2_SCALE \
    --reg_kl_scale $REG_KL_SCALE \
    --reg_kl_sparsity $REG_KL_SPARSITY \
    --train_chkpt_dir $TRAIN_CHKPT_DIR \
    --train_learning_rate $TRAIN_LEARNING_RATE \
    --train_max_steps $TRAIN_MAX_STEPS \
    --train_optimizer_str $TRAIN_OPTIMIZER_STR \
    --train_save_ckpt_secs $TRAIN_SAVE_CKPT_SECS \
    --train_save_summ_secs $TRAIN_SAVE_SUMM_SECS \
    &>> $TRAIN_DIR/training.log"

echo -e "\t$TRAINCMD"
mkdir -p $TRAIN_DIR
echo $TRAINCMD > $TRAIN_DIR/training.log
eval $TRAINCMD

DEV_DATASET_NAME="dev05.ctl_cl10.top50"
DEV_DATA_DIR="/home/tfuser/data/serialized_examples/top50/$DATASET_NAME/"
LOG_DIR="$TRAIN_CHKPT_DIR/concur_eval_$DEV_DATASET_NAME"
EVAL_INTERVAL_SECS=30
EVAL_RUN_MODE='stats'

CONCURDEVCMD = "python3 $CODEDIR/eval.py \
    --data_dir $DEV_DATA_DIR \
    --data_batch_size $DATA_BATCH_SIZE \
    --train_chkpt_dir $TRAIN_CHKPT_DIR \
    --log_dir $LOG_DIR
    --eval_run_mode $EVAL_RUN_MODE
    --eval_interval_secs $EVAL_INTERVAL_SECS
    &>> $LOG_DIR/concur_eval.log"

echo -e "\t$CONCURDEVCMD"
echo $CONCURDEVCMD > $LOG_DIR/concur_eval.log
eval $CONCURDEVCMD &
CONCURDEV_PID=$!
echo $CONCURDEV_PID

```
