# Train Model for Top 50 dataset

```
CODEDIR='/home/tfuser/code/models'

for OUT_LABEL_COUNT in 50 51; do

    COLL_NAME="top$OUT_LABEL_COUNT"
    DATASET_NAME="train80.ctl_cl10.$COLL_NAME"
    DATA_DIR="/home/tfuser/data/serialized_examples/$COLL_NAME/$DATASET_NAME/"

    DATA_BATCH_SIZE="128" # change for test
    DATA_MODE="all"
    DATA_SERIALIZED="--data_serialized"
    DATA_BATCH_NORM="--data_batch_norm" # change for test

    CONV_DEPTH="15"
    CONV_ACTV_STR="relu"
    CONV_BATCH_NORM="--conv_batch_norm"
    CONV_GENE_PAIR=""

    FCS_DIMENSION_STR="975,975,975,975,975"
    FCS_ACTV_STR="relu"
    FCS_BATCH_NORM="--fcs_batch_norm"
    FCS_RES_BLOCK_SIZE="0"

    OUT_ACTV_STR="relu"

    REG_DO_KEEP_PROB="0.7"
    REG_L1_SCALE="0"
    REG_L2_SCALE="0"
    REG_KL_SCALE="0"
    REG_KL_SPARSITY="0.2"

    TRAIN_LEARNING_RATE="0.001"
    TRAIN_MAX_STEPS="250000" # change for test
    TRAIN_OPTIMIZER_STR="Adam"
    TRAIN_SAVE_CKPT_SECS="3600"
    TRAIN_SAVE_SUMM_SECS="180"
    TRAIN_DIR="/home/tfuser/models/$DATASET_NAME-dm_$DATA_MODE-cd_$CONV_DEPTH-dim_$FCS_DIMENSION_STR-do_$REG_DO_KEEP_PROB"
    TRAIN_CHKPT_DIR="$TRAIN_DIR/chkpts/"

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
    eval time $TRAINCMD

    ## final evaluations
    for KEY in \
    dev05.ctl_cl10.$COLL_NAME:preds \
    train80.ctl_cl10.$COLL_NAME:stats \
    test15.ctl_cl10.$COLL_NAME:stats \
    test15.ctl_cl10.$COLL_NAME-um:stats \
    test15.$COLL_NAME.ctl_cl10:stats \
    test15.$COLL_NAME.$COLL_NAME:stats \
    test15.$COLL_NAME.$COLL_NAME-um:stats \
    ASC.ctl.$COLL_NAME:preds \
    ; do

        EVAL_DATASET_NAME=`echo $KEY | cut -f1 -d:`
        EVAL_RUN_MODE=`echo $KEY | cut -f2 -d:`

        EVAL_DATA_DIR="/home/tfuser/data/serialized_examples/$COLL_NAME/$EVAL_DATASET_NAME/"
        LOG_DIR="$TRAIN_DIR/final_eval_$EVAL_DATASET_NAME"
        mkdir -p $LOG_DIR

        EVALCMD="python3 $CODEDIR/eval.py \
            --data_dir $EVAL_DATA_DIR \
            --data_batch_size $DATA_BATCH_SIZE \
            --data_mode $DATA_MODE \
            $DATA_SERIALIZED \
            $DATA_BATCH_NORM \
            --train_chkpt_dir $TRAIN_CHKPT_DIR \
            --log_dir $LOG_DIR
            --eval_run_mode $EVAL_RUN_MODE
            &>> $LOG_DIR/final_eval.log"

        echo -e "\t$EVALCMD"
        echo $EVALCMD > $LOG_DIR/final_eval.log
        eval time $EVALCMD
    done

    grep "eval final_evals:" $TRAIN_DIR/final_eval_*/* | sed 's#:#\t#g' \
        | sed 's#, #\t#g' |sed 's#\[##g' | sed 's#\]# #g' | sed 's#/#\t#g' \
        > $TRAIN_DIR/all_evals_final.txt

done # COLL_NAME
```

## Run concurrent evaluation
```
DEV_DATASET_NAME="dev05.ctl_cl10.top50"
DEV_DATA_DIR="/home/tfuser/data/serialized_examples/top50/$DATASET_NAME/"
LOG_DIR="$TRAIN_CHKPT_DIR/concur_eval_$DEV_DATASET_NAME"
EVAL_INTERVAL_SECS=30
EVAL_RUN_MODE='stats'

CONCURDEVCMD="python3 $CODEDIR/eval.py \
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


## extract predictions
```
BASEDIR='/home/tfuser/models/'
for SET in dev05 ASC; do
    for FILEPREF in `ls $BASEDIR/*/final_eval_$SET*/final_eval*log | sed "s#.log##g"`; do
        echo $FILEPREF
        grep "outmatrix\[\[" $FILEPREF.log | sed "s#\[#\n#g" | sed "s# #\t#g" | sed "s#\]#\t#g" \
            | sed "s#outmatrix##g" | sed '/^$/d' > $FILEPREF.preds
    done;
done
```


# Train Model for Top 400 dataset

```
CODEDIR='/home/tfuser/code/models'

for OUT_LABEL_COUNT in 400; do

    COLL_NAME="top$OUT_LABEL_COUNT"
    DATASET_NAME="train80.ctl_cl10.$COLL_NAME"
    DATA_DIR="/home/tfuser/data/serialized_examples/$COLL_NAME/$DATASET_NAME/"

    DATA_BATCH_SIZE="128" # change for test
    DATA_MODE="all"
    DATA_SERIALIZED="--data_serialized"
    DATA_BATCH_NORM="--data_batch_norm" # change for test

    CONV_DEPTH="15"
    CONV_ACTV_STR="relu"
    CONV_BATCH_NORM="--conv_batch_norm"
    CONV_GENE_PAIR=""

    FCS_DIMENSION_STR="978,978,978,978,978"
    FCS_ACTV_STR="relu"
    FCS_BATCH_NORM="--fcs_batch_norm"
    FCS_RES_BLOCK_SIZE="0"

    OUT_ACTV_STR="relu"

    REG_DO_KEEP_PROB="0.7"
    REG_L1_SCALE="0"
    REG_L2_SCALE="0"
    REG_KL_SCALE="0"
    REG_KL_SPARSITY="0.2"

    TRAIN_LEARNING_RATE="0.001"
    TRAIN_MAX_STEPS="250000" # change for test
    TRAIN_OPTIMIZER_STR="Adam"
    TRAIN_SAVE_CKPT_SECS="3600"
    TRAIN_SAVE_SUMM_SECS="150"
    TRAIN_DIR="/home/tfuser/models/$DATASET_NAME-dm_$DATA_MODE-cd_$CONV_DEPTH-dim_$FCS_DIMENSION_STR-do_$REG_DO_KEEP_PROB"
    TRAIN_CHKPT_DIR="$TRAIN_DIR/chkpts/"

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
    eval time $TRAINCMD

    ## final evaluations
    for KEY in \
    dev05.ctl_cl10.$COLL_NAME:preds \
    train80.ctl_cl10.$COLL_NAME:stats \
    test15.ctl_cl10.$COLL_NAME:stats \
    test15.ctl_cl10.$COLL_NAME-um:stats \
    test15.$COLL_NAME.ctl_cl10:stats \
    test15.$COLL_NAME.$COLL_NAME:stats \
    test15.$COLL_NAME.$COLL_NAME-um:stats \
    ASC.ctl.$COLL_NAME:preds \
    ; do

        EVAL_DATASET_NAME=`echo $KEY | cut -f1 -d:`
        EVAL_RUN_MODE=`echo $KEY | cut -f2 -d:`

        EVAL_DATA_DIR="/home/tfuser/data/serialized_examples/$COLL_NAME/$EVAL_DATASET_NAME/"
        LOG_DIR="$TRAIN_DIR/final_eval_$EVAL_DATASET_NAME"
        mkdir -p $LOG_DIR

        EVALCMD="python3 $CODEDIR/eval.py \
            --data_dir $EVAL_DATA_DIR \
            --data_batch_size $DATA_BATCH_SIZE \
            --data_mode $DATA_MODE \
            $DATA_SERIALIZED \
            $DATA_BATCH_NORM \
            --train_chkpt_dir $TRAIN_CHKPT_DIR \
            --log_dir $LOG_DIR
            --eval_run_mode $EVAL_RUN_MODE
            &>> $LOG_DIR/final_eval.log"

        echo -e "\t$EVALCMD"
        echo $EVALCMD > $LOG_DIR/final_eval.log
        eval time $EVALCMD
    done

    grep "eval final_evals:" $TRAIN_DIR/final_eval_*/* | sed 's#:#\t#g' \
        | sed 's#, #\t#g' |sed 's#\[##g' | sed 's#\]# #g' | sed 's#/#\t#g' \
        > $TRAIN_DIR/all_evals_final.txt

done # COLL_NAME
```

# evaluate MCF10A
```
CODEDIR='/home/tfuser/code/models'
for TRAIN_DIR in \
    /home/tfuser/models/train80.ctl_cl10.top400-dm_all-cd_15-dim_978,978,978,978,978-do_0.7 \
    /home/tfuser/models/train80.ctl_cl10.top51-dm_all-cd_15-dim_975,975,975,975,975-do_0.7 \
    /home/tfuser/models/train80.ctl_cl10.top50-dm_all-cd_15-dim_975,975,975,975,975-do_0.7 \
    ; do

    TRAIN_CHKPT_DIR="$TRAIN_DIR/chkpts/"
    DATA_BATCH_SIZE="144"
    DATA_MODE="all"
    DATA_SERIALIZED="--data_serialized"
    DATA_BATCH_NORM="--data_batch_norm"

    for KEY in \
    mcf10a_progression:probs \
    ; do

        EVAL_DATASET_NAME=`echo $KEY | cut -f1 -d:`
        EVAL_RUN_MODE=`echo $KEY | cut -f2 -d:`

        EVAL_DATA_DIR="/home/tfuser/data/serialized_examples/$EVAL_DATASET_NAME/"
        LOG_DIR="$TRAIN_DIR/final_eval_$EVAL_DATASET_NAME"
        mkdir -p $LOG_DIR

        EVALCMD="python3 $CODEDIR/eval.py \
            --data_dir $EVAL_DATA_DIR \
            --data_batch_size $DATA_BATCH_SIZE \
            --data_mode $DATA_MODE \
            $DATA_SERIALIZED \
            $DATA_BATCH_NORM \
            --train_chkpt_dir $TRAIN_CHKPT_DIR \
            --log_dir $LOG_DIR
            --eval_run_mode $EVAL_RUN_MODE
            &>> $LOG_DIR/final_eval.log"

        echo -e "\t$EVALCMD"
        echo $EVALCMD > $LOG_DIR/final_eval.log
        eval time $EVALCMD

        grep "outmatrix\[\[" $LOG_DIR/final_eval.log | sed "s#\[#\n#g" | sed "s# #\t#g" | sed "s#\]#\t#g" \
            | sed "s#outmatrix##g" | sed '/^$/d' > $LOG_DIR/final_eval.preds
        grep "probabilities\[\[" $LOG_DIR/final_eval.log | sed "s#\[#\n#g" | sed "s# #\t#g" | sed "s#\]#\t#g" \
            | sed "s#outmatrix##g" | sed '/^$/d' > $LOG_DIR/final_eval.probs
    done
done
```

# tensorboard
```
source activate tensorflow_p36
tensorboard --logdir=/home/tfuser/models/ --port 6006
```