# wget https://raw.github.com/lehmannro/assert.sh/v1.1/assert.sh
LOAD_TESTS=1
MODEL_TESTS=1
TRAIN_TESTS=1
TEST_TESTS=1

. assert.sh

LOGDIR="/mnt/knowdnn_hdd/tfuser/logs/tests/"
mkdir -p $LOGDIR
TESTCT=1

#python3 serialize_features.py /home/tfuser/code/tests/data/test_tsv/ /home/tfuser/code/tests/data/test_tfrecord/ --label_col 7

# 1: check param dict
CMD="python3 params.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
    --log_dir $LOGDIR &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
((LOAD_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))

# 2: check basic plain text shuffle load data
CMD="python3 load_data.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
    --training --data_mode all --data_batch_size 6 --train_max_steps 1 &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
((LOAD_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))

# 3-5: check all load data modes working
for SHUF in ""; do
    for DMODE in all diff exp_only; do
        for STEPS in 1; do
            CMD="python3 load_data.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
            $SHUF --data_mode $DMODE --data_batch_size 6 --train_max_steps $STEPS &> $LOGDIR/$TESTCT.log"
            # echo -e "\n****$TESTCT: $CMD"
            ((LOAD_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))
        done;
    done;
done

# 6-7: check shuffle and non shuffle load data work for overflow
for SHUF in --training ""; do
    for DMODE in exp_only; do
        for STEPS in 4; do
            CMD="python3 load_data.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
            $SHUF --data_mode $DMODE --data_batch_size 6 --train_max_steps $STEPS &> $LOGDIR/$TESTCT.log"
            # echo -e "\n****$TESTCT: $CMD"
            ((LOAD_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))
        done;
    done;
done

# 8: check basic model construction
CMD="python3 models.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
     --data_batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
     &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
((MODEL_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))

# 9: check input layer batch norm
CMD="python3 models.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
     --data_batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
     --data_batch_norm &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
((MODEL_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))

# 10: check normal conv layer options
CMD="python3 models.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
     --data_batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
     --conv_actv_str relu --conv_batch_norm --conv_depth 3 &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
((MODEL_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))

# 11: check normal conv layer options
CMD="python3 models.py --data_dir /home/tfuser/code/tests/data/test_tfrecord/ --data_serialized --out_label_count 10 \
     --data_batch_size 4 --data_mode all  --train_max_steps 1 --reg_do_keep_prob 0.7 \
     --conv_actv_str relu --conv_batch_norm --conv_depth 3 &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
((MODEL_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))


# 11: check conv_gene_pair_layer options
CMD="python3 models.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
     --data_batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
     --conv_batch_norm --conv_gene_pair --conv_depth 3 &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
((MODEL_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))

# 12-17: check fully connected options
for CONVD in 0 3; do
    for DIMS in 0 5 7,5; do
        CMD="python3 models.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
             --data_batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
             --conv_depth $CONVD --fcs_dimension_str $DIMS --fcs_actv_str relu --fcs_batch_norm \
             &> $LOGDIR/$TESTCT.log"
        # echo -e "\n****$TESTCT: $CMD"
        ((MODEL_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))
    done;
done

# 18: check residual blocks
CMD="python3 models.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
    --data_batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
    --conv_depth 3 --fcs_dimension_str 8,8,8,8,8,8 --fcs_res_block_size 2 --fcs_batch_norm \
    &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
((MODEL_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))

# 20: check training
rm -r "$LOGDIR/chkptdir_$TESTCT"
MYTRAINNUM=$TESTCT
CMD="python3 train.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
    --train_chkpt_dir $LOGDIR/chkptdir_$TESTCT --train_save_summ_secs 4 --train_save_ckpt_secs 8\
    --data_batch_size 8 --data_mode all --out_actv_str None --train_max_steps 5000 \
    --training --reg_do_keep_prob 0.7 --train_learning_rate 0.1  \
    --fcs_dimension_str 0 --data_batch_norm \
    &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
((TRAIN_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))

# 21: check testing once through for all run modes
for RM in stats preds probs; do
    rm -r "$LOGDIR/chkptdir_$TESTCT"
    CMD="python3 eval.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --data_mode all \
        --train_chkpt_dir $LOGDIR/chkptdir_$MYTRAINNUM --log_dir $LOGDIR/chkptdir_$MYTRAINNUM/log_$TESTCT \
        --data_batch_size 6 --eval_run_mode $RM \
        &> $LOGDIR/$TESTCT.log"
    # echo -e "\n****$TESTCT: $CMD"
    ((TEST_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))
done

# 21-23: check train loss functions
for LF in reg_l1_scale reg_l2_scale reg_kl_scale; do
    rm -r "$LOGDIR/chkptdir_$TESTCT"
    CMD="python3 train.py --data_dir /home/tfuser/code/tests/data/test_tsv/ --out_label_count 10 \
        --train_chkpt_dir $LOGDIR/chkptdir_$TESTCT --train_save_summ_secs 2 \
        --data_batch_size 8 --data_mode all  --train_max_steps 2000 --training --reg_do_keep_prob 0.7 \
        --conv_depth 3 --fcs_dimension_str 5,5 --$LF 2 --data_batch_norm \
        &> $LOGDIR/$TESTCT.log"
    # echo -e "\n****$TESTCT: $CMD"
    ((TRAIN_TESTS)) && assert_raises "$CMD" && ((TESTCT=TESTCT+1))
done

# end of test suite
assert_end examples