# wget https://raw.github.com/lehmannro/assert.sh/v1.1/assert.sh
. assert.sh

LOGDIR="/mnt/knowdnn_hdd/tfuser/logs/tests/"
mkdir -p $LOGDIR
TESTCT=0

# 0: check param dict
CMD="python3 params.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
    --log_dir $LOGDIR &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
assert_raises "$CMD"
((TESTCT=TESTCT+1))

# 1: check basic plain text shuffle load data
CMD="python3 load_data.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
    --training --data_mode all --batch_size 6 --train_max_steps 1 &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
assert_raises "$CMD"
((TESTCT=TESTCT+1))

# 2-4: check all load data modes working
for SHUF in ""; do
    for DMODE in all diff exp_only; do
        for STEPS in 1; do
            CMD="python3 load_data.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
            $SHUF --data_mode $DMODE --batch_size 6 --train_max_steps $STEPS &> $LOGDIR/$TESTCT.log"
            # echo -e "\n****$TESTCT: $CMD"
            assert_raises "$CMD"
            ((TESTCT=TESTCT+1))
        done;
    done;
done

# 5-6: check shuffle and non shuffle load data work for overflow
for SHUF in --training ""; do
    for DMODE in exp_only; do
        for STEPS in 4; do
            CMD="python3 load_data.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
            $SHUF --data_mode $DMODE --batch_size 6 --train_max_steps $STEPS &> $LOGDIR/$TESTCT.log"
            # echo -e "\n****$TESTCT: $CMD"
            assert_raises "$CMD"
            ((TESTCT=TESTCT+1))
        done;
    done;
done

# 7: check basic model construction
CMD="python3 models.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
     --batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
     &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
assert_raises "$CMD"
((TESTCT=TESTCT+1))

# 8: check input layer batch norm
CMD="python3 models.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
     --batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
     --in_batch_norm &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
assert_raises "$CMD"
((TESTCT=TESTCT+1))

# 9: check normal conv layer options
CMD="python3 models.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
     --batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
     --conv_actv_str relu --conv_batch_norm --conv_depth 3 &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
assert_raises "$CMD"
((TESTCT=TESTCT+1))

# 10: check conv_gene_pair_layer options
CMD="python3 models.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
     --batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
     --conv_batch_norm --conv_gene_pair --conv_depth 3 &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
assert_raises "$CMD"
((TESTCT=TESTCT+1))

# 11-16: check fully connected options
for CONVD in 0 3; do
    for DIMS in 0 5 7,5; do
        CMD="python3 models.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
             --batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
             --conv_depth $CONVD --fcs_dimension_str $DIMS --fcs_actv_str relu --fcs_batch_norm \
             &> $LOGDIR/$TESTCT.log"
        # echo -e "\n****$TESTCT: $CMD"
        assert_raises "$CMD"
        ((TESTCT=TESTCT+1))
    done;
done

# 17: check residual blocks
CMD="python3 models.py --data_dir /home/tfuser/data/test_tsv/ --out_label_count 10 \
    --batch_size 4 --data_mode all  --train_max_steps 1 --training --reg_do_keep_prob 0.7 \
    --conv_depth 3 --fcs_dimension_str 8,7,6,5,4,3 --fcs_res_block_size 2 --fcs_batch_norm \
    &> $LOGDIR/$TESTCT.log"
# echo -e "\n****$TESTCT: $CMD"
assert_raises "$CMD"
((TESTCT=TESTCT+1))



# end of test suite
assert_end examples