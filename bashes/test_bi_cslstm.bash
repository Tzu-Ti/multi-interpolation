#!/bin/bash

python test_kth_trape_bi_jstlstm.py --model_name inter_trape_bi_cslstm_t1s3 --log_dir ../tensorflow/logs/kth_bi_lstm_${1} --batch_size 1 --num_hidden 32,32,32,32 --save_dir checkpoints/kth_bi_lstm_${1} --gen_frm_dir results/test/kth_bi_lstm_${1} --pretrained_model checkpoints/kth_bi_lstm_${1}/model.ckpt-200000 --gen_num 5 --patch_size 1