#!/bin/bash

python train_kth_trape_bi_jstlstm.py --model_name inter_trape_bi_cslstm_t1s3 --log_dir ../tensorflow/logs/kth_bi_lstm_t1s3_patch_1 --batch_size 1 --num_hidden 32,32,32,32 --save_dir checkpoints/kth_bi_lstm_t1s3_patch_1 --gen_frm_dir results/kth_bi_lstm_t1s3_patch_1 --patch_size 1