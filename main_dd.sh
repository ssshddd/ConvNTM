CUDA_VISIBLE_DEVICES=1 python train_convntm.py \
-data_tag dailydialogues_ulen150_unum_25 \
-topic_num 20 \
-epochs 100 \
-batch_size 100 \
-ntm_dim 500 \
-target_rec_M 32
