# ConvNTM

This repository is a PyTorch implementation of our AAAI 2023 paper **"[ConvNTM: Conversational Neural Topic Model](https://ojs.aaai.org/index.php/AAAI/article/view/26595)"**

## Dependencies

* Python>=3.7
* torch>=1.10.1
* gensim>=3.8.3

## Preprocessed Data
We have uploaded the preprocessed data for DailyDialog and EmpatheticDialogues in the folder `processed_data/` to facilitate model training.

## Model Training
* data_tag: dailydialogues_ulen150_unum_25/emp_ulen150_unum_8

* topic_num:10/20/30/50/70/100

For DailyDialog:
```
python train_convntm.py \
-data_tag dailydialogues_ulen150_unum_25 \
-topic_num 20 \
-epochs 100 \
-batch_size 100 \
-ntm_dim 500
```
For EmpatheticDialogues:
```
python train_convntm.py \
-data_tag emp_ulen150_unum_8 \
-topic_num 20 \
-epochs 100 \
-batch_size 100 \
-ntm_dim 500 \
```

## Model Inference
To evaluate an existing model, set the "load_pretrain_ntm" and "only_eval" to True, and and place checkpoints for the NTM model and the Encoder model on suitable paths.

For DailyDialog:
```
python train_convntm.py \
-data_tag dailydialogues_ulen150_unum_25 \
-topic_num 20 \
-epochs 100 \
-batch_size 100 \
-ntm_dim 500 \
-only_eval \
-load_pretrain_ntm \
```

## Citation
```bibtex
@inproceedings{sun2023convntm,
  title={ConvNTM: Conversational Neural Topic Model},
  author={Sun, Hongda and Tu, Quan and Li, Jinpeng and Yan, Rui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={11},
  pages={13609--13617},
  year={2023}
}
```

Feel free to contact me sunhongda98@ruc.edu.cn for any question.

Partial credit to previous reprostories:
* https://github.com/yuewang-cuhk/TAKG
