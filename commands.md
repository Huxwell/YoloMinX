# Commands

## Download COCO val2017

```bash
cd ~/Documents/YoloMinX/datasets/COCO && wget --progress=bar:force http://images.cocodataset.org/zips/val2017.zip && wget --progress=bar:force http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip val2017.zip && unzip annotations_trainval2017.zip && rm val2017.zip annotations_trainval2017.zip
```

## Smoke test: YOLOX-Nano on val2017 (train=val, overfit to verify mAP pipeline)

Symlinks already in place:
- `datasets/train2017 -> val2017`
- `datasets/annotations/instances_train2017.json` (real file from zip)

```bash
cd ~/Documents/YoloMinX && \
PYTHONPATH=~/Documents/YoloMinX \
yolox_venv/bin/python tools/train.py \
  -f exps/default/yolox_nano.py \
  -d 1 -b 2 --fp16 -o \
  data_dir ~/Documents/YoloMinX/datasets \
  train_ann instances_val2017.json \
  input_size "(128,128)" \
  test_size "(128,128)" \
  multiscale_range 0 \
  max_epoch 50 \
  eval_interval 1 \
  no_aug_epochs 5 \
  warmup_epochs 0
```

## Smoke test v2: 256x256, no mosaic/mixup, overfit check

```bash
cd ~/Documents/YoloMinX && \
PYTHONPATH=~/Documents/YoloMinX \
yolox_venv/bin/python tools/train.py \
  -f exps/default/yolox_nano.py \
  -d 1 -b 1 --fp16 -o \
  data_dir ~/Documents/YoloMinX/datasets \
  train_ann instances_val2017.json \
  input_size "(256,256)" \
  test_size "(256,256)" \
  multiscale_range 0 \
  mosaic_prob 0 \
  enable_mixup False \
  max_epoch 50 \
  eval_interval 1 \
  no_aug_epochs 10 \
  warmup_epochs 0
```

## Smoke test v3: 256x256, all augmentation off, overfit check

Note: `enable_mixup False` via CLI is broken (bool("False")==True in Python). Use `enable_mixup 0` instead.

```bash
cd ~/Documents/YoloMinX && \
PYTHONPATH=~/Documents/YoloMinX \
yolox_venv/bin/python tools/train.py \
  -f exps/default/yolox_nano.py \
  -d 1 -b 1 --fp16 -o \
  data_dir ~/Documents/YoloMinX/datasets \
  train_ann instances_val2017.json \
  input_size "(256,256)" \
  test_size "(256,256)" \
  multiscale_range 0 \
  mosaic_prob 0 \
  enable_mixup 0 \
  mixup_prob 0 \
  hsv_prob 0 \
  flip_prob 0 \
  degrees 0 \
  translate 0 \
  shear 0 \
  max_epoch 50 \
  eval_interval 1 \
  no_aug_epochs 50 \
  warmup_epochs 1
```

## Person-only overfit: YOLOX-Nano on val2017_person (2693 images, 1 class)

```bash
cd ~/Documents/YoloMinX && \
PYTHONPATH=~/Documents/YoloMinX \
yolox_venv/bin/python tools/train.py \
  -f exps/default/yolox_nano_person.py \
  -d 1 -b 1 --fp16 \
  data_dir ~/Documents/YoloMinX/datasets \
  input_size "(256,256)" \
  test_size "(256,256)" \
  multiscale_range 0 \
  mosaic_prob 0 \
  max_epoch 50 \
  eval_interval 1 \
  no_aug_epochs 5 \
  warmup_epochs 1
```

## Overfit: YOLOX-Nano on val2017_person_200 (200 images, 1 class)

```bash
cd ~/Documents/YoloMinX && \
PYTHONPATH=~/Documents/YoloMinX \
yolox_venv/bin/python tools/train.py \
  -f exps/default/yolox_nano_person.py \
  -d 1 -b 1 --fp16 \
  data_dir ~/Documents/YoloMinX/datasets \
  train_ann instances_val2017_person_50.json \
  val_ann instances_val2017_person_50.json \
  input_size "(256,256)" \
  test_size "(256,256)" \
  multiscale_range 0 \
  mosaic_prob 0 \
  max_epoch 50 \
  eval_interval 1 \
  no_aug_epochs 5 \
  warmup_epochs 1
```

## Overfit: val2017_person_50 + LR fix (basic_lr_per_img restored for batch 1)

```bash
cd ~/Documents/YoloMinX && \
PYTHONPATH=~/Documents/YoloMinX \
yolox_venv/bin/python tools/train.py \
  -f exps/default/yolox_nano_person.py \
  -d 1 -b 1 --fp16 \
  data_dir ~/Documents/YoloMinX/datasets \
  train_ann instances_val2017_person_50.json \
  val_ann instances_val2017_person_50.json \
  input_size "(256,256)" \
  test_size "(256,256)" \
  multiscale_range 0 \
  mosaic_prob 0 \
  basic_lr_per_img 0.001 \
  max_epoch 50 \
  eval_interval 1 \
  no_aug_epochs 5 \
  warmup_epochs 1
```

## Smoke test v4: hardcoded overfit exp (no CLI arg uncertainty)

All settings locked in `exps/default/yolox_nano_overfit.py`.

```bash
cd ~/Documents/YoloMinX && \
PYTHONPATH=~/Documents/YoloMinX \
yolox_venv/bin/python tools/train.py \
  -f exps/default/yolox_nano_overfit.py \
  -d 1 -b 1 --fp16 \
  data_dir ~/Documents/YoloMinX/datasets
```




```bash
cd ~/Documents/YoloMinX && \
PYTHONPATH=~/Documents/YoloMinX \
yolox_venv/bin/python tools/train.py \
  -f exps/default/yolox_nano_person.py \
  -d 1 -b 4 \
  data_dir ~/Documents/YoloMinX/datasets \
  train_ann instances_val2017_person_50.json \
  val_ann instances_val2017_person_50.json \
  multiscale_range 0 \
  mosaic_prob 0 \
  max_epoch 50 \
  eval_interval 1 \
  no_aug_epochs 5 \
  warmup_epochs 1
```


cd ~/Documents/YoloMinX && PYTHONPATH=~/Documents/YoloMinX yolox_venv/bin/python tools/train.py   -f exps/default/yolox_nano_person.py   -d 1 -b 4   data_dir ~/Documents/YoloMinX/datasets   train_ann instances_val2017_person_50.json   val_ann instances_val2017_person_50.json   multiscale_range 0   mosaic_prob 0   max_epoch 500   eval_interval 1   no_aug_epochs 5   warmup_epochs 1

## Smoke test baseline: 200 images, 100 epochs — 128x128 (fast) and 416x416 (full res)

cd ~/Documents/YoloMinX && PYTHONPATH=~/Documents/YoloMinX yolox_venv/bin/python tools/train.py -f exps/default/yolox_nano_person.py -d 1 -b 4 data_dir ~/Documents/YoloMinX/datasets train_ann instances_val2017_person_200.json val_ann instances_val2017_person_200.json input_size "(128,128)" test_size "(128,128)" multiscale_range 0 mosaic_prob 0 max_epoch 100 eval_interval 1 no_aug_epochs 5 warmup_epochs 1

cd ~/Documents/YoloMinX && PYTHONPATH=~/Documents/YoloMinX yolox_venv/bin/python tools/train.py -f exps/default/yolox_nano_person.py -d 1 -b 4 data_dir ~/Documents/YoloMinX/datasets train_ann instances_val2017_person_200.json val_ann instances_val2017_person_200.json multiscale_range 0 mosaic_prob 0 max_epoch 100 eval_interval 1 no_aug_epochs 5 warmup_epochs 1

## Single-batch overfit (4 images): pipeline smoke test — loss should reach near-zero

cd ~/Documents/YoloMinX && PYTHONPATH=~/Documents/YoloMinX yolox_venv/bin/python tools/train.py -f exps/default/yolox_nano_person.py -d 1 -b 4 data_dir ~/Documents/YoloMinX/datasets train_ann instances_val2017_person_4.json val_ann instances_val2017_person_4.json multiscale_range 0 mosaic_prob 0 max_epoch 2000 eval_interval 10 no_aug_epochs 5 warmup_epochs 1

## the best variant:

cd ~/Documents/YoloMinX && PYTHONPATH=~/Documents/YoloMinX yolox_venv/bin/python tools/train.py   -f exps/default/yolox_nano_person.py   -d 1 -b 4   data_dir ~/Documents/YoloMinX/datasets   train_ann instances_val2017_person_50.json   val_ann instances_val2017_person_50.json   multiscale_range 0   mosaic_prob 0   max_epoch 2000   eval_interval 1   no_aug_epochs 5   warmup_epochs 1