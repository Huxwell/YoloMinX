# Setup

Fresh clone to mAP on 4 images. Paste commands one by one.

**Requirements:** Python 3.10, CUDA 12.1, ~1 GB disk for COCO val2017.

---

## 1. Clone

```bash
git clone https://github.com/Huxwell/YoloMinX.git
cd YoloMinX
```

## 2. Virtual environment

```bash
python3.10 -m venv yolox_venv
source yolox_venv/bin/activate
```

## 3. Install PyTorch (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Verify GPU

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected: `True` and your GPU name.

## 6. Download COCO val2017 (~1 GB)

```bash
mkdir -p datasets
cd datasets
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
rm val2017.zip annotations_trainval2017.zip
cd ..
```

## 7. Create 4-image person subset

```bash
python3 datasets/filter_coco_person.py \
  --ann datasets/annotations/instances_val2017.json \
  --images-dir datasets/val2017 \
  --out-ann datasets/annotations/instances_val2017_person_4.json \
  --out-images-dir datasets/val2017_person_4 \
  --max-images 4
ln -s val2017 datasets/train2017
```

## 8. Train (4-image overfit, ~2000 epochs)

```bash
PYTHONPATH=. yolox_venv/bin/python tools/train.py \
  -f exps/default/yolox_nano_person.py \
  -d 1 -b 4 \
  data_dir $(pwd)/datasets \
  train_ann instances_val2017_person_4.json \
  val_ann instances_val2017_person_4.json \
  multiscale_range 0 \
  mosaic_prob 0 \
  max_epoch 2000 \
  eval_interval 10 \
  no_aug_epochs 5 \
  warmup_epochs 1
```

mAP is printed after each eval interval. With 4 images the model should overfit to 0.1-0.2 mAP (depending on random weights) in 1-2k epochs and closer to 0.7 (for IoU 0.5) over 8-10k epochs (~10min on 4gb vRAM T600)