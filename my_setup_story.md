# My YOLOX Setup Story

> **Environment**: Ubuntu Linux | Python 3.10.16 | PyTorch 2.5.1 | CUDA 12.1 | GPU: NVIDIA Tesla T4 (15 GB VRAM)

---

## Phase 0 — Environment Bootstrap

### Step 0.1 — Create and activate a Python virtual environment

```bash
cd /home/dev/ai-playground
python3 -m venv .venv
source .venv/bin/activate
```

### Step 0.2 — Install PyTorch with CUDA 12.1 support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 0.3 — Verify GPU is visible to PyTorch

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected output:
# 2.5.1+cu121
# True
# Tesla T4
```

---

## Phase 1 — Clone and Install YOLOX

### Step 1.1 — Clone the YOLOX repository

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```

### Step 1.2 — Install YOLOX dependencies (without installing the package itself)

```bash
pip install -r YOLOX/requirements.txt
```

> **Note**: We intentionally did NOT run `pip install -e YOLOX/` to keep the source editable and importable via `PYTHONPATH`.

### Step 1.3 — Install additional required packages

```bash
pip install pycocotools loguru tabulate tqdm thop
pip install Pillow
```

### Step 1.4 — Verify the YOLOX source import works

```bash
PYTHONPATH=YOLOX python3 -c "from yolox.exp import Exp; print('YOLOX import OK')"
```

---

## Phase 2 — Mock COCO Dataset (Smoke Test Data)

### Step 2.1 — Create the mock dataset script

```bash
cat > /home/dev/ai-playground/create_mock_coco.py << 'PYEOF'
import os
import json
from PIL import Image


def main():
    base_dir = "/home/dev/ai-playground/datasets/COCO"
    os.makedirs(os.path.join(base_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train2017"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val2017"), exist_ok=True)

    # COCO has 80 classes
    categories = [{"id": i, "name": f"class_{i}", "supercategory": "none"} for i in range(1, 81)]

    for split in ["train2017", "val2017"]:
        images = []
        annotations = []
        ann_id = 1

        # 4 dummy images per split
        for img_id in range(1, 5):
            file_name = f"{img_id:012d}.jpg"
            img_path = os.path.join(base_dir, split, file_name)

            # Save 640x640 dummy image
            color = (img_id * 50 % 256, img_id * 100 % 256, img_id * 150 % 256)
            img = Image.new("RGB", (640, 640), color=color)
            img.save(img_path)

            images.append({
                "id": img_id,
                "width": 640,
                "height": 640,
                "file_name": file_name
            })

            # Box annotations
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "area": 40000.0,
                "iscrowd": 0
            })
            ann_id += 1

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 80,
                "bbox": [300.0, 300.0, 150.0, 250.0],
                "area": 37500.0,
                "iscrowd": 0
            })
            ann_id += 1

        json_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        json_file = os.path.join(base_dir, "annotations", f"instances_{split}.json")
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)

    print("Mock COCO dataset successfully created at:", base_dir)


if __name__ == "__main__":
    main()
PYEOF
```

### Step 2.2 — Run the mock dataset generator

```bash
source .venv/bin/activate
PYTHONPATH=YOLOX python3 create_mock_coco.py
# Output: Mock COCO dataset successfully created at: /home/dev/ai-playground/datasets/COCO
```

### Step 2.3 — Verify dataset structure

```bash
find datasets/COCO -type f | sort
# Expected:
# datasets/COCO/annotations/instances_train2017.json
# datasets/COCO/annotations/instances_val2017.json
# datasets/COCO/train2017/000000000001.jpg  ... 000000000004.jpg
# datasets/COCO/val2017/000000000001.jpg    ... 000000000004.jpg
```

---

## Phase 3 — Smoke-Test Training (YOLOX-S, 1 epoch, mock data)

### Step 3.1 — Run 1-epoch smoke test

```bash
source .venv/bin/activate
PYTHONPATH=YOLOX python3 YOLOX/tools/train.py \
    -n yolox-s \
    -d 1 \
    -b 4 \
    --fp16 \
    -o \
    max_epoch 1 \
    data_dir /home/dev/ai-playground/datasets/COCO \
    train_ann instances_train2017.json \
    val_ann instances_val2017.json
```

> **Result**: Training completed 1 epoch / 1 iteration successfully.  
> Checkpoints saved to `YOLOX_outputs/yolox_s/`.  
> Loss metrics: `total_loss=60.6`, `iou_loss=5.0`, `conf_loss=49.6`, `cls_loss=1.0`.  
> Peak GPU VRAM: ~5394 MB. ✅

---

## Phase 4 — Real COCO val2017 Dataset Download

### Step 4.1 — Create the val download script

```bash
cat > /home/dev/ai-playground/download_coco_val.sh << 'SHEOF'
#!/usr/bin/env bash
set -e

mkdir -p datasets/COCO
cd datasets/COCO

echo "Downloading COCO annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Downloading COCO val2017 images..."
wget -c http://images.cocodataset.org/zips/val2017.zip

echo "Extracting annotations..."
unzip -o annotations_trainval2017.zip
rm annotations_trainval2017.zip

echo "Extracting val2017 images..."
unzip -o val2017.zip
rm val2017.zip

echo "COCO val2017 setup finished!"
SHEOF
chmod +x download_coco_val.sh
```

### Step 4.2 — Create the train download script (for future use)

```bash
cat > /home/dev/ai-playground/download_coco_train.sh << 'SHEOF'
#!/usr/bin/env bash
set -e

# Clean up mock training directory if it exists
if [ -d "datasets/COCO/train2017" ]; then
    echo "Removing old mock train2017 directory..."
    rm -rf datasets/COCO/train2017
fi

mkdir -p datasets/COCO
cd datasets/COCO

echo "Downloading COCO train2017 images (~18GB)..."
wget -c http://images.cocodataset.org/zips/train2017.zip

echo "Extracting train2017 images..."
unzip -o train2017.zip
rm train2017.zip

echo "COCO train2017 setup finished!"
SHEOF
chmod +x download_coco_train.sh
```

### Step 4.3 — Run the val download

```bash
cd /home/dev/ai-playground
bash download_coco_val.sh
# Downloads ~1GB of images + annotations from cocodataset.org
# Extracts to datasets/COCO/val2017/ (5000 images) and datasets/COCO/annotations/
```

---

## Phase 5 — YOLOX-S Quick Validation Run (val2017)

### Step 5.1 — Create the yolox_s_val experiment file

```bash
cat > /home/dev/ai-playground/yolox_s_val.py << 'PYEOF'
import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "yolox_s_val"

        # Override paths to use val2017 for both training and validation
        self.train_ann = "instances_val2017.json"
        self.val_ann = "instances_val2017.json"

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="val2017",  # Hardcoded dataset image folder name
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )
PYEOF
```

### Step 5.2 — Run YOLOX-S on val2017 (quick 2.5-minute check, killed manually)

```bash
source .venv/bin/activate
PYTHONPATH=YOLOX python3 YOLOX/tools/train.py \
    -f yolox_s_val.py \
    -d 1 \
    -b 16 \
    --fp16 \
    -o \
    max_epoch 5 \
    data_dir /home/dev/ai-playground/datasets/COCO
# Run terminated manually at ~95 iterations (epoch 1/5) after ~2.5 minutes
# VRAM: 11,342 MB / 15,360 MB
# Multi-scale training: sizes ranged 480–800px dynamically per batch ✅
```

---

## Phase 6 — YOLOX-Nano Custom Experiment Files

### Step 6.1 — Create yolox_nano_train.py (train on full COCO train2017)

```bash
cat > /home/dev/ai-playground/yolox_nano_train.py << 'PYEOF'
import os
import torch.nn as nn
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Nano model settings
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.exp_name = "yolox_nano_train_coco"

        # Full COCO train/val splits
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.data_dir = "/home/dev/ai-playground/datasets/COCO"
        self.eval_interval = 1  # run validation after every epoch

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train2017",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )
PYEOF
```

### Step 6.2 — Create yolox_nano_val.py (train/eval loop on val2017 only)

```bash
cat > /home/dev/ai-playground/yolox_nano_val.py << 'PYEOF'
import os
import torch.nn as nn
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Nano settings
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.exp_name = "yolox_nano_val"

        # Override paths to use val2017 for both training and validation
        self.train_ann = "instances_val2017.json"
        self.val_ann = "instances_val2017.json"

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model uses depthwise=True
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="val2017",  # Hardcoded dataset image folder name
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )
PYEOF
```

---

## Phase 7 — YOLOX-Nano 3-Epoch Training + Evaluation (val2017)

### Step 7.1 — Run 3-epoch YOLOX-Nano training on val2017

```bash
source .venv/bin/activate
mkdir -p logs
PYTHONPATH=YOLOX python3 YOLOX/tools/train.py \
    -f yolox_nano_val.py \
    -d 1 \
    -b 32 \
    --fp16 \
    -o \
    max_epoch 3 \
    data_dir /home/dev/ai-playground/datasets/COCO \
    2>&1 | tee logs/nano_coco_train.log
```

> **Result**: 3 epochs / 471 total iterations (157 per epoch) completed successfully.  
> Peak VRAM: ~7.6 GB.  
> C++ JIT COCO evaluator compiled and ran without errors.  
> mAP results (expected low — 3 epochs from scratch):  
> - Person AP: 0.021 | Person AR: 1.863  
> - Overall AP: 0.00 ✅ (proves the eval pipeline works end-to-end)

Checkpoints saved to `YOLOX_outputs/yolox_nano_val/`:
- `epoch_1_ckpt.pth`, `epoch_2_ckpt.pth`, `epoch_3_ckpt.pth`
- `best_ckpt.pth`, `last_epoch_ckpt.pth`, `latest_ckpt.pth`
- `train_log.txt`

### Step 7.2 — Verify checkpoint files exist

```bash
ls -lh YOLOX_outputs/yolox_nano_val/
```

---

## Phase 8 — Commands That Failed / Were Killed

| Command | What Happened | Fix Applied |
|---|---|---|
| First `create_mock_coco.py` version imported `numpy` | `import numpy as np` was present but `np` was never used; caused no actual error but was cleaned up in final version | Removed the unused `numpy` import |
| `YOLOX/tools/train.py` with `-n yolox-s` on real COCO val data | Ran fine but was **killed manually** after 2.5 min at iteration 95 to save time | Intentional kill — just a timing check |
| Various `pip install` attempts without the venv activated | Packages installed to system Python instead of `.venv` | Always `source .venv/bin/activate` first |
| `git clone` on slow connection | Occasionally stalled | Re-ran with `git clone --depth 1` for speed |

---

## Summary of Files Created

| File | Purpose |
|---|---|
| `create_mock_coco.py` | Generates 4-image COCO-format dataset for smoke tests |
| `yolox_nano_train.py` | YOLOX-Nano experiment config pointing to full COCO train2017 |
| `yolox_nano_val.py` | YOLOX-Nano experiment config using val2017 for both train and eval |
| `yolox_s_val.py` | YOLOX-S experiment config using val2017 (used for quick check) |
| `download_coco_val.sh` | Downloads COCO val2017 images + annotations (~1 GB) |
| `download_coco_train.sh` | Downloads COCO train2017 images (~18 GB) — for future use |
| `logs/nano_coco_train.log` | Full training log from the 3-epoch nano run |

---

## Key Environment Variables / Patterns Used Throughout

```bash
# Always needed when running YOLOX from source (not installed as package)
export PYTHONPATH=/home/dev/ai-playground/YOLOX

# Activate venv before any python/pip command
source /home/dev/ai-playground/.venv/bin/activate

# Standard training command pattern
PYTHONPATH=YOLOX python3 YOLOX/tools/train.py \
    -f <experiment_file.py> \
    -d 1 \            # number of GPUs
    -b <batch_size> \
    --fp16 \          # mixed precision
    -o \              # occupy GPU memory upfront
    max_epoch <N> \
    data_dir /home/dev/ai-playground/datasets/COCO
```
