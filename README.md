# EduYOLO: A Classroom Behavior Recognition Framework Based on High-Resolution Feature Attention Fusion

Official implementation of **EduYOLO**, a YOLOv11-based detector tailored for student classroom behavior recognition in dense, heavily-occluded classroom scenes.

This repository is a modified fork of [Ultralytics](https://github.com/ultralytics/ultralytics) **v8.3.9** — the complete training/validation code is included, no extra plugins required.

## Highlights

- **RFAConv backbone & neck** — replaces standard convolutions with Receptive-Field Attention Convolution (`RFAConv`, `C3k2_RFAConv`) to strengthen spatial perception of small and occluded objects;
- **C2BRA** — a Bi-Level Routing Attention module after SPPF for context modeling in crowded scenes;
- **P2 high-resolution detection head** — adds a P2/4 head (four heads total: P2–P5) for the many small objects typical of back rows in classrooms;
- **FM-IoU loss** — a Focaler-style multi-point-distance IoU loss (hyperparameters `d=0.0`, `u=0.95`, normalization factor `s=1`) that focuses regression on hard-to-refine samples.

## Main Results

Input 640×640, trained **from scratch** (no pretrained weights), 150 epochs.

| Dataset | Model | mAP50 | mAP50:95 | Recall | Precision | F1 | Params | FLOPs | FPS |
|---|---|---|---|---|---|---|---|---|---|
| CSCB-Dataset | YOLOv11m (baseline) | 76.49 | 58.44 | 74.54 | 73.27 | 73.90 | 20.03M | 67.7G | 185 |
| CSCB-Dataset | **EduYOLO** | **78.96** | **61.97** | **76.90** | **75.54** | **76.21** | 21.21M | 93.0G | 145 |
| SCB-Dataset3 | YOLOv11m (baseline) | 93.84 | 75.52 | 95.51 | 91.25 | 93.33 | 20.03M | 67.7G | 188 |
| SCB-Dataset3 | **EduYOLO** | **95.45** | **78.89** | **96.82** | **91.72** | **94.20** | 21.21M | 93.0G | 148 |

## ⚠️ Notes for Reproducing the Paper Results

Please read these three points first — they are the most common causes of a reproduction gap:

1. **Enable FM-IoU manually.** For clean baseline comparison, the released code defaults to CIoU. To train EduYOLO with the paper's FM-IoU loss, open `EduYOLO/ultralytics/utils/loss.py`, find `BboxLoss.forward` and swap the two lines:

   ```python
   # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
   iou = bbox_focaler_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False,
                             mpdiou_hw=mpdiou_hw[fg_mask], d=0.0, u=0.95)
   ```

   `d=0.0, u=0.95` are the grid-searched optima reported in the paper (Sec. 4.5.2); the normalization factor `s=1` is the code's default behavior and needs no extra setting.

2. **Use the m scale.** EduYOLO in the paper is the **m** scale (~21.2M params / 93.0 GFLOPs). Ultralytics infers the scale from the config **filename**; loading `EduYOLO.yaml` directly falls back to the *n* scale with only a one-line warning. Copy the config to `EduYOLOm.yaml` (or otherwise specify scale `m`) before training, and check the parameter count printed at the start of `train.py` — it should be ≈ 21M.

3. **No pretrained weights.** All models in the paper are trained from scratch (`pretrained=False`; `train.py` also blocks automatic weight downloads). Loading COCO-pretrained weights changes the training dynamics and the results.

## Installation

```bash
git clone https://github.com/datadance/EduYOLO.git
cd EduYOLO/EduYOLO
pip install -e .            # or: pip install ultralytics==8.3.9 and use this repo's source
```

Tested environment: Python 3.12, PyTorch 2.5.1, CUDA 12.4, single NVIDIA L20 (48GB).

## Dataset Preparation

### SCB-Dataset3 (public)

We use the **official release as-is** — the university-classroom subset (671 images, 6 classes: `Raise_hand`, `Read`, `Write`, `OnPhone`, `Bow_head`, `Sleep`) from
[F. Yang, T. Wang, *SCB-dataset3: A benchmark for detecting student classroom behavior*, arXiv:2310.02522](https://arxiv.org/abs/2310.02522), downloadable from the authors' repo [Whiffe/SCB-dataset](https://github.com/Whiffe/SCB-dataset). **No re-splitting, cleaning, resampling or class re-weighting is applied** — keep the official data organization and splits.

Write your own `data.yaml`:

```yaml
path: /path/to/SCB-Dataset3
train: images/train
val: images/val
nc: 6
names: ['Raise_hand', 'Read', 'Write', 'OnPhone', 'Bow_head', 'Sleep']  # order must match the official labels
```

### CSCB-Dataset (self-constructed)

Collected from real college classrooms (1,764 images, ~27,000 annotations, 6 behaviors), split 8:1:1 at the video-segment level to prevent temporal leakage. Contact us for access.

> **Note:** the files under `EduYOLO/dataset/` (100 sample images, `split_data.py`, `xml2txt.py`, a `ship` demo `data.yaml`) are **only** a format demo for smoke-testing the pipeline. They are **not** related to the SCB experiments — do not reuse `split_data.py`'s split for SCB-Dataset3.

## Training

Fill in the paths in `EduYOLO/train.py` (model yaml / data yaml / output dir), then:

```bash
python train.py
```

All hyperparameters used in the paper (already set in `train.py` / `ultralytics/cfg/default.yaml`):

| Item | Value |
|---|---|
| Epochs | 150 |
| Batch size | 8 |
| Image size | 640×640 |
| Optimizer | AdamW (lr0=0.01, momentum=0.937, weight_decay=0.0005) |
| Warmup | 3 epochs |
| Seed | 42 |
| AMP | False |
| Pretrained | False (from scratch) |
| Augmentation | Ultralytics defaults: mosaic=1.0, close_mosaic=10, fliplr=0.5, hsv_h/s/v=0.015/0.7/0.4, scale=0.5, translate=0.1 (no mixup / copy-paste) |

Multi-seed robustness results (seeds 7, 11, 13, 17, 19) are reported in the paper, Sec. 4.5.3.

## Validation

```bash
python val.py   # fill in the weight path and data yaml; default split='val'
```

`val.py` prints per-class Precision / Recall / F1 / mAP50 / mAP75 / mAP50-95 plus FPS and model info in the same format as the paper's tables, and saves them to `paper_data.txt`.

## Model Configuration

The EduYOLO architecture is defined in `EduYOLO/ultralytics/cfg/models/11/EduYOLO.yaml`; the added modules live in `EduYOLO/ultralytics/nn/ADD_modules/` (`RFAconv.py`, `attention.py`, `block.py`) and are registered in `ultralytics/nn/tasks.py`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{eduyolo,
  title   = {EduYOLO: A Classroom Behavior Recognition Framework Based on High-Resolution Feature Attention Fusion},
  author  = {<authors>},
  journal = {<journal, under review>},
  year    = {2026}
}
```

## Acknowledgements

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Thanks to the authors of [SCB-Dataset](https://github.com/Whiffe/SCB-dataset), RFAConv and BiFormer for their open-source contributions.

## License

AGPL-3.0, following the upstream Ultralytics license.


