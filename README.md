# PPE Detection and Tracking

All project code and generated outputs belong beside `Fork Dataset/`; none of these scripts writes into or edits that source folder. The supplied Roboflow export has 2,356 images, 1,885/236/235 in train/valid/test, four classes in ID order `boots, gloves, helmet, vest`, and 87 empty-label negative frames.

## Dataset Audit

The existing `Fork Dataset/data.yaml` uses `../train/images`, `../valid/images`, and `../test/images`. Those resolve outside `Fork Dataset/` and are missing. The root [data.yaml](data.yaml) fixes the paths without changing the export.

Initial read-only audit found 61 repeated normalized filename stems and 27 named source groups spanning existing splits, plus one unresolved numeric-only bucket. It found no byte-identical image files. The pHash scan found 2,404 cross-split candidate pairs at distance <= 5, including many adjacent frames from named videos; these candidates are not all duplicate images, but they strongly reinforce that the current random split leaks video context.

```powershell
python check_dataset.py --dataset "Fork Dataset"
```

`split_by_source.py` copies images and labels to `ppe_video_split/`, groups `video_4_frame_0012` as source `video_4` and `ppe_0850` as source `ppe`, then assigns whole groups to approximately 80/10/10 splits. It writes `source_groups.csv` and its own `data.yaml`. Check the CSV before training. Numeric-only names have no source signal and are conservatively kept together as one unresolved group; provide a mapping CSV with `filename,group` columns to recover their actual video identities.

```powershell
python split_by_source.py --dataset "Fork Dataset" --output ppe_video_split --seed 42
python check_dataset.py --dataset ppe_video_split
```

Group counts can be small or uneven because video groups cannot be divided to hit exact percentages. For deployment claims, use the new video-disjoint test set and keep it out of training/model selection.

## Windows Environment

Use 64-bit Python 3.12. BoxMOT 25 supports Python 3.10-3.13. CUDA wheels bundle the CUDA runtime; install a current NVIDIA driver, not a separate CUDA toolkit for PyTorch.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

NVIDIA GPU (official PyTorch CUDA 12.8 wheel index):

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CPU-only fallback:

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The package pins are `ultralytics==8.4.166` and `boxmot==25.0.0` (current releases checked 2026-09-30). YOLO26 first appeared in Ultralytics 8.4.x; use at least 8.4.0, but the scripts here rely on the current `nms` argument and so pin 8.4.166. Check `torch.cuda.is_available()` before training. Single-GPU Windows training uses `workers=0` and a main guard; Windows multi-GPU DDP has PyTorch/libuv limitations, so use one GPU or WSL2/Linux for multi-GPU.

## Train and Evaluate

First re-split by source. Then start with nano at 640:

```powershell
python train.py --model yolo26n.pt --data ppe_video_split/data.yaml --imgsz 640 --batch 8 --epochs 100 --device 0
```

Choose `--model yolo26s.pt` for a speed/accuracy balance or `yolo26m.pt` when VRAM allows. If GPU memory is tight, lower `--batch`; for CPU use `--device cpu --batch 2`. `workers=0` is deliberately conservative on Windows. The script explicitly uses AdamW, `lr0=0.001`, early stopping, moderate mosaic/scale/color augmentation and no mixup. Ultralytics' `optimizer=auto` may select MuSGD for long runs; AdamW is explicit here for this relatively small dataset. Set `--cls-pw 0.25` only if per-class validation shows persistent imbalance.

Start at 640 for a baseline. Compare 960 when small PPE occupies only a few pixels, lowering batch (often 2-4) to fit memory:

```powershell
python train.py --model runs/ppe/ppe_yolo26n_640/weights/best.pt --data ppe_video_split/data.yaml --imgsz 960 --batch 2 --epochs 40 --name ppe_yolo26n_960 --device 0
```

Optional P2 small-object experiment (not a released pretrained checkpoint; it builds a YOLO26 P2 YAML and transfers compatible pretrained weights):

```powershell
python train.py --model yolo26n.pt --data ppe_video_split/data.yaml --imgsz 640 --batch 4 --epochs 100 --p2 --name ppe_yolo26n_p2 --device 0
```

YOLO26 P2/P6 YAML architectures are supported by current Ultralytics, but there are no scale-specific pretrained P2 `.pt` weights. `train.py` writes a scale-specific copy of the official `yolo26-p2.yaml` under `runs/ppe/model_configs/`, then transfers compatible weights. P2 increases compute/memory. Compare it against 960px and sliced inference before keeping it. For gloves/boots, larger pixels, camera-specific training frames, and slices are often simpler wins.

To add realistic training-only low-resolution examples, keep validation and test pristine:

```powershell
python degrade_dataset.py --source ppe_video_split --output ppe_lowres --variants 1 --seed 42
python train.py --model yolo26n.pt --data ppe_lowres/data.yaml --imgsz 640 --batch 8 --device 0 --name ppe_yolo26n_lowres
```

This copies all files and adds training-only downscale/upscale, blur, noise, and JPEG-quality variants. It does not rewrite labels or the original dataset.

Evaluate on the held-out test split after model selection:

```powershell
python evaluate.py --model runs/ppe/ppe_yolo26n_640/weights/best.pt --data ppe_video_split/data.yaml --split test --imgsz 640 --device 0
```

The output includes per-class precision, recall, mAP50-95, mAP50/mAP75, confusion-matrix CSV, PR/confusion plots, and JSON. Per-class P/R are reported at the validator's selected operating threshold; mAP is the threshold-swept ranking metric. Low AP with low recall suggests missed/tiny/occluded labels or low confidence; low precision suggests confusing backgrounds or class overlap. Check support counts and label quality before changing thresholds. Across the full export, gloves (2,409 instances) and boots (3,278) are rarer than helmet/vest; glove/boot weakness may also be resolution and annotation quality, not just frequency.

## YOLO26 Notes

- YOLO26 needs Ultralytics 8.4.0 or newer. Current docs define YOLO26n/s/m and NMS-free `nms=False` inference.
- The model trains dual one-to-many and one-to-one heads. Normal prediction/validation defaults to one-to-many plus NMS; `nms=False` selects the one-to-one end-to-end path and avoids a second suppression pass. The code feeds Ultralytics' decoded `Results.boxes.data` into BoxMOT, not raw network tensors.
- YOLO26 removes Distribution Focal Loss bins (`reg_max=1`); the legacy `dfl` loss-weight field is repurposed for normalized L1 box-distance regression. Do not assume the old YOLO DFL interpretation.
- YOLO26 uses MuSGD in its long-run recipe, while current `optimizer=auto` selects based on run size. The training script pins AdamW and its learning rate for repeatable small-dataset fine-tuning.
- Export with `nms=False` gives supported ONNX/TensorRT graphs an end-to-end output shaped like `(batch, max_det, 6)`: `xyxy, confidence, class_id` (typically `(1, 300, 6)`). `nms=None` exports raw one-to-many candidates; `nms=True` embeds conventional NMS where supported. Some constrained formats fall back to one-to-many. Output graph metadata matters: `nms` at load time cannot change an already-exported graph.

## Video Inference and Compliance

```powershell
python infer_video.py --video "videos\shift1.mp4" --ppe-model runs/ppe/ppe_yolo26n_640/weights/best.pt --person-model yolo26n.pt --tracker botsort --reid-model osnet_x1_0_msmt17 --device 0 --imgsz 640 --output outputs\shift1
```

Person detection uses COCO YOLO26 class 0; PPE uses your four-class checkpoint. BoxMOT gets rows `(x1,y1,x2,y2,confidence,class_id)` and a BGR frame. For BoxMOT v25, NumPy tracker input is `N x 6`; output is `M x 8` in `xyxy, track_id, confidence, class_id, detection_index` order. This uses the current factory/config API and lets ReID-enabled trackers lazily encode current person crops.

BoT-SORT + OSNet MSMT17 is the default. Also supported: StrongSORT and Deep OC-SORT with the same ReID model, and ByteTrack without appearance embeddings. Try `--detect-every 2` to reduce detector cost; tracker updates still run on skipped frames with empty detections and the current image. Use `--device cpu` on systems without CUDA. `--slice-size 640` opts into SAHI tiled inference for both person/PPE boxes; tiles are mapped back to frame coordinates and add substantial latency. Begin with full-frame 960 or slice only when small objects are otherwise missed.

Compliance associates boxes by overlap/containment in category-specific upper-head, torso, arm/hand-side, and lower-leg/boot regions. A per-ID rolling vote updates only on detector frames; status persists between skipped frames. Defaults: 9 samples, minimum 3 observations, 60% present vote. Tune these thresholds on annotated video. The dataset's single `gloves` class means the simple logic treats one or more glove detections as glove presence; it does not prove both hands are protected.

Outputs are `ppe_annotated.mp4`, `violations.csv`, and `violations.json`. A violation begins when voting changes to missing PPE and closes when voting recovers; an unresolved event at end-of-video has blank end fields. These are decision-support signals, not a substitute for inspecting false negatives.

### Single-Photo Compliance

Use [infer_image_compliance.py](infer_image_compliance.py) when you need a per-person PPE estimate for one still photo. It runs the same person detector and the existing trained PPE model, associates equipment with body regions, draws a person box with helmet/vest/gloves/boots status, and writes JPEG, CSV, and JSON outputs. Single-image person numbers are local to that photo, not persistent tracking IDs. For wide CCTV views with distant people, enable `--slice-size 640`; on `test4.jpeg` this found 22 people compared with 4 using full-frame 640 inference, at a higher runtime cost.

```powershell
python infer_image_compliance.py --image "C:\path\to\photo.webp" --ppe-model "C:\path\to\best.pt" --person-model yolo26n.pt --imgsz 640 --slice-size 640 --device cpu --output outputs\photo_compliance
```

Photo-compliance defaults to `--person-conf 0.20 --ppe-conf 0.30` to avoid labeling weak low-confidence detections as compliant PPE. Lower `--ppe-conf` (for example, `0.15`) to favor recall, at the cost of more false detections. `YES` means the detector found that PPE class in the expected body region; `NO` means no qualifying detection was associated. It does not prove the person is not wearing the item. This distinction matters especially for boots and gloves, whose test-set recall is weak.

## Tracker Benchmark

```powershell
python benchmark_trackers.py --videos "videos\shift1.mp4" "videos\shift2.mp4" --person-model yolo26n.pt --trackers botsort,strongsort,deepocsort,bytetrack --reid-model osnet_x1_0_msmt17 --device 0 --output runs\tracker_benchmark
```

All trackers replay identical cached person detections; benchmark output separates tracker+ReID FPS from detector-inclusive effective FPS. The fork dataset has PPE boxes, not person identities, so it cannot produce IDF1/HOTA/ID switches. Supply `--gt-dir mot_gt` with one standard MOTChallenge person-identity CSV per video named `<video-stem>.txt` to score those metrics. The script reports IDF1, HOTA averaged over IoU 0.05-0.95, and ID switches at IoU 0.5. Without GT it reports speed and leaves identity metrics null. This is a fair comparison of the tracker stage, not detector accuracy; add manually verified identity tracks for defensible MOT scores.

If ID switches remain high, label the same people consistently across multiple cameras/videos, crop person boxes, and split train/query/gallery by video/camera to avoid identity leakage. Arrange crops in a Market1501-style directory (`bounding_box_train`, `query`, `bounding_box_test`) with identity/camera encoded in filenames, then adapt [reid_data.example.yaml](reid_data.example.yaml):

```powershell
boxmot train-reid --data reid_data.yaml --model osnet_x1_0 --loss triplet --epochs 100 --p-ids 16 --k-instances 4 --device 0 --num-workers 0 --project runs\reid --name ppe_osnet
```

Evaluate the learned embedding with `boxmot eval-reid --weights <best.pt> --dataset market1501 --data-dir <reid-root> --device 0`, then pass its checkpoint as `--reid-model <best.pt>`. ReID requires identity labels; PPE labels alone are not sufficient.

## Export and Speed

```powershell
python export_model.py --model runs/ppe/ppe_yolo26n_640/weights/best.pt --format onnx --imgsz 640
python export_model.py --model runs/ppe/ppe_yolo26n_640/weights/best.pt --format engine --imgsz 640 --fp16
```

ONNX export is the straightforward Windows route. To use it for CPU inference, install `onnxruntime` and pass the `.onnx` path as `--ppe-model`; pass `--device cpu`. TensorRT `.engine` is GPU-, driver-, CUDA-, and TensorRT-version-specific; install a compatible NVIDIA TensorRT Windows runtime, or export/run under WSL2/Linux if the native Windows stack is unavailable. Use FP16 on supported NVIDIA hardware and benchmark the exported artifact. For CPU, use YOLO26n at 640, ONNX Runtime CPU, or OpenVINO; lower frame size or detect every second frame when acceptable. Avoid 960 and SAHI on CPU unless latency permits.

## Current API and Further Work

BoxMOT 25 is a breaking Python integration release: use `create_tracker`, typed `ReIDConfig`, and `tracker.update(detections, frame)`. Older snippets using `boxmot.api`, `Tracker`, `img=`, `embs=`, or positional legacy tracker constructors need migration. The script uses the v25 API documented for `BotSortConfig`, `DeepOcSortConfig`, and `ReIDConfig`.

Current references: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/), [end-to-end detection](https://docs.ultralytics.com/guides/end2end-detection/), [export](https://docs.ultralytics.com/modes/export/), [BoxMOT Python API](https://mikel-brostrom.github.io/boxmot/python/), [BoxMOT installation](https://mikel-brostrom.github.io/boxmot/getting-started/installation/), [ReID training](https://mikel-brostrom.github.io/boxmot/modes/train/), and [PyTorch installation](https://pytorch.org/get-started/locally/).

Useful next comparisons: YOLO11n/s on the exact same video-disjoint split; RF-DETR only if its deployment latency fits; low-resolution frames from your actual cameras; person detector fine-tuning if COCO person boxes fail; and manual video-level MOT annotations to assess identity continuity. The dataset is CC BY 4.0; retain attribution. Ultralytics and BoxMOT are AGPL-3.0 projects, so review licensing before proprietary/commercial deployment.