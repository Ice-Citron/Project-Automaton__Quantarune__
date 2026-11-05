# perception

This folder holds our work on finding the plug and the target port in
the camera images. We tried four methods. Most of this work did not go
into our final policy. We keep it here as a record. This folder is an
archive, so the code does not run as is.

## Layout

```
perception/
  classical/            OpenCV features of the plug and the port
  segmentation-model/   a learned segmenter for the plug and the port
  cnos/                 SAM mask proposals with DINOv2 selection
  megapose/             6D pose of the port with MegaPose
```

## classical/

| File | What it does |
|---|---|
| `vision_target_features.py` | Finds plug and port keypoints in the centre camera and estimates their geometry |
| `learned_segmentation_cv.py` | Gets the same features from the masks of the learned segmenter |

The policies in `../policies/vision-gated/` use these features.

## segmentation-model/

This is a U-Net that segments the plug and the target port.
`train_port_segmentation.py` trains it.

| Path | What it holds |
|---|---|
| `model/` | The model, the datasets, the losses, the metrics and the inference code |
| `config/config.json` | The model config |
| `utils/dataset/` | Export of the annotation pack, the pseudo-labels and the train and validation manifests |
| `utils/manual-correction/` | Selection of a review subset, a browser tool to correct masks, validation and fine-tune manifests |

The recorded result after 14 epochs was a mean IoU of 0.41 on the
validation set. The plug IoU was 0.43 and the target IoU was 0.38. The
model found the plug. It was not accurate enough to centre on a port.

## cnos/

| File | What it does |
|---|---|
| `sam_demo.py` | Makes automatic mask proposals with SAM (ViT-B) |
| `cnos_style_selection.py` | Selects one proposal by DINOv2 similarity to templates |
| `segmenter_overlays.py` | Draws the segmenter output on held-out frames |
| `output/` | The result images and the two summary JSON files |

SAM usually made a good mask. The best proposal had an IoU of 0.54 to
0.93 against the pseudo-label. The DINOv2 selection did not pick it
often. On 12 frames the top pick had a mean IoU of 0.19.

## megapose/

MegaPose estimates the 6D pose of an object from one RGB image and a
mesh. The weights were not available, so every call went through a
public Hugging Face Space on CPU.

| Path | What it holds |
|---|---|
| `scripts/` | The shell runners for the capture, the trials and the evidence chain |
| `utils/data-preparation/` | Frame matching to the bag, ground-truth extraction and the input lists |
| `utils/huggingface-client/` | The client for the Space and a copy of the Space app |
| `utils/evaluation/` | Comparison with ground truth and the trial summary |
| `output/run-1` to `run-5` | Results of the two smoke tests, the two offline runs and the in-the-loop trials |

Offline on SFP scene 137, MegaPose found the port to about 1 cm in the
image plane. The depth error was tens of millimetres. In the loop, the
policy logged the MegaPose estimate but did not use it to move the arm.
See `../policies/megapose-scripted-insert/` for that policy.

## Notes

- Some scripts still import `quantarune_training.vision`. That code is
  now in `segmentation-model/model/`.
- The datasets, the checkpoints and the raw frames are not in this
  repository.
