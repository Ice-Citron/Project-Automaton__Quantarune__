# computer-vision

This folder holds the perception work (there are three components to this).

Note that while a lot of this was not successfully incorporated in our final design, we have decided to include it in this repo for the sake of completeness. 

```
classical/: holds the OpenCV port finder and the vision gate. 
segmentation-model/: holds the learned port segmenter with its code and best checkpoint and a 200-episode sample of the annotation pack. 
cnos/: holds SAM and DINOv2 
megapose/: a simple readme discussing ideas.
```

## Research and exploration

### Learned port and plug segmentation

<a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/computer-vision/segmentation-model/"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/computer-vision/cnos/segmenter_val_overlays.png" width="100%"></a>

A U-Net trained on the demonstration frames to segment the plug and the
target port, with the pseudo-labels drawn in red and the prediction outlined
in green. After 14 epochs the validation mean IoU is 0.41, the plug IoU 0.43
and the target IoU 0.38, with a plug centre error of 18 px and a target
centre error of 49 px (`segmentation-model/model/metrics.jsonl`). Good
enough to find the plug, not yet good enough to centre a port.

<img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/computer-vision/cnos/segmenter_training_curves.png" width="60%">

### SAM proposals against the pseudo-labels

<a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/computer-vision/cnos/"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/computer-vision/cnos/sam_proposals_vs_pseudolabel.png" width="60%"></a>

Segment Anything (ViT-B) on three frames gives 39 to 54 mask proposals per
frame, and the best proposal overlaps the pseudo-label with an IoU of 0.54,
0.92 and 0.93 (`cnos/sam_demo_summary.json`). The right mask is usually
among the proposals, so the open question was how to pick it.

### CNOS-style selection with DINOv2

<a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/computer-vision/cnos/"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/computer-vision/cnos/cnos_style_dinov2_selection.png" width="100%"></a>

Picking the proposal by DINOv2 similarity to 24 templates, in the style of
CNOS. On 12 test frames the top-1 pick overlaps the label by 0.19 IoU on
average and clears 0.5 IoU in only 2 frames, while the best available
proposal averages 0.50 (`cnos/cnos_style_summary.json`). 

### MegaPose

<a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/computer-vision/megapose/OFFLINE_EVALUATION_20261104.md"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/computer-vision/megapose/figures/offline_smoke_overlay_sheet.png" width="100%"></a>

MegaPose was run as an experiment. Every call went through a public Hugging Face
Space running the RGB model on CPU. Offline, against the ground-truth port
pose on SFP scene 137, the estimates land 8 to 12 mm off in the image plane
but tens of millimetres off in depth, which is not enough to drive the last
millimetre (`megapose/OFFLINE_EVALUATION_20261104.md`).

<a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/computer-vision/megapose/IN_THE_LOOP_20261104.md"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/computer-vision/megapose/figures/in_loop_trial001_looks_sheet.png" width="100%"></a>

MegaPose was then called inside a running policy with ground truth off on
the same scene, 21 trials in all. The three evidence trials inserted the
cable and scored 87.21, 87.20 and 87.15, but the insertion was driven by
taught cage templates, an image servo at the hand-off and the scripted
insert. MegaPose's estimates at the hand-off were about 1 cm off in the
plane and 4 cm high, so they were logged, gated and never allowed to move
the arm. Everything was taught and tested on scene 137 only
(`megapose/IN_THE_LOOP_20261104.md`).
