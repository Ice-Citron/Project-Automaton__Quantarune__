"""Runs the trained port segmenter on held-out frames and saves an overlay grid.

Red = predicted target-port mask, blue = predicted plug mask,
green outline = pseudo-label used for training.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/mnt/sienarune/Black-Projects/Project-Automaton/Santa_Maria/"
                   "code/aic-quantarune-isaac-training/quantarune_training")
from quantarune_training.vision.inference import PortSegmenterPredictor  # noqa: E402

ROOT = Path("/mnt/sienarune/Black-Projects/Project-Automaton/Santa_Maria/code/"
            "aic-quantarune-isaac-training/quantarune_training/outputs")
PACK = ROOT / "segmentation_annotation_pack_sfp_20260506"
VAL = ROOT / "segmentation_manifests_sfp_pseudolabel_20260506/val.jsonl"
CKPT = ROOT / "port_segmenter_sfp_pseudolabel_resume12_gpu_20260507/best.pt"
OUT = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6

rows = [json.loads(l) for l in open(VAL)]
print("manifest keys:", sorted(rows[0].keys()))
picks = [rows[i] for i in np.linspace(0, len(rows) - 1, N).astype(int)]
predictor = PortSegmenterPredictor(CKPT, device="cuda")
tiles = []
for row in picks:
    img = cv2.cvtColor(cv2.imread(str(PACK / row["image_path"])), cv2.COLOR_BGR2RGB)
    task = row.get("task") or {k: v for k, v in row.items() if k.startswith("task")}
    pred = predictor.predict(img, task)
    over = img.copy()
    for prob, cand, color in ((pred.target_prob, pred.target_candidate, (255, 40, 40)),
                              (pred.plug_prob, pred.plug_candidate, (40, 90, 255))):
        m = cv2.resize(prob.astype(np.float32), (img.shape[1], img.shape[0])) >= predictor.threshold
        over[m] = (0.45 * over[m] + 0.55 * np.array(color)).astype(np.uint8)
        if cand is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in cand.bbox_xyxy]
            cv2.rectangle(over, (x0, y0), (x1, y1), color, 1)
            cx, cy = cand.center_px
            cv2.drawMarker(over, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 10, 1)
    gt_path = row.get("target_mask_path")
    if gt_path and (PACK / gt_path).is_file():
        gt = cv2.imread(str(PACK / gt_path), cv2.IMREAD_GRAYSCALE)
        cnts, _ = cv2.findContours((gt > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (40, 255, 40), 2)
    label = f'{row["episode_id"]} f{row["frame_idx"]} {row.get("port_name", "")}'
    conf = f"sel={pred.selection_confidence:.2f}" if hasattr(pred, "selection_confidence") else ""
    cv2.putText(over, f"{label} {conf}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(cv2.resize(over, (384, 384)))
cols = 3
rows_n = int(np.ceil(len(tiles) / cols))
grid = np.zeros((rows_n * 384, cols * 384, 3), np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    grid[r * 384:(r + 1) * 384, c * 384:(c + 1) * 384] = t
cv2.imwrite(str(OUT / "segmenter_val_overlays.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print("saved", OUT / "segmenter_val_overlays.png", "image shape", img.shape)
