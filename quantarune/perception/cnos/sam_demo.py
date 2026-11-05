"""SAM (ViT-B) automatic mask proposals on AIC centre-camera frames.

Left: all SAM proposals. Right: the single proposal with the best IoU against
the pseudo-label target-port mask (an oracle stand-in for CNOS's
template-matching selection step, which is NOT run here).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])  # scratchpad pylib with segment_anything
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # noqa: E402

CKPT = sys.argv[2]
OUT = Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3
ROOT = Path("/mnt/sienarune/Black-Projects/Project-Automaton/Santa_Maria/code/"
            "aic-quantarune-isaac-training/quantarune_training/outputs")
PACK = ROOT / "segmentation_annotation_pack_sfp_20260506"
VAL = ROOT / "segmentation_manifests_sfp_pseudolabel_20260506/val.jsonl"

rows = [json.loads(l) for l in open(VAL)]
picks = [rows[i] for i in np.linspace(0, len(rows) - 1, N).astype(int)]
sam = sam_model_registry["vit_b"](checkpoint=CKPT).to("cuda")
gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.86,
                                stability_score_thresh=0.92, min_mask_region_area=50)
rng = np.random.default_rng(0)
tiles = []
summary = []
for row in picks:
    img = cv2.cvtColor(cv2.imread(str(PACK / row["image_path"])), cv2.COLOR_BGR2RGB)
    masks = gen.generate(img)
    allm = img.copy()
    for m in masks:
        col = rng.integers(60, 255, 3)
        sel = m["segmentation"]
        allm[sel] = (0.5 * allm[sel] + 0.5 * col).astype(np.uint8)
    gt = cv2.imread(str(PACK / row["target_mask_path"]), cv2.IMREAD_GRAYSCALE) > 127
    best, best_iou = None, 0.0
    for m in masks:
        sel = m["segmentation"]
        inter = np.logical_and(sel, gt).sum()
        union = np.logical_or(sel, gt).sum()
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best, best_iou = m, iou
    pick = img.copy()
    if best is not None:
        sel = best["segmentation"]
        pick[sel] = (0.4 * pick[sel] + 0.6 * np.array([255, 40, 40])).astype(np.uint8)
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pick, cnts, -1, (40, 255, 40), 2)
    for im, txt in ((allm, f"SAM ViT-B: {len(masks)} proposals"),
                    (pick, f"best proposal vs pseudo-label, IoU {best_iou:.2f}")):
        cv2.putText(im, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(np.hstack([cv2.resize(allm, (448, 448)), cv2.resize(pick, (448, 448))]))
    summary.append({"frame": row["image_path"], "proposals": len(masks), "best_iou": round(best_iou, 3)})
grid = np.vstack(tiles)
cv2.imwrite(str(OUT / "sam_proposals_vs_pseudolabel.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
json.dump(summary, open(OUT / "sam_demo_summary.json", "w"), indent=1)
print(json.dumps(summary))
