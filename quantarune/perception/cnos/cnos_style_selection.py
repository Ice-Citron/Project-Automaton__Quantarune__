"""CNOS-style target selection: SAM proposals scored by DINOv2 similarity to
exemplar templates of the target port, no pseudo-label used at test time.

Templates: pseudo-label crops of the target port from TRAIN frames.
Test: VAL frames -> SAM proposals -> DINOv2 CLS embedding per proposal crop ->
max cosine similarity to the template bank -> best proposal.  IoU against the
val pseudo-label is reported only as the evaluation metric.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])  # scratchpad pylib with segment_anything
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # noqa: E402

SAM_CKPT, OUT = sys.argv[2], Path(sys.argv[3])
N_TEMPL = int(sys.argv[4]) if len(sys.argv) > 4 else 24
N_TEST = int(sys.argv[5]) if len(sys.argv) > 5 else 12
ROOT = Path("/mnt/sienarune/Black-Projects/Project-Automaton/Santa_Maria/code/"
            "aic-quantarune-isaac-training/quantarune_training/outputs")
PACK = ROOT / "segmentation_annotation_pack_sfp_20260506"
MAN = ROOT / "segmentation_manifests_sfp_pseudolabel_20260506"
DEV = "cuda"


def rows(name):
    return [json.loads(l) for l in open(MAN / name)]


def load(row):
    img = cv2.cvtColor(cv2.imread(str(PACK / row["image_path"])), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(PACK / row["target_mask_path"]), cv2.IMREAD_GRAYSCALE) > 127
    return img, gt


def crop(img, mask, pad=4):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    c = img[y0:y1, x0:x1].copy()
    c[~mask[y0:y1, x0:x1]] = 127  # grey out background like a masked template
    return cv2.resize(c, (224, 224))


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


import timm  # noqa: E402

dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                         num_classes=0, img_size=224).to(DEV)
dino.train(False)
mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


@torch.no_grad()
def embed(crops):
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(DEV) / 255.0
    return torch.nn.functional.normalize(dino((x - mean) / std), dim=1)


train = rows("train.jsonl")
templ = []
for r in [train[i] for i in np.linspace(0, len(train) - 1, N_TEMPL).astype(int)]:
    img, gt = load(r)
    c = crop(img, gt)
    if c is not None:
        templ.append(c)
bank = embed(templ)
print("templates:", bank.shape[0])

sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(DEV)
gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.86,
                                stability_score_thresh=0.92, min_mask_region_area=50)
val = rows("val.jsonl")
tests = [val[i] for i in np.linspace(0, len(val) - 1, N_TEST).astype(int)]
summary, tiles = [], []
for r in tests:
    img, gt = load(r)
    keep, crops = [], []
    frame_area = img.shape[0] * img.shape[1]
    for m in gen.generate(img):
        frac = m["segmentation"].sum() / frame_area
        if not 0.003 <= frac <= 0.2:  # drop specks and the board/table
            continue
        c = crop(img, m["segmentation"])
        if c is not None:
            keep.append(m["segmentation"])
            crops.append(c)
    sims = (embed(crops) @ bank.T).max(dim=1).values.cpu().numpy()
    order = np.argsort(-sims)
    ious = [iou(k, gt) for k in keep]
    best = keep[order[0]]
    summary.append({"frame": r["image_path"], "proposals": len(keep),
                    "top1_sim": round(float(sims[order[0]]), 3),
                    "top1_iou": round(ious[order[0]], 3),
                    "oracle_iou": round(max(ious), 3),
                    "oracle_rank": int(np.argmax([ious[i] for i in order])) + 1})
    vis = img.copy()
    vis[best] = (0.4 * vis[best] + 0.6 * np.array([255, 40, 40])).astype(np.uint8)
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (40, 255, 40), 2)
    cv2.putText(vis, f"sim {sims[order[0]]:.2f} IoU {ious[order[0]]:.2f}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(cv2.resize(vis, (384, 384)))
cols = 4
grid = np.zeros((int(np.ceil(len(tiles) / cols)) * 384, cols * 384, 3), np.uint8)
for i, t in enumerate(tiles):
    rr, cc = divmod(i, cols)
    grid[rr * 384:(rr + 1) * 384, cc * 384:(cc + 1) * 384] = t
cv2.imwrite(str(OUT / "cnos_style_dinov2_selection.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
hits = sum(1 for s in summary if s["top1_iou"] >= 0.5)
res = {"templates": int(bank.shape[0]), "test_frames": len(summary),
       "top1_hits_iou_ge_0.5": hits,
       "mean_top1_iou": round(float(np.mean([s["top1_iou"] for s in summary])), 3),
       "mean_oracle_iou": round(float(np.mean([s["oracle_iou"] for s in summary])), 3),
       "per_frame": summary}
json.dump(res, open(OUT / "cnos_style_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}))
