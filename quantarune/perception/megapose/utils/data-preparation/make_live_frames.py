"""Build the MegaPose input list for live full-resolution frames.

Usage: python make_live_frames.py <frames_dir> <gt.json> <mesh_extent_xyz_m> <out_dir> [max_frames]
The detection box is the projection of the port link's axis-aligned box of
the given mesh extent (an oracle box from ground truth), padded by 8 px.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


def project(K, pts_cam):
    uv = (K @ pts_cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def main(frames_dir, gt_json, extent, out_dir, max_frames=12):
    os.makedirs(out_dir, exist_ok=True)
    info = json.load(open(f"{frames_dir}/camera_info.json"))
    K = np.array(info["k"]).reshape(3, 3)
    W, H = info["width"], info["height"]
    ex = np.array([float(v) for v in extent.split(",")]) / 2
    corners = np.array([[sx * ex[0], sy * ex[1], sz * ex[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    gt = json.load(open(gt_json))
    rows = []
    for fr in gt["frames"]:
        T = np.array(fr["T_cam_port"])
        if T[2, 3] <= 0.02:
            continue
        pts = (T[:3, :3] @ corners.T).T + T[:3, 3]
        uv = project(K, pts)
        c = project(K, T[:3, 3][None])[0]
        if not (0 <= c[0] < W and 0 <= c[1] < H):
            continue
        pad = 8
        bbox = [int(max(0, uv[:, 0].min() - pad)), int(max(0, uv[:, 1].min() - pad)),
                int(min(W - 1, uv[:, 0].max() + pad)), int(min(H - 1, uv[:, 1].max() + pad))]
        png = f"{out_dir}/{os.path.splitext(fr['file'])[0]}.png"
        if not os.path.exists(png):
            Image.open(f"{frames_dir}/{fr['file']}").convert("RGB").save(png)
        rows.append({"file": fr["file"], "image": png, "bbox": bbox,
                     "port_px": [round(float(c[0]), 1), round(float(c[1]), 1)],
                     "depth_m": round(float(T[2, 3]), 4), "T_cam_port": fr["T_cam_port"],
                     "stamp_ns": fr["stamp_ns"]})
    step = max(1, len(rows) // int(max_frames))
    rows = rows[::step][:int(max_frames)]
    spec = {"frames": rows, "K": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                                  "note": f"camera_info from the sim, {W}x{H}"},
            "mesh": "sfp_mount.obj (Intrinsic-AI/meshes/mujoco/manual_sfp_mount.obj, metres)",
            "bbox_source": f"oracle: projected {extent} m box at the GT port link, pad 8 px"}
    json.dump(spec, open(f"{out_dir}/frames.json", "w"), indent=1)
    tiles = []
    for r in rows:
        im = Image.open(r["image"]).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle(r["bbox"], outline=(255, 255, 0), width=3)
        d.ellipse([r["port_px"][0] - 5, r["port_px"][1] - 5, r["port_px"][0] + 5, r["port_px"][1] + 5], fill=(255, 0, 0))
        d.text((8, 8), f"{r['file']} z={r['depth_m']:.3f}", fill=(255, 255, 255))
        tiles.append(im.resize((576, 512), Image.BILINEAR))
    cols = 4
    sheet = Image.new("RGB", (576 * cols, 512 * ((len(tiles) + cols - 1) // cols)))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 576, (i // cols) * 512))
    sheet.save(f"{out_dir}/input_sheet.png")
    print("frames:", len(rows), "of", len(gt["frames"]), "-> ", f"{out_dir}/input_sheet.png")
    for r in rows:
        print(r["file"], "bbox", r["bbox"], "depth", r["depth_m"])


if __name__ == "__main__":
    main(*sys.argv[1:6])
