"""Build the MegaPose input list with oracle boxes from the projected mesh.

Usage: python make_live_frames_mesh.py <frames_dir> <gt.json> <mesh.obj> <tx,ty,tz> <out_dir> [max_frames]
<tx,ty,tz> is the mesh origin in the GT link frame (from the model SDF).
The box is the projection of every mesh vertex, padded by 8 px.
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def main(frames_dir, gt_json, mesh_path, offset, out_dir, max_frames=12):
    os.makedirs(out_dir, exist_ok=True)
    info = json.load(open(f"{frames_dir}/camera_info.json"))
    K = np.array(info["k"]).reshape(3, 3)
    W, H = info["width"], info["height"]
    mesh = trimesh.load(mesh_path, force="mesh")
    verts = np.asarray(mesh.vertices)[::7]
    T_link_mesh = np.eye(4)
    T_link_mesh[:3, 3] = [float(v) for v in offset.split(",")]
    gt = json.load(open(gt_json))
    rows = []
    for fr in gt["frames"]:
        T_cam_mesh = np.array(fr["T_cam_port"]) @ T_link_mesh
        pts = (T_cam_mesh[:3, :3] @ verts.T).T + T_cam_mesh[:3, 3]
        if pts[:, 2].min() <= 0.02:
            continue
        uv = (K @ pts.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        c = (K @ T_cam_mesh[:3, 3])
        c = c[:2] / c[2]
        if not (0 <= c[0] < W and 0 <= c[1] < H):
            continue
        pad = 8
        bbox = [int(max(0, uv[:, 0].min() - pad)), int(max(0, uv[:, 1].min() - pad)),
                int(min(W - 1, uv[:, 0].max() + pad)), int(min(H - 1, uv[:, 1].max() + pad))]
        png = f"{out_dir}/{os.path.splitext(fr['file'])[0]}.png"
        if not os.path.exists(png):
            Image.open(f"{frames_dir}/{fr['file']}").convert("RGB").save(png)
        rows.append({"file": fr["file"], "image": png, "bbox": bbox,
                     "mesh_px": [round(float(c[0]), 1), round(float(c[1]), 1)],
                     "depth_m": round(float(T_cam_mesh[2, 3]), 4),
                     "T_cam_port": fr["T_cam_port"], "T_cam_mesh_gt": np.round(T_cam_mesh, 6).tolist(),
                     "stamp_ns": fr["stamp_ns"]})
    step = max(1, len(rows) // int(max_frames))
    rows = rows[::step][:int(max_frames)]
    spec = {"frames": rows, "K": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                                  "note": f"camera_info from the sim, {W}x{H}"},
            "mesh": os.path.basename(mesh_path), "mesh_file": os.path.basename(mesh_path),
            "T_link_mesh_translation": offset,
            "bbox_source": "oracle: projection of the GT-posed mesh vertices, pad 8 px"}
    json.dump(spec, open(f"{out_dir}/frames.json", "w"), indent=1)
    tiles = []
    for r in rows:
        im = Image.open(r["image"]).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle(r["bbox"], outline=(255, 255, 0), width=3)
        x, y = r["mesh_px"]
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 0, 0))
        d.text((8, 8), f"{r['file']} z={r['depth_m']:.3f}", fill=(255, 255, 255))
        tiles.append(im.resize((576, 512), Image.BILINEAR))
    cols = 4
    sheet = Image.new("RGB", (576 * cols, 512 * ((len(tiles) + cols - 1) // cols)))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 576, (i // cols) * 512))
    sheet.save(f"{out_dir}/input_sheet.png")
    print("frames:", len(rows), "of", len(gt["frames"]), "->", f"{out_dir}/input_sheet.png")
    for r in rows:
        print(r["file"], "bbox", r["bbox"], "depth", r["depth_m"])


if __name__ == "__main__":
    main(*sys.argv[1:7])
