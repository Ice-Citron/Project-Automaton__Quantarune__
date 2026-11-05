"""Run MegaPose on sim frames through the public Hugging Face Space.

Usage: python run_space_smoke.py <frames.json> <out_dir>
"""
import json
import re
import shutil
import sys
import time

from gradio_client import Client, handle_file

SPACE = "hedonismbot24/megapose-6d-pose-estimation"


def parse_pose(text):
    m = re.search(r"4x4 Pose Matrix.*?\n(.*)", text, re.S)
    if not m:
        return None
    nums = re.findall(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?|[-+]?\d+\.?", m.group(1))
    vals = [float(v) for v in nums][:16]
    return [vals[i:i + 4] for i in range(0, 16, 4)] if len(vals) == 16 else None


def main(frames_json, out_dir):
    spec = json.load(open(frames_json))
    K = spec["K"]
    client = Client(SPACE)
    mesh = f"{out_dir}/{spec.get('mesh_file', 'sfp_mount.obj')}"
    t0 = time.time()
    status = client.predict(handle_file(mesh), "m", "RGB", api_name="/handle_initialize")
    print(f"init {time.time() - t0:.1f}s: {status!r}", flush=True)
    results = []
    for f in spec["frames"]:
        x1, y1, x2, y2 = f["bbox"]
        t0 = time.time()
        text, vis = client.predict(
            handle_file(f["image"]), None, x1, y1, x2, y2,
            K["fx"], K["fy"], K["cx"], K["cy"], api_name="/handle_estimate")
        dt = time.time() - t0
        if "episode_id" in f:
            tag = f"{f['episode_id']}_f{f['frame_idx']:06d}"
        else:
            tag = f["file"].rsplit(".", 1)[0]
        vis_path = None
        if vis:
            vis_src = vis if isinstance(vis, str) else vis.get("path") or vis.get("url")
            if vis_src and not str(vis_src).startswith("http"):
                vis_path = f"{out_dir}/{tag}_vis.png"
                shutil.copy(vis_src, vis_path)
        open(f"{out_dir}/{tag}.txt", "w").write(text)
        pose = parse_pose(text)
        ok = text.startswith("✅")
        results.append({**f, "ok": ok, "seconds": round(dt, 1), "pose_T_cam_obj": pose, "vis": vis_path})
        pos = [round(p[3], 4) for p in pose[:3]] if pose else None
        print(f"{tag} ok={ok} {dt:.1f}s pos={pos}", flush=True)
    json.dump({"space": SPACE, "mode": "RGB", "K": K, "mesh": spec["mesh"],
               "bbox_source": spec.get("bbox_source"), "results": results},
              open(f"{out_dir}/results.json", "w"), indent=1)
    print("done", sum(r["ok"] for r in results), "/", len(results))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
