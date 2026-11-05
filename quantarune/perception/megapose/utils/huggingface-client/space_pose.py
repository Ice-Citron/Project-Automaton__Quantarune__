"""One MegaPose estimate through the public Hugging Face Space.

Usage: python space_pose.py <image.png> <x1> <y1> <x2> <y2> <fx> <fy> <cx> <cy> <mesh.obj> <units> <out.json>
Initialises the Space with the mesh, runs one estimate and writes the
4x4 camera-to-object pose plus the Space's text to out.json. Exit 0 on a
pose, 2 when the Space returned no pose.
"""
import json
import re
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


def main(argv):
    image, x1, y1, x2, y2, fx, fy, cx, cy, mesh, units, out = argv
    t0 = time.time()
    client = Client(SPACE, verbose=False)
    status = client.predict(handle_file(mesh), units, "RGB", api_name="/handle_initialize")
    t_init = time.time() - t0
    text, vis = client.predict(
        handle_file(image), None, float(x1), float(y1), float(x2), float(y2),
        float(fx), float(fy), float(cx), float(cy), api_name="/handle_estimate")
    pose = parse_pose(text)
    vis_path = None
    if isinstance(vis, dict):
        vis_path = vis.get("path")
    elif isinstance(vis, str):
        vis_path = vis
    json.dump({"ok": pose is not None, "pose_T_cam_obj": pose, "init_status": status,
               "text": text, "vis_path": vis_path, "seconds": round(time.time() - t0, 1),
               "init_seconds": round(t_init, 1)}, open(out, "w"), indent=1)
    sys.exit(0 if pose is not None else 2)


if __name__ == "__main__":
    main(sys.argv[1:13])
