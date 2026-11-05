"""Match recorded dataset frames to bag sim stamps through the arm joints.

Usage: python match_frames_to_bag.py <parquet> <info.json> <bag_dir> <stride> <out_frames.csv>
Each parquet row i is video frame i. For every stride-th row the bag
/joint_states message with the nearest joint vector gives the stamp. The
dataset's joint_positions follow the bag's alphabetical joint order
(elbow, gripper left finger, shoulder_lift, shoulder_pan, wrist_1..3).
"""
import csv
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def main(parquet, info_json, bag_dir, stride, out_csv):
    df = pd.read_parquet(parquet)
    names = json.load(open(info_json))["features"]["observation.state"]["names"]
    jidx = [i for i, n in enumerate(names) if n.startswith("joint_positions")]
    ts = get_typestore(Stores.ROS2_JAZZY)
    stamps, joints = [], []
    with Reader(pathlib.Path(bag_dir)) as r:
        conns = [c for c in r.connections if c.topic == "/joint_states"]
        for c, t, raw in r.messages(connections=conns):
            m = ts.deserialize_cdr(raw, c.msgtype)
            if len(m.position) != len(jidx):
                continue
            stamps.append(m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)
            joints.append(list(m.position))
    stamps = np.array(stamps)
    joints = np.array(joints)
    print("bag joint_states:", len(stamps), "span s:", (stamps[-1] - stamps[0]) / 1e9)
    rows = []
    for i in range(0, len(df), int(stride)):
        q = np.array(df["observation.state"].iloc[i])[jidx]
        d = np.linalg.norm(joints - q, axis=1)
        k = int(np.argmin(d))
        rows.append((int(stamps[k]), f"f{i:06d}.png", 1152, 1024, i, round(float(d[k]), 5)))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp_ns", "file", "width", "height", "frame_index", "joint_residual_rad"])
        w.writerows(rows)
    res = [r[5] for r in rows]
    st = [r[0] for r in rows]
    print("frames:", len(rows), "residual max:", max(res), "median:", float(np.median(res)))
    print("stamps monotonic:", all(a < b for a, b in zip(st, st[1:])),
          "dt median s:", float(np.median(np.diff(st)) / 1e9) if len(st) > 1 else None)


if __name__ == "__main__":
    main(*sys.argv[1:6])
