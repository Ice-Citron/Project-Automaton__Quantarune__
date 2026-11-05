"""Ground-truth port pose in the centre-camera optical frame per saved frame.

Usage: python extract_port_gt.py <bag_dir> <frames.csv> <port_link> <out.json>
Reads /tf, /tf_static and /scoring/tf from a trial bag, composes
world -> ... -> center_camera/optical and world -> ... -> <port_link> at
each frame stamp and writes T_cam_port per frame.
"""
import bisect
import csv
import json
import pathlib
import sys

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

CAMERA = "center_camera/optical"


def quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def to_mat(tr):
    t, q = tr.transform.translation, tr.transform.rotation
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def load_edges(bag_dir):
    """edges[child] = (parent, times, mats); static edges have one entry."""
    ts = get_typestore(Stores.ROS2_JAZZY)
    edges = {}
    with Reader(pathlib.Path(bag_dir)) as r:
        conns = [c for c in r.connections if c.topic in ("/tf", "/tf_static", "/scoring/tf")]
        for c, t, raw in r.messages(connections=conns):
            m = ts.deserialize_cdr(raw, c.msgtype)
            for tr in m.transforms:
                stamp = tr.header.stamp.sec * 10**9 + tr.header.stamp.nanosec
                if c.topic == "/tf_static":
                    stamp = -1
                e = edges.setdefault(tr.child_frame_id, (tr.header.frame_id, [], []))
                e[1].append(stamp)
                e[2].append(to_mat(tr))
    for child, (parent, times, mats) in edges.items():
        order = np.argsort(times)
        edges[child] = (parent, [times[i] for i in order], [mats[i] for i in order])
    return edges


def edge_at(edges, child, stamp):
    parent, times, mats = edges[child]
    if times[0] == -1:
        return parent, mats[0]
    i = bisect.bisect_right(times, stamp) - 1
    i = max(0, min(i, len(times) - 1))
    return parent, mats[i]


def world_T(edges, frame, stamp):
    m = np.eye(4)
    chain = [frame]
    while frame in edges:
        parent, e = edge_at(edges, frame, stamp)
        m = e @ m
        frame = parent
        chain.append(frame)
    return m, chain


def main(bag_dir, frames_csv, port_link, out_json):
    edges = load_edges(bag_dir)
    rows = list(csv.DictReader(open(frames_csv)))
    out = []
    for row in rows:
        stamp = int(row["stamp_ns"])
        T_w_cam, cam_chain = world_T(edges, CAMERA, stamp)
        T_w_port, port_chain = world_T(edges, port_link, stamp)
        T_cam_port = np.linalg.inv(T_w_cam) @ T_w_port
        out.append({"file": row["file"], "stamp_ns": stamp,
                    "T_cam_port": np.round(T_cam_port, 6).tolist(),
                    "T_world_cam": np.round(T_w_cam, 6).tolist(),
                    "T_world_port": np.round(T_w_port, 6).tolist()})
    roots = {cam_chain[-1], port_chain[-1]}
    meta = {"bag": bag_dir, "camera_frame": CAMERA, "port_link": port_link,
            "camera_chain": cam_chain, "port_chain": port_chain, "roots": sorted(roots)}
    json.dump({"meta": meta, "frames": out}, open(out_json, "w"), indent=1)
    print("roots:", sorted(roots), "frames:", len(out))
    for o in out[:3]:
        print(o["file"], "port in cam:", [round(v, 4) for v in np.array(o["T_cam_port"])[:3, 3]])


if __name__ == "__main__":
    main(*sys.argv[1:5])
