"""
MegaPose 6D Object Pose Estimation Demo (RGB-D + Occlusion-Robust)

Uses HappyPose/MegaPose (CoRL 2022) for 6D object pose estimation.
Input: RGB image + optional depth image + 3D mesh (.obj/.ply/.stl) + 2D bounding box
Output: 6D pose (4x4 matrix), visualized with projected axes

Three modes:
- RGB: RGB-only, single hypothesis
- RGBD: RGB + depth, single hypothesis (better Z accuracy)
- RGBD + Multi-hypothesis + ICP: 5 hypotheses + ICP depth refinement (best for occlusion)
"""

import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image

os.environ.setdefault("HAPPYPOSE_DATA_DIR", "/app/happypose_data")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global model cache ───────────────────────────────────────────────────────
_pose_estimator = None
_model_info = None
_current_object_dataset = None
_current_mode = None
_dense_mesh_path = None

# Maps UI mode names to HappyPose model names
MODE_MAP = {
    "RGB": "megapose-1.0-RGB",
    "RGBD": "megapose-1.0-RGBD",
    "RGBD + Multi-Hyp + ICP (occlusion-robust)": "megapose-1.0-RGB-multi-hypothesis-icp",
}


def _ensure_dense_mesh(mesh_path: str, min_vertices: int = 2500) -> str:
    """Load a mesh and subdivide if needed to ensure enough vertices for MegaPose."""
    mesh = trimesh.load(mesh_path, force="mesh")
    iterations = 0
    while len(mesh.vertices) < min_vertices and iterations < 6:
        mesh = mesh.subdivide()
        iterations += 1
    out_path = Path(tempfile.mkdtemp()) / "dense_mesh.ply"
    mesh.export(str(out_path))
    logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces (subdivided {iterations}x)")
    return str(out_path)


def load_model(mesh_path: str, mode: str = "RGBD", mesh_label: str = "object", mesh_units: str = "m"):
    """Load MegaPose model with the given object mesh."""
    global _pose_estimator, _model_info, _current_object_dataset, _current_mode, _dense_mesh_path

    from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
    from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model

    dense_path = _ensure_dense_mesh(mesh_path)
    _dense_mesh_path = dense_path

    rigid_objects = [
        RigidObject(label=mesh_label, mesh_path=Path(dense_path), mesh_units=mesh_units)
    ]
    _current_object_dataset = RigidObjectDataset(rigid_objects)

    model_name = MODE_MAP[mode]
    _model_info = NAMED_MODELS[model_name]
    _current_mode = mode

    logger.info(f"Loading {model_name}...")
    _pose_estimator = load_named_model(model_name, _current_object_dataset).to(device)
    # Subsample SO3 grid for faster inference (162 -> ~20 views)
    _pose_estimator._SO3_grid = _pose_estimator._SO3_grid[::8]
    logger.info(f"Model loaded on {device} (mode={mode}, model={model_name})")


def _prepare_depth(depth_image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Convert uploaded depth image to float32 meters at the correct resolution."""
    depth = depth_image.copy()

    # Handle color depth images -> single channel
    if len(depth.shape) == 3:
        if depth.shape[2] == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
        elif depth.shape[2] == 4:
            depth = cv2.cvtColor(depth, cv2.COLOR_RGBA2GRAY)
        else:
            depth = depth[:, :, 0]

    # Convert to float meters
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0  # mm -> m
    elif depth.dtype == np.uint8:
        depth = depth.astype(np.float32) / 255.0 * 5.0  # normalize to 0-5m
    else:
        depth = depth.astype(np.float32)

    # Resize if needed
    h, w = target_shape[:2]
    if depth.shape[:2] != (h, w):
        logger.warning(f"Resizing depth from {depth.shape[:2]} to ({h}, {w})")
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    logger.info(f"Depth prepared: shape={depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]m")
    return depth


def estimate_pose(rgb: np.ndarray, K: np.ndarray, bbox: list,
                  depth: Optional[np.ndarray] = None, label: str = "object"):
    """Run MegaPose inference. Returns 4x4 pose matrix."""
    import pandas as pd
    from happypose.toolbox.inference.types import ObservationTensor, PandasTensorCollection

    observation = ObservationTensor.from_numpy(rgb, depth, K).to(device)

    det_data = {"label": [label], "batch_im_id": [0], "instance_id": [0]}
    det_tensors = {"bboxes": torch.tensor([bbox], dtype=torch.float32)}
    detections = PandasTensorCollection(infos=pd.DataFrame(det_data), **det_tensors).to(device)

    with torch.no_grad():
        data_TCO, extra_data = _pose_estimator.run_inference_pipeline(
            observation,
            detections=detections,
            **_model_info["inference_parameters"],
        )

    pose = data_TCO.poses.cpu().numpy()[0]  # (4, 4)
    timing = extra_data.get("timing_str", "")
    return pose, timing


# ── Visualization ────────────────────────────────────────────────────────────

def draw_pose_axes(image, pose_matrix, K, axis_length=0.05, thickness=3):
    vis = image.copy()
    R, t = pose_matrix[:3, :3], pose_matrix[:3, 3]
    origin = t.reshape(3, 1)
    axes_3d = origin + R @ (np.eye(3) * axis_length)

    def proj(pt):
        p = (K @ pt).flatten()
        if abs(p[2]) < 1e-6:
            return None
        return int(p[0] / p[2]), int(p[1] / p[2])

    o = proj(origin)
    if o is None:
        return vis

    for i, (color, label) in enumerate(zip(
        [(0, 0, 255), (0, 255, 0), (255, 0, 0)], ["X", "Y", "Z"]
    )):
        end = proj(axes_3d[:, i:i+1])
        if end:
            cv2.arrowedLine(vis, o, end, color, thickness, tipLength=0.2)
            cv2.putText(vis, label, (end[0]+5, end[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.circle(vis, o, 5, (255, 255, 255), -1)
    cv2.circle(vis, o, 5, (0, 0, 0), 2)
    return vis


def draw_bbox(image, bbox, color=(0, 255, 255), thickness=2):
    vis = image.copy()
    x1, y1, x2, y2 = [int(b) for b in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def quat_from_matrix(R):
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return 0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2]) * 2
        return (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2]) * 2
        return (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1]) * 2
        return (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s


def quat_to_euler(w, x, y, z):
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    sinp = 2*(w*y-z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def format_pose(pose, mode):
    t = pose[:3, 3]
    w, x, y, z = quat_from_matrix(pose[:3, :3])
    roll, pitch, yaw = quat_to_euler(w, x, y, z)
    return f"""━━━ Mode: {mode} ━━━

━━━ Position (meters) ━━━
  X: {t[0]:+.4f}
  Y: {t[1]:+.4f}
  Z: {t[2]:+.4f}

━━━ Orientation (quaternion) ━━━
  W: {w:+.6f}   X: {x:+.6f}
  Y: {y:+.6f}   Z: {z:+.6f}

━━━ Euler Angles (degrees) ━━━
  Roll:  {roll:+.2f}°
  Pitch: {pitch:+.2f}°
  Yaw:   {yaw:+.2f}°

━━━ 4x4 Pose Matrix ━━━
{np.array2string(pose, precision=4, suppress_small=True)}"""


# ── Gradio handlers ──────────────────────────────────────────────────────────

def handle_initialize(mesh_file, mesh_units, mode):
    if mesh_file is None:
        return "❌ Please upload a 3D mesh file"
    try:
        path = mesh_file.name if hasattr(mesh_file, "name") else mesh_file
        load_model(path, mode=mode, mesh_units=mesh_units)
        model_name = MODE_MAP[mode]
        n_hyp = _model_info["inference_parameters"].get("n_pose_hypotheses", 1)
        icp = _model_info["inference_parameters"].get("run_depth_refiner", False)
        details = f"Model: {model_name}\nHypotheses: {n_hyp}" + (" + ICP refinement" if icp else "")
        return f"✅ Object initialized!\n{details}\n\nGo to tab ② to estimate pose."
    except Exception as e:
        logger.error(f"Init error: {e}", exc_info=True)
        return f"❌ Error: {e}"


def handle_estimate(query_image, depth_image, bbox_x1, bbox_y1, bbox_x2, bbox_y2, fx, fy, cx, cy):
    if query_image is None:
        return "❌ Upload an image", None
    if _pose_estimator is None:
        return "❌ Initialize an object first (tab ①)", None

    try:
        bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Check if this mode needs depth
        needs_depth = _model_info.get("requires_depth", False)
        depth = None

        if depth_image is not None:
            depth = _prepare_depth(depth_image, query_image.shape)
        elif needs_depth:
            logger.warning("Mode requires depth but none provided — results may be degraded")

        pose, timing = estimate_pose(query_image, K, bbox, depth=depth)

        vis = draw_bbox(query_image, bbox)
        vis = draw_pose_axes(vis, pose, K)

        notes = []
        if depth is not None:
            notes.append(f"📐 Depth: [{depth.min():.3f}, {depth.max():.3f}]m")
        elif needs_depth:
            notes.append("⚠️ Depth image recommended for this mode but not provided")
        notes_str = "\n".join(notes)

        text = f"✅ Pose Estimated\n{notes_str}\n\n{format_pose(pose, _current_mode)}\n\n⏱ {timing}"
        return text, vis

    except Exception as e:
        logger.error(f"Estimation error: {e}", exc_info=True)
        return f"❌ Error: {e}", query_image


# ── UI ───────────────────────────────────────────────────────────────────────

DESCRIPTION = """
# 🎯 MegaPose — 6D Object Pose Estimation

**[MegaPose](https://megapose6d.github.io/)** (Labbé et al., CoRL 2022) estimates the 6D pose of **any object** 
from a single RGB or RGB-D image + a 3D CAD mesh — no object-specific training required.

### How it works
1. **Initialize**: Upload a 3D mesh (.obj/.ply/.stl) and select a mode
2. **Estimate**: Upload an RGB image + optional depth map, specify a bounding box → get the 6D pose
3. **Visualize**: See projected 3D coordinate axes (Red=X, Green=Y, Blue=Z)

| Feature | Detail |
|---------|--------|
| Method | Render-and-compare with SO(3) grid search + iterative refinement |
| Input | RGB or RGB-D image + CAD mesh + 2D bounding box |
| Output | 4×4 pose matrix (rotation + translation) |
| Paper | [CoRL 2022](https://arxiv.org/abs/2212.06173) |
| Code | [HappyPose](https://github.com/agimus-project/happypose) |
"""

with gr.Blocks(title="MegaPose 6D Pose Estimation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.Tab("① Initialize Object"):
            gr.Markdown("""Upload a 3D mesh of the object you want to detect.

> 💡 The mesh is auto-subdivided if needed (MegaPose requires ≥2500 vertices).
> Untextured meshes work fine — only geometry is needed.""")
            with gr.Row():
                with gr.Column():
                    init_mesh = gr.File(label="3D Mesh (.obj / .ply / .stl)", file_types=[".obj", ".ply", ".stl"])
                    init_units = gr.Radio(["m", "mm"], value="m", label="Mesh units")
                    init_mode = gr.Radio(
                        choices=list(MODE_MAP.keys()),
                        value="RGBD + Multi-Hyp + ICP (occlusion-robust)",
                        label="Estimation Mode",
                        info="Choose based on your input and scene complexity",
                    )
                    gr.Markdown("""
| Mode | Depth needed | Hypotheses | Best for |
|------|:---:|:---:|------|
| **RGB** | No | 1 | Quick test, no depth sensor |
| **RGBD** | Yes | 1 | Full objects with depth sensor |
| **RGBD + Multi-Hyp + ICP** | Yes | 5 + ICP | **Partially visible / occluded objects** |
""")
                    init_btn = gr.Button("🚀 Initialize", variant="primary", size="lg")
                with gr.Column():
                    init_result = gr.Textbox(label="Status", lines=5, interactive=False)
            init_btn.click(handle_initialize, [init_mesh, init_units, init_mode], init_result)

        with gr.Tab("② Estimate Pose"):
            gr.Markdown("""Upload a query image and specify the bounding box of the object.

> ⚠️ Bounding box coordinates are in **pixels**. Adjust to tightly enclose the **visible portion** of the object.""")
            with gr.Row():
                with gr.Column():
                    est_image = gr.Image(label="Query Image (RGB)", type="numpy")
                    est_depth = gr.Image(
                        label="Depth Image (16-bit PNG in mm, or 8-bit grayscale)",
                        type="numpy",
                    )
                    gr.Markdown("""<details><summary>📋 Depth image format guide</summary>

- **16-bit PNG** (most depth sensors): Values in **millimeters**, auto-converted to meters
- **8-bit grayscale**: Linearly scaled to 0–5 meters
- **Float**: Used as-is (assumed meters)
- **Color depth maps**: Auto-converted to single channel
- Resolution mismatch with RGB is handled automatically

</details>""")
                    gr.Markdown("#### Bounding Box (pixels)")
                    with gr.Row():
                        bb_x1 = gr.Number(label="x1 (left)", value=80)
                        bb_y1 = gr.Number(label="y1 (top)", value=40)
                    with gr.Row():
                        bb_x2 = gr.Number(label="x2 (right)", value=240)
                        bb_y2 = gr.Number(label="y2 (bottom)", value=200)
                    gr.Markdown("#### Camera Intrinsics")
                    with gr.Row():
                        est_fx = gr.Number(label="fx", value=500.0)
                        est_fy = gr.Number(label="fy", value=500.0)
                    with gr.Row():
                        est_cx = gr.Number(label="cx", value=320.0)
                        est_cy = gr.Number(label="cy", value=240.0)
                    est_btn = gr.Button("🎯 Estimate Pose", variant="primary", size="lg")
                with gr.Column():
                    est_vis = gr.Image(label="Pose Visualization", type="numpy")
                    est_result = gr.Textbox(label="Pose Output", lines=22, interactive=False)
            est_btn.click(
                handle_estimate,
                [est_image, est_depth, bb_x1, bb_y1, bb_x2, bb_y2, est_fx, est_fy, est_cx, est_cy],
                [est_result, est_vis],
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## MegaPose Architecture

1. **Coarse model**: Samples 162 rotations on SO(3), renders the CAD model at each, compares to the query crop via a CNN → selects top-K hypotheses
2. **Refiner model**: Iteratively renders the object at each pose hypothesis, computes residual updates → 5 refinement iterations per hypothesis
3. **Scoring**: Re-scores all refined hypotheses, picks the best
4. **ICP** (optional): Iterative Closest Point alignment on depth point cloud for final geometric refinement

### Handling Occlusion

The default single-hypothesis mode assumes the full object is visible. For **partially visible objects**:

- **Multi-hypothesis mode** generates **5 coarse pose candidates** instead of 1. With more candidates, at least one is likely to align with the visible portion of the object.
- **ICP refinement** uses the depth point cloud to geometrically align the 3D model to the visible surface, which is robust to missing parts since ICP only matches *observed* points.

### Available Models

| Model | Coarse | Refiner | Hypotheses | ICP | Requires Depth |
|-------|--------|---------|:---:|:---:|:---:|
| `megapose-1.0-RGB` | ✅ | RGB | 1 | ❌ | No |
| `megapose-1.0-RGBD` | ✅ | RGBD | 1 | ❌ | Yes |
| `megapose-1.0-RGB-multi-hypothesis-icp` | ✅ | RGB | 5 | ✅ | Yes |

All use the same coarse model (`coarse-rgb-906902141`).

### Key Features
- **Zero-shot**: No per-object training — works on any object with a CAD mesh
- **Untextured meshes**: Only geometry is needed
- **Occlusion-robust**: Multi-hypothesis + ICP handles partial visibility
- **Renderer**: Panda3D offscreen (no custom CUDA compilation)

### Citation
```bibtex
@inproceedings{labbe2022megapose,
    title={MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare},
    author={Labb\\'e, Yann and Manuelli, Lucas and Cosber, Arsalan and Jatavallabhula, KM and Dieter, Miles and Fox, Dieter},
    booktitle={CoRL},
    year={2022}
}
```

### Links
- [Paper (CoRL 2022)](https://arxiv.org/abs/2212.06173)
- [Project Page](https://megapose6d.github.io/)
- [HappyPose GitHub](https://github.com/agimus-project/happypose)
- [BOP Benchmark](https://bop.felk.cvut.cz/)
            """)

    gr.Markdown("---\n<center>Built with [MegaPose](https://megapose6d.github.io/) via [HappyPose](https://github.com/agimus-project/happypose) and [Gradio](https://gradio.app)</center>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
