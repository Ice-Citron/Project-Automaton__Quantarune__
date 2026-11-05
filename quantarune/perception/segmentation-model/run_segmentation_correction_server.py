from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local browser-based mask correction server for the segmentation "
            "review subset or one correction batch."
        )
    )
    parser.add_argument("--root", required=True, help="Review subset root or one correction batch root.")
    parser.add_argument("--manifest", default=None, help="Optional manifest file. Defaults to root/manifest.jsonl or root/review_manifest.jsonl.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _default_manifest(root: Path) -> Path:
    candidates = (
        root / "manifest.jsonl",
        root / "review_manifest.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No manifest found under {root}")


def _progress_key(row: dict[str, Any]) -> str:
    return str(row["image_path"])


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Segmentation Correction</title>
  <style>
    body { font-family: sans-serif; margin: 0; background: #111; color: #eee; }
    .wrap { display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }
    .side { padding: 16px; border-right: 1px solid #333; overflow-y: auto; }
    .main { padding: 16px; }
    .row { margin-bottom: 12px; }
    button, select, input { font: inherit; }
    button { padding: 8px 10px; margin-right: 8px; margin-bottom: 8px; }
    canvas { border: 1px solid #333; image-rendering: pixelated; cursor: crosshair; }
    .meta { font-size: 13px; line-height: 1.4; color: #ccc; white-space: pre-wrap; }
    .toolbar button.active { background: #2d6cdf; color: white; }
    .small { font-size: 12px; color: #aaa; }
    .done-badge { color: #7be495; font-weight: bold; }
    .preview { max-width: 100%; border: 1px solid #333; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="side">
      <div class="row"><strong>Segmentation Correction</strong></div>
      <div class="row">
        <button id="prevBtn">Prev</button>
        <button id="nextBtn">Next</button>
        <button id="saveBtn">Save</button>
      </div>
      <div class="row">
        <label><input id="reviewedBox" type="checkbox"> Mark reviewed</label>
      </div>
      <div class="row toolbar">
        <button id="targetBtn" class="active">Target</button>
        <button id="plugBtn">Plug</button>
        <button id="eraseBtn">Erase</button>
      </div>
      <div class="row">
        <label>Brush size
          <input id="brushSize" type="range" min="2" max="48" value="12">
          <span id="brushSizeValue">12</span>
        </label>
      </div>
      <div class="row">
        <label><input id="showTarget" type="checkbox" checked> Show target</label><br>
        <label><input id="showPlug" type="checkbox" checked> Show plug</label>
      </div>
      <div class="row">
        <label>Frame
          <select id="frameSelect"></select>
        </label>
      </div>
      <div class="row small" id="statusLine">Loading…</div>
      <div class="row small" id="progressLine"></div>
      <div class="row meta" id="metaBox"></div>
      <div class="row">
        <a id="overlayLink" href="#" target="_blank" rel="noopener noreferrer">Open overlay reference</a>
        <img id="overlayPreview" class="preview" alt="overlay preview">
      </div>
    </div>
    <div class="main">
      <canvas id="canvas"></canvas>
    </div>
  </div>
  <script>
    const state = {
      manifest: [],
      index: 0,
      image: null,
      targetMask: null,
      plugMask: null,
      activeLayer: "target",
      erase: false,
      brushSize: 12,
      drawing: false,
      dirty: false,
      width: 0,
      height: 0,
    };

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const targetBtn = document.getElementById("targetBtn");
    const plugBtn = document.getElementById("plugBtn");
    const eraseBtn = document.getElementById("eraseBtn");
    const brushSize = document.getElementById("brushSize");
    const brushSizeValue = document.getElementById("brushSizeValue");
    const showTarget = document.getElementById("showTarget");
    const showPlug = document.getElementById("showPlug");
    const frameSelect = document.getElementById("frameSelect");
    const statusLine = document.getElementById("statusLine");
    const progressLine = document.getElementById("progressLine");
    const metaBox = document.getElementById("metaBox");
    const reviewedBox = document.getElementById("reviewedBox");
    const overlayLink = document.getElementById("overlayLink");
    const overlayPreview = document.getElementById("overlayPreview");

    function setStatus(msg) { statusLine.textContent = msg; }
    function progressCount() {
      return state.manifest.filter((row) => row.review_done).length;
    }
    function setProgressLine() {
      progressLine.textContent = `Reviewed ${progressCount()}/${state.manifest.length}`;
    }
    function setButtons() {
      targetBtn.classList.toggle("active", state.activeLayer === "target" && !state.erase);
      plugBtn.classList.toggle("active", state.activeLayer === "plug" && !state.erase);
      eraseBtn.classList.toggle("active", state.erase);
    }
    function optionLabel(row, idx) {
      const done = row.review_done ? " [done]" : "";
      return `${idx + 1}/${state.manifest.length}${done} ${row.target_slot} ${row.episode_id}:${row.frame_idx}`;
    }
    function refreshSelectOption(idx) {
      const opt = frameSelect.options[idx];
      if (!opt) return;
      opt.textContent = optionLabel(state.manifest[idx], idx);
    }

    function imageFromUrl(url) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
      });
    }

    async function loadManifest() {
      const res = await fetch("/api/manifest");
      state.manifest = await res.json();
      frameSelect.innerHTML = "";
      state.manifest.forEach((row, idx) => {
        const opt = document.createElement("option");
        opt.value = String(idx);
        opt.textContent = optionLabel(row, idx);
        frameSelect.appendChild(opt);
      });
      setProgressLine();
      await loadFrame(0);
    }

    async function loadFrame(index) {
      if (index < 0 || index >= state.manifest.length) return;
      state.index = index;
      frameSelect.value = String(index);
      const row = state.manifest[index];
      const [img, target, plug] = await Promise.all([
        imageFromUrl(`/files/${row.image_path}`),
        imageFromUrl(`/files/${row.target_mask_path}`),
        imageFromUrl(`/files/${row.plug_mask_path}`),
      ]);
      state.image = img;
      state.targetMask = target;
      state.plugMask = plug;
      state.width = img.width;
      state.height = img.height;
      canvas.width = img.width;
      canvas.height = img.height;
      reviewedBox.checked = Boolean(row.review_done);
      overlayLink.href = `/files/${row.overlay_path}`;
      overlayPreview.src = `/files/${row.overlay_path}`;
      metaBox.textContent = JSON.stringify(row, null, 2);
      state.dirty = false;
      redraw();
      setProgressLine();
      setStatus(`Loaded ${index + 1}/${state.manifest.length}`);
    }

    function redraw() {
      if (!state.image) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(state.image, 0, 0);
      if (showTarget.checked && state.targetMask) {
        ctx.save();
        ctx.globalAlpha = 0.35;
        ctx.drawImage(state.targetMask, 0, 0);
        ctx.restore();
      }
      if (showPlug.checked && state.plugMask) {
        ctx.save();
        ctx.globalAlpha = 0.35;
        ctx.drawImage(state.plugMask, 0, 0);
        ctx.restore();
      }
    }

    function drawStroke(x, y) {
      const layer = state.activeLayer === "target" ? state.targetMask : state.plugMask;
      const off = document.createElement("canvas");
      off.width = state.width;
      off.height = state.height;
      const offCtx = off.getContext("2d");
      offCtx.drawImage(layer, 0, 0);
      offCtx.save();
      offCtx.fillStyle = state.erase ? "rgba(0,0,0,1)" : "rgba(255,255,255,1)";
      offCtx.globalCompositeOperation = state.erase ? "destination-out" : "source-over";
      offCtx.beginPath();
      offCtx.arc(x, y, state.brushSize, 0, Math.PI * 2);
      offCtx.fill();
      offCtx.restore();
      const img = new Image();
      img.onload = () => {
        if (state.activeLayer === "target") state.targetMask = img;
        else state.plugMask = img;
        state.dirty = true;
        redraw();
      };
      img.src = off.toDataURL("image/png");
    }

    function pointerPos(evt) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = state.width / rect.width;
      const scaleY = state.height / rect.height;
      return {
        x: Math.max(0, Math.min(state.width - 1, Math.round((evt.clientX - rect.left) * scaleX))),
        y: Math.max(0, Math.min(state.height - 1, Math.round((evt.clientY - rect.top) * scaleY))),
      };
    }

    async function saveCurrent() {
      if (!state.image) return;
      const row = state.manifest[state.index];
      const payload = {
        image_path: row.image_path,
        target_mask_png: imageToPng(state.targetMask),
        plug_mask_png: imageToPng(state.plugMask),
      };
      const res = await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        setStatus(`Save failed: ${await res.text()}`);
        return;
      }
      state.dirty = false;
      setStatus(`Saved ${row.image_path}`);
    }

    async function saveProgressFlag(reviewDone) {
      const row = state.manifest[state.index];
      const payload = {
        image_path: row.image_path,
        review_done: Boolean(reviewDone),
      };
      const res = await fetch("/api/progress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        setStatus(`Progress save failed: ${await res.text()}`);
        return;
      }
      row.review_done = Boolean(reviewDone);
      refreshSelectOption(state.index);
      setProgressLine();
      setStatus(`Marked ${row.image_path} as ${reviewDone ? "reviewed" : "not reviewed"}`);
    }

    function imageToPng(img) {
      const off = document.createElement("canvas");
      off.width = state.width;
      off.height = state.height;
      const offCtx = off.getContext("2d");
      offCtx.clearRect(0, 0, off.width, off.height);
      offCtx.drawImage(img, 0, 0);
      return off.toDataURL("image/png");
    }

    canvas.addEventListener("pointerdown", (evt) => {
      state.drawing = true;
      const {x, y} = pointerPos(evt);
      drawStroke(x, y);
    });
    canvas.addEventListener("pointermove", (evt) => {
      if (!state.drawing) return;
      const {x, y} = pointerPos(evt);
      drawStroke(x, y);
    });
    window.addEventListener("pointerup", () => { state.drawing = false; });

    document.getElementById("prevBtn").onclick = async () => {
      if (state.dirty) await saveCurrent();
      await loadFrame(Math.max(0, state.index - 1));
    };
    document.getElementById("nextBtn").onclick = async () => {
      if (state.dirty) await saveCurrent();
      await loadFrame(Math.min(state.manifest.length - 1, state.index + 1));
    };
    document.getElementById("saveBtn").onclick = saveCurrent;
    targetBtn.onclick = () => { state.activeLayer = "target"; state.erase = false; setButtons(); };
    plugBtn.onclick = () => { state.activeLayer = "plug"; state.erase = false; setButtons(); };
    eraseBtn.onclick = () => { state.erase = !state.erase; setButtons(); };
    brushSize.oninput = () => {
      state.brushSize = Number(brushSize.value);
      brushSizeValue.textContent = String(state.brushSize);
    };
    showTarget.onchange = redraw;
    showPlug.onchange = redraw;
    frameSelect.onchange = async () => {
      if (state.dirty) await saveCurrent();
      await loadFrame(Number(frameSelect.value));
    };
    reviewedBox.onchange = async () => { await saveProgressFlag(reviewedBox.checked); };
    window.addEventListener("keydown", async (evt) => {
      if (evt.target && ["INPUT", "SELECT", "TEXTAREA"].includes(evt.target.tagName)) return;
      if (evt.key === "s") {
        evt.preventDefault();
        await saveCurrent();
      } else if (evt.key === "ArrowRight") {
        evt.preventDefault();
        if (state.dirty) await saveCurrent();
        await loadFrame(Math.min(state.manifest.length - 1, state.index + 1));
      } else if (evt.key === "ArrowLeft") {
        evt.preventDefault();
        if (state.dirty) await saveCurrent();
        await loadFrame(Math.max(0, state.index - 1));
      } else if (evt.key === "1") {
        state.activeLayer = "target";
        state.erase = false;
        setButtons();
      } else if (evt.key === "2") {
        state.activeLayer = "plug";
        state.erase = false;
        setButtons();
      } else if (evt.key.toLowerCase() === "e") {
        state.erase = !state.erase;
        setButtons();
      }
    });

    setButtons();
    loadManifest().catch((err) => setStatus(String(err)));
  </script>
</body>
</html>
"""


class CorrectionHandler(BaseHTTPRequestHandler):
    root_dir: Path
    manifest_rows: list[dict[str, Any]]
    progress_path: Path
    progress_state: dict[str, Any]

    def _send_bytes(self, payload: bytes, *, status: int = 200, content_type: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json_response(self, payload: Any, *, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"),
            status=status,
            content_type="application/json",
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(INDEX_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/manifest":
            payload: list[dict[str, Any]] = []
            for row in self.manifest_rows:
                rel_image = Path(str(row["image_path"])).relative_to("images")
                key = _progress_key(row)
                payload.append(
                    {
                        **row,
                        "target_mask_path": str(Path("masks") / "target" / rel_image),
                        "plug_mask_path": str(Path("masks") / "plug" / rel_image),
                        "overlay_path": str(Path("overlays") / rel_image),
                        "review_done": bool(self.progress_state.get(key, {}).get("review_done", False)),
                    }
                )
            self._json_response(payload)
            return
        if parsed.path.startswith("/files/"):
            rel_path = Path(unquote(parsed.path.removeprefix("/files/")))
            file_path = (self.root_dir / rel_path).resolve()
            try:
                file_path.relative_to(self.root_dir.resolve())
            except ValueError:
                self._send_bytes(b"Forbidden", status=HTTPStatus.FORBIDDEN)
                return
            if not file_path.is_file():
                self._send_bytes(b"Not found", status=HTTPStatus.NOT_FOUND)
                return
            content_type = "image/png" if file_path.suffix.lower() == ".png" else "application/octet-stream"
            self._send_bytes(file_path.read_bytes(), content_type=content_type)
            return
        self._send_bytes(b"Not found", status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/save", "/api/progress"}:
            self._send_bytes(b"Not found", status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body.decode("utf-8"))
        image_rel = Path(str(payload["image_path"])).relative_to("images")

        if parsed.path == "/api/progress":
            key = str(payload["image_path"])
            self.progress_state[key] = {
                "review_done": bool(payload.get("review_done", False)),
            }
            self.progress_path.write_text(
                json.dumps(self.progress_state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._json_response({"ok": True})
            return

        target_rel = Path("masks") / "target" / image_rel
        plug_rel = Path("masks") / "plug" / image_rel
        for rel_path, key in ((target_rel, "target_mask_png"), (plug_rel, "plug_mask_png")):
            data_url = str(payload[key])
            if "," not in data_url:
                self._send_bytes(b"Bad PNG data", status=HTTPStatus.BAD_REQUEST)
                return
            _, encoded = data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            out_path = self.root_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(image_bytes)

        self._json_response({"ok": True})


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest is not None
        else _default_manifest(root)
    )
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    rows = _load_jsonl(manifest_path)
    if not rows:
        raise SystemExit(f"Manifest is empty: {manifest_path}")
    progress_path = root / "correction_progress.json"
    progress_state = {}
    if progress_path.is_file():
        progress_state = json.loads(progress_path.read_text(encoding="utf-8"))

    handler_cls = type(
        "BoundCorrectionHandler",
        (CorrectionHandler,),
        {
            "root_dir": root,
            "manifest_rows": rows,
            "progress_path": progress_path,
            "progress_state": progress_state,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(
        json.dumps(
            {
                "root": str(root),
                "manifest": str(manifest_path),
                "rows": len(rows),
                "url": f"http://{args.host}:{args.port}/",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
