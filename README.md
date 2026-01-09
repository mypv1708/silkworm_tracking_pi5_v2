## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run

Defaults are read from `config.yaml`.

### PyTorch (`.pt`)

```bash
python3 main.py --benchmark --model best.pt
```

### ONNX (`.onnx`) (recommended on Raspberry Pi)

Ultralytics may not infer the task from exported formats like ONNX, so `--task pose` (or `model.task: "pose"`)
is required for `result.keypoints` to be available.

**Mode 1: Pose tracking (default)** - Full tracking with keypoints, freeze detection, and overlap detection:
```bash
python3 main.py --benchmark --model best.onnx
```

**Mode 2: Heatmap visualization** - Detect-only mode with density heatmap overlay (no pose/tracking):
```bash
python3 main.py --benchmark --model best.onnx --mode heatmap
```