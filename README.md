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

```bash
python3 main.py --benchmark --model best.onnx
```