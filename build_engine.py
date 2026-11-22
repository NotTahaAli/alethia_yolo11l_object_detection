"""Build the TensorRT FP16 engine that run.py uses on CUDA.

    python build_engine.py

Run once on the machine that will run run.py: TensorRT engines only work on the GPU and TensorRT version they
were built with. Needs an NVIDIA GPU and yolo/yolo11l.pt (see README).
"""
import os

import torch
from ultralytics import YOLO

from run import BATCH_SIZE, ENGINE_MODEL_FILE, MODEL_INPUT_SIZE, PT_MODEL_FILE


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("TensorRT export needs a CUDA GPU.")
    exported = YOLO(PT_MODEL_FILE, task="detect").export(
        format="engine",
        imgsz=MODEL_INPUT_SIZE,
        quantize=16,  # FP16
        # Dynamic shapes keep Ultralytics' minimal letterbox (e.g. 384x640 for 16:9), exactly as with the .pt model.
        # A static engine would pad every frame to 640x640 and drift from the reference results.
        dynamic=True,
        batch=BATCH_SIZE,  # max batch of the optimisation profile; run.py never sends more
        simplify=False,  # TensorRT fuses the graph itself; skips the onnxslim/onnxruntime dependencies
        device=0,
    )
    if os.path.abspath(str(exported)) != os.path.abspath(ENGINE_MODEL_FILE):
        raise SystemExit(f"Engine written to {exported}, expected {ENGINE_MODEL_FILE}")
    print(f"Built {ENGINE_MODEL_FILE}")


if __name__ == "__main__":
    main()
