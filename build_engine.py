"""Build the TensorRT FP16 engine that run.py uses on CUDA.

    python build_engine.py

Run once on the machine that will run run.py: TensorRT engines only work on the GPU and TensorRT version they
were built with. Needs an NVIDIA GPU and yolo/yolo11l.pt (see README).
"""
import os

import torch
from ultralytics import YOLO

from run import BATCH_SIZE, ENGINE_MODEL_FILE, PT_MODEL_FILE

# TensorRT tunes its kernels for the optimisation profile's "opt" shape, which Ultralytics sets to the export shape:
# a full batch of 16:9 landscape frames (640x360 letterboxed to 384x640), by far the most common input in FineVideo.
# Ultralytics sets the profile's max to twice that (768x1280), which still covers every rect shape run.py sends
# (at most 640 on each side), and the min to 32x32.
OPT_INPUT_SHAPE = (384, 640)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("TensorRT export needs a CUDA GPU.")
    exported = YOLO(PT_MODEL_FILE, task="detect").export(
        format="engine",
        imgsz=OPT_INPUT_SHAPE,
        quantize=16,  # FP16
        # Dynamic shapes let each video keep Ultralytics' minimal letterbox, exactly as with the .pt model.
        # A static engine would pad every frame to one size and drift from the reference results.
        dynamic=True,
        batch=BATCH_SIZE,  # batch of the opt and max shapes; run.py never sends more
        simplify=False,  # TensorRT fuses the graph itself; skips the onnxslim/onnxruntime dependencies
        device=0,
    )
    if os.path.abspath(str(exported)) != os.path.abspath(ENGINE_MODEL_FILE):
        raise SystemExit(f"Engine written to {exported}, expected {ENGINE_MODEL_FILE}")
    print(f"Built {ENGINE_MODEL_FILE}")


if __name__ == "__main__":
    main()
