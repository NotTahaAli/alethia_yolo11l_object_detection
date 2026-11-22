# Video Indexing with YOLO11l

For every second of every video in a folder of [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
Parquet files, `run.py` records which of five objects (person, car, dog, laptop, cell phone) appear and how many
people are visible, using Ultralytics YOLO11l ([`yolo11l.pt`](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11l.pt)).
It runs on NVIDIA GPUs (TensorRT or PyTorch), on Apple Silicon (GPU and Neural Engine together) and on CPU.
The assignment is described in `AI_Engineering_TakeHome_Brief.pdf`.

## Setup

Requires Python 3.11. No `apt-get` packages are needed.

```sh
pip install -r requirements.txt
```

On Linux this installs the CUDA 12 build of PyTorch and TensorRT 10, from the PyTorch and NVIDIA package indexes
listed at the top of `requirements.txt`. On macOS it installs the standard PyTorch build and `coremltools`.

### Preparation command

Run once from the repository root, with internet access, on the machine that will run `run.py`:

```sh
hf download hf://Ultralytics/YOLO11@bee92e2/yolo11l.pt --local-dir yolo && python build_engine.py
```

1. `hf download` fetches the official weights to `yolo/yolo11l.pt`.
2. `build_engine.py` (NVIDIA only) builds a TensorRT FP16 engine at `yolo/yolo11l.engine`. Engines only work on the
   GPU model and TensorRT version that built them, so build it on the machine that runs `run.py`. On a machine without
   CUDA it skips the build with a message, and `run.py` uses `yolo11l.pt`.

On Apple Silicon there is nothing to build: CoreML models are exported to `yolo/coreml/` the first time each input
shape is seen (about a minute each, then cached).

## Running

```sh
python run.py --input <folder of .parquet files> --output results.jsonl
```

| Option | Effect |
| --- | --- |
| `--workers B,B,...` | Backend of each worker process (see below). Default `cuda,cuda` with CUDA, `mps,mps,coreml,coreml` on Apple Silicon, `cpu` otherwise |
| `--model FILE` | Model for `cuda`, `mps` and `cpu` workers. Default `yolo/yolo11l.engine` on CUDA if built, else `yolo/yolo11l.pt` |
| `--fp32` | Run `.pt` models in FP32 instead of FP16, e.g. to make a reference |
| `--max-files N`, `--max-rows N` | Only process the first N Parquet files, or the first N videos of each file |
| `--dev` | Same as `--max-files 1 --max-rows 1` |

| Backend | Runs | Model |
| --- | --- | --- |
| `cuda` | NVIDIA GPU | TensorRT FP16 engine if built, else `yolo11l.pt` in FP16 |
| `mps` | Apple GPU | `yolo11l.pt` in FP16 |
| `coreml` | Apple Neural Engine | FP16 CoreML model per input shape |
| `cpu` | CPU | `yolo11l.pt` in FP32 |

## How it works

```text
main process                       worker process (one per --workers entry)
------------                       -----------------------------------------
read Parquet rows  --- video --->  decoder thread: decode, rotate, resize ---> frames
write result lines <--- lines ---  main thread:    batch, pad + normalise on device, YOLO, NMS, per-second results
```

1. **Reading.** The main process streams each Parquet file one row at a time and puts every video on a task queue.
   A file that cannot be read gets an error line for the row it failed on.
2. **Decoding** (worker, background thread). PyAV decodes each frame to BGR, applies the rotation tag, and assigns
   the frame to second `floor((pts - first_pts) * time_base)`. Frames that need resizing for the letterbox are resized
   here with the same OpenCV call Ultralytics uses, so the model thread never waits for it.
3. **Inference** (worker, main thread). Frames of the same size are batched (up to 64, across seconds) and padded,
   converted to RGB and normalised on the device, then run through the model and Ultralytics' own
   `non_max_suppression`. This is exactly what `YOLO.predict` does, step for step, without its per-call overhead:
   the output is byte-identical. CoreML workers pad on the CPU and hand the Neural Engine uint8 images directly,
   which are the same pixels Ultralytics' CoreML backend would rebuild from the normalised tensor.
4. **Results.** A second's line is written once all its frames are scored. Each worker sends its lines to the main
   process, which writes the single output file. The main process ends the run: once every video it fed is finished,
   it tells each worker to stop.
5. **Failures.** A video that fails to decode keeps the lines already written and gets one error line. If a worker
   process dies (for example the decoder crashes on a damaged file), its unfinished videos get error lines and a new
   worker takes over.

### Model settings

As the brief requires: input size 640 with Ultralytics' minimal ("rect") letterbox, confidence 0.4, class-aware NMS
with IoU 0.7 (set explicitly: Ultralytics 8.4 `predict` otherwise falls back to 0.5), BGR frames, and rotation tags
applied. Detection runs only for the five classes.

### Why it is fast

On an Apple M5 Pro the default runs at about 430 frames per second over a whole Parquet file, about 5x the first
working version, with output
matching full-precision PyTorch well inside the brief's thresholds. [`OPTIMISATION_LOG.md`](OPTIMISATION_LOG.md) has
every change that was tried, in order, with measurements.

- **No per-call overhead.** Padding and normalisation run on the device; the model and NMS are called directly.
- **Overlapped work.** Decoding and resizing run in a thread alongside inference, and several worker processes keep
  the accelerator busy while others do CPU work.
- **FP16 everywhere.** TensorRT FP16 on NVIDIA (optimised for the most common batch, 64 frames at 384x640), FP16
  PyTorch on Apple GPUs, FP16 CoreML on the Neural Engine.
- **Apple GPU and Neural Engine together.** They only run in parallel from separate processes, hence the
  `mps,mps,coreml,coreml` default. CoreML models are static, so there is one per letterbox shape, which keeps the
  padding identical to the `.pt` model.
- **Pinned memory on CUDA.** Frames are stacked straight into a reusable page-locked buffer and copied to the GPU
  asynchronously.

## Checking correctness

`compare.py` scores a results file against a reference with the brief's three checks and exits non-zero on failure.
To check the fast path against full-precision PyTorch on a few videos:

```sh
python run.py --input <folder> --output reference.jsonl --workers cpu --max-rows 3        # or --workers cuda --model yolo/yolo11l.pt --fp32
python run.py --input <folder> --output fast.jsonl --max-rows 3
python compare.py reference.jsonl fast.jsonl
```

## Output format

One JSON line per second of each video, in no particular order:

```jsonl
{"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people": 2}
{"video": "train-00003-of-01357.parquet:4", "second": 13, "objects": [], "people": 0}
{"video": "train-00003-of-01357.parquet:5", "error": "could not decode video"}
```

- `video` is the Parquet file name without its folder, a colon, and the row number in that file, counting from 0.
- Second 12 holds the frames whose timestamp is at least 12.0 and below 13.0 seconds, counted from the first decoded
  frame of the video. Every second up to and including the one that holds the last frame has a line.
- An object is listed if the model finds it in at least half of that second's frames.
- `people` is the median number of people found per frame in that second, rounded down when it falls halfway.
- A video that cannot be processed gets one error line; seconds already written for it are kept.

## Files

| File | Purpose |
| --- | --- |
| `run.py` | The pipeline |
| `build_engine.py` | Builds the TensorRT FP16 engine (preparation command) |
| `compare.py` | Scores results against a reference |
| `OPTIMISATION_LOG.md` | Every optimisation tried, in order, with measurements |
| `requirements.in` | Direct dependencies; `requirements.txt` is generated from it (command at its top) |
