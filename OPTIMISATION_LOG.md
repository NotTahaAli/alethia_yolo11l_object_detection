# Optimisation log

Everything tried to make `run.py` faster, in the order it was tried, with how it performed and whether it was kept.

**Machines.** Apple M5 Pro (18 CPU cores, 20 GPU cores, 16 Neural Engine cores) unless stated otherwise. One NVIDIA
data point comes from a Kaggle T4. The grader's RTX 4090 was not available, so the NVIDIA-only changes (marked
*untested*) are measured nowhere yet.

**Test data.** `train-00000-of-01357.parquet` from FineVideo:

- *dev clip*: row 0, 209 s, 4,994 frames, 640x360
- *3 videos*: rows 0–2, 1,082 seconds of results, 29,782 frames
- *10 videos*: rows 0–9, 83,826 frames
- *damaged fixtures*: a good video, garbage bytes, a null cell, a video cut in half, and a corrupt Parquet file
- *shape fixtures*: the shortest local video for each of the 17 letterbox input shapes in the first 3 files

**Correctness.** `compare.py` scores a run against a reference with the brief's checks: object F1 ≥ 0.95, people
exact ≥ 85 %, people within 1 ≥ 95 %. The reference is FP32 PyTorch with IoU 0.7 (see step 12).

## 1. Starting point

Per-second `model.predict` on one second of frames at a time, FP32 on MPS.

| Measure | Result |
|---|---|
| dev clip | 60 s (about 83 fps) |

## 2. Is decoding a bottleneck?

Decoded the dev clip with PyAV, single-threaded and with `thread_type = "AUTO"`, plus conversion to BGR arrays.

| Measure | Result |
|---|---|
| decode only, 1 thread | 0.93 s |
| decode, threaded, + `to_ndarray` | 0.34 s (about 15,000 fps) |

**Kept:** threaded decoding. Decoding is negligible next to inference (about 0.6 % of the time).

## 3. Decode once, bucket frames by timestamp

Replaced the frame-count arithmetic with one decode pass that puts each frame in second
`floor((pts - first_pts) * time_base)`. A correctness fix more than a speed change.

| Measure | Result |
|---|---|
| dev clip | 56.6 s |

**Kept.**

## 4. Decoder thread + batches across seconds

A background thread decodes while the main thread runs YOLO on batches of up to 64 frames that can span seconds.

| Measure | Result |
|---|---|
| dev clip | 57.4 s (no change: the Mac GPU was the limit) |
| 3 videos vs previous version | identical output (1,082 seconds, F1 1.0) |

**Kept:** it costs nothing here and matters when the GPU is faster than one-second batches can feed it.

## 5. Batch size on MPS

| Batch | Time for 600 frames |
|---|---|
| 30 | 6.4 s |
| 120 | 6.7 s |

**No effect on MPS.** Kept 64, which suits the TensorRT engine.

## 6. FP16 on the Apple GPU

| Measure | FP32 | FP16 |
|---|---|---|
| raw model throughput | 91 fps | 205 fps |
| 3 videos | 330 s | 150 s |
| vs FP32 reference | | F1 0.9986, people exact 1.0, within 1 1.0 |

**Kept:** 2.2x faster, well inside the thresholds.

## 7. CoreML on the Neural Engine

Exported FP16 CoreML at 384x640 (batch 1, static).

| Measure | Result |
|---|---|
| through Ultralytics `predict` | 142 fps |
| raw `coremltools`, `CPU_AND_NE` | 181 fps |
| raw `coremltools`, `ALL` | 187 fps |
| 3 videos vs FP32 reference | F1 1.0, people exact 0.982, within 1 1.0 |

Batch > 1 needs a dynamic export, which loses the Neural Engine, so each image is its own request.
**Kept** as a second backend (step 9).

## 8. GPU and Neural Engine in one process (threads)

| Measure | Result |
|---|---|
| MPS + CoreML threads, Ultralytics | 200 fps total (no better than MPS alone) |
| raw model, GPU thread + ANE thread | GPU 25 fps + ANE 179 fps = 204 fps |

**Rejected:** the GPU is starved when both run in one process.

## 9. GPU and Neural Engine as separate processes

Raw model throughput, processes running at the same time:

| Processes | Total |
|---|---|
| 1 GPU | 196 fps |
| 1 GPU + 1 ANE | 196 + 165 = 361 fps |
| 2 GPU | 129 + 130 = 259 fps |
| 1 GPU + 2 ANE | 194 + 105 + 107 = 406 fps |

**Kept:** led to the multi-process design (step 10).

## 10. Worker processes

The main process reads Parquet files and hands videos to worker processes, each with its own backend, decoder
thread and batching. CoreML workers export one static model per letterbox shape so padding matches the `.pt` model.

10 videos:

| `--workers` | Time | Throughput |
|---|---|---|
| `mps,coreml` | 282 s | 297 fps |
| `mps,mps,coreml` | 312 s | 269 fps |
| `mps,mps,coreml,coreml` | 253 s | 331 fps |

| Check | Result |
|---|---|
| 3 videos vs FP32 baseline | F1 0.9993, people exact 0.998 |
| damaged fixtures | same error lines as before |
| worker killed mid-video (`kill -9`) | its video kept 5 seconds + 1 error line, a new worker took over, run finished |

**Kept,** default `mps,mps,coreml,coreml`. It also stops a crashing decoder from killing the whole run.

## 11. More worker processes

Raw model throughput, any mix of 4–6 GPU and ANE processes: **407–466 fps**, varying about ±10 % between identical
runs. Full pipeline, 10 videos:

| `--workers` | Time |
|---|---|
| `mps,mps,mps,coreml,coreml` | 271 s |
| `mps,mps,coreml,coreml` (rerun) | 295 s (253 s earlier) |

**No reliable gain.** The GPU, Neural Engine and CPU share one memory system and power budget; the difference between
mixes is smaller than run-to-run noise (heat). Kept 4 workers.

## 12. Correctness fix: NMS IoU 0.7

Ultralytics 8.4 `predict` uses IoU **0.5** when `iou` is not set; the brief's reference uses 0.7. Every earlier run
used 0.5. Now set explicitly.

| 3 videos, IoU 0.5 vs IoU 0.7 (both FP32) | Result |
|---|---|
| objects F1 | 1.0 |
| people exact | 0.9963 (0.4 % of seconds changed) |

**Fixed.** All later references use IoU 0.7.

## 13. Preprocess on the device, call the model directly

Reproduces `YOLO.predict` step for step without its per-call overhead: resize in the decoder thread with
Ultralytics' exact OpenCV call; pad with 114, BGR→RGB and normalise on the device; the same `AutoBackend`; and
Ultralytics' `non_max_suppression` with the predictor's arguments.

| Check | Result |
|---|---|
| 3 videos, MPS FP16, vs `YOLO.predict` path | byte-identical (0 differing lines) |
| 3 videos, CoreML, vs `YOLO.predict` path | byte-identical |
| damaged fixtures | byte-identical |

The first measurement was **slower**: one MPS worker on 3 videos took 181 s against 148 s. Timing each stage inside the
worker showed NMS taking 57 s. Ultralytics only uses torchvision's NMS kernel if `torchvision` has already been
imported; `YOLO.predict` imports it during warm-up, the direct path did not, so NMS fell back to a slow pure-PyTorch
implementation (same results). Fixed by importing torchvision.

One worker, 3 videos:

| Backend | `YOLO.predict` path | Direct path |
|---|---|---|
| MPS | 148 s | 144 s (−3 %) |
| CoreML | 214 s | 173 s (−19 %) |

Padding order on MPS (permute → pad vs pad → permute): 6.3 vs 6.5 ms per batch, no difference.

**Kept.** The Mac GPU is the bottleneck, so MPS gains little; CoreML gains because its CPU-side work shrank. On CUDA
the GPU is much faster, so removing CPU work should matter more there (*untested*).

## 14. Pinned memory on CUDA (*untested*)

Frames are stacked straight into a reusable page-locked buffer and copied with `non_blocking=True`, removing one
~45 MB copy per batch. PyTorch's guidance: pin a buffer once and reuse it (pinning on every call is slower) and do not
reuse it until the copy has finished; here the detections are read back before the next batch, which waits for it.

## 15. TensorRT engine shape (*untested*)

TensorRT tunes kernels for the optimisation profile's "opt" shape. Ultralytics' default export made that 64x640x640
with a maximum of 64x1280x1280; real batches are almost all 64x384x640. Now built with `imgsz=(384, 640)`: opt
64x384x640, max 64x768x1280 (still covers every input shape). Expected: faster kernels for the common case and less
build memory. On a Kaggle T4 the old maximum caused many "skipping tactic, insufficient memory" messages.

## 16. Exactness across every input shape

The first 3 local files contain 65 resolutions, which letterbox to 17 model input shapes; 11 of them need a resize.
Ran the shortest video of each shape (1,450 seconds) through both paths.

| Check | Result |
|---|---|
| MPS: `YOLO.predict` path vs direct path | byte-identical |
| CoreML: `YOLO.predict` path vs direct path | byte-identical |
| CoreML vs MPS | F1 0.995, people exact 0.992, within 1 1.0 |

The first run on a Mac exports one CoreML model per new shape (about a minute each, 16 for this data), then caches it.

## 17. Correctness fix: when workers stop

Workers exited once the feeder had finished and the task queue looked empty. `multiprocessing.Queue.empty()` only
checks the pipe, so the last video could still be in the feeder's buffer and be dropped without an error line. Now the
main process counts videos fed and finished, and sends each worker a stop signal only when they match.

| Check | Result |
|---|---|
| damaged fixtures | identical output |
| folder with only a corrupt file | error line, exits in 1.7 s |
| more workers than videos | complete output |
| worker killed mid-video | 117 seconds kept + 1 error line, other videos complete |

## 18. CoreML: uint8 images straight to the Neural Engine

Ultralytics' CoreML backend turns the normalised float tensor back into one uint8 image per request. Building those
images directly (`copyMakeBorder` + BGR→RGB) skips two float conversions of the batch. Checked first that uint8 → /255
→ ×255 round-trips exactly in float32 for all 256 values, so the pixels are identical.

| Check | Result |
|---|---|
| 3 videos and all 17 shapes vs previous CoreML path | byte-identical |
| one CoreML worker, 3 videos | 173 s → 163 s (−6 %) |

**Kept.**

## 19. Worker mix, re-tuned with all of the above

10 videos (83,826 frames):

| `--workers` | Time | Throughput |
|---|---|---|
| `mps,mps,coreml,coreml` | 238.6 s | 351 fps |
| `mps,mps,mps,coreml,coreml` | 238.0 s | 352 fps |
| `mps,mps,coreml,coreml,coreml` | 246.7 s | 340 fps |

All three agree (F1 0.9998). **Kept** `mps,mps,coreml,coreml`: as fast as five workers with one process less.
Up from 331 fps in step 10, and about 4.2x the starting point (step 1).

## 20. Faster model execution on the Apple GPU

Raw model, batch 64 at 384x640, FP16:

| Variant | Throughput | Output vs standard |
|---|---|---|
| standard (contiguous) | 246 fps | |
| `channels_last` memory format | 149 fps | max abs diff 1.6 |
| `torch.compile` | 190 fps | max abs diff 2.3 |

**Rejected:** both slower on MPS, and both change the model's numbers.

## 21. Considered and not done

| Idea | Why not |
|---|---|
| Hardware video decoding (NVDEC / VideoToolbox) | Every video is H.264 at 640 px or less; decoding is 0.6 % of the time |
| Resizing on the GPU | GPU resizing differs slightly from OpenCV's, so results would drift from the reference; only 22 % of videos need a resize and it now runs in the decoder thread anyway |
| NVIDIA Multi-Process Service | Lets kernels from several processes overlap, but needs a daemon running on the grader's VM |
| CUDA graphs | Do not fit dynamic input shapes |
| Skipping frames | The reference uses every frame; would trade accuracy for speed |
| A CPU worker on the Mac | YOLO11l on CPU manages roughly 5–10 fps and takes power from the GPU |

## Other data points

| Machine | Setup | Result |
|---|---|---|
| Kaggle T4 | `.pt` FP16, 1 worker, dev clip | a little over a minute (about 75 fps) |
