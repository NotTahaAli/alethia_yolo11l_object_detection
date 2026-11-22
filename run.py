"""YOLO11l object detection over every second of every video in a folder of FineVideo Parquet files.

    python run.py --input <folder of .parquet files> --output results.jsonl

The main process reads the Parquet files and hands each video to a pool of worker processes. Each worker runs one
inference backend (CUDA, MPS, CoreML or CPU); inside a worker a thread decodes videos into BGR frames while the main
thread batches them through YOLO. Workers send result lines back and the main process writes them to one file.
If a worker process dies (e.g. a damaged video crashes the decoder), its unfinished videos get error lines and a
replacement worker is started.
"""
import argparse
import functools
import io
import json
import multiprocessing as mp
import os
import queue
import shutil
import statistics
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from time import time
from typing import Literal, TypeAlias

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision  # noqa: F401  Ultralytics' NMS uses torchvision's fast kernel only once it is imported
from av.video.frame import VideoFrame
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.nms import non_max_suppression

PT_MODEL_FILE = "yolo/yolo11l.pt"
ENGINE_MODEL_FILE = "yolo/yolo11l.engine"  # built by build_engine.py
COREML_DIR = "yolo/coreml"  # CoreML models, one per letterboxed input shape, exported on first use
MODEL_INPUT_SIZE = 640
MODEL_STRIDE = 32
MODEL_CONFIDENCE_THRESHOLD = 0.4
# The brief's reference uses IoU 0.7. Set it explicitly: Ultralytics 8.4 predict falls back to 0.5 when unset.
NMS_IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300  # Ultralytics default
LETTERBOX_PADDING_VALUE = 114  # Ultralytics default
OBJECT_DETECTION_CLASSES = ("person", "car", "dog", "laptop", "cell phone")
# Frames per model call. The TensorRT engine is built with this as its max batch, so keep them in sync.
BATCH_SIZE = 64
FRAME_QUEUE_SIZE = 4 * BATCH_SIZE  # decoded frames buffered ahead of the model, per worker
ERROR_MESSAGE = "could not decode video"

Image: TypeAlias = NDArray[np.uint8]
Backend: TypeAlias = Literal["cuda", "mps", "coreml", "cpu"]
BACKENDS: tuple[Backend, ...] = ("cuda", "mps", "coreml", "cpu")
Shape: TypeAlias = tuple[int, int]


# ---------------------------------------------------------------------------------------------------------------------
# Inference
#
# Predictor reproduces what YOLO.predict does for a list of BGR frames, step for step, but cheaper:
#   letterbox  - Ultralytics' minimal ("rect") letterbox: resize with the same OpenCV call (in the decoder thread, see
#                prepare_frame), then pad with 114 to a multiple of the stride (on the device instead of the CPU)
#   normalise  - BHWC uint8 -> BCHW, BGR -> RGB, FP16/FP32, / 255, in the same order as Predictor.preprocess
#   model      - the same AutoBackend the predictor builds
#   NMS        - ultralytics.utils.nms.non_max_suppression with the arguments DetectionPredictor.postprocess passes
# so the model sees identical input and the detections are identical, without YOLO.predict's per-call setup, CPU
# padding and Results objects.


@dataclass(frozen=True, slots=True)
class Letterbox:
    """Ultralytics' rect letterbox for one frame size: resize to `resized`, then pad to `padded` (both height, width)."""
    resized: Shape
    padded: Shape
    top: int
    bottom: int
    left: int
    right: int


@functools.lru_cache(maxsize=None)
def letterbox(height: int, width: int) -> Letterbox:
    params = LetterBox(MODEL_INPUT_SIZE, auto=True, stride=MODEL_STRIDE).get_params(
        {"img": np.empty((height, width, 3), np.uint8)})
    new_width, new_height = params["new_unpad"]
    top, bottom, left, right = params["top"], params["bottom"], params["left"], params["right"]
    return Letterbox((new_height, new_width), (new_height + top + bottom, new_width + left + right),
                     top, bottom, left, right)


def coreml_model(shape: Shape) -> str:
    """Path of the FP16 CoreML model for this input shape, exporting it on first use.
    A static model at exactly the rect shape letterboxes identically to the .pt model."""
    path = os.path.join(COREML_DIR, f"yolo11l_{shape[0]}x{shape[1]}.mlpackage")
    if os.path.exists(path):
        return path
    os.makedirs(COREML_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=COREML_DIR) as tmp:  # export privately, then rename: safe across workers
        pt_copy = shutil.copy(PT_MODEL_FILE, tmp)
        exported = YOLO(pt_copy, task="detect").export(format="coreml", imgsz=shape, quantize=16, nms=False)
        try:
            os.rename(str(exported), path)
        except OSError:
            if not os.path.exists(path):  # lost a race to another worker otherwise
                raise
    return path


class Predictor:
    """Runs YOLO on batches of same-shape, already resized frames for one backend."""

    def __init__(self, backend: Backend, model_file: str | None, fp16: bool):
        self.backend = backend
        self.device = torch.device({"cuda": "cuda:0", "mps": "mps", "coreml": "cpu", "cpu": "cpu"}[backend])
        self.fp16 = fp16 and backend in ("cuda", "mps")  # applies to .pt models; engines and CoreML set their own
        self.pinned: dict[tuple[int, ...], torch.Tensor] = {}  # CUDA: reusable page-locked upload buffer per shape
        self.coreml_models: dict[Shape, AutoBackend] = {}  # CoreML: one static model per input shape
        if backend == "coreml":
            self.model_file = "CoreML"
            model = self._model_for(letterbox(360, 640).padded)  # the most common shape; exported on first run
        else:
            self.model_file = model_file or (ENGINE_MODEL_FILE if backend == "cuda" and os.path.exists(ENGINE_MODEL_FILE)
                                             else PT_MODEL_FILE)
            model = self.model = self._load(self.model_file)
        names: dict[int, str] = model.names
        name_to_id = {name: class_id for class_id, name in names.items()}
        missing = [name for name in OBJECT_DETECTION_CLASSES if name not in name_to_id]
        if missing:
            raise ValueError(f"Classes {missing} not found in the model's class names.")
        self.names = names
        self.class_ids = [name_to_id[name] for name in OBJECT_DETECTION_CLASSES]

    def _load(self, path: str) -> AutoBackend:
        model = AutoBackend(path, device=str(self.device), fp16=self.fp16, verbose=False)
        model.eval()
        return model

    def _model_for(self, padded: Shape) -> AutoBackend:
        if self.backend != "coreml":
            return self.model
        if padded not in self.coreml_models:
            self.coreml_models[padded] = self._load(coreml_model(padded))
        return self.coreml_models[padded]

    def _upload(self, images: list[Image]) -> torch.Tensor:
        """Stack the frames into one BHWC uint8 tensor on the device."""
        if self.device.type != "cuda":
            return torch.from_numpy(np.stack(images)).to(self.device)
        # Stack straight into a page-locked buffer so the copy to the GPU is a single asynchronous DMA.
        shape = images[0].shape
        if shape not in self.pinned:
            self.pinned[shape] = torch.empty((BATCH_SIZE, *shape), dtype=torch.uint8).pin_memory()
        buffer = self.pinned[shape][:len(images)]
        np.stack(images, out=buffer.numpy())
        # Reusing the buffer next batch is safe: predict() reads the detections back, which waits for this copy.
        return buffer.to(self.device, non_blocking=True)

    @torch.inference_mode()
    def predict(self, images: list[Image]) -> list[tuple[set[str], int]]:
        """Return (objects found, people count) for each frame. Frames must share one shape and come from
        prepare_frame, so they are already resized and only need padding."""
        box = letterbox(images[0].shape[0], images[0].shape[1])
        if box.resized != images[0].shape[:2]:
            raise ValueError(f"frame of shape {images[0].shape} was not resized by prepare_frame")
        model = self._model_for(box.padded)

        x = self._upload(images).permute(0, 3, 1, 2)  # BHWC -> BCHW
        x = F.pad(x, (box.left, box.right, box.top, box.bottom), value=LETTERBOX_PADDING_VALUE)
        x = x.flip(1)  # BGR -> RGB
        x = (x.half() if model.fp16 else x.float()).div_(255)
        detections = non_max_suppression(model(x), MODEL_CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, self.class_ids,
                                          agnostic=False, max_det=MAX_DETECTIONS, nc=0,
                                          end2end=getattr(model, "end2end", False))
        if not (isinstance(detections, list) and all(isinstance(d, torch.Tensor) for d in detections)):
            raise TypeError(f"unexpected NMS output {type(detections).__name__}")

        # One transfer for the whole batch: column 5 of each (n, 6) detection tensor is the class id.
        class_ids: list[int] = torch.cat([d[:, 5] for d in detections]).int().tolist()
        results: list[tuple[set[str], int]] = []
        start = 0
        for d in detections:
            names = [self.names[class_id] for class_id in class_ids[start:start + len(d)]]
            start += len(d)
            results.append((set(names), names.count("person")))
        return results


# ---------------------------------------------------------------------------------------------------------------------
# Decoding


def prepare_frame(frame: VideoFrame) -> Image:
    """BGR array, rotated by the display matrix (frame.rotation is counter-clockwise degrees) and resized for the
    model with the exact OpenCV call of Ultralytics' letterbox. Runs in the decoder thread, off the inference path."""
    image = np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)  # bgr24 is uint8; no copy
    k = (frame.rotation // 90) % 4
    if k:
        image = np.ascontiguousarray(np.rot90(image, k=k))
    height, width = letterbox(image.shape[0], image.shape[1]).resized
    if (height, width) != image.shape[:2]:
        image = np.asarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR), dtype=np.uint8)
    return image


def decode_video(video_bytes: bytes) -> Iterator[tuple[int, Image | None]]:
    """Yield (second, image) per frame, and (second, None) once each second is complete, empty seconds included.
    A frame's second is floor((pts - first_pts) * time_base), i.e. counted from the first decoded frame."""
    with av.open(io.BytesIO(video_bytes), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        first_pts: int | None = None
        current_second = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("frame without timestamp")
            if first_pts is None:
                first_pts = frame.pts
            # shortcut: out-of-order timestamps are folded into the current second
            second = max(int((frame.pts - first_pts) * frame.time_base), current_second)
            while second > current_second:
                yield current_second, None
                current_second += 1
            yield second, prepare_frame(frame)
        if first_pts is None:
            raise ValueError("no frames decoded")
        yield current_second, None


# ---------------------------------------------------------------------------------------------------------------------
# Messages. Main process -> workers: VideoTask. Worker decoder thread -> worker main thread: the frame events.
# Workers -> main process: WorkerReady, VideoStarted, ResultLine, VideoFinished, WorkerExited.


@dataclass(frozen=True, slots=True)
class VideoTask:
    video: str  # "<parquet file name>:<row>"
    data: object  # the mp4 cell; checked to be bytes by the worker


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    video: str
    second: int
    image: Image


@dataclass(frozen=True, slots=True)
class SecondEnd:
    video: str
    second: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class VideoEnd:
    video: str
    failed: bool


@dataclass(frozen=True, slots=True)
class DecodingDone:
    crash: BaseException | None = None  # set if the decoder thread itself died


FrameEvent: TypeAlias = DecodedFrame | SecondEnd | VideoEnd | DecodingDone


@dataclass(frozen=True, slots=True)
class WorkerReady:
    worker: int
    description: str


@dataclass(frozen=True, slots=True)
class VideoStarted:
    worker: int
    video: str


@dataclass(frozen=True, slots=True)
class ResultLine:
    text: str


@dataclass(frozen=True, slots=True)
class VideoFinished:
    worker: int
    video: str


@dataclass(frozen=True, slots=True)
class WorkerExited:
    worker: int


WorkerMessage: TypeAlias = WorkerReady | VideoStarted | ResultLine | VideoFinished | WorkerExited
TaskQueue: TypeAlias = "MpQueue[VideoTask]"
ResultQueue: TypeAlias = "MpQueue[WorkerMessage]"


# ---------------------------------------------------------------------------------------------------------------------
# Worker process


@dataclass(slots=True)
class SecondStats:
    frames_done: int = 0
    frame_count: int | None = None  # known once the decoder has finished the second
    object_frames: Counter[str] = field(default_factory=Counter[str])
    people: list[int] = field(default_factory=list[int])


class ResultWriter:
    """Aggregates per-frame detections into per-second lines, emitting each second once all its frames are in."""

    def __init__(self, emit: Callable[[str], None]):
        self.emit = emit
        self.pending: dict[tuple[str, int], SecondStats] = {}

    def add_detections(self, frames: list[DecodedFrame], detections: list[tuple[set[str], int]]) -> None:
        touched: set[tuple[str, int]] = set()
        for frame, (objects, people) in zip(frames, detections, strict=True):
            key = (frame.video, frame.second)
            stats = self.pending.setdefault(key, SecondStats())
            stats.frames_done += 1
            stats.object_frames.update(objects)
            stats.people.append(people)
            touched.add(key)
        for key in touched:
            self._emit_if_complete(key)

    def end_second(self, event: SecondEnd) -> None:
        key = (event.video, event.second)
        self.pending.setdefault(key, SecondStats()).frame_count = event.frame_count
        self._emit_if_complete(key)

    def _emit_if_complete(self, key: tuple[str, int]) -> None:
        stats = self.pending[key]
        if stats.frame_count is None or stats.frames_done < stats.frame_count:
            return
        del self.pending[key]
        n = stats.frame_count
        objects = [name for name in OBJECT_DETECTION_CLASSES if n and stats.object_frames[name] >= n / 2]
        people = int(statistics.median(stats.people)) if stats.people else 0  # int() rounds a .5 median down
        self.emit(json.dumps({"video": key[0], "second": key[1], "objects": objects, "people": people}))

    def fail_video(self, video: str) -> None:
        """Drop the unfinished seconds of a failed video and emit its error line. Call after flushing the batch."""
        for key in [k for k in self.pending if k[0] == video]:
            del self.pending[key]
        self.emit(error_line(video))


def error_line(video: str) -> str:
    return json.dumps({"video": video, "error": ERROR_MESSAGE})


def decode_tasks(worker: int, tasks: TaskQueue, all_fed: MpEvent, results: ResultQueue,
                 events: "queue.Queue[FrameEvent]") -> None:
    """Worker decoder thread: pull videos until the main process has fed them all, turn each into frame events."""
    try:
        while True:
            try:
                task = tasks.get(timeout=0.5)
            except queue.Empty:
                if all_fed.is_set() and tasks.empty():
                    break
                continue
            results.put(VideoStarted(worker, task.video))
            frame_count = 0
            try:
                if not isinstance(task.data, bytes):  # pyarrow is untyped: check the boundary at runtime
                    raise TypeError(f"mp4 cell holds {type(task.data).__name__}, not bytes")
                for second, image in decode_video(task.data):
                    if image is None:
                        events.put(SecondEnd(task.video, second, frame_count))
                        frame_count = 0
                    else:
                        events.put(DecodedFrame(task.video, second, image))
                        frame_count += 1
                events.put(VideoEnd(task.video, failed=False))
            except Exception as e:
                print(f"Error decoding {task.video}: {e}", flush=True)
                events.put(VideoEnd(task.video, failed=True))
        events.put(DecodingDone())
    except BaseException as e:
        events.put(DecodingDone(crash=e))


def worker_main(worker: int, backend: Backend, model_file: str | None, fp16: bool,
                tasks: TaskQueue, all_fed: MpEvent, results: ResultQueue) -> None:
    predictor = Predictor(backend, model_file, fp16)
    results.put(WorkerReady(worker, f"{backend} {predictor.model_file}{' FP16' if predictor.fp16 else ''}"))
    events: queue.Queue[FrameEvent] = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    decoder = threading.Thread(target=decode_tasks, args=(worker, tasks, all_fed, results, events), daemon=True)
    decoder.start()

    writer = ResultWriter(lambda text: results.put(ResultLine(text)))
    batch: list[DecodedFrame] = []

    def flush() -> None:
        if batch:
            writer.add_detections(batch, predictor.predict([f.image for f in batch]))
            batch.clear()

    while True:
        event = events.get()
        match event:
            case DecodedFrame():
                # Same-shape batches keep Ultralytics' minimal letterbox, matching single-video inference.
                if batch and batch[0].image.shape != event.image.shape:
                    flush()
                batch.append(event)
                if len(batch) >= BATCH_SIZE:
                    flush()
            case SecondEnd():
                writer.end_second(event)
            case VideoEnd(video=video, failed=failed):
                flush()  # every second of this video is now complete
                if failed:
                    writer.fail_video(video)
                results.put(VideoFinished(worker, video))
            case DecodingDone(crash=crash):
                flush()
                if crash is not None:
                    raise crash
                break
    decoder.join()
    results.put(WorkerExited(worker))


# ---------------------------------------------------------------------------------------------------------------------
# Main process


def get_parquet_files(folder_path: str, max_files: int) -> list[str]:
    files = sorted(os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in the folder: {folder_path}")
    return files[:max_files] if max_files > 0 else files


def feed_tasks(parquet_files: list[str], max_rows: int, tasks: TaskQueue, all_fed: MpEvent,
               write: Callable[[str], None]) -> None:
    """Reader thread: put every video on the task queue. Unreadable files get an error line for the row they broke on."""
    try:
        for parquet_file in parquet_files:
            name = os.path.basename(parquet_file)
            row_index = 0
            try:
                batches = pq.ParquetFile(parquet_file).iter_batches(batch_size=1, columns=["mp4"])
                for row_index, batch in enumerate(batches):
                    if 0 < max_rows <= row_index:
                        break
                    tasks.put(VideoTask(f"{name}:{row_index}", batch.column(0)[0].as_py()))
            except Exception as e:
                print(f"Error reading {parquet_file}: {e}", flush=True)
                write(error_line(f"{name}:{row_index}"))
    finally:
        all_fed.set()


def default_backends() -> list[Backend]:
    if torch.cuda.is_available():
        return ["cuda", "cuda"]  # one process's CPU pre/post-processing overlaps the other's GPU work
    if torch.backends.mps.is_available():
        # Measured on an M5 Pro: the GPU and the Neural Engine run in parallel only as separate processes.
        return ["mps", "mps", "coreml", "coreml"]
    return ["cpu"]


def run(parquet_files: list[str], output_file: str, backends: list[Backend], model_file: str | None, fp16: bool,
        max_rows: int) -> None:
    ctx = mp.get_context("spawn")
    tasks: TaskQueue = ctx.Queue(maxsize=2 * len(backends))
    results: ResultQueue = ctx.Queue()
    all_fed = ctx.Event()
    lock = threading.Lock()

    with open(output_file, "w") as out:
        def write(text: str) -> None:
            with lock:
                out.write(text + "\n")

        def start(worker: int) -> BaseProcess:
            process = ctx.Process(target=worker_main, name=f"worker-{worker}",
                                  args=(worker, backends[worker], model_file, fp16, tasks, all_fed, results))
            process.start()
            return process

        processes = {worker: start(worker) for worker in range(len(backends))}
        ready: set[int] = set()
        exited: set[int] = set()
        in_flight: dict[int, set[str]] = {worker: set() for worker in processes}
        feeder = threading.Thread(target=feed_tasks, args=(parquet_files, max_rows, tasks, all_fed, write), daemon=True)
        feeder.start()

        def handle(message: WorkerMessage) -> None:
            match message:
                case WorkerReady(worker=worker, description=description):
                    ready.add(worker)
                    print(f"Worker {worker}: {description}", flush=True)
                case VideoStarted(worker=worker, video=video):
                    print(f"Worker {worker}: {video}", flush=True)
                    in_flight[worker].add(video)
                case ResultLine(text=text):
                    write(text)
                case VideoFinished(worker=worker, video=video):
                    in_flight[worker].discard(video)
                case WorkerExited(worker=worker):
                    exited.add(worker)

        try:
            while len(exited) < len(processes):
                try:
                    handle(results.get(timeout=1))
                    continue
                except queue.Empty:
                    pass
                for worker, process in processes.items():
                    if worker in exited or process.is_alive():
                        continue
                    while True:  # take everything it sent before dying
                        try:
                            handle(results.get_nowait())
                        except queue.Empty:
                            break
                    if worker in exited:
                        continue
                    if worker not in ready:
                        raise RuntimeError(f"Worker {worker} ({backends[worker]}) died during start-up")
                    for video in sorted(in_flight[worker]):
                        print(f"Worker {worker} died while processing {video}", flush=True)
                        write(error_line(video))
                    in_flight[worker].clear()
                    ready.discard(worker)
                    processes[worker] = start(worker)
        finally:
            for process in processes.values():
                if process.is_alive():
                    process.terminate()
        feeder.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11l Object Detection on FineVideo Parquet Files")
    parser.add_argument("--input", required=True, help="Folder containing .parquet files")
    parser.add_argument("--output", default="results.jsonl", help="jsonl file to write the results to")
    parser.add_argument("--workers", help="Comma-separated backend per worker process, from "
                        f"{', '.join(BACKENDS)} (default: cuda,cuda on NVIDIA, mps,mps,coreml,coreml on Apple Silicon)")
    parser.add_argument("--model", help=f"Model file for cuda/mps/cpu workers (default: {ENGINE_MODEL_FILE} on CUDA "
                        f"if built, else {PT_MODEL_FILE})")
    parser.add_argument("--max-files", type=int, default=0, help="Only process this many parquet files (0 = all)")
    parser.add_argument("--max-rows", type=int, default=0, help="Only process this many videos per file (0 = all)")
    parser.add_argument("--dev", action="store_true", help="Shortcut for --max-files 1 --max-rows 1")
    parser.add_argument("--fp32", action="store_true", help="Run .pt models in FP32 (default FP16 on CUDA and MPS)")
    args = parser.parse_args()
    if args.dev:
        args.max_files, args.max_rows = 1, 1

    backends: list[Backend] = default_backends()
    if args.workers:
        names: list[str] = args.workers.split(",")
        unknown = [n for n in names if n not in BACKENDS]
        if unknown:
            parser.error(f"unknown backends {unknown}; choose from {', '.join(BACKENDS)}")
        backends = [b for n in names for b in BACKENDS if b == n]
    run(get_parquet_files(args.input, args.max_files), args.output, backends, args.model, not args.fp32, args.max_rows)


if __name__ == "__main__":
    start_time = time()
    main()
    print(f"Total processing time: {time() - start_time:.2f} seconds")
