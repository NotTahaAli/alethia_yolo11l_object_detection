"""YOLO11l object detection over every second of every video in a folder of FineVideo Parquet files.

    python run.py --input <folder of .parquet files> --output results.jsonl

A background thread decodes videos into BGR frames and pushes them onto a bounded queue. The main thread
batches frames (across seconds of the same video) through YOLO and writes one JSON line per second.
"""
import argparse
import io
import json
import os
import queue
import statistics
import threading
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from time import time
from typing import TextIO, TypeAlias

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from av.video.frame import VideoFrame
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

PT_MODEL_FILE = "yolo/yolo11l.pt"
ENGINE_MODEL_FILE = "yolo/yolo11l.engine"  # built by build_engine.py
MODEL_INPUT_SIZE = 640
MODEL_CONFIDENCE_THRESHOLD = 0.4
OBJECT_DETECTION_CLASSES = ("person", "car", "dog", "laptop", "cell phone")
# Frames per model call. The TensorRT engine is built with this as its max batch, so keep them in sync.
BATCH_SIZE = 64
QUEUE_SIZE = 4 * BATCH_SIZE  # decoded frames buffered ahead of the model
ERROR_MESSAGE = "could not decode video"

Image: TypeAlias = NDArray[np.uint8]


# Events sent from the decoder thread to the main thread, in decode order.
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
class VideoFailed:
    video: str


@dataclass(frozen=True, slots=True)
class DecodingDone:
    crash: BaseException | None = None  # set if the decoder thread itself died


Event = DecodedFrame | SecondEnd | VideoFailed | DecodingDone


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_model(device: str) -> str:
    return ENGINE_MODEL_FILE if device == "0" and os.path.exists(ENGINE_MODEL_FILE) else PT_MODEL_FILE


class Predictor:
    def __init__(self, model_file: str, device: str):
        self.device = device
        self.model = YOLO(model_file, task="detect")
        names: dict[int, str] = self.model.names
        name_to_id = {name: class_id for class_id, name in names.items()}
        missing = [name for name in OBJECT_DETECTION_CLASSES if name not in name_to_id]
        if missing:
            raise ValueError(f"Classes {missing} not found in the model's class names.")
        self.names = names
        self.class_ids = [name_to_id[name] for name in OBJECT_DETECTION_CLASSES]

    def predict(self, images: list[Image]) -> list[tuple[set[str], int]]:
        """Return (objects found, people count) for each image."""
        results = self.model.predict(images,
                                     imgsz=MODEL_INPUT_SIZE,
                                     conf=MODEL_CONFIDENCE_THRESHOLD,
                                     classes=self.class_ids,
                                     quantize=16 if self.device == "0" else None,  # FP16 on CUDA only
                                     verbose=False,
                                     device=self.device)
        detections: list[tuple[set[str], int]] = []
        for result in results:
            if not isinstance(result, Results):
                raise TypeError(f"unexpected prediction type {type(result)}")
            class_ids = [int(c) for c in result.boxes.cls.tolist()] if result.boxes is not None else []
            names = [self.names[class_id] for class_id in class_ids]
            detections.append((set(names), names.count("person")))
        return detections


def to_bgr(frame: VideoFrame) -> Image:
    """BGR array with the display-matrix rotation applied (frame.rotation is counter-clockwise degrees)."""
    image = np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)  # bgr24 is uint8; no copy
    k = (frame.rotation // 90) % 4
    return np.ascontiguousarray(np.rot90(image, k=k)) if k else image


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
            yield second, to_bgr(frame)
        if first_pts is None:
            raise ValueError("no frames decoded")
        yield current_second, None


def iter_videos(parquet_file: str, max_rows: int) -> Iterator[tuple[int, object]]:
    """Yield (row index, mp4 cell), one row at a time so a whole file never sits in memory."""
    batches = pq.ParquetFile(parquet_file).iter_batches(batch_size=1, columns=["mp4"])
    for row_index, batch in enumerate(batches):
        if 0 < max_rows <= row_index:
            return
        yield row_index, batch.column(0)[0].as_py()


def decode_all(parquet_files: list[str], max_rows: int, events: "queue.Queue[Event]") -> None:
    """Decoder thread: turn every video into events. Never lets an exception escape."""
    try:
        for parquet_file in parquet_files:
            name = os.path.basename(parquet_file)
            row_index = 0
            try:
                for row_index, video_bytes in iter_videos(parquet_file, max_rows):
                    video = f"{name}:{row_index}"
                    print(f"Decoding {video}")
                    frame_count = 0
                    try:
                        if not isinstance(video_bytes, bytes):  # pyarrow is untyped: check the boundary at runtime
                            raise TypeError(f"mp4 cell holds {type(video_bytes).__name__}, not bytes")
                        for second, image in decode_video(video_bytes):
                            if image is None:
                                events.put(SecondEnd(video, second, frame_count))
                                frame_count = 0
                            else:
                                events.put(DecodedFrame(video, second, image))
                                frame_count += 1
                    except Exception as e:
                        print(f"Error decoding {video}: {e}")
                        events.put(VideoFailed(video))
            except Exception as e:  # unreadable or damaged parquet file: blame the row we were on
                print(f"Error reading {parquet_file}: {e}")
                events.put(VideoFailed(f"{name}:{row_index}"))
        events.put(DecodingDone())
    except BaseException as e:
        events.put(DecodingDone(crash=e))


@dataclass(slots=True)
class SecondStats:
    frames_done: int = 0
    frame_count: int | None = None  # known once the decoder has finished the second
    object_frames: Counter[str] = field(default_factory=Counter[str])
    people: list[int] = field(default_factory=list[int])


class ResultWriter:
    """Aggregates per-frame detections into per-second lines, writing each second once all its frames are in."""

    def __init__(self, out: TextIO):
        self.out = out
        self.pending: dict[tuple[str, int], SecondStats] = {}

    def _write(self, line: dict[str, object]) -> None:
        self.out.write(json.dumps(line) + "\n")

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
            self._write_if_complete(key)

    def end_second(self, event: SecondEnd) -> None:
        key = (event.video, event.second)
        self.pending.setdefault(key, SecondStats()).frame_count = event.frame_count
        self._write_if_complete(key)

    def _write_if_complete(self, key: tuple[str, int]) -> None:
        stats = self.pending[key]
        if stats.frame_count is None or stats.frames_done < stats.frame_count:
            return
        del self.pending[key]
        n = stats.frame_count
        objects = [name for name in OBJECT_DETECTION_CLASSES if n and stats.object_frames[name] >= n / 2]
        people = int(statistics.median(stats.people)) if stats.people else 0  # int() rounds a .5 median down
        self._write({"video": key[0], "second": key[1], "objects": objects, "people": people})

    def fail_video(self, video: str) -> None:
        """Drop the unfinished seconds of a failed video and write its error line. Call after flushing the batch."""
        for key in [k for k in self.pending if k[0] == video]:
            del self.pending[key]
        self._write({"video": video, "error": ERROR_MESSAGE})


def run(parquet_files: list[str], output_file: str, predictor: Predictor, max_rows: int) -> None:
    events: queue.Queue[Event] = queue.Queue(maxsize=QUEUE_SIZE)
    decoder = threading.Thread(target=decode_all, args=(parquet_files, max_rows, events), daemon=True)
    decoder.start()

    with open(output_file, "w") as out:
        writer = ResultWriter(out)
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
                case VideoFailed():
                    flush()
                    writer.fail_video(event.video)
                case DecodingDone(crash=crash):
                    flush()
                    if crash is not None:
                        raise crash
                    break
    decoder.join()


def get_parquet_files(folder_path: str, max_files: int) -> list[str]:
    files = sorted(os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in the folder: {folder_path}")
    return files[:max_files] if max_files > 0 else files


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11l Object Detection on FineVideo Parquet Files")
    parser.add_argument("--input", required=True, help="Folder containing .parquet files")
    parser.add_argument("--output", default="results.jsonl", help="jsonl file to write the results to")
    parser.add_argument("--model", help=f"YOLO model file (default: {ENGINE_MODEL_FILE} on CUDA if built, else {PT_MODEL_FILE})")
    parser.add_argument("--max-files", type=int, default=0, help="Only process this many parquet files (0 = all)")
    parser.add_argument("--max-rows", type=int, default=0, help="Only process this many videos per file (0 = all)")
    parser.add_argument("--dev", action="store_true", help="Shortcut for --max-files 1 --max-rows 1")
    args = parser.parse_args()
    if args.dev:
        args.max_files, args.max_rows = 1, 1

    device = pick_device()
    model_file: str = args.model or default_model(device)
    print(f"Using {model_file} on device {device}")
    predictor = Predictor(model_file, device)
    run(get_parquet_files(args.input, args.max_files), args.output, predictor, args.max_rows)


if __name__ == "__main__":
    start_time = time()
    main()
    print(f"Total processing time: {time() - start_time:.2f} seconds")
