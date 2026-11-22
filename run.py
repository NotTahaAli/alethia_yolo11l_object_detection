from time import time
from ultralytics import YOLO
import pyarrow.parquet as pq
import numpy as np
import argparse
import os
import io
import av
import json
import statistics
import torch

from av.video.frame import VideoFrame

MODEL_FILE: str = "yolo/yolo11l.pt"
MODEL_INPUT_SIZE: int = 640
MODEL_CONFIDENCE_THRESHOLD: float = 0.4
OBJECT_DETECTION_CLASSES: list = ["person", "car", "dog", "laptop", "cell phone"]

# Production settings
MAX_FILES_TO_PROCESS: int = -1
ROWS_TO_PROCESS_PER_FILE: int = -1
SHOW_VERBOSE_PREDICTIONS: bool = False
SAVE_FRAMES: bool = False

# Development settings
DEV_MAX_FILES_TO_PROCESS: int = 1
DEV_ROWS_TO_PROCESS_PER_FILE: int = 1
DEV_SHOW_VERBOSE_PREDICTIONS: bool = False
DEV_SAVE_FRAMES: bool = False

class Predictor:
    def __init__(self, model_file: str, input_size: int, confidence_threshold: float, class_names: list[str]):
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names
        self._load_model(model_file)
        self.class_ids = self._get_class_ids(class_names)

    def _load_model(self, model_file: str):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "0"
        if torch.backends.mps.is_available():
            self.device = "mps"

        self.model = YOLO(model_file)

    def _get_class_ids(self, class_names: list[str]) -> list[int]:
        class_ids = []
        for class_name in class_names:
            if class_name in self.model.names.values():
                class_id = list(self.model.names.keys())[list(self.model.names.values()).index(class_name)]
                class_ids.append(class_id)
            else:
                raise ValueError(f"Class name '{class_name}' not found in the model's class names.")
        return class_ids

    @staticmethod
    def _to_bgr(frame: VideoFrame) -> np.ndarray:
        """BGR array with the display-matrix rotation applied (frame.rotation is counter-clockwise degrees)."""
        image = frame.to_ndarray(format="bgr24")
        k = (frame.rotation // 90) % 4
        return np.ascontiguousarray(np.rot90(image, k=k)) if k else image

    def predict(self, frames: list[VideoFrame]):
        return self.model.predict([self._to_bgr(f) for f in frames],
                                  imgsz=self.input_size,
                                  conf=self.confidence_threshold,
                                  classes=self.class_ids,
                                  quantize=16 if self.device == "0" else None,  # FP16 on CUDA only
                                  verbose=SHOW_VERBOSE_PREDICTIONS,
                                  device=self.device)

    def evaluate(self, result) -> tuple[set[str], int]:
        unique_objects = set()
        people_count = 0
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            unique_objects.add(class_name)
            if class_name == "person":
                people_count += 1
        return unique_objects, people_count

def get_parquet_files(folder_path: str) -> list[str]:
    """Get a list of all Parquet files in the specified folder."""
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")])
    if not files:
        raise FileNotFoundError(f"No Parquet files found in the folder: {folder_path}")
    if MAX_FILES_TO_PROCESS > 0 and len(files) > MAX_FILES_TO_PROCESS:
        print(f"Limiting to {MAX_FILES_TO_PROCESS} files for processing.")
        files = files[:MAX_FILES_TO_PROCESS]
    return files
class Logger:
    def __init__(self, output_file: str):
        self.output_file = output_file
        open(output_file, "w").close()  # start fresh so reruns don't duplicate lines

    def _append(self, file: str, row_index: int, data: dict):
        file_name = os.path.basename(file)
        data["video"] = f"{file_name}:{row_index}"
        with open(self.output_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')

    def log_error(self, file: str, row_index: int):
        self._append(file, row_index, {"error": "could not decode video"})

    def log_second(self, file: str, row_index: int, second: int, objects: list[str], people: int):
        self._append(file, row_index, {"second": second, "objects": objects, "people": people})

class VideoProcessor:
    def __init__(self, predictor: Predictor, logger: Logger, video: bytes, file: str, row: int):
        self.predictor = predictor
        self.logger = logger
        self.video_bytes = video
        self.file = file
        self.row = row

    def _save_frame(self, frame: VideoFrame, second: int, frame_index: int):
        if not SAVE_FRAMES:
            return
        output_dir = f"frames/{os.path.basename(self.file)}/row_{self.row}"
        os.makedirs(output_dir, exist_ok=True)
        frame.save(os.path.join(output_dir, f"second_{second}_frame_{frame_index}.jpg"))

    def _frames_by_second(self):
        """Decode the video once and yield (second, frames) for every second, including empty ones.
        A frame's second is floor((pts - first_pts) * time_base), i.e. counted from the first decoded frame."""
        with av.open(io.BytesIO(self.video_bytes), mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            first_pts = None
            current_second, frames = 0, []
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    raise ValueError("frame without timestamp")
                if first_pts is None:
                    first_pts = frame.pts
                # ponytail: out-of-order timestamps are folded into the current second
                second = max(int((frame.pts - first_pts) * frame.time_base), current_second)
                while second > current_second:
                    yield current_second, frames
                    current_second, frames = current_second + 1, []
                frames.append(frame)
                self._save_frame(frame, second, frame_index)
            if first_pts is None:
                raise ValueError("no frames decoded")
            yield current_second, frames

    def _process_second(self, second: int, frames: list[VideoFrame]):
        item_counts = {x: 0 for x in OBJECT_DETECTION_CLASSES}
        people_counts = []
        for result in self.predictor.predict(frames) if frames else []:
            unique_objects, people_count = self.predictor.evaluate(result)
            for obj in unique_objects:
                item_counts[obj] += 1
            people_counts.append(people_count)

        people = int(statistics.median(people_counts)) if people_counts else 0
        objects_detected = [obj for obj, count in item_counts.items() if frames and count >= len(frames) / 2]
        self.logger.log_second(self.file, self.row, second, objects_detected, people)

    def process(self):
        print(f"Processing video from file: {self.file}, row: {self.row}")
        for second, frames in self._frames_by_second():
            self._process_second(second, frames)

class ParquetProcessor:
    def __init__(self, predictor: Predictor, logger: Logger, file_path: str):
        self.predictor = predictor
        self.logger = logger
        self.file_path = file_path
        self.table = pq.read_table(file_path, columns=["mp4"])

    def _get_mp4_count(self) -> int:
        if ROWS_TO_PROCESS_PER_FILE > 0 and self.table.num_rows > ROWS_TO_PROCESS_PER_FILE:
            print(f"Limiting to {ROWS_TO_PROCESS_PER_FILE} mp4s for processing.")
            return ROWS_TO_PROCESS_PER_FILE
        return self.table.num_rows

    def process(self):
        mp4_column = self.table.column("mp4")
        for row_index in range(self._get_mp4_count()):
            try:
                video_bytes = mp4_column[row_index].as_py()
                VideoProcessor(self.predictor, self.logger, video_bytes, self.file_path, row_index).process()
            except Exception as e:
                self.logger.log_error(self.file_path, row_index)
                print(f"Error processing row {row_index}: {e}")

def main():
    parser = argparse.ArgumentParser(description="YOLO11l Object Detection on FineVideo Parquet Files")
    parser.add_argument("--input", type=str, required=True, help="Folder containing .parquet files")
    parser.add_argument("--output", type=str, default="results.jsonl", help="jsonl file to write the results to")
    parser.add_argument("--model", type=str, default=MODEL_FILE, help="Path to the YOLO model file")
    parser.add_argument("--dev", action="store_true", help="Run in development mode with limited files and rows for testing")

    args = parser.parse_args()
    if args.dev:
        global MAX_FILES_TO_PROCESS, ROWS_TO_PROCESS_PER_FILE, SHOW_VERBOSE_PREDICTIONS, SAVE_FRAMES
        MAX_FILES_TO_PROCESS = DEV_MAX_FILES_TO_PROCESS
        ROWS_TO_PROCESS_PER_FILE = DEV_ROWS_TO_PROCESS_PER_FILE
        SHOW_VERBOSE_PREDICTIONS = DEV_SHOW_VERBOSE_PREDICTIONS
        SAVE_FRAMES = DEV_SAVE_FRAMES

    input_folder = args.input
    output_file = args.output
    predictor = Predictor(args.model, MODEL_INPUT_SIZE, MODEL_CONFIDENCE_THRESHOLD, OBJECT_DETECTION_CLASSES)

    parquet_files = get_parquet_files(input_folder)
    logger = Logger(output_file)

    for file_path in parquet_files:
        try:
            parquet_processor = ParquetProcessor(predictor, logger, file_path)
        except Exception as e:
            logger.log_error(file_path, 0)
            print(f"Error reading parquet file {file_path}: {e}")
            continue
        parquet_processor.process()

if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")