import math
from time import time
from ultralytics import YOLO
import pyarrow.parquet as pq
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

    def predict(self, frames: list[VideoFrame]):
        return self.model.predict([f.to_ndarray(format="bgr24") for f in frames],
                                  imgsz=self.input_size,
                                  conf=self.confidence_threshold,
                                  classes=self.class_ids,
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

    def _append(self, file: str, row_index: int, data: dict):
        file_name = os.path.basename(file)
        data["file"] = f"{file_name}:{row_index}"
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

        byte_stream = io.BytesIO(video)
        self.av_container = av.open(byte_stream, mode='r', format='mp4')
        self.video = self.av_container.streams.video[0]
        self.frame_rate = self.video.average_rate
        self.total_frames = self.video.frames
        self.duration_seconds = self.total_frames / self.frame_rate

    def _save_frame(self, frame: VideoFrame, second: int, frame_index: int):
        if not SAVE_FRAMES:
            return
        file_base_name = os.path.basename(self.file)
        output_dir = f"frames/{file_base_name}/row_{self.row}"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"second_{second}_frame_{frame_index}.jpg")
        frame.save(output_file_path)

    def _process_second(self, second: int):
        start_frame = int(second * self.frame_rate)
        end_frame = min(int((second + 1) * self.frame_rate), self.total_frames)
        frame_count = end_frame - start_frame
        item_counts = {x: 0 for x in OBJECT_DETECTION_CLASSES if x != "person"}
        people_counts = []
        frames = []
        for frame_index in range(start_frame, end_frame):
            frame = next(self.av_container.decode(video=0))
            frames.append(frame)
            self._save_frame(frame, second, frame_index)

        predictions = self.predictor.predict(frames)
        for result in predictions:
            unique_objects, people_count = self.predictor.evaluate(result)
            for obj in unique_objects:
                if obj != "person":
                    item_counts[obj] += 1
            people_counts.append(people_count)

        avg_people_count = int(statistics.median(people_counts))
        objects_detected = [obj for obj, count in item_counts.items() if count >= frame_count / 2]
        self.logger.log_second(self.file, self.row, second, objects_detected, avg_people_count)


    def process(self):
        print(f"Processing video from file: {self.file}, row: {self.row}")
        for i in range(math.ceil(self.duration_seconds)):
            second = i
            self._process_second(second)
class ParquetProcessor:
    def __init__(self, predictor: Predictor, logger: Logger, file_path: str):
        self.predictor = predictor
        self.logger = logger
        self.table = pq.read_table(file_path)
        self.file_path = file_path

    def _get_mp4_count(self) -> int:
        if self.table.num_rows <= 0:
            raise ValueError("Number of mp4s cannot be less than 1.")
        if ROWS_TO_PROCESS_PER_FILE > 0 and self.table.num_rows > ROWS_TO_PROCESS_PER_FILE:
            print(f"Limiting to {ROWS_TO_PROCESS_PER_FILE} mp4s for processing.")
            return ROWS_TO_PROCESS_PER_FILE
        return self.table.num_rows

    def _get_video(self, row_index: int) -> bytes:
        if row_index < 0 or row_index >= self._get_mp4_count():
            raise IndexError("Row index is out of bounds.")
        return self.table.column("mp4")[row_index].as_py()

    def process(self):
        mp4_count = self._get_mp4_count()
        for row_index in range(mp4_count):
            try:
                video_bytes = self._get_video(row_index)
                video_processor = VideoProcessor(self.predictor, self.logger, video_bytes, self.file_path, row_index)
                video_processor.process()
            except Exception as e:
                self.logger.log_error(self.file_path, row_index)
                print(f"Error processing row {row_index}: {e}")
                break

def main():
    parser = argparse.ArgumentParser(description="YOLO11l Object Detection on FineVideo Parquet Files")
    parser.add_argument("--input", type=str, required=True, help="Folder containing .parquet files")
    parser.add_argument("--output", type=str, default="results.jsonl", help="jsonl file to append the results to")
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
        parquet_processor = ParquetProcessor(predictor, logger, file_path)
        parquet_processor.process()

if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")