from sympy import li
from ultralytics import YOLO
import pyarrow.parquet as pq
from pyarrow import Table
import argparse
import os

MODEL_WEIGHTS: str = "yolo/yolo11l.pt"
MODEL_INPUT_SIZE: int = 640
MODEL_CONFIDENCE_THRESHOLD: float = 0.4
OBJECT_DETECTION_CLASSES: list = ["person", "car", "dog", "laptop", "cell phone"]

# Production settings
MAX_FILES_TO_PROCESS: int = -1
ROWS_TO_PROCESS_PER_FILE: int = -1
SHOW_VERBOSE_PREDICTIONS: bool = False

# Development settings
DEV_MAX_FILES_TO_PROCESS: int = 1
DEV_ROWS_TO_PROCESS_PER_FILE: int = 1
DEV_SHOW_VERBOSE_PREDICTIONS: bool = True

def get_parquet_files(folder_path: str) -> list[str]:
    """Get a list of all Parquet files in the specified folder."""
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No Parquet files found in the folder: {folder_path}")
    if MAX_FILES_TO_PROCESS > 0 and len(files) > MAX_FILES_TO_PROCESS:
        print(f"Limiting to {MAX_FILES_TO_PROCESS} files for processing.")
        files = files[:MAX_FILES_TO_PROCESS]
    return files

def get_row_count(table: Table) -> int:
    """Get the number of rows in a Parquet file."""
    if table.num_rows <= 0:
        raise ValueError("Number of rows cannot be less than 1.")
    if ROWS_TO_PROCESS_PER_FILE > 0 and table.num_rows > ROWS_TO_PROCESS_PER_FILE:
        print(f"Limiting to {ROWS_TO_PROCESS_PER_FILE} rows for processing.")
        return ROWS_TO_PROCESS_PER_FILE
    return table.num_rows

def get_video_bytes(table: Table, row_index: int) -> bytes:
    """Get the base64 string of the video from a specific row in the Parquet file."""
    if row_index < 0 or row_index >= table.num_rows:
        raise IndexError("Row index is out of bounds.")
    video_bytes = table.column("mp4")[row_index].as_py()
    return video_bytes

def load_model(weights_path: str) -> YOLO:
    model =  YOLO(weights_path)
    return model

def predict_frame(model: YOLO, frame):
    return model.predict(frame,
                         imgsz=MODEL_INPUT_SIZE,
                         conf=MODEL_CONFIDENCE_THRESHOLD,
                         classes=OBJECT_DETECTION_CLASSES,
                         verbose=SHOW_VERBOSE_PREDICTIONS)

def process_video(video, model: YOLO, output_file):
    pass

def process_parquet_file(file_path, model: YOLO, output_file):
    print(f"Processing file: {file_path}")
    table = pq.read_table(file_path)
    row_count = get_row_count(table)

    for row_index in range(row_count):
        video_bytes = get_video_bytes(table, row_index)
        print(type(video_bytes))
        print(f"Processing row {row_index + 1}/{row_count}")
        print(f"Video bytes length: {len(video_bytes)}")

def main():
    parser = argparse.ArgumentParser(description="YOLO11l Object Detection on FineVideo Parquet Files")
    parser.add_argument("--input", type=str, required=True, help="Folder containing .parquet files")
    parser.add_argument("--output", type=str, default="results.jsonl", help="jsonl file to append the results to")
    parser.add_argument("--model", type=str, default=MODEL_WEIGHTS, help="Path to the YOLO model weights")
    parser.add_argument("--dev", action="store_true", help="Run in development mode with limited files and rows for testing")

    args = parser.parse_args()
    if args.dev:
        global MAX_FILES_TO_PROCESS, ROWS_TO_PROCESS_PER_FILE, SHOW_VERBOSE_PREDICTIONS
        MAX_FILES_TO_PROCESS = DEV_MAX_FILES_TO_PROCESS
        ROWS_TO_PROCESS_PER_FILE = DEV_ROWS_TO_PROCESS_PER_FILE
        SHOW_VERBOSE_PREDICTIONS = DEV_SHOW_VERBOSE_PREDICTIONS

    input_folder = args.input
    output_file = args.output
    model = load_model(args.model)

    parquet_files = get_parquet_files(input_folder)

    for file_path in parquet_files:
        process_parquet_file(file_path, model, output_file)

if __name__ == "__main__":
    main()