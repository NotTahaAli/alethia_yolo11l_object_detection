# YOLO11l Object Detection in HuggingFace FineVideo Dataset for Alethia AI Take Home Assignment

- Dataset: `data/train-00000-of-01357.parquet` to `data/train-00001-of-01357.parquet` of [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
- Detection Model: Ultralytics YOLO11l specifically [yolo11l.pt](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11l.pt), download in the preparatory commands.
- Assignment details in AI_Engineering_TakeHome_Brief.pdf

## Installing requirements

```sh
pip install -r requirements.txt
```

## Preparatory Commands

```sh
hf download hf://Ultralytics/YOLO11@bee92e2/yolo11l.pt --local-dir yolo
```

## Running the Model

```sh
python run.py --input <folder of .parquet files> --output results.jsonl
```

## Output Format

```jsonl
{"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people":
2}
{"video": "train-00003-of-01357.parquet:4", "second": 13, "objects": [], "people": 0}
{"video": "train-00003-of-01357.parquet:5", "error": "could not decode video"}
```

- `video` is the Parquet file name without its folder, a colon, and the row number in that file, counting from 0.
- Second 12 holds the frames whose timestamp is at least 12.0 and below 13.0 seconds, counted from the first decoded frame of the video.
- There is a line for every second up to and including the second that holds the last frame. A second with nothing found has an empty list and "people": 0.
- Object names are exactly `person, car, dog, laptop, cell phone`. They are irrespective of order.
- If a video can't be processed at any point, there exists an error message `could not decode video`. earlier seconds are already logged, after failure that video is aborted.

## Detection Criteria

- An object is present in a second if the model finds it in at least half of that second's frames.
- The people count for a second is the median of the number of people found in each frame of that second, rounded down when it falls halfway between two numbers.

## Testing Criteria (as per Alethia)

- Objects must have an F1 score of at least 0.95 against the reference.
- People count must match the reference exactly on at least 85% of seconds.
- People count must be within 1 of the reference on at least 95% of seconds.
- The test set is about 13 hours of FineVideo video in the same Parquet format, Some test files may be damaged or unusual. Total test runtime must be less than 6 hours on NVIDIA RTX 4090 with 24 GB of memory, 32 thread CPU with 64 GB of RAM with Ubuntu 22.04, Python 3.11, and an NVIDIA driver that supports CUDA 12. Some test files may be damaged or unusual.