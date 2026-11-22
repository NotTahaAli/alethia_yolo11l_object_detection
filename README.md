# YOLO11l Object Detection in HuggingFace FineVideo Dataset for Alethia AI Take Home Assignment

- Dataset: `data/train-00000-of-01357.parquet` to `data/train-00001-of-01357.parquet` of [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
- Detection Model: Ultralytics YOLO11l specifically [yolo11l.pt](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11l.pt), download in the preparatory commands.
- Assignment details in AI_Engineering_TakeHome_Brief.pdf

## Installing requirements

Python 3.11. No `apt-get` packages are needed.

```sh
pip install -r requirements.txt
```

On Linux this installs the CUDA 12 build of PyTorch and TensorRT 10 (from the PyTorch and NVIDIA package indexes listed in `requirements.txt`).

## Preparatory Commands

Run once, from the repository root, on the machine that will run `run.py` (needs internet and the NVIDIA GPU):

```sh
hf download hf://Ultralytics/YOLO11@bee92e2/yolo11l.pt --local-dir yolo
python build_engine.py
```

`build_engine.py` exports `yolo/yolo11l.pt` to a TensorRT FP16 engine at `yolo/yolo11l.engine`. TensorRT engines only work on the GPU model and TensorRT version they were built with, so build it on the target machine. The engine has dynamic input shapes so frames get the same minimal letterbox as the `.pt` model, and a max batch of `BATCH_SIZE` from `run.py`.

## Running the Model

```sh
python run.py --input <folder of .parquet files> --output results.jsonl
```

On CUDA, `run.py` uses `yolo/yolo11l.engine` if it exists, otherwise `yolo/yolo11l.pt` (FP16). Elsewhere it uses `yolo/yolo11l.pt` in FP32. `--model <file>` overrides this.

Development options: `--max-files N`, `--max-rows N` (videos per file), `--dev` (one video) and `--fp32` (no FP16 for the `.pt` model on CUDA).

## Checking Correctness

`compare.py` scores a results file against a reference with the three checks below. To check the engine against full-precision PyTorch:

```sh
python run.py --input <folder> --output reference.jsonl --model yolo/yolo11l.pt --fp32 --max-rows 5
python run.py --input <folder> --output engine.jsonl --max-rows 5
python compare.py reference.jsonl engine.jsonl
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