# Answers to Writeup Questions (questions from Alethia)

## 1. Your timings. How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time? (100 words)

- The first version took 57 seconds for a 209 second clip on my machine (Estimated 34.09 hours for full 50 files). I didn't run on all 50 files as it would have taken so long and I didn't have time. The final version took 3042 seconds (Estimated 8.3 hours for full 50 files) on 150 videos (1.3 million frames) (3 randomly sampled rows from each file). The single best change was using FP16 quantized weights instead of the raw .pt weights.

## 2. Your pipeline. Describe what your script does, step by step, from opening a Parquet file to writing an output line. (150 words)

1. Reads Command line arguments and opens the input folder.
2. Opens Paraquet files and Extracts the video bytes from the mp4 column of every selected row.
3. Creates a task per video, added to the Task Queue.
5. A worker picks up the task from queue and decodes it using pyAV.
6. Decodes full video into frames in one go, and links frames to each second.
7. Each frame is transformed to NDArray in BGR and rotated (if needed) and sent to an internal event queue.
8. Picks up batches of 64 frames from the queue and runs the YOLO model on them after preprocessing.
9. Parses and aggregate the results per second based on rules in the brief.
10. Converts the result to JSON line and send to main process which then writes to the file.
11. Stops workers and exits cleanly when all videos processed.

## 3. Where the time goes. Which part of your final version takes the most time? How did you measure this? (150 ## words)

- On my hardware, the prediction through the YOLO model is still the bottleneck. It was measured by timing each step. The decoding step works nearly 15k frames per second on the CPU, but currently the best throughput I have gotten is 426 frames per second on the prediction model.

## 4. Checking correctness. How did you check that your results stayed correct after you made the script faster? (100 words)

- The first slow and correct script was run on `train-00000-of-01357.parquet` from row 0 to 9. The results were stored and any upcoming changes were measured against the stored results using the compare.py script.

## 5. Working with AI. Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it? (100 words)

- AI recommended exporting the yolo weights to the CUDA engine format. However, I decided not to use it because I tested it on Kaggle and the model failed to recognise any objects.

## 6. Next steps. If you had one more week, what would you change, and why? (100 words)

- If I had more time, I would run the tests properly on all files. I would also try to test different frame skips to see what is the best number of frames I can cut without compromising on accuracy. Because these are natural videos, ignoring cuts object and person movement takes time and thus the state change between two consecutive frames isn't that much. Additionally I would try to see how low I can get the quantization and figure out why the CUDA engine format failed.
