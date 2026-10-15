# Real-Time Crowd Detection & Occupancy Analytics

A Python computer vision tool that detects and counts people in images, video files, or a live webcam using a YOLO model and OpenCV. It then tracks occupancy over time and flags when a space goes over capacity.

## What it does

- **Detects people** in each frame with a pretrained YOLOv8 model, keeping only the "person" class
- **Counts occupancy over time** for video, sampling every Nth frame for speed
- **Smooths the count** with a rolling average, so one-frame detection flickers don't distort the trend
- **Flags capacity breaches**: boxes and the on-screen banner turn red when the count exceeds a limit you set
- **Summarizes each video**: average, median, and peak count, when the peak happened, and how long the space was over capacity

## Outputs

For a video like `mall.mp4`, results go to `outputs/`:

| File | Contents |
|---|---|
| `mall_annotated.mp4` | Video with bounding boxes and a live people count |
| `mall_occupancy.csv` | Frame, timestamp, raw count, and smoothed count |
| `mall_occupancy.png` | Chart of occupancy over time, with the capacity line |
| `mall_summary.json` | Average/peak counts, peak time, time over capacity |

For an image, you get an annotated copy and a `summary.json` with the count.

## Run it

```bash
pip install -r requirements.txt

# Image
python detect.py --source path/to/photo.jpg

# Video, with a capacity limit of 20 people
python detect.py --source path/to/video.mp4 --capacity 20

# Webcam (press Ctrl+C to stop)
python detect.py --source 0 --capacity 10
```

The YOLOv8n weights (about 6 MB) download automatically the first time you run it.

Useful options:

| Flag | Default | Meaning |
|---|---|---|
| `--every` | 5 | Analyze every Nth frame (higher = faster) |
| `--conf` | 0.4 | Minimum confidence for a detection |
| `--model` | `yolov8n.pt` | Swap in a larger model like `yolov8s.pt` for better accuracy |
| `--no-video` | off | Skip writing the annotated video |

## How it works

```
video frame ──► YOLOv8 (person class only) ──► bounding boxes
                                                   │
              ┌────────────────────────────────────┤
              ▼                                    ▼
   annotated frame (OpenCV)            OccupancyTracker.record(time, count)
                                                   │
                                                   ▼
                                CSV  +  chart  +  summary stats
```

- `crowd/detector.py`: wraps the YOLO model and returns person boxes with confidence scores
- `crowd/occupancy.py`: stores counts over time and computes the summary statistics and chart
- `crowd/annotate.py`: draws boxes and the count banner with OpenCV
- `crowd/pipeline.py`: reads images or video, samples frames, and writes all outputs
- `detect.py`: the command-line interface

## Tests

The tests use a scripted fake detector, so they run without downloading a model:

```bash
python -m unittest discover tests
```

They check the occupancy statistics, the rolling average, frame sampling, capacity flags, and that every output file is created.

## Next steps

- Add object tracking (e.g. ByteTrack) to count unique people entering and leaving, not just people in frame
- Define zones in the frame and count occupancy per zone
- Build a Streamlit dashboard for uploading a video and viewing the results
