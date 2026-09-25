"""Traffic event detection for a fixed road camera (WIUT Hackathon 2026, CV track)."""
import os

# The evaluation machine has no internet: never let ultralytics try to reach it.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")
