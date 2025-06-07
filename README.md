Overview

This project implements a pedestrian crossing analysis system that utilizes YOLOv5 for pedestrian detection and DeepSORT for tracking. The system processes video input, allows manual zone selection, tracks pedestrian movement, logs data into a PostgreSQL database, and generates a graph of pedestrian counts over time.

Installation

Ensure Python 3.8+ is installed.

Install required packages:

pip install opencv-python torch ultralytics deep_sort_realtime matplotlib psycopg2-binary numpy annotated-types

Set up PostgreSQL:

Create a database named pedestrian_analysis.

Update the get_connection function in db.py with your PostgreSQL credentials (e.g., user, password, host, port).

Usage

Run the script:

python app.py

A video window will open displaying the video feed.

Select a crossing zone by clicking and dragging the mouse to define a rectangle.

Press SPACE to confirm the zone and start the analysis.

Press r to reset the analysis (clears zone and data).

Press ESC to exit and generate the final graph.

The system will save a graph as pedestrian_analysis.png and display it (if possible).

Requirements

Python 3.8+
Libraries:
opencv-python (cv2)
torch
ultralytics (for YOLOv5)
deep_sort_realtime (for DeepSORT)
matplotlib (for graphing)
psycopg2-binary (for PostgreSQL)
numpy
annotated-types

Features

Real-time pedestrian detection using YOLOv5.
Tracking of pedestrians using DeepSORT.
Interactive zone selection with mouse input.
Data logging to a PostgreSQL database (videos, zones, pedestrians, analysis).
Visualization of pedestrian counts per second in a graph.

Database Schema

videos: Stores video file paths and names (id, file_path, video_name).
zones: Stores zone polygons linked to videos (id, video_id, polygon).
pedestrians: Records pedestrian entry/exit times and zones (id, zone_id, entry_time, exit_time, video_id).
analysis: Logs analysis data (id, pedestrian_id, duration).

Configuration

Edit db.py to match your PostgreSQL setup.
Video path is hardcoded in main(); update video_path as needed.

Troubleshooting

Video not opening: Ensure the video file path is correct and accessible.
FPS error: Verify the video file is valid and contains metadata.
Graph not displaying: Check if matplotlib is installed and there are no display issues (e.g., running on a headless server).