# KickVision

This repository features a comprehensive computer vision and machine learning project dedicated to football analysis. It employs various cutting-edge techniques to provide detailed insights into player movements and actions.

## Features

1. **Object Detection with YOLOv8**
   - Utilizes ultralytics and YOLOv8 to detect objects in images and videos.
   - Fine-tune and train YOLO on a custom dataset for tailored detection.

2. **Pixel Segmentation with KMeans**
   - Clusters pixels to segment players from the background.
   - Accurate t-shirt color detection using advanced K-means clustering.

3. **Motion Tracking with Optical Flow**
   - Measures camera movement through optical flow analysis.
   - Tracks player motion seamlessly across frames.

4. **Perspective Transformation with OpenCV**
   - Uses CV2's perspective transformation to represent depth and perspective of the scene.
   - Enhances visual analysis by adjusting for perspective changes.

5. **Player Movement Analysis**
   - Measures player speed and distance covered in the image.
   - Provides quantitative data on player performance and movement.

## Datasets
- 30s football clips: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips
- Fine tuning and training of YOLO: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1

## Iteration 1
![Iteration 1](images/iteration1.png)

## Iteration 2
![Iteration 2](images/iteration2.png)

## Iteration 3
![Iteration 3](images/iteration3.png)

Work in progress...
