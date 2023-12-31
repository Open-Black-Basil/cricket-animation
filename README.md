<!--
<div align="center">
-->
YOLOv5 🚀 is the world's most loved vision AI, representing <a href="https://ultralytics.com">Ultralytics</a> open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

![ball](https://github.com/Black-Basil-Technologies-Pvt-Ltd/cricket-animation/blob/387dbf5a0e15b60eee70b863f6e748faa4d765ff/video/cricketBallTracking.gif)
 

We have made changes in this repo <a href="https://github.com/ultralytics/yolov5">Ultralytics_yolov5</a> and trained these yolov5s.pt models from scratch on our cricket dataset.

you can download our trained weights from google drive link <a href='https://drive.google.com/drive/folders/1F6dBoGrqOBGPV_4efOYsNGH96RKpFFRo?usp=sharing'> Drive link</a> 

Different yolov5 model that are available which can further be used to enhanched the training accuracy -- 

| Model                                                                                           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| ----------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ---------------------------- | ----------------------------- | ------------------------------ | ------------------ | ---------------------- |
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)              | 640                   | 28.0                 | 45.7              | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)              | 640                   | 37.4                 | 56.8              | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)              | 640                   | 45.4                 | 64.1              | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)              | 640                   | 49.0                 | 67.3              | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)              | 640                   | 50.7                 | 68.9              | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  |
|                                                                                                 |                       |                      |                   |                              |                               |                                |                    |                        |
| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt)            | 1280                  | 36.0                 | 54.4              | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt)            | 1280                  | 44.8                 | 63.7              | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt)            | 1280                  | 51.3                 | 69.3              | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt)            | 1280                  | 53.7                 | 71.3              | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |




Basic Requirements 
```commandline
pip install ultralytics

git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

-------------Train on your custom DATA ---------------

```bash
python train.py --data dataset.yaml --epochs 150 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```
------------------------Inference on Video and Image Data------------------------ </br>
For Ball Tracking
```bash
python detect.py --weights yolov5s.pt --balltrek --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

For Ball and Bat Impact Detection
```bash
python detect.py --weights yolov5s.pt --balltrek --vid.mp4                         # video
```
To cartoonize a video or picture 
```bash
python detect.py --weights yolov5s.pt --cartoon --vid.mp4                         # video
```

# SHOT Detection

```bash
cd cricket_pose_estimation/                       
```

<h1></h1>

<ul>

<li> 
  whitout keypoints
</li>
</ul>

![video1](https://github.com/Black-Basil-Technologies-Pvt-Ltd/cricket-animation/blob/a061d0dbe85ff83ccd8a2a611cee350b43e73917/cricket_pose_estimation/results/shot_det.gif)

<ul>
<li>
  with keypoints and bounding box
</li>
</ul>

![video2](https://github.com/Black-Basil-Technologies-Pvt-Ltd/cricket-animation/blob/edd0b7ed3db8387c5d0a6f29c2a999f1630ff569/cricket_pose_estimation/results/keypoint_shot_detection.gif)



