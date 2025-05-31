# YOLO-BEV-Simulation

A real-time object detection and visualization system that combines YOLOv5 with Bird’s Eye View (BEV) transformation. This project processes video input to detect vehicles and pedestrians, transforms their locations into a top-down view, and overlays visual representations to simulate traffic from an aerial perspective.

## 🔍 Features

- ✅ Object detection using YOLOv11 (https://github.com/ultralytics/yolov11)
- ✅ Centroid extraction and optional tracking
- ✅ Perspective transform to Bird’s Eye View (BEV)
- ✅ Real-time overlay of transparent icons for each object class
- ✅ Ego-vehicle overlay to simulate top-down context
- ✅ Frame-by-frame processing with video export

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy
