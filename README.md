# Gesture-Controlled Mouse

A Python project that enables mouse control using hand gestures detected via a webcam. This project uses computer vision and deep learning to recognize gestures and translate them into mouse actions.

## Features
- Real-time hand gesture detection
- Mouse movement and click simulation
- Easy to set up and run

## Requirements
- Python 3.8+
- OpenCV
- PyAutoGUI
- NumPy
- (Optional) YOLOv8 for advanced gesture detection

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/jaswanthsanjay88/gesture-controlled-mouse.git
   cd gesture-controlled-mouse
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   Or install manually:
   ```sh
   pip install opencv-python pyautogui numpy
   ```

## Usage
Run the main gesture controller script:
```sh
python src/Gesture_Controller.py
```

You can also test the camera or run minimal tests:
```sh
python src/test_camera.py
python src/minimal_test.py
```

## File Structure
- `src/Gesture_Controller.py` - Main script for gesture-controlled mouse
- `src/gesture_with_mask_detection.py` - Gesture detection with mask support
- `src/simple_gesture_controller.py` - Simplified version
- `src/test_camera.py` - Camera test utility
- `src/minimal_test.py` - Minimal test script
- `src/yolov8n.pt` - YOLOv8 model weights (if used)

## License
This project is licensed under the MIT License.

## Author
jaswanthsanjay88
