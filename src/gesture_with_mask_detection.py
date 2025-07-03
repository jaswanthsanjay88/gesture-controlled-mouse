# Gesture Controller with Face Mask Detection
import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
import numpy as np
import string
import ctypes
import sys
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[INFO] Ultralytics YOLOv8 not installed. Install with: pip install ultralytics")

print("Starting gesture controller with face mask detection...")

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Gesture Encodings 
class Gest(IntEnum):
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    V_GEST = 33
    TWO_FINGER_CLOSED = 34

class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

class HandRecog:
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
        
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign
        
    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
        
    def get_dz(self,point):
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
        
    def set_finger_state(self):
        if self.hand_result == None:
            return
        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger = 0
        self.finger = self.finger | 0
        for idx,point in enumerate(points):
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            
            try:
                ratio = round(dist/dist2,1)
            except:
                ratio = round(dist/0.01,1)

            self.finger = self.finger << 1
            if ratio > 0.5 :
                self.finger = self.finger | 1

    def get_gesture(self):
        if self.hand_result == None:
            return Gest.PALM
        current_gesture = Gest.PALM
        if Gest.FIRST2 == self.finger:
            point = [[8,12],[5,9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1/dist2
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8,12]) < 0.1:
                    current_gesture = Gest.TWO_FINGER_CLOSED
                else:
                    current_gesture = Gest.MID
        else:
            current_gesture = self.finger
            
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 4:
            self.ori_gesture = current_gesture
        return self.ori_gesture

class Controller:
    flag = False
    grabflag = False
    prev_hand = None
    
    @staticmethod
    def get_position(hand_result):
        point = 9
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0]*sx)
        y = int(position[1]*sy)
        if Controller.prev_hand is None:
            Controller.prev_hand = x, y
        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]

        distsq = delta_x**2 + delta_y**2
        ratio = 1
        Controller.prev_hand = [x, y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1/2))
        else:
            ratio = 2.1
        x, y = x_old + delta_x*ratio, y_old + delta_y*ratio
        return (x, y)

    @staticmethod
    def handle_controls(gesture, hand_result):
        x, y = None, None
        if gesture != Gest.PALM:
            x, y = Controller.get_position(hand_result)
            
        # flag reset
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")

        # implementation
        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=0.1)

        elif gesture == Gest.FIST:
            if not Controller.grabflag:
                Controller.grabflag = True
                pyautogui.mouseDown(button="left")
            pyautogui.moveTo(x, y, duration=0.1)

        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False

        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button='right')
            Controller.flag = False

        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False

class VirtualKeyboard:
    def __init__(self, layout=None, pos=(50, 100), key_size=(60, 60), spacing=10):
        # Enhanced QWERTY layout with space, backspace, enter, shift
        if layout is None:
            self.layout = [
                list("QWERTYUIOP"),
                list("ASDFGHJKL"),
                ["Shift"] + list("ZXCVBNM") + ["Back", "Enter"],
                ["Space"]
            ]
        else:
            self.layout = layout
        self.pos = pos
        self.key_size = key_size
        self.spacing = spacing
        self.keys = self._generate_keys()
        self.last_pressed = None
        self.last_press_time = 0
        self.dwell_start = None
        self.dwell_key = None
        self.shift = False

    def _generate_keys(self):
        keys = []
        x0, y0 = self.pos
        w, h = self.key_size
        for row_idx, row in enumerate(self.layout):
            for col_idx, key in enumerate(row):
                # Space bar is wider
                if key == "Space":
                    width = w * 5 + self.spacing * 4
                    x = x0
                    y = y0 + row_idx * (h + self.spacing)
                    keys.append({'key': key, 'rect': (x, y, width, h)})
                    continue
                # Backspace and Enter are wider
                if key in ("Back", "Enter"):
                    width = w * 1.5
                else:
                    width = w
                x = x0 + col_idx * (w + self.spacing)
                y = y0 + row_idx * (h + self.spacing)
                keys.append({'key': key, 'rect': (x, y, width, h)})
        return keys

    def draw(self, image):
        for k in self.keys:
            x, y, w, h = map(int, k['rect'])
            color = (200, 200, 200)
            if self.dwell_key == k['key']:
                # Show dwell progress as a green bar
                color = (0, 255, 0)
                if self.dwell_start:
                    elapsed = min((cv2.getTickCount() - self.dwell_start) / cv2.getTickFrequency(), 0.5)
                    bar_width = int(w * (elapsed / 0.5))
                    cv2.rectangle(image, (x, y + h - 8), (x + bar_width, y + h), (0, 255, 0), -1)
            if self.last_pressed == k['key']:
                color = (0, 180, 255)
            cv2.rectangle(image, (x, y), (x + int(w), y + int(h)), color, -1)
            cv2.rectangle(image, (x, y), (x + int(w), y + int(h)), (50, 50, 50), 2)
            label = k['key']
            if self.shift and len(label) == 1 and label.isalpha():
                label = label.upper()
            elif not self.shift and len(label) == 1 and label.isalpha():
                label = label.lower()
            font_scale = 1.2 if len(label) == 1 else 0.8
            cv2.putText(image, label, (x + 10, y + int(h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    def get_key_at(self, x, y):
        for k in self.keys:
            x0, y0, w, h = map(int, k['rect'])
            if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
                return k['key']
        return None

    def process_fingertip(self, x, y):
        import time
        key = self.get_key_at(x, y)
        now = cv2.getTickCount()
        if key:
            if self.dwell_key != key:
                self.dwell_key = key
                self.dwell_start = now
            else:
                elapsed = (now - self.dwell_start) / cv2.getTickFrequency()
                if elapsed > 0.5:
                    self.press_key(key)
                    self.dwell_key = None
                    self.dwell_start = None
        else:
            self.dwell_key = None
            self.dwell_start = None

    def press_key(self, key):
        # Handle special keys
        if key == "Space":
            pyautogui.press('space')
        elif key == "Back":
            pyautogui.press('backspace')
        elif key == "Enter":
            pyautogui.press('enter')
        elif key == "Shift":
            self.shift = not self.shift
        elif len(key) == 1:
            char = key.upper() if self.shift else key.lower()
            pyautogui.press(char)
            if self.shift:
                self.shift = False
        self.last_pressed = key
        self.last_press_time = cv2.getTickCount()

class GestureController:
    gc_mode = 0
    cap = None
    hr_major = None
    hr_minor = None
    dom_hand = True
    face_mask_detected = False
    face_detection_confidence = 0.5
    mask_detection_history = []
    keyboard_mode = False
    vk = VirtualKeyboard()
    yolo_mode = False
    yolo_model = None
    
    def __init__(self):
        print("Initializing Gesture Controller with Face Mask Detection...")
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)
        if not GestureController.cap.isOpened():
            print("Error: Could not open camera!")
            return
        
        # Set camera resolution for better performance
        GestureController.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        GestureController.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera opened successfully")

        # Load YOLOv8 model if available
        if YOLO is not None:
            try:
                GestureController.yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
                print("YOLOv8 model loaded.")
            except Exception as e:
                print(f"Could not load YOLOv8 model: {e}")
                GestureController.yolo_model = None

    @staticmethod
    def detect_face_mask(image, face_results):
        """
        Enhanced face mask detection using multiple color ranges and morphological operations
        """
        mask_detected = False
        mask_confidence = 0.0
        
        if face_results.detections:
            for detection in face_results.detections:
                # Get face bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, w-1))
                y = max(0, min(y, h-1))
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 0 and height > 0:
                    # Extract face region
                    face_roi = image[y:y+height, x:x+width]
                    
                    if face_roi.size > 0:
                        # Convert to HSV for better mask detection
                        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                        
                        # Define comprehensive mask color ranges
                        color_ranges = [
                            # Blue surgical masks
                            ([100, 50, 50], [130, 255, 255]),
                            # Light blue masks
                            ([85, 50, 70], [115, 255, 255]),
                            # White/light masks
                            ([0, 0, 180], [180, 30, 255]),
                            # Gray masks
                            ([0, 0, 50], [180, 50, 150]),
                            # Dark masks (black/navy)
                            ([0, 0, 0], [180, 255, 80]),
                            # Green masks
                            ([35, 40, 40], [85, 255, 255])
                        ]
                        
                        # Create combined mask for all colors
                        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                        
                        for lower, upper in color_ranges:
                            color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                            combined_mask = cv2.bitwise_or(combined_mask, color_mask)
                        
                        # Apply morphological operations to reduce noise
                        kernel = np.ones((3,3), np.uint8)
                        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                        
                        # Focus on lower 2/3 of face (mouth and nose area)
                        lower_face_start = height // 3
                        lower_face = combined_mask[lower_face_start:, :]
                        
                        # Calculate mask coverage percentage
                        mask_pixels = cv2.countNonZero(lower_face)
                        total_pixels = lower_face.shape[0] * lower_face.shape[1]
                        mask_confidence = mask_pixels / total_pixels if total_pixels > 0 else 0
                        
                        # Threshold for mask detection (12% coverage)
                        if mask_confidence > 0.12:
                            mask_detected = True
                            # Draw green rectangle for mask detected
                            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            cv2.putText(image, f"MASK DETECTED ({mask_confidence:.1%})", 
                                       (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(image, "Face tracking enhanced", 
                                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            # Draw red rectangle for no mask
                            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
                            cv2.putText(image, f"NO MASK ({mask_confidence:.1%})", 
                                       (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Show detection confidence
                        detection_conf = detection.score[0] if detection.score else 0
                        cv2.putText(image, f"Face: {detection_conf:.2f}", 
                                   (x, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return mask_detected, mask_confidence

    @staticmethod
    def classify_hands(results):
        left, right = None, None
        try:
            from google.protobuf.json_format import MessageToDict
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else:
                left = results.multi_hand_landmarks[0]
        except:
            pass

        try:
            handedness_dict = MessageToDict(results.multi_handedness[1])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[1]
            else:
                left = results.multi_hand_landmarks[1]
        except:
            pass
            
        if GestureController.dom_hand == True:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else:
            GestureController.hr_major = left
            GestureController.hr_minor = right

    def start(self):
        print("Starting gesture controller with enhanced face mask detection and virtual keyboard...")
        if not GestureController.cap.isOpened():
            print("Error: Camera is not opened!")
            return
        window_width, window_height = 900, 700
        cv2.namedWindow('Gesture Controller with Enhanced Face Mask Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gesture Controller with Enhanced Face Mask Detection', window_width, window_height)
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, 'Gesture Controller with Enhanced Face Mask Detection')
            if hwnd:
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
        except Exception as e:
            print(f"Window always-on-top failed: {e}")
        handmajor = HandRecog(HLabel.MAJOR)
        print("Hand recognition objects created")
        face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            print("MediaPipe hands and face detection initialized, starting main loop...")
            frame_count = 0
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, image = GestureController.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                frame_count += 1
                if frame_count % 60 == 0:
                    mask_status = "DETECTED" if GestureController.face_mask_detected else "NOT DETECTED"
                    print(f"Frame {frame_count} - Mask status: {mask_status}")
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                hand_results = hands.process(image)
                face_results = face_detection.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mask_detected, mask_confidence = GestureController.detect_face_mask(image, face_results)
                GestureController.face_mask_detected = mask_detected
                GestureController.mask_detection_history.append(mask_detected)
                if len(GestureController.mask_detection_history) > 10:
                    GestureController.mask_detection_history.pop(0)
                stable_mask_detected = sum(GestureController.mask_detection_history) > len(GestureController.mask_detection_history) // 2
                # Facial behavior and mood detection
                mood_label = "Unknown"
                if face_results.detections:
                    for detection in face_results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = image.shape
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        x = max(0, min(x, w-1))
                        y = max(0, min(y, h-1))
                        width = min(width, w - x)
                        height = min(height, h - y)
                        if width > 0 and height > 0:
                            face_roi = image[y:y+height, x:x+width]
                            if face_roi.size > 0:
                                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                                mouth_roi = gray[int(height*0.65):, :]
                                _, mouth_thresh = cv2.threshold(mouth_roi, 80, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                                white = cv2.countNonZero(mouth_thresh)
                                total = mouth_thresh.shape[0] * mouth_thresh.shape[1]
                                mouth_ratio = white / total if total > 0 else 0
                                if mouth_ratio > 0.25:
                                    mood_label = "Happy"
                                elif mouth_ratio > 0.15:
                                    mood_label = "Neutral"
                                else:
                                    mood_label = "Sad"
                                cv2.putText(image, f"Mood: {mood_label}", (x, y + height + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                # Removed overlay drawing and overlay mode toggle
                # ...existing code for keyboard/mouse mode, overlays removed...
                key = cv2.waitKey(1) & 0xFF
                if key == ord('k'):
                    GestureController.keyboard_mode = not GestureController.keyboard_mode
                    print(f"Keyboard mode: {GestureController.keyboard_mode}")
                if key == ord('y'):
                    GestureController.yolo_mode = not GestureController.yolo_mode
                    print(f"YOLOv8 Object Detection: {'ON' if GestureController.yolo_mode else 'OFF'}")
                if key == 13:
                    print("Exit key pressed")
                    break
                if GestureController.keyboard_mode:
                    GestureController.vk.draw(image)
                    if hand_results.multi_hand_landmarks:
                        index_tip = hand_results.multi_hand_landmarks[0].landmark[8]
                        h, w, _ = image.shape
                        fx, fy = int(index_tip.x * w), int(index_tip.y * h)
                        cv2.circle(image, (fx, fy), 10, (255, 0, 0), -1)
                        GestureController.vk.process_fingertip(fx, fy)
                else:
                    if hand_results.multi_hand_landmarks:
                        GestureController.classify_hands(hand_results)
                        handmajor.update_hand_result(GestureController.hr_major)
                        handmajor.set_finger_state()
                        gest_name = handmajor.get_gesture()
                        Controller.handle_controls(gest_name, handmajor.hand_result)
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    else:
                        Controller.prev_hand = None
                # YOLOv8 object detection
                if GestureController.yolo_mode and GestureController.yolo_model is not None:
                    results = GestureController.yolo_model(image, verbose=False)
                    for r in results:
                        boxes = r.boxes
                        names = r.names if hasattr(r, 'names') else None
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            label = names[cls] if names else str(cls)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                status_color = (0, 255, 0) if stable_mask_detected else (0, 0, 255)
                status_text = f"MASK: {'ON' if stable_mask_detected else 'OFF'}"
                cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                if mask_confidence > 0:
                    cv2.putText(image, f"Confidence: {mask_confidence:.1%}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                instructions = [
                    "CONTROLS:",
                    "V-Gesture: Move cursor",
                    "Fist: Click & drag",
                    "Middle finger (after V): Left click",
                    "Index finger (after V): Right click",
                    "Two fingers closed (after V): Double click",
                    "Press ENTER to exit"
                ]
                for i, instruction in enumerate(instructions):
                    y_pos = image.shape[0] - 20 - (len(instructions) - i - 1) * 20
                    cv2.putText(image, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if stable_mask_detected:
                    cv2.putText(image, " TRACKING ACTIVE", (image.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                mode_text = "KEYBOARD MODE" if GestureController.keyboard_mode else "MOUSE MODE"
                cv2.putText(image, mode_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                yolo_text = "YOLOv8: ON" if GestureController.yolo_mode else "YOLOv8: OFF"
                cv2.putText(image, yolo_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow('Gesture Controller with Enhanced Face Mask Detection', image)
                if cv2.waitKey(5) & 0xFF == 13:
                    print("Exit key pressed")
                    break
        print("Cleaning up...")
        GestureController.cap.release()
        cv2.destroyAllWindows()
        print("Gesture Controller with Face Mask Detection stopped.")

if __name__ == "__main__":
    print("Starting Enhanced Gesture Controller with Face Mask Detection...")
    try:
        gc = GestureController()
        gc.start()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Application ending...")
        try:
            cv2.destroyAllWindows()
        except:
            pass
