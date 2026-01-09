"""
Camera utilities for silkworm detection
Handles camera setup, configuration, and management
"""

import cv2
from typing import List


def list_cameras(max_index: int = 5) -> List[int]:
    """List available cameras."""
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    print(f"Available cameras: {available if available else 'None detected (tried 0..' + str(max_index) + ')'}")
    return available


def setup_camera(camera_index: int, target_fps: int = 15, width: int = 640, height: int = 480):
    """Setup camera with optimized settings."""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}, trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open any camera (tried {camera_index} and 0)")
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # Verify actual resolution (camera may not support requested resolution)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Test camera read
    ret, _ = cap.read()
    if not ret:
        print("Camera cannot read frames, retrying...")
        cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera resolution: {actual_width}x{actual_height} (requested: {width}x{height})")
    return cap


def get_model_info(model_path: str) -> str:
    """Get model information and performance hints."""
    if "ncnn" in model_path.lower():
        return "NCNN model - optimized for Pi"
    elif "onnx" in model_path.lower():
        return "ONNX model - optimized for Pi"
    else:
        return "PyTorch model - may be slow on Pi"
