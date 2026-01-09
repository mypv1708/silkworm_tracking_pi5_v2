"""
Main detection engine
Handles the core detection loop and processing pipeline with threaded capture
Optimized for Raspberry Pi 4/5 (no manual resize - YOLO handles letterbox)
"""

import time
from threading import Thread, Lock
from pathlib import Path
from ultralytics import YOLO
import cv2

from config import Config
from camera_utils import setup_camera, get_model_info
from silkworm_detection import process_detections
from display_utils import draw_performance_info, draw_scale_bar, display_frame
from performance_monitor import PerformanceMonitor
from object_tracker import SilkwormTracker
from freeze_detection import process_freeze
from overlap_detection import process_overlap
from heatmap_visualizer import HeatmapVisualizer


class DetectionEngine:
    """Main detection engine for silkworm detection."""

    def __init__(self, args, cfg: Config):
        self.args = args
        self.cfg = cfg
        self.mode = getattr(args, "mode", "pose")

        # Initialize
        self.model = self._setup_model()
        self.cap = self._setup_camera()
        self.performance_monitor = PerformanceMonitor()

        # Mode-specific initialization
        if self.mode == "pose":
            self.tracker = self._setup_tracker()
            self.overlap_counters = {}
            self.head_history = {}
            self.freeze_counters = {}
        elif self.mode == "heatmap":
            self.tracker = self._setup_tracker()  # Use tracker for stable IDs
            self.heatmap_visualizer = None  # Lazy init with actual frame size
            # Not needed in heatmap mode
            self.overlap_counters = None
            self.head_history = None
            self.freeze_counters = None

        # Threaded capture
        self.running = True
        self._frame_lock = Lock()
        self._latest_frame = None
        self._capture_thread = Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.device = cfg.device if cfg.device is not None else 'cpu'

        print(f"Mode: {self.mode}")
        print(f"Camera: {cfg.camera_width}x{cfg.camera_height} @ {args.fps} FPS")
        print(f"Inference size: {args.imgsz}")
        print(f"Config: conf={args.conf}, skip={args.skip}, pose_conf={cfg.pose_conf}")

        # ✅ Raspberry Pi CPU optimization
        if self.device == 'cpu':
            try:
                import torch
                torch.set_num_threads(2)  # best for Pi4/5
                print("[Torch] threads:", torch.get_num_threads())
            except Exception:
                pass

        # ✅ Reduce camera buffer latency
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass

    def _setup_model(self):
        model_path = self.args.model if getattr(self.args, "model", None) else self.cfg.model_path
        
        # Use "pose" task for both modes - model was trained for pose detection
        # In heatmap mode, we just use the boxes (ignore keypoints)
        model_task = getattr(self.args, "task", None) or getattr(self.cfg, "model_task", "pose")

        # Ultralytics sometimes cannot infer the task from exported formats (e.g. .onnx),
        # which breaks pose pipelines (result.keypoints becomes None). Force the task.
        #
        # Also disable Ultralytics "AutoUpdate" (pip installs) on low-power / restricted
        # environments (e.g. Raspberry Pi) to avoid permission/network issues.
        if str(model_path).lower().endswith(".onnx"):
            try:
                from ultralytics.utils import checks
                checks.AUTOINSTALL = False
            except Exception:
                pass

        print(get_model_info(model_path))
        print(f"Model task: {model_task}")

        # Normalize relative paths to this project directory for convenience
        try:
            p = Path(model_path)
            if not p.is_absolute():
                base_dir = Path(__file__).resolve().parent
                model_path = str((base_dir / p).resolve())
        except Exception:
            pass

        return YOLO(model_path, task=model_task)

    def _setup_camera(self):
        return setup_camera(
            self.args.camera, 
            self.args.fps,
            self.cfg.camera_width,
            self.cfg.camera_height
        )

    def _setup_tracker(self):
        return SilkwormTracker(
            max_distance=self.cfg.max_distance,
            max_disappeared=self.cfg.max_disappeared
        )

    def run(self):
        try:
            while True:
                # Get frame
                with self._frame_lock:
                    frame = None if self._latest_frame is None else self._latest_frame.copy()
                    self._latest_frame = None
                if frame is None:
                    time.sleep(0.001)
                    continue

                # Skip for performance
                if (self.performance_monitor.total_frames_read % max(1, self.args.skip)) != 0:
                    self.performance_monitor.start_frame()
                    continue

                # Process
                self._process_frame(frame)
                self.performance_monitor.start_frame()

                if self.performance_monitor.should_cleanup():
                    self.performance_monitor.cleanup()

                self._control_fps()

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self._cleanup()
            self._print_summary()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._latest_frame = frame
            self.performance_monitor.record_frame_read()

    # ✅ Optimized for Raspberry Pi — no manual resize
    def _process_frame(self, frame):
        inference_start = time.time()
        results = self.model(
            frame,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.cfg.iou_thresh,
            device=self.device,
            verbose=False
        )
        inference_time = time.time() - inference_start

        result = results[0]

        if self.mode == "heatmap":
            # Heatmap mode: use pose model but only extract boxes for heatmap
            # Lazy initialize heatmap visualizer with actual frame size
            if self.heatmap_visualizer is None:
                h, w = frame.shape[:2]
                self.heatmap_visualizer = HeatmapVisualizer(
                    frame_shape=(h, w),
                    grid_shape=self.cfg.heatmap_grid_shape,
                    opacity=self.cfg.heatmap_opacity,
                    draw_grid=self.cfg.heatmap_draw_grid
                )
                # Heatmap visualizer initialized with actual frame size
            
            # Extract and filter detections from pose model
            boxes_list = []
            tracked_objects = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Filter by confidence threshold
                confs = result.boxes.conf.cpu().numpy()
                conf_mask = confs >= self.args.conf
                boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(float)
                boxes_xyxy = boxes_xyxy[conf_mask]
                
                # Prepare detections for tracker: create dummy keypoints from bbox center
                # Tracker expects (head, body, tail, bbox) format
                detections_for_tracker = []
                for box in boxes_xyxy:
                    x1, y1, x2, y2 = box
                    
                    # Filter invalid boxes (too small or invalid dimensions)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (x2 - x1) < 3 or (y2 - y1) < 3:  # Minimum size threshold
                        continue
                    
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    # Create dummy keypoints: head at top, body at center, tail at bottom
                    # Ensure head, body, tail are distinct (at least 1 pixel apart)
                    head_y = int(y1)
                    body_y = int(cy)
                    tail_y = int(y2)
                    # If box is too small, adjust keypoints to be distinct
                    if tail_y <= head_y:
                        tail_y = head_y + 1
                    if body_y <= head_y:
                        body_y = head_y + 1
                    if body_y >= tail_y:
                        body_y = tail_y - 1
                    
                    head = (int(cx), head_y)
                    body = (int(cx), body_y)
                    tail = (int(cx), tail_y)
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    detections_for_tracker.append((head, body, tail, bbox))
                
                # Update tracker to get stable IDs
                tracked_objects = self.tracker.update(detections_for_tracker)
                
                # Draw bounding boxes (green) with stable ID labels
                # Pre-compute frame bounds once
                frame_h, frame_w = frame.shape[0], frame.shape[1]
                for obj_id, head, body, tail, bbox in tracked_objects:
                    x1, y1, x2, y2 = bbox
                    # Clamp to frame bounds (optimized: single min/max per coordinate)
                    x1 = max(0, min(x1, frame_w - 1))
                    y1 = max(0, min(y1, frame_h - 1))
                    x2 = max(0, min(x2, frame_w - 1))
                    y2 = max(0, min(y2, frame_h - 1))
                    if x2 > x1 and y2 > y1:
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw stable ID label (cache text size calculation)
                        label = f"ID {obj_id}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        label_y = max(y1 - 5, label_size[1] + 5)
                        label_bg_h = label_size[1] + 10
                        label_bg_w = label_size[0] + 5
                        # Draw background for text
                        cv2.rectangle(frame, (x1, label_y - label_bg_h), 
                                     (x1 + label_bg_w, label_y + 5), (0, 255, 0), -1)
                        # Draw text
                        cv2.putText(frame, label, (x1 + 2, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Prepare boxes list for heatmap (just bbox coordinates)
                # tracked_objects format: (obj_id, head, body, tail, bbox)
                # bbox format: (x1, y1, x2, y2)
                boxes_list = [[float(x1), float(y1), float(x2), float(y2)] 
                             for _, _, _, _, (x1, y1, x2, y2) in tracked_objects]
                self.performance_monitor.record_detections(len(tracked_objects))
            else:
                # No detections - update tracker with empty list
                tracked_objects = self.tracker.update([])
            
            # Update heatmap overlay (even if empty, to show grid)
            frame = self.heatmap_visualizer.update(frame, boxes_list)
            
            # Display performance - pass tracked objects for correct count
            if self.args.benchmark:
                self.performance_monitor.record_inference_time(inference_time)
                draw_performance_info(frame, inference_time, tracked_objects,
                                      self.performance_monitor.frame_count, self.performance_monitor)

            draw_scale_bar(frame)

            if not self.args.no_display:
                if display_frame(frame):
                    raise KeyboardInterrupt("Exit")
        
        else:
            # Pose mode: tracking with keypoints (original behavior)
            # Fail-fast for pose pipeline: if model doesn't output keypoints, downstream tracking is meaningless.
            expected_task = getattr(self.args, "task", None) or getattr(self.cfg, "model_task", "pose")
            if expected_task == "pose" and getattr(result, "keypoints", None) is None:
                raise RuntimeError(
                    "Model output has no keypoints. This project expects a pose model.\n"
                    "Fix: set `model.task: \"pose\"` in config.yaml (or run `python3 main.py --task pose`), "
                    "and ensure the exported model actually contains keypoints."
                )
            silkworms = process_detections(result, frame, self.cfg, self.tracker)

            if result.boxes is not None:
                self.performance_monitor.record_detections(len(result.boxes))

            # Freeze logic
            for obj_id, head, body, tail, bbox in silkworms:
                process_freeze(obj_id, head, 1.0, bbox, self.cfg,
                               self.head_history, self.freeze_counters, frame)

            # Overlap
            if len(silkworms) >= 2:
                process_overlap(silkworms, self.overlap_counters, frame, self.cfg)

            # Display performance
            if self.args.benchmark:
                self.performance_monitor.record_inference_time(inference_time)
                draw_performance_info(frame, inference_time, silkworms,
                                      self.performance_monitor.frame_count, self.performance_monitor)

            draw_scale_bar(frame)

            if not self.args.no_display:
                if display_frame(frame):
                    raise KeyboardInterrupt("Exit")

    def _control_fps(self):
        current_fps = self.performance_monitor.get_current_fps()
        if current_fps > self.args.fps:
            time.sleep(0.1)

    def _cleanup(self):
        self.running = False
        try:
            if self._capture_thread.is_alive():
                self._capture_thread.join(timeout=0.5)
        except:
            pass

        self.cap.release()
        if not self.args.no_display:
            cv2.destroyAllWindows()

    def _print_summary(self):
        if self.args.benchmark:
            self.performance_monitor.print_summary()