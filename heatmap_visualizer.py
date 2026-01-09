"""
Heatmap visualizer for silkworm density analysis
Simple and efficient implementation for Raspberry Pi
"""

import cv2
import numpy as np
from typing import List, Tuple


class HeatmapVisualizer:
    """Visualizes detection density as a heatmap overlay."""
    
    def __init__(self,
                 frame_shape: Tuple[int, int],
                 grid_shape: Tuple[int, int] = (4, 6),
                 opacity: float = 0.3,
                 draw_grid: bool = True):
        """
        Args:
            frame_shape: (height, width) of the frame
            grid_shape: (rows, cols) - number of grid cells
            opacity: Heatmap overlay opacity (0-1)
            draw_grid: Whether to draw grid lines
        """
        self.h, self.w = frame_shape
        self.grid_rows, self.grid_cols = grid_shape
        self.opacity = float(opacity)
        self.draw_grid = bool(draw_grid)
        
        # Calculate cell size
        self.cell_w = self.w // self.grid_cols
        self.cell_h = self.h // self.grid_rows
        
        # Heatmap buffer: grid_rows x grid_cols
        self.heatmap = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Create grid lines image if needed
        self.grid_img = None
        if self.draw_grid:
            self.grid_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            # Draw vertical lines
            for i in range(1, self.grid_cols):
                x = i * self.cell_w
                cv2.line(self.grid_img, (x, 0), (x, self.h - 1), (200, 200, 200), 1)
            # Draw horizontal lines
            for i in range(1, self.grid_rows):
                y = i * self.cell_h
                cv2.line(self.grid_img, (0, y), (self.w - 1, y), (200, 200, 200), 1)
    
    def update(self, frame: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
        """
        Update heatmap with new detections and overlay on frame.
        
        Args:
            frame: Input frame (H, W, 3)
            boxes: List of bounding boxes as [x1, y1, x2, y2]
        
        Returns:
            Frame with heatmap overlay
        """
        # Reset heatmap
        self.heatmap.fill(0.0)
        
        # Count detections in each grid cell
        for box in boxes:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = box[:4]
            
            # Clamp to frame bounds
            x1 = max(0, min(float(x1), self.w - 1))
            y1 = max(0, min(float(y1), self.h - 1))
            x2 = max(0, min(float(x2), self.w - 1))
            y2 = max(0, min(float(y2), self.h - 1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Calculate which cells this box overlaps
            # Use floor division to get cell indices
            cell_x1 = int(x1 / self.cell_w)
            cell_y1 = int(y1 / self.cell_h)
            cell_x2 = int(x2 / self.cell_w)
            cell_y2 = int(y2 / self.cell_h)
            
            # Clamp cell indices to valid range
            cell_x1 = max(0, min(cell_x1, self.grid_cols - 1))
            cell_y1 = max(0, min(cell_y1, self.grid_rows - 1))
            cell_x2 = max(0, min(cell_x2, self.grid_cols - 1))
            cell_y2 = max(0, min(cell_y2, self.grid_rows - 1))
            
            # Count this detection in all overlapping cells
            # This represents density: more detections = higher value
            for cy in range(cell_y1, cell_y2 + 1):
                for cx in range(cell_x1, cell_x2 + 1):
                    self.heatmap[cy, cx] += 1.0
        
        # Normalize heatmap to [0, 1] for visualization
        # This makes the highest density cell = 1.0 (red), others relative to it
        max_val = self.heatmap.max()
        if max_val > 0:
            # Normalize: each cell value / max value
            # This preserves relative density: cell with most detections = red, others scaled proportionally
            self.heatmap = self.heatmap / max_val
        else:
            # No detections - return frame with grid only
            if self.grid_img is not None:
                return cv2.addWeighted(frame, 1.0, self.grid_img, 0.6, 0)
            return frame
        
        # Create color heatmap: red (high density) to yellow (low density)
        # Red = high density (many silkworms), Yellow = low density (few silkworms)
        # Use a more visible color gradient
        heatmap_colored = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.uint8)
        heatmap_normalized = (self.heatmap * 255).astype(np.uint8)
        
        # Apply colormap: red (high) -> orange -> yellow (low)
        # Red channel: full intensity for high density
        heatmap_colored[:, :, 2] = heatmap_normalized
        # Green channel: increases with density for orange/yellow gradient
        # Low density: more green = yellow, High density: less green = red
        heatmap_colored[:, :, 1] = (heatmap_normalized * 0.6).astype(np.uint8)
        # Blue channel: zero (red-yellow gradient)
        heatmap_colored[:, :, 0] = 0
        
        # Resize to frame size using nearest neighbor to preserve grid boundaries
        heatmap_resized = cv2.resize(heatmap_colored, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        
        # Overlay heatmap on frame with opacity
        result = cv2.addWeighted(frame, 1.0 - self.opacity, heatmap_resized, self.opacity, 0)
        
        # Add grid lines if enabled
        if self.grid_img is not None:
            result = cv2.addWeighted(result, 1.0, self.grid_img, 0.6, 0)
        
        return result

