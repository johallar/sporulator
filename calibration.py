import cv2
import numpy as np
from PIL import Image
import re

class StageCalibration:
    """Handle stage calibration for automatic pixel scale detection"""
    
    def __init__(self):
        self.known_standards = {
            "10um_scale_bar": {"length_um": 10, "description": "10 micrometer scale bar"},
            "20um_scale_bar": {"length_um": 20, "description": "20 micrometer scale bar"},
            "50um_scale_bar": {"length_um": 50, "description": "50 micrometer scale bar"},
            "100um_scale_bar": {"length_um": 100, "description": "100 micrometer scale bar"},
            "pollen_grain": {"length_um": 25, "description": "Average pollen grain (25μm)"},
            "red_blood_cell": {"length_um": 7.5, "description": "Human red blood cell (7.5μm)"},
        }
    
    def detect_scale_bar(self, image, min_length_pixels=50, max_length_pixels=500):
        """Detect scale bars in calibration images using improved line detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Focus on bottom-right quadrant where scale bars are typically located
        roi_y_start = h // 2
        roi_x_start = w // 2
        roi = gray[roi_y_start:, roi_x_start:]
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        roi_cleaned = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        
        # Apply edge detection with different parameters for better line detection
        edges = cv2.Canny(roi_cleaned, 30, 100, apertureSize=3)
        
        # Detect lines using Hough transform with stricter parameters
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=min_length_pixels, maxLineGap=5)
        
        if lines is None:
            # Try detection on full image as fallback
            edges_full = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges_full, 1, np.pi/180, threshold=50, 
                                   minLineLength=min_length_pixels, maxLineGap=10)
            roi_offset_x, roi_offset_y = 0, 0
        else:
            roi_offset_x, roi_offset_y = roi_x_start, roi_y_start
        
        if lines is None:
            return None
        
        # Filter and score horizontal lines
        candidate_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Adjust coordinates if from ROI
            x1, x2 = x1 + roi_offset_x, x2 + roi_offset_x
            y1, y2 = y1 + roi_offset_y, y2 + roi_offset_y
            
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            # Consider lines that are roughly horizontal (within 10 degrees)
            if angle < 10 or angle > 170:
                # Calculate line quality score based on:
                # - Length (longer is better)
                # - Horizontal-ness (more horizontal is better)
                # - Position (bottom-right preferred)
                length_score = min(length / max_length_pixels, 1.0)
                angle_score = 1.0 - (min(angle, 180-angle) / 10.0)
                position_score = (x1 + x2) / (2 * w) + (y1 + y2) / (2 * h)  # Favor bottom-right
                
                quality_score = (length_score * 0.5 + angle_score * 0.3 + position_score * 0.2)
                
                candidate_lines.append({
                    'coords': (x1, y1, x2, y2),
                    'length': length,
                    'angle': angle,
                    'quality_score': quality_score
                })
        
        if not candidate_lines:
            return None
        
        # Sort by quality score and return the best line
        candidate_lines.sort(key=lambda x: x['quality_score'], reverse=True)
        return candidate_lines[0]
    
    def detect_micrometer_divisions(self, image, min_tick_length=30, max_tick_length=200):
        """Detect graduated micrometer divisions (tick marks) and calculate spacing"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Apply morphological operations to enhance line detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        enhanced = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Apply edge detection with higher thresholds to reduce noise
        edges = cv2.Canny(enhanced, 100, 200, apertureSize=3)
        
        # Detect lines using Hough transform with much higher threshold to reduce noise
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                               minLineLength=min_tick_length, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Accept lines at any angle - division marks can be oriented in any direction
        division_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Calculate angle of the line
            if x2 - x1 == 0:  # Vertical line
                angle = 90
            else:
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            # Accept lines of sufficient length (filter out noise)
            if min_tick_length <= length <= max_tick_length:
                # Calculate center point of the line
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Calculate line direction for grouping
                dx = x2 - x1
                dy = y2 - y1
                
                division_lines.append({
                    'coords': (x1, y1, x2, y2),
                    'center_x': center_x,
                    'center_y': center_y,
                    'length': length,
                    'angle': angle,
                    'dx': dx,
                    'dy': dy
                })
        
        if len(division_lines) < 3:  # Need at least 3 division marks
            return None
        
        # Group lines by orientation to find the dominant direction
        angle_groups = {}
        angle_tolerance = 15  # degrees
        
        for line in division_lines:
            angle = line['angle']
            
            # Find existing group or create new one
            group_found = False
            for group_angle in angle_groups:
                if abs(angle - group_angle) <= angle_tolerance:
                    angle_groups[group_angle].append(line)
                    group_found = True
                    break
            
            if not group_found:
                angle_groups[angle] = [line]
        
        # Find the largest group (most consistent orientation)
        if not angle_groups:
            return None
            
        largest_group = max(angle_groups.values(), key=len)
        
        if len(largest_group) < 3:
            return None
        
        # Determine dominant direction and sort accordingly
        sample_line = largest_group[0]
        
        # If lines are more horizontal, sort by x-position
        # If lines are more vertical, sort by y-position
        if abs(sample_line['angle']) < 45 or abs(sample_line['angle']) > 135:
            # More horizontal - sort by x-position
            largest_group.sort(key=lambda x: x['center_x'])
            spacings = []
            for i in range(1, len(largest_group)):
                spacing = largest_group[i]['center_x'] - largest_group[i-1]['center_x']
                spacings.append(spacing)
        else:
            # More vertical - sort by y-position
            largest_group.sort(key=lambda x: x['center_y'])
            spacings = []
            for i in range(1, len(largest_group)):
                spacing = largest_group[i]['center_y'] - largest_group[i-1]['center_y']
                spacings.append(spacing)
        
        # Filter out spacings that are too small (likely noise) or too large (likely missed divisions)
        if spacings:
            median_spacing = np.median(spacings)
            # Keep spacings within 50% of median to remove outliers
            filtered_spacings = [s for s in spacings if 0.5 * median_spacing <= s <= 1.5 * median_spacing]
            
            if len(filtered_spacings) >= 2:  # Need at least 2 consistent spacings
                final_spacing = np.median(filtered_spacings)
            else:
                final_spacing = median_spacing
        else:
            return None
        
        if final_spacing <= 5:  # Spacing too small, likely noise
            return None
        
        return {
            'tick_marks': largest_group,
            'spacing_pixels': final_spacing,
            'num_divisions': len(largest_group) - 1,
            'quality_score': min(len(largest_group) / 10.0, 1.0),  # Better with more tick marks
            'dominant_angle': sample_line['angle']
        }

    def detect_circular_objects(self, image, min_radius=10, max_radius=100):
        """Detect circular objects like pollen grains or cells"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                  param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Sort by radius (larger circles first)
            circles = sorted(circles, key=lambda x: x[2], reverse=True)
            return circles
        
        return None
    
    
    def calculate_pixel_scale(self, scale_bar_length_pixels, known_length_um):
        """Calculate pixel scale from detected scale bar"""
        if scale_bar_length_pixels <= 0 or known_length_um <= 0:
            return None
        
        pixel_scale = scale_bar_length_pixels / known_length_um
        return pixel_scale
    
    def auto_detect_scale(self, image, reference_type="scale_bar", known_length_um=None):
        """Automatically detect and calculate pixel scale from calibration image"""
        results = {
            'pixel_scale': None,
            'detection_method': None,
            'confidence': 0,
            'detected_objects': [],
            'visualization': None
        }
        
        if reference_type == "scale_bar":
            # Detect scale bars
            scale_bar = self.detect_scale_bar(image)
            if scale_bar and known_length_um:
                pixel_scale = self.calculate_pixel_scale(scale_bar['length'], known_length_um)
                
                # Calculate confidence based on quality score and length
                h, w = image.shape[:2]
                relative_length = scale_bar['length'] / w
                quality_confidence = scale_bar.get('quality_score', 0.5)
                length_confidence = min(relative_length / 0.1, 1.0)  # Good if >10% of image width
                confidence = (quality_confidence * 0.7 + length_confidence * 0.3)
                
                results.update({
                    'pixel_scale': pixel_scale,
                    'detection_method': 'scale_bar',
                    'confidence': confidence,
                    'detected_objects': [scale_bar],
                    'visualization': self.create_calibration_visualization(image, [scale_bar], 'scale_bar')
                })
        
        elif reference_type == "micrometer_divisions":
            # Detect micrometer tick marks
            divisions = self.detect_micrometer_divisions(image)
            if divisions and known_length_um:
                # known_length_um should be the distance per division (e.g., 10μm for 0.01mm divisions)
                pixel_scale = self.calculate_pixel_scale(divisions['spacing_pixels'], known_length_um)
                
                # Calculate confidence based on number of divisions and consistency
                quality_confidence = divisions.get('quality_score', 0.5)
                consistency_confidence = min(divisions['num_divisions'] / 10.0, 1.0)  # Better with more divisions
                confidence = (quality_confidence * 0.6 + consistency_confidence * 0.4) * 0.9  # High confidence for micrometer
                
                results.update({
                    'pixel_scale': pixel_scale,
                    'detection_method': 'micrometer_divisions',
                    'confidence': confidence,
                    'detected_objects': [divisions],
                    'visualization': self.create_calibration_visualization(image, [divisions], 'micrometer')
                })
        
        elif reference_type == "circular_object":
            # Detect circular objects
            circles = self.detect_circular_objects(image)
            if circles is not None and len(circles) > 0 and known_length_um:
                # Use the largest circle as reference
                largest_circle = circles[0]
                diameter_pixels = largest_circle[2] * 2
                pixel_scale = diameter_pixels / known_length_um
                
                # Calculate confidence based on circle quality and number of circles detected
                num_circles = len(circles)
                circle_confidence = min(num_circles / 5.0, 1.0)  # Better if multiple similar circles
                size_confidence = min(largest_circle[2] / 50.0, 1.0)  # Better if reasonably sized
                confidence = (circle_confidence * 0.4 + size_confidence * 0.6) * 0.7  # Cap at 0.7 for circles
                
                results.update({
                    'pixel_scale': pixel_scale,
                    'detection_method': 'circular_object',
                    'confidence': confidence,
                    'detected_objects': circles,
                    'visualization': self.create_calibration_visualization(image, circles, 'circular')
                })
        
        return results
    
    def create_calibration_visualization(self, image, detected_objects, detection_type):
        """Create visualization of detected calibration objects"""
        # Convert to RGB if needed
        if len(image.shape) == 3:
            vis_image = image.copy()
        else:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if detection_type == 'scale_bar':
            for obj in detected_objects:
                if isinstance(obj, dict) and 'coords' in obj:
                    x1, y1, x2, y2 = obj['coords']
                    cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # Add text
                    cv2.putText(vis_image, f"Scale Bar: {obj['length']:.1f}px", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif detection_type == 'circular':
            for circle in detected_objects[:5]:  # Show max 5 circles
                x, y, r = circle
                cv2.circle(vis_image, (x, y), r, (255, 0, 0), 2)
                cv2.circle(vis_image, (x, y), 2, (255, 0, 0), 3)
                # Add text
                cv2.putText(vis_image, f"D: {r*2}px", (x+r+5, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        elif detection_type == 'micrometer':
            # Handle micrometer divisions visualization
            for divisions_obj in detected_objects:
                if isinstance(divisions_obj, dict) and 'tick_marks' in divisions_obj:
                    tick_marks = divisions_obj['tick_marks']
                    spacing = divisions_obj['spacing_pixels']
                    
                    # Draw each detected tick mark
                    for i, tick in enumerate(tick_marks):
                        x1, y1, x2, y2 = tick['coords']
                        cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow lines
                        
                        # Add tick number
                        cv2.putText(vis_image, str(i+1), 
                                   (int(tick['center_x']), y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # Draw spacing indicators between first few tick marks
                    for i in range(min(3, len(tick_marks)-1)):
                        tick1 = tick_marks[i]
                        tick2 = tick_marks[i+1]
                        
                        # Draw spacing line
                        start_x = int(tick1['center_x'])
                        end_x = int(tick2['center_x'])
                        mid_y = min(int(tick1['center_y']), int(tick2['center_y'])) - 20
                        
                        cv2.line(vis_image, (start_x, mid_y), (end_x, mid_y), (255, 255, 0), 1)  # Cyan spacing line
                        
                        # Add spacing text
                        mid_x = (start_x + end_x) // 2
                        cv2.putText(vis_image, f"{spacing:.1f}px", 
                                   (mid_x-30, mid_y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # Add summary text
                    h, w = vis_image.shape[:2]
                    summary_text = f"Detected {len(tick_marks)} tick marks, avg spacing: {spacing:.1f}px"
                    cv2.putText(vis_image, summary_text, (10, h-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis_image
    
    def manual_calibration(self, image, point1, point2, known_length_um):
        """Manual calibration by clicking two points on a known distance"""
        if not point1 or not point2 or known_length_um <= 0:
            return None
        
        # Calculate distance between points
        distance_pixels = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        
        if distance_pixels <= 0:
            return None
        
        pixel_scale = distance_pixels / known_length_um
        
        # Create visualization
        vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.line(vis_image, point1, point2, (0, 255, 255), 2)
        cv2.circle(vis_image, point1, 5, (0, 255, 255), -1)
        cv2.circle(vis_image, point2, 5, (0, 255, 255), -1)
        cv2.putText(vis_image, f"{distance_pixels:.1f}px = {known_length_um}μm", 
                   (min(point1[0], point2[0]), min(point1[1], point2[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return {
            'pixel_scale': pixel_scale,
            'distance_pixels': distance_pixels,
            'visualization': vis_image
        }
    
    def validate_calibration(self, pixel_scale, image_context="microscopy"):
        """Validate if the calculated pixel scale is reasonable"""
        # Typical pixel scales for different magnifications
        reasonable_ranges = {
            "microscopy_low": (0.1, 2.0),    # Low magnification: 0.1-2 pixels/μm
            "microscopy_medium": (2.0, 10.0), # Medium magnification: 2-10 pixels/μm  
            "microscopy_high": (10.0, 50.0),  # High magnification: 10-50 pixels/μm
            "microscopy": (0.1, 50.0)         # General range
        }
        
        if image_context not in reasonable_ranges:
            image_context = "microscopy"
        
        min_scale, max_scale = reasonable_ranges[image_context]
        
        is_valid = min_scale <= pixel_scale <= max_scale
        
        return {
            'is_valid': is_valid,
            'pixel_scale': pixel_scale,
            'expected_range': reasonable_ranges[image_context],
            'warning': None if is_valid else f"Scale seems unusually {'high' if pixel_scale > max_scale else 'low'} for {image_context}"
        }