import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology
from skimage.segmentation import clear_border, watershed
from scipy import ndimage
import math
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple, Any

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

class Detector(ABC):
    """
    Abstract base class for spore detection backends.
    
    Detection backends are responsible for finding potential spore candidates in an image.
    They should return a list of dictionaries containing contours and confidence scores.
    """
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect potential spore candidates in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection candidates, each containing:
            - 'contour': OpenCV contour (numpy array)
            - 'confidence': Detection confidence score (0.0 to 1.0)
            - Additional detector-specific metadata (optional)
        """
        pass


class SporeAnalyzer:
    def __init__(self):
        self.pixel_scale = 1.0  # pixels per micrometer
        self.min_area = 10  # minimum area in um^2
        self.max_area = 500  # maximum area in um^2
        self.circularity_range = (0.3, 0.9)  # min, max circularity
        self.aspect_ratio_range = (1.0, 5.0)  # min, max aspect ratio (length/width)
        self.solidity_range = (0.5, 1.0)  # min, max solidity (contour area / convex hull area)
        self.convexity_range = (0.7, 1.0)  # min, max convexity (convex hull perimeter / contour perimeter)
        self.extent_range = (0.3, 1.0)  # min, max extent (contour area / bounding rect area)
        self.exclude_edges = True
        self.blur_kernel = 5
        self.threshold_method = "Otsu"
        self.threshold_value = None
        self.exclude_touching = True  # exclude touching/merged spores
        self.touching_aggressiveness = "Balanced"  # Conservative, Balanced, or Aggressive
        self.separate_touching = False  # separate touching spores using watershed
        self.separation_min_distance = 5  # minimum distance between peaks in watershed
        self.separation_sigma = 1.0  # gaussian sigma for peak detection
        self.separation_erosion_iterations = 1  # erosion iterations before distance transform
    
    def set_parameters(self, pixel_scale=1.0, min_area=10, max_area=500, 
                      circularity_range=(0.3, 0.9), aspect_ratio_range=(1.0, 5.0),
                      solidity_range=(0.5, 1.0), convexity_range=(0.7, 1.0), 
                      extent_range=(0.3, 1.0), exclude_edges=True,
                      blur_kernel=5, threshold_method="Otsu", threshold_value=None,
                      exclude_touching=True, touching_aggressiveness="Balanced",
                      separate_touching=False, separation_min_distance=5, 
                      separation_sigma=1.0, separation_erosion_iterations=1):
        """Set analysis parameters"""
        self.pixel_scale = pixel_scale
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_range = circularity_range
        self.aspect_ratio_range = aspect_ratio_range
        self.solidity_range = solidity_range
        self.convexity_range = convexity_range
        self.extent_range = extent_range
        self.exclude_edges = exclude_edges
        self.blur_kernel = blur_kernel
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.exclude_touching = exclude_touching
        self.touching_aggressiveness = touching_aggressiveness
        self.separate_touching = separate_touching
        self.separation_min_distance = separation_min_distance
        self.separation_sigma = separation_sigma
        self.separation_erosion_iterations = separation_erosion_iterations
    
    def preprocess_image(self, image):
        """Preprocess image for spore detection"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        if self.blur_kernel > 1:
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        else:
            blurred = gray
        
        # Apply thresholding
        if self.threshold_method == "Otsu":
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.threshold_method == "Adaptive":
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        else:  # Manual
            threshold_val = self.threshold_value if self.threshold_value is not None else 127
            _, binary = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Invert if needed (spores should be white on black background)
        # Check if we have more white pixels than black
        if np.sum(binary == 255) > np.sum(binary == 0):
            binary = cv2.bitwise_not(binary)
        
        return gray, binary
    
    def find_contours(self, binary_image):
        """Find contours in binary image"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def calculate_circularity(self, contour):
        """Calculate circularity of a contour (4π*area/perimeter²)"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return min(circularity, 1.0)  # Cap at 1.0 for perfect circle
    
    def calculate_solidity(self, contour):
        """Calculate solidity (contour area / convex hull area)"""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return 0
        return area / hull_area
    
    def calculate_convexity(self, contour):
        """Calculate convexity (convex hull perimeter / contour perimeter)"""
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        if perimeter == 0:
            return 0
        return hull_perimeter / perimeter
    
    def calculate_extent(self, contour):
        """Calculate extent (contour area / bounding rectangle area)"""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area == 0:
            return 0
        return area / rect_area
    
    def compute_convexity_defects_metrics(self, contour):
        """
        Count deep convexity defects to detect touching spores.
        
        Returns number of deep defects where depth is >= 4% of equivalent diameter.
        Deep concavities often indicate multiple touching spores.
        """
        if len(contour) < 4:
            return 0
            
        try:
            # Get convex hull and defects
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) < 4:
                return 0
                
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                return 0
            
            # Calculate equivalent diameter for normalization
            area = cv2.contourArea(contour)
            if area <= 0:
                return 0
            equivalent_diameter = 2 * np.sqrt(area / np.pi)
            
            # Count deep defects (depth >= 4% of equivalent diameter)
            deep_defect_threshold = 0.04 * equivalent_diameter
            num_deep_defects = 0
            
            for defect in defects:
                depth = defect[0][3] / 256.0  # OpenCV returns depth in fixed point
                if depth >= deep_defect_threshold:
                    num_deep_defects += 1
                    
            return num_deep_defects
            
        except Exception:
            return 0
    
    def compute_ellipse_residual(self, contour, ellipse=None):
        """
        Compute irregularity score based on deviation from fitted ellipse.
        
        Higher values indicate more irregular shapes that may be touching spores.
        Uses RMS radial deviation normalized by minor axis.
        """
        if len(contour) < 5:
            return 0.0
            
        try:
            # Fit ellipse if not provided
            if ellipse is None:
                ellipse = cv2.fitEllipse(contour)
            
            center, axes, angle = ellipse
            cx, cy = center
            major_axis, minor_axis = max(axes), min(axes)
            
            if minor_axis <= 0:
                return 0.0
            
            # Convert angle to radians
            angle_rad = np.radians(angle)
            
            # Calculate deviations for each contour point
            deviations = []
            for point in contour:
                px, py = point[0]
                
                # Translate to ellipse center
                dx = px - cx
                dy = py - cy
                
                # Rotate to ellipse coordinate system
                cos_a = np.cos(-angle_rad)
                sin_a = np.sin(-angle_rad)
                x_rot = dx * cos_a - dy * sin_a
                y_rot = dx * sin_a + dy * cos_a
                
                # Calculate expected radius on ellipse at this angle
                theta = np.arctan2(y_rot, x_rot)
                expected_radius = (major_axis/2) * (minor_axis/2) / np.sqrt(
                    ((minor_axis/2) * np.cos(theta))**2 + ((major_axis/2) * np.sin(theta))**2
                )
                
                # Calculate actual radius
                actual_radius = np.sqrt(x_rot**2 + y_rot**2)
                
                # Store deviation
                deviation = abs(actual_radius - expected_radius)
                deviations.append(deviation)
            
            # Calculate RMS deviation normalized by minor axis
            if len(deviations) > 0:
                rms_deviation = np.sqrt(np.mean(np.array(deviations)**2))
                normalized_residual = rms_deviation / (minor_axis / 2)
                return min(normalized_residual, 2.0)  # Cap at 2.0 for extreme cases
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def compute_area_outlier(self, area_um2, batch_stats):
        """
        Flag area outliers using robust z-score with MAD.
        
        Returns True if the area is an outlier (z > 2.5), which may indicate
        multiple touching spores detected as one large object.
        """
        if batch_stats is None or 'areas' not in batch_stats:
            return False
            
        areas = batch_stats['areas']
        if len(areas) < 3:  # Need at least 3 spores for meaningful statistics
            return False
            
        try:
            # Calculate median and MAD (Median Absolute Deviation)
            median_area = np.median(areas)
            mad = np.median(np.abs(areas - median_area))
            
            if mad == 0:  # All areas are identical
                return False
                
            # Calculate robust z-score
            robust_z = 0.6745 * (area_um2 - median_area) / mad
            
            # Flag as outlier if z > 2.5 (indicating unusually large area)
            return robust_z > 2.5
            
        except Exception:
            return False
    
    def detect_touching_hard_rules(self, contour, solidity, convexity, ellipse_residual, area_outlier):
        """
        Apply hard rules to detect touching spores.
        
        Returns True if any hard rule indicates touching spores.
        """
        try:
            # Count deep convexity defects
            num_deep_defects = self.compute_convexity_defects_metrics(contour)
            
            # Hard rules for touching detection
            rule_a = (solidity < 0.85 and num_deep_defects >= 1)
            rule_b = (num_deep_defects >= 2)
            rule_c = (convexity < 0.90 and ellipse_residual > 0.15)
            rule_d = area_outlier
            
            return rule_a or rule_b or rule_c or rule_d
            
        except Exception:
            return False
    
    def detect_touching_score_based(self, contour, solidity, convexity, ellipse_residual, aggressiveness="Balanced"):
        """
        Score-based touching detection with adjustable aggressiveness.
        
        Returns True if composite score indicates touching spores.
        """
        try:
            # Count deep convexity defects
            num_deep_defects = self.compute_convexity_defects_metrics(contour)
            
            # Calculate base composite score
            S = (0.4 * max(0, 0.9 - solidity) + 
                 0.2 * max(0, 0.95 - convexity) + 
                 0.3 * min(1, num_deep_defects / 2) + 
                 0.1 * ellipse_residual)
            
            # Adjust threshold based on aggressiveness
            if aggressiveness == "Conservative":
                threshold = 0.6  # Higher threshold, fewer detections
            elif aggressiveness == "Aggressive":
                threshold = 0.4  # Lower threshold, more detections
            else:  # Balanced
                threshold = 0.5  # Default threshold
            
            return S >= threshold
            
        except Exception:
            return False
    
    def is_touching_spore(self, contour, solidity, convexity, area_um2, batch_stats=None):
        """
        Main touching spore detection method combining hard rules and score-based detection.
        
        Returns tuple: (is_touching, detection_details)
        """
        try:
            # Calculate ellipse residual
            ellipse_residual = self.compute_ellipse_residual(contour)
            
            # Check area outlier
            area_outlier = self.compute_area_outlier(area_um2, batch_stats)
            
            # Apply hard rules
            hard_rules_touching = self.detect_touching_hard_rules(
                contour, solidity, convexity, ellipse_residual, area_outlier
            )
            
            # Apply score-based detection
            score_based_touching = self.detect_touching_score_based(
                contour, solidity, convexity, ellipse_residual, self.touching_aggressiveness
            )
            
            # Combine results (touching if either method detects it)
            is_touching = hard_rules_touching or score_based_touching
            
            # Prepare detection details
            num_deep_defects = self.compute_convexity_defects_metrics(contour)
            
            detection_details = {
                'num_deep_defects': num_deep_defects,
                'ellipse_residual': ellipse_residual,
                'area_outlier': area_outlier,
                'hard_rules_touching': hard_rules_touching,
                'score_based_touching': score_based_touching,
                'is_touching': is_touching
            }
            
            return is_touching, detection_details
            
        except Exception:
            # Return safe defaults on error
            return False, {
                'num_deep_defects': 0,
                'ellipse_residual': 0.0,
                'area_outlier': False,
                'hard_rules_touching': False,
                'score_based_touching': False,
                'is_touching': False
            }
    
    def is_touching_edge(self, contour, image_shape):
        """Check if contour touches image edges"""
        h, w = image_shape[:2]
        for point in contour:
            x, y = point[0]
            if x <= 1 or y <= 1 or x >= w-2 or y >= h-2:
                return True
        return False
    
    def _create_spore_mask(self, contour, image_shape):
        """Create binary mask from contour"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], (255,))
        return mask
    
    def _preprocess_for_watershed(self, mask):
        """Preprocess mask for watershed segmentation"""
        # Apply morphological erosion to thin connected regions
        if self.separation_erosion_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=self.separation_erosion_iterations)
        
        # Compute distance transform
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Apply gaussian filter to smooth distance transform
        if self.separation_sigma > 0:
            distance = ndimage.gaussian_filter(distance, sigma=self.separation_sigma)
        
        return distance
    
    def _find_watershed_seeds(self, distance_transform):
        """Find seeds (local maxima) for watershed segmentation"""
        try:
            # Find local maxima using ndimage.maximum_filter
            # Create a mask of local maxima
            footprint = np.ones((self.separation_min_distance, self.separation_min_distance))
            local_maxima = (distance_transform == ndimage.maximum_filter(distance_transform, footprint=footprint))
            
            # Apply threshold to filter weak maxima
            threshold = 0.3 * distance_transform.max()
            local_maxima = local_maxima & (distance_transform >= threshold)
            
            # Get coordinates of local maxima
            coordinates = np.argwhere(local_maxima)
            
            if len(coordinates) == 0:
                return None
            
            # Create markers for watershed
            markers = np.zeros(distance_transform.shape, dtype=np.int32)
            for i, (y, x) in enumerate(coordinates):
                markers[y, x] = i + 1
            
            return markers
        except Exception:
            return None
    
    def separate_touching_spores(self, contour, image_shape):
        """
        Separate touching spores using watershed segmentation.
        
        Args:
            contour: Input contour representing touching spores
            image_shape: Shape of the original image
            
        Returns:
            List of separated contours or None if separation fails
        """
        try:
            # Create mask from contour
            mask = self._create_spore_mask(contour, image_shape)
            
            # Preprocess mask for watershed
            distance = self._preprocess_for_watershed(mask)
            
            # Find watershed seeds
            markers = self._find_watershed_seeds(distance)
            if markers is None:
                return None
            
            # Apply watershed segmentation
            # Use negative distance as the topography
            labels = watershed(-distance, markers, mask=mask)
            
            # Extract individual contours from labeled regions
            separated_contours = []
            for label in np.unique(labels):
                if label == 0:  # Skip background
                    continue
                
                # Create mask for this label
                label_mask = (labels == label).astype(np.uint8) * 255
                
                # Find contours in the labeled region
                contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Add valid contours (filter out very small ones)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 10:  # Minimum area threshold for separated contours
                        separated_contours.append(cnt)
            
            # Return separated contours if we got more than one
            if len(separated_contours) > 1:
                return separated_contours
            else:
                # Separation didn't produce multiple valid contours
                return None
                
        except Exception as e:
            # Log the error and return None for graceful fallback
            logging.warning(f"Watershed separation failed: {e}")
            return None
    
    def calculate_ellipse_dimensions(self, contour):
        """Calculate major and minor axis lengths from fitted ellipse"""
        if len(contour) < 5:
            return None, None, None
        
        try:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # OpenCV fitEllipse returns (width, height) where angle corresponds to the width direction
            # We need to ensure the angle corresponds to the major axis (length)
            width, height = axes
            
            if width >= height:
                # Width is major axis (length)
                major_axis = width
                minor_axis = height
                # Angle already corresponds to major axis
                major_angle = angle
            else:
                # Height is major axis (length)
                major_axis = height
                minor_axis = width
                # Adjust angle to correspond to major axis (add 90 degrees)
                major_angle = angle + 90
                
            return major_axis, minor_axis, major_angle
        except:
            return None, None, None
    
    def get_extreme_points(self, contour):
        """Get extreme points of contour for dimension calculation"""
        # Find leftmost, rightmost, topmost, and bottommost points
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
        
        return leftmost, rightmost, topmost, bottommost
    
    def calculate_dimensions_from_contour(self, contour):
        """Calculate length and width from contour using multiple methods"""
        # Method 1: Fitted ellipse (preferred for spores)
        major_axis, minor_axis, angle = self.calculate_ellipse_dimensions(contour)
        
        if major_axis is not None and minor_axis is not None:
            return major_axis, minor_axis, angle
        
        # Method 2: Bounding rectangle as fallback
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        
        # Calculate distances between opposite corners
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])
        
        # Return length (max) and width (min)
        if width > height:
            return width, height, rect[2]
        else:
            return height, width, rect[2] + 90
    
    def analyze_spore(self, contour, image_shape, batch_stats=None):
        """Analyze a single spore contour with optional touching detection"""
        # Calculate basic properties
        area_pixels = cv2.contourArea(contour)
        area_um2 = area_pixels / (self.pixel_scale ** 2)
        
        # Check area constraints
        if area_um2 < self.min_area or area_um2 > self.max_area:
            return None
        
        # Check circularity
        circularity = self.calculate_circularity(contour)
        if circularity < self.circularity_range[0] or circularity > self.circularity_range[1]:
            return None
        
        # Calculate additional shape metrics
        solidity = self.calculate_solidity(contour)
        convexity = self.calculate_convexity(contour)
        extent = self.calculate_extent(contour)
        
        # Check solidity filter
        if solidity < self.solidity_range[0] or solidity > self.solidity_range[1]:
            return None
        
        # Check convexity filter
        if convexity < self.convexity_range[0] or convexity > self.convexity_range[1]:
            return None
        
        # Check extent filter
        if extent < self.extent_range[0] or extent > self.extent_range[1]:
            return None
        
        # Check edge exclusion
        if self.exclude_edges and self.is_touching_edge(contour, image_shape):
            return None
        
        # Calculate dimensions
        length_pixels, width_pixels, angle = self.calculate_dimensions_from_contour(contour)
        
        if length_pixels is None or width_pixels is None:
            return None
        
        length_um = length_pixels / self.pixel_scale
        width_um = width_pixels / self.pixel_scale
        
        # Calculate aspect ratio
        aspect_ratio = length_um / width_um if width_um > 0 else 0
        
        # Check aspect ratio filter
        if aspect_ratio < self.aspect_ratio_range[0] or aspect_ratio > self.aspect_ratio_range[1]:
            return None
        
        # Touching spore detection
        touching_detected = False
        touching_details = {
            'num_deep_defects': 0,
            'ellipse_residual': 0.0,
            'area_outlier': False,
            'hard_rules_touching': False,
            'score_based_touching': False,
            'is_touching': False
        }
        
        if self.exclude_touching or self.separate_touching:
            touching_detected, touching_details = self.is_touching_spore(
                contour, solidity, convexity, area_um2, batch_stats
            )
            
            if touching_detected:
                if self.separate_touching:
                    # Return special marker indicating separation should be attempted
                    return {'requires_separation': True, 'contour': contour, 'image_shape': image_shape}
                elif self.exclude_touching:
                    # Filter out touching spores if exclusion is enabled
                    return None
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Prepare result dictionary
        result = {
            'contour': contour,
            'centroid': (cx, cy),
            'area_pixels': area_pixels,
            'area_um2': area_um2,
            'length_pixels': length_pixels,
            'width_pixels': width_pixels,
            'length_um': length_um,
            'width_um': width_um,
            'angle': angle,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'convexity': convexity,
            'extent': extent,
            'perimeter': cv2.arcLength(contour, True),
            'touching_detected': touching_detected,
            'touching_details': touching_details
        }
        
        return result
    
    def analyze_candidates(self, image: np.ndarray, candidates: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Analyze spore candidates and return measurements.
        
        This method takes detection candidates from any detector backend and applies
        the measurement and filtering logic to return spore analysis results.
        
        Args:
            image: Input image as numpy array
            candidates: List of detection candidates from a Detector, each containing:
                       - 'contour': OpenCV contour (numpy array)
                       - 'confidence': Detection confidence score (optional)
                       - Additional metadata (optional)
        
        Returns:
            List of spore measurement dictionaries, or None if no valid spores found.
            Each result contains measurements, dimensions, shape metrics, etc.
        """
        if not candidates:
            return None
        
        # Two-pass analysis for touching detection with batch statistics
        batch_stats = None
        
        if self.exclude_touching or self.separate_touching:
            # First pass: collect areas for batch statistics (basic filtering only)
            areas = []
            for candidate in candidates:
                contour = candidate.get('contour')
                if contour is None:
                    continue
                
                # Basic area check
                area_pixels = cv2.contourArea(contour)
                area_um2 = area_pixels / (self.pixel_scale ** 2)
                
                if self.min_area <= area_um2 <= self.max_area:
                    areas.append(area_um2)
            
            # Prepare batch statistics for area outlier detection
            if len(areas) >= 3:  # Need at least 3 for meaningful statistics
                batch_stats = {'areas': areas}
        
        # Second pass: analyze each candidate with complete filtering
        spore_results = []
        for candidate in candidates:
            # Extract contour from candidate
            contour = candidate.get('contour')
            if contour is None:
                continue
                
            # Apply complete measurement and filtering logic with batch stats
            spore_data = self.analyze_spore(contour, image.shape, batch_stats)
            
            if spore_data is not None:
                # Check if this requires separation
                if isinstance(spore_data, dict) and spore_data.get('requires_separation', False):
                    # Attempt watershed separation
                    separated_contours = self.separate_touching_spores(contour, image.shape)
                    
                    if separated_contours is not None:
                        # Analyze each separated contour individually
                        for sep_contour in separated_contours:
                            # Temporarily disable touching detection for separated contours
                            old_exclude = self.exclude_touching
                            old_separate = self.separate_touching
                            self.exclude_touching = False
                            self.separate_touching = False
                            
                            sep_data = self.analyze_spore(sep_contour, image.shape, batch_stats)
                            
                            # Restore original settings
                            self.exclude_touching = old_exclude
                            self.separate_touching = old_separate
                            
                            if sep_data is not None:
                                # Add confidence score if provided
                                if 'confidence' in candidate:
                                    sep_data['detection_confidence'] = candidate['confidence']
                                # Mark as separated
                                sep_data['separated_from_touching'] = True
                                spore_results.append(sep_data)
                    else:
                        # Separation failed, handle based on exclude_touching setting
                        if not self.exclude_touching:
                            # Temporarily disable touching detection to get the original contour analyzed
                            old_exclude = self.exclude_touching
                            old_separate = self.separate_touching
                            self.exclude_touching = False
                            self.separate_touching = False
                            
                            original_data = self.analyze_spore(contour, image.shape, batch_stats)
                            
                            # Restore original settings
                            self.exclude_touching = old_exclude
                            self.separate_touching = old_separate
                            
                            if original_data is not None:
                                if 'confidence' in candidate:
                                    original_data['detection_confidence'] = candidate['confidence']
                                original_data['separation_failed'] = True
                                spore_results.append(original_data)
                else:
                    # Normal case - not requiring separation
                    # Add confidence score if provided
                    if 'confidence' in candidate:
                        spore_data['detection_confidence'] = candidate['confidence']
                    spore_results.append(spore_data)
        
        return spore_results if spore_results else None

    def analyze_image(self, image: np.ndarray, detector: Optional['Detector'] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Analyze image for spores and return measurements.
        
        Args:
            image: Input image as numpy array  
            detector: Optional Detector instance. If None, uses traditional detection.
        
        Returns:
            List of spore measurement dictionaries, or None if no spores found.
        """
        if detector is not None:
            # Use pluggable detector with fallback to traditional method
            try:
                candidates = detector.detect(image)
                return self.analyze_candidates(image, candidates)
            except Exception as e:
                # Log the error and fallback to traditional detection
                logging.warning(f"Detector failed with error: {e}. Falling back to traditional detection.")
                return self._analyze_image_traditional(image)
        else:
            # Use traditional detection pipeline (backward compatibility)
            return self._analyze_image_traditional(image)
    
    def _analyze_image_traditional(self, image: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """
        Original analyze_image implementation for backward compatibility.
        """
        # Preprocess image
        gray, binary = self.preprocess_image(image)
        
        # Find contours
        contours = self.find_contours(binary)
        
        if not contours:
            return None
        
        # Two-pass analysis for touching detection with batch statistics
        batch_stats = None
        
        if self.exclude_touching or self.separate_touching:
            # First pass: collect areas for batch statistics (basic filtering only)
            areas = []
            for contour in contours:
                # Basic area check
                area_pixels = cv2.contourArea(contour)
                area_um2 = area_pixels / (self.pixel_scale ** 2)
                
                if self.min_area <= area_um2 <= self.max_area:
                    areas.append(area_um2)
            
            # Prepare batch statistics for area outlier detection
            if len(areas) >= 3:  # Need at least 3 for meaningful statistics
                batch_stats = {'areas': areas}
        
        # Second pass: analyze each contour with complete filtering
        spore_results = []
        for contour in contours:
            spore_data = self.analyze_spore(contour, image.shape, batch_stats)
            
            if spore_data is not None:
                # Check if this requires separation
                if isinstance(spore_data, dict) and spore_data.get('requires_separation', False):
                    # Attempt watershed separation
                    separated_contours = self.separate_touching_spores(contour, image.shape)
                    
                    if separated_contours is not None:
                        # Analyze each separated contour individually
                        for sep_contour in separated_contours:
                            # Temporarily disable touching detection for separated contours
                            old_exclude = self.exclude_touching
                            old_separate = self.separate_touching
                            self.exclude_touching = False
                            self.separate_touching = False
                            
                            sep_data = self.analyze_spore(sep_contour, image.shape, batch_stats)
                            
                            # Restore original settings
                            self.exclude_touching = old_exclude
                            self.separate_touching = old_separate
                            
                            if sep_data is not None:
                                # Mark as separated
                                sep_data['separated_from_touching'] = True
                                spore_results.append(sep_data)
                    else:
                        # Separation failed, handle based on exclude_touching setting
                        if not self.exclude_touching:
                            # Temporarily disable touching detection to get the original contour analyzed
                            old_exclude = self.exclude_touching
                            old_separate = self.separate_touching
                            self.exclude_touching = False
                            self.separate_touching = False
                            
                            original_data = self.analyze_spore(contour, image.shape, batch_stats)
                            
                            # Restore original settings
                            self.exclude_touching = old_exclude
                            self.separate_touching = old_separate
                            
                            if original_data is not None:
                                original_data['separation_failed'] = True
                                spore_results.append(original_data)
                else:
                    # Normal case - not requiring separation
                    spore_results.append(spore_data)
        
        return spore_results if spore_results else None
    
    def get_measurement_lines(self, spore_data):
        """Generate line coordinates for measurement visualization"""
        contour = spore_data['contour']
        angle = spore_data['angle']
        centroid = spore_data['centroid']
        length_pixels = spore_data['length_pixels']
        width_pixels = spore_data['width_pixels']
        
        cx, cy = centroid
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate line endpoints for length (major axis)
        half_length = length_pixels / 2
        length_x1 = int(cx - half_length * np.cos(angle_rad))
        length_y1 = int(cy - half_length * np.sin(angle_rad))
        length_x2 = int(cx + half_length * np.cos(angle_rad))
        length_y2 = int(cy + half_length * np.sin(angle_rad))
        
        # Calculate line endpoints for width (minor axis, perpendicular)
        width_angle_rad = angle_rad + np.pi/2
        half_width = width_pixels / 2
        width_x1 = int(cx - half_width * np.cos(width_angle_rad))
        width_y1 = int(cy - half_width * np.sin(width_angle_rad))
        width_x2 = int(cx + half_width * np.cos(width_angle_rad))
        width_y2 = int(cy + half_width * np.sin(width_angle_rad))
        
        return {
            'length_line': ((length_x1, length_y1), (length_x2, length_y2)),
            'width_line': ((width_x1, width_y1), (width_x2, width_y2)),
            'centroid': centroid
        }


class TraditionalDetector(Detector):
    """
    Traditional OpenCV-based contour detection backend.
    
    This detector wraps the existing contour detection pipeline (preprocessing + contour finding)
    to conform to the Detector interface, allowing it to be used as a pluggable backend.
    """
    
    def __init__(self, analyzer: SporeAnalyzer):
        """
        Initialize TraditionalDetector with a SporeAnalyzer instance.
        
        Args:
            analyzer: SporeAnalyzer instance to use for preprocessing parameters
        """
        self.analyzer = analyzer
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect spore candidates using traditional OpenCV contour detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection candidates, each containing:
            - 'contour': OpenCV contour (numpy array) 
            - 'confidence': Always 1.0 for traditional detection
        """
        # Use the analyzer's preprocessing pipeline
        gray, binary = self.analyzer.preprocess_image(image)
        
        # Find contours using existing method
        contours = self.analyzer.find_contours(binary)
        
        # Convert contours to candidate format
        candidates = []
        for contour in contours:
            # Traditional detection has no confidence scoring, so set to 1.0
            candidate = {
                'contour': contour,
                'confidence': 1.0
            }
            candidates.append(candidate)
        
        return candidates


def mask_to_contours(mask: np.ndarray, confidence_threshold: float = 0.5) -> List[np.ndarray]:
    """
    Convert a segmentation mask to OpenCV contours.
    
    This utility function can be reused by different detectors that output segmentation masks.
    
    Args:
        mask: Segmentation mask as numpy array (H, W) with values 0-1 or 0-255
        confidence_threshold: Threshold for binarizing the mask (for values 0-1)
        
    Returns:
        List of OpenCV contours (numpy arrays)
    """
    try:
        # Normalize mask to 0-255 if needed
        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                # Values are in 0-1 range
                binary_mask = (mask > confidence_threshold).astype(np.uint8) * 255
            else:
                # Values are in 0-255 range
                binary_mask = (mask > confidence_threshold * 255).astype(np.uint8) * 255
        else:
            binary_mask = (mask > confidence_threshold * 255).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return list(contours)
    except Exception as e:
        logging.warning(f"Error converting mask to contours: {e}")
        return []


class YOLOSegOnnxDetector(Detector):
    """
    ONNX-based YOLO segmentation detector for spore detection.
    
    This detector uses a YOLO segmentation model via ONNX Runtime to detect spores
    and provides segmentation masks that are converted to contours for measurement.
    """
    
    def __init__(self, model_path: str = "attached_assets/models/spore_yolov8n-seg.onnx",
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                 target_size: int = 1024):
        """
        Initialize YOLOSegOnnxDetector.
        
        Args:
            model_path: Path to the ONNX model file
            confidence_threshold: Minimum confidence score for detections (default: 0.25)
            iou_threshold: IoU threshold for Non-Maximum Suppression (default: 0.45)
            target_size: Target size for the long side of input image (default: 1024)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_size = target_size
        self.session = None
        self.input_name = None
        self.output_names = None
        self.model_loaded = False
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model and cache it."""
        if not ONNX_AVAILABLE:
            self.logger.warning("ONNX Runtime not available. YOLOSegOnnxDetector will return empty results.")
            return
        
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Model file not found at {self.model_path}. "
                              f"YOLOSegOnnxDetector will return empty results.")
            return
        
        try:
            # Create ONNX Runtime session with CPU provider
            providers = ['CPUExecutionProvider']
            session_options = ort.SessionOptions() if ort else None
            if ort and session_options:
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            if ort:
                self.session = ort.InferenceSession(self.model_path, 
                                                  sess_options=session_options,
                                                  providers=providers)
            
            # Get input and output information
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.model_loaded = True
            self.logger.info(f"Successfully loaded ONNX model from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            self.model_loaded = False
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO inference.
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            
        Returns:
            Tuple of (preprocessed_image, scale_factor, original_shape)
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume input is RGB
            img = image.copy()
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            # Convert grayscale to RGB
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img = image.copy()
        
        # Get original shape
        original_shape = img.shape[:2]  # (H, W)
        
        # Calculate scale factor to resize image while preserving aspect ratio
        h, w = original_shape
        scale = min(self.target_size / h, self.target_size / w)
        
        # Resize image
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create square image with padding
        padded_img = np.full((self.target_size, self.target_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets (center the image)
        pad_y = (self.target_size - new_h) // 2
        pad_x = (self.target_size - new_w) // 2
        
        padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
        
        # Convert to float and normalize to [0, 1]
        padded_img = padded_img.astype(np.float32) / 255.0
        
        # Convert to CHW format for YOLO
        padded_img = np.transpose(padded_img, (2, 0, 1))  # (C, H, W)
        
        # Add batch dimension
        padded_img = np.expand_dims(padded_img, axis=0)  # (1, C, H, W)
        
        return padded_img, scale, original_shape
    
    def _apply_nms(self, detections: np.ndarray, masks: Optional[np.ndarray] = None) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Apply Non-Maximum Suppression to filter overlapping detections.
        
        Args:
            detections: Detection array with shape (N, 6) containing [x1, y1, x2, y2, conf, class]
            masks: Optional mask array with shape (N, H, W)
            
        Returns:
            Tuple of (filtered_detections, filtered_masks)
        """
        if len(detections) == 0:
            return [], []
        
        # Extract bounding boxes and scores
        boxes_xyxy = detections[:, :4]  # [x1, y1, x2, y2]
        scores = detections[:, 4]  # confidence scores
        
        # Convert boxes from xyxy to xywh format for cv2.dnn.NMSBoxes
        boxes_xywh = np.column_stack([
            boxes_xyxy[:, 0],  # x
            boxes_xyxy[:, 1],  # y
            boxes_xyxy[:, 2] - boxes_xyxy[:, 0],  # width = x2 - x1
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1]   # height = y2 - y1
        ])
        
        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), 
            scores.tolist(), 
            self.confidence_threshold, 
            self.iou_threshold
        )
        
        filtered_detections = []
        filtered_masks = []
        
        if len(indices) > 0:
            # Flatten indices if it's a nested array
            if isinstance(indices, np.ndarray) and indices.ndim > 1:
                indices = indices.flatten()
            
            for i in indices:
                detection_info = {
                    'bbox': boxes_xyxy[i],  # Keep original xyxy format for downstream processing
                    'confidence': float(scores[i]),
                    'class': int(detections[i, 5]) if detections.shape[1] > 5 else 0
                }
                filtered_detections.append(detection_info)
                
                if masks is not None:
                    filtered_masks.append(masks[i])
        
        return filtered_detections, filtered_masks
    
    def _postprocess_output(self, outputs: List[np.ndarray], scale: float, 
                          original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Postprocess YOLO outputs to extract detections and masks.
        
        Args:
            outputs: List of ONNX model outputs
            scale: Scale factor used during preprocessing
            original_shape: Original image shape (H, W)
            
        Returns:
            List of detection candidates with contours and confidence scores
        """
        try:
            if not outputs or len(outputs) < 1:
                return []
            
            # Extract main detection output (typically first output)
            detections = outputs[0]  # Shape: (batch, detections, features)
            
            if len(detections.shape) == 3:
                detections = detections[0]  # Remove batch dimension
            
            # Extract segmentation masks if available (YOLOv8-seg format)
            # Must be done BEFORE confidence filtering to maintain index alignment
            masks = None
            original_detections = detections.copy()  # Keep original for mask extraction
            
            if len(outputs) > 1:
                # YOLOv8-seg: outputs[1] contains prototype masks (batch, 32, mask_h, mask_w)
                proto_masks = outputs[1]  # Prototype masks
                if len(proto_masks.shape) == 4:
                    proto_masks = proto_masks[0]  # Remove batch dimension: (32, mask_h, mask_w)
                
                # Extract mask coefficients from original (unfiltered) detection tensor
                if original_detections.shape[1] >= 4 + 1 + 32:  # xyxy + conf + 32 mask coeffs
                    mask_coeffs = original_detections[:, -32:]  # Shape: (num_detections, 32)
                    
                    # Decode masks using original detections for proper alignment
                    if len(mask_coeffs) > 0:
                        self.logger.debug(f"Decoding {len(mask_coeffs)} masks from prototypes shape {proto_masks.shape}")
                        masks = self._decode_yolov8_masks(proto_masks, mask_coeffs, 
                                                         original_detections[:, :4], original_shape, scale)
                        if masks is not None and len(masks) > 0:
                            self.logger.debug(f"Successfully decoded {len(masks)} masks")
                        else:
                            self.logger.warning("Mask decoding returned empty result")
            
            # Now apply confidence filtering to both detections and masks
            conf_mask = None
            if detections.shape[1] >= 5:  # Has confidence column
                conf_mask = detections[:, 4] >= self.confidence_threshold
                detections = detections[conf_mask]
                
                # Apply same confidence filtering to masks if they exist
                if masks is not None:
                    masks = masks[conf_mask]
            
            if len(detections) == 0:
                return []
            
            # Apply NMS
            filtered_detections, filtered_masks = self._apply_nms(detections, masks)
            
            # Convert to contours
            candidates = []
            for i, detection in enumerate(filtered_detections):
                bbox = detection['bbox']
                confidence = detection['confidence']
                
                # Scale back to original image coordinates
                x1, y1, x2, y2 = bbox
                
                # Account for padding offset
                pad_y = (self.target_size - int(original_shape[0] * scale)) // 2
                pad_x = (self.target_size - int(original_shape[1] * scale)) // 2
                
                # Remove padding offset
                x1 = (x1 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                x2 = (x2 - pad_x) / scale
                y2 = (y2 - pad_y) / scale
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))
                
                # Create contour from mask or bounding box
                if filtered_masks and i < len(filtered_masks):
                    # Use segmentation mask
                    mask = filtered_masks[i]
                    
                    # Resize mask to original image size
                    mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]), 
                                            interpolation=cv2.INTER_LINEAR)
                    
                    # Convert mask to contours
                    contours = mask_to_contours(mask_resized, confidence_threshold=0.5)
                    
                    # Add each contour as a separate candidate
                    for contour in contours:
                        if cv2.contourArea(contour) > 0:  # Filter out empty contours
                            candidates.append({
                                'contour': contour,
                                'confidence': confidence,
                                'detection_type': 'segmentation'
                            })
                else:
                    # Create contour from bounding box as fallback
                    bbox_contour = np.array([
                        [[int(x1), int(y1)]],
                        [[int(x2), int(y1)]],
                        [[int(x2), int(y2)]],
                        [[int(x1), int(y2)]]
                    ], dtype=np.int32)
                    
                    candidates.append({
                        'contour': bbox_contour,
                        'confidence': confidence,
                        'detection_type': 'bbox'
                    })
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error in postprocessing: {e}")
            return []
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect spore candidates using YOLO segmentation model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection candidates, each containing:
            - 'contour': OpenCV contour (numpy array)
            - 'confidence': Detection confidence score (0.0 to 1.0)
            - 'detection_type': 'segmentation' or 'bbox' (optional metadata)
        """
        # Return empty list if model is not loaded
        if not self.model_loaded or self.session is None:
            self.logger.warning("Model not loaded. Returning empty detection results.")
            return []
        
        try:
            # Preprocess image
            preprocessed_img, scale, original_shape = self._preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: preprocessed_img})
            
            # Postprocess outputs
            candidates = self._postprocess_output(list(outputs), scale, original_shape)
            
            self.logger.info(f"Detected {len(candidates)} spore candidates")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error during YOLO detection: {e}")
            return []
    
    def _decode_yolov8_masks(self, proto_masks: np.ndarray, mask_coeffs: np.ndarray, 
                           bboxes: np.ndarray, original_shape: Tuple[int, int], scale: float) -> np.ndarray:
        """
        Decode YOLOv8 segmentation masks from prototypes and coefficients.
        
        Args:
            proto_masks: Prototype masks (32, mask_h, mask_w)
            mask_coeffs: Mask coefficients (num_detections, 32)
            bboxes: Bounding boxes in xyxy format (num_detections, 4)
            original_shape: Original image shape (H, W)
            scale: Scale factor used during preprocessing
            
        Returns:
            Decoded masks (num_detections, original_h, original_w)
        """
        try:
            # Validate inputs
            if proto_masks is None or mask_coeffs is None:
                self.logger.warning("Null inputs to mask decoder")
                return np.array([])
            
            if len(proto_masks.shape) != 3 or proto_masks.shape[0] != 32:
                self.logger.warning(f"Invalid proto_masks shape: {proto_masks.shape}, expected (32, H, W)")
                return np.array([])
            
            if len(mask_coeffs.shape) != 2 or mask_coeffs.shape[1] != 32:
                self.logger.warning(f"Invalid mask_coeffs shape: {mask_coeffs.shape}, expected (N, 32)")
                return np.array([])
            
            # Get dimensions
            mask_h, mask_w = proto_masks.shape[1], proto_masks.shape[2]
            num_detections = mask_coeffs.shape[0]
            
            self.logger.debug(f"Decoding {num_detections} masks from prototypes {proto_masks.shape}")
            
            # Compute masks: proto @ coeffs.T
            # Reshape proto_masks: (32, mask_h * mask_w)
            proto_flat = proto_masks.reshape(32, -1)
            # mask_coeffs: (num_detections, 32)
            # Result: (num_detections, mask_h * mask_w)
            masks_flat = np.matmul(mask_coeffs, proto_flat)
            
            # Reshape back to (num_detections, mask_h, mask_w)
            masks = masks_flat.reshape(num_detections, mask_h, mask_w)
            
            # Apply sigmoid activation
            masks = 1.0 / (1.0 + np.exp(-masks))
            
            # Upsample masks to target size
            upsampled_masks = []
            for i in range(num_detections):
                mask = masks[i]
                
                # Resize to target_size (same size as preprocessing)
                mask_resized = cv2.resize(mask, (self.target_size, self.target_size), 
                                        interpolation=cv2.INTER_LINEAR)
                
                # Remove padding and scale to original image size
                pad_y = (self.target_size - int(original_shape[0] * scale)) // 2
                pad_x = (self.target_size - int(original_shape[1] * scale)) // 2
                
                # Crop to remove padding
                scaled_h = int(original_shape[0] * scale)
                scaled_w = int(original_shape[1] * scale)
                mask_cropped = mask_resized[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w]
                
                # Scale to original image size
                mask_original = cv2.resize(mask_cropped, (original_shape[1], original_shape[0]), 
                                         interpolation=cv2.INTER_LINEAR)
                
                # Optional: Crop mask to bbox area for better precision
                x1, y1, x2, y2 = bboxes[i]
                # Convert to original coordinates
                x1_orig = max(0, int((x1 - pad_x) / scale))
                y1_orig = max(0, int((y1 - pad_y) / scale))
                x2_orig = min(original_shape[1], int((x2 - pad_x) / scale))
                y2_orig = min(original_shape[0], int((y2 - pad_y) / scale))
                
                # Apply bbox cropping to mask (optional, can improve precision)
                mask_bbox_cropped = np.zeros_like(mask_original)
                if x2_orig > x1_orig and y2_orig > y1_orig:
                    mask_bbox_cropped[y1_orig:y2_orig, x1_orig:x2_orig] = \
                        mask_original[y1_orig:y2_orig, x1_orig:x2_orig]
                    upsampled_masks.append(mask_bbox_cropped)
                else:
                    upsampled_masks.append(mask_original)
            
            return np.array(upsampled_masks) if upsampled_masks else np.array([])
            
        except Exception as e:
            self.logger.error(f"Error decoding YOLOv8 masks: {e}")
            return np.array([])
    
    def set_thresholds(self, confidence_threshold: float, iou_threshold: float):
        """Update detection thresholds."""
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
