import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology
from skimage.segmentation import clear_border
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple, Any

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
    
    def set_parameters(self, pixel_scale=1.0, min_area=10, max_area=500, 
                      circularity_range=(0.3, 0.9), aspect_ratio_range=(1.0, 5.0),
                      solidity_range=(0.5, 1.0), convexity_range=(0.7, 1.0), 
                      extent_range=(0.3, 1.0), exclude_edges=True,
                      blur_kernel=5, threshold_method="Otsu", threshold_value=None):
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
    
    def is_touching_edge(self, contour, image_shape):
        """Check if contour touches image edges"""
        h, w = image_shape[:2]
        for point in contour:
            x, y = point[0]
            if x <= 1 or y <= 1 or x >= w-2 or y >= h-2:
                return True
        return False
    
    def calculate_ellipse_dimensions(self, contour):
        """Calculate major and minor axis lengths from fitted ellipse"""
        if len(contour) < 5:
            return None, None, None
        
        try:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            return major_axis, minor_axis, angle
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
    
    def analyze_spore(self, contour, image_shape):
        """Analyze a single spore contour"""
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
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        return {
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
            'perimeter': cv2.arcLength(contour, True)
        }
    
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
        
        # Analyze each candidate using existing measurement logic
        spore_results = []
        for candidate in candidates:
            # Extract contour from candidate
            contour = candidate.get('contour')
            if contour is None:
                continue
                
            # Apply existing measurement and filtering logic
            spore_data = self.analyze_spore(contour, image.shape)
            if spore_data is not None:
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
            # Use pluggable detector
            candidates = detector.detect(image)
            return self.analyze_candidates(image, candidates)
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
        
        # Analyze each contour
        spore_results = []
        for contour in contours:
            spore_data = self.analyze_spore(contour, image.shape)
            if spore_data is not None:
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
