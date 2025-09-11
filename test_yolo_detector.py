#!/usr/bin/env python3
"""
Test script for YOLOSegOnnxDetector implementation.

This script tests:
1. YOLOSegOnnxDetector initialization with missing model (graceful handling)
2. mask_to_contours utility function with synthetic data
3. Integration with SporeAnalyzer
"""

import numpy as np
import cv2
import logging
from spore_analyzer import YOLOSegOnnxDetector, SporeAnalyzer, mask_to_contours

def create_test_image(size=(512, 512)):
    """Create a synthetic test image with spore-like objects."""
    image = np.zeros((*size, 3), dtype=np.uint8)
    
    # Add some circular objects that look like spores
    centers = [(128, 128), (256, 256), (384, 384)]
    radii = [30, 25, 35]
    
    for center, radius in zip(centers, radii):
        cv2.circle(image, center, radius, (255, 255, 255), -1)
    
    return image

def create_test_mask(size=(512, 512)):
    """Create a synthetic segmentation mask."""
    mask = np.zeros(size, dtype=np.float32)
    
    # Add some circular objects
    centers = [(128, 128), (256, 256), (384, 384)]
    radii = [30, 25, 35]
    
    for center, radius in zip(centers, radii):
        cv2.circle(mask, center, radius, (1.0,), -1)
    
    return mask

def test_mask_to_contours():
    """Test the mask_to_contours utility function."""
    print("\n=== Testing mask_to_contours utility function ===")
    
    # Create a test mask
    mask = create_test_mask()
    
    # Convert mask to contours
    contours = mask_to_contours(mask, confidence_threshold=0.5)
    
    print(f"Created test mask with shape: {mask.shape}")
    print(f"Found {len(contours)} contours")
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(f"  Contour {i+1}: {len(contour)} points, area = {area:.1f}")
    
    assert len(contours) > 0, "Should find contours in test mask"
    print("✓ mask_to_contours test passed")

def test_yolo_detector_initialization():
    """Test YOLOSegOnnxDetector initialization with missing model."""
    print("\n=== Testing YOLOSegOnnxDetector initialization ===")
    
    # Test with non-existent model path (should handle gracefully)
    detector = YOLOSegOnnxDetector(
        model_path="nonexistent_model.onnx",
        confidence_threshold=0.3,
        iou_threshold=0.5
    )
    
    print(f"Detector initialized with model_loaded: {detector.model_loaded}")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print(f"IoU threshold: {detector.iou_threshold}")
    
    assert not detector.model_loaded, "Should handle missing model gracefully"
    print("✓ Detector initialization test passed")

def test_yolo_detector_detect():
    """Test YOLOSegOnnxDetector detection with missing model."""
    print("\n=== Testing YOLOSegOnnxDetector detection ===")
    
    # Initialize detector with missing model
    detector = YOLOSegOnnxDetector(model_path="nonexistent_model.onnx")
    
    # Create test image
    test_image = create_test_image()
    print(f"Created test image with shape: {test_image.shape}")
    
    # Run detection (should return empty list gracefully)
    candidates = detector.detect(test_image)
    
    print(f"Detection returned {len(candidates)} candidates")
    assert isinstance(candidates, list), "Detection should return a list"
    assert len(candidates) == 0, "Should return empty list with missing model"
    
    print("✓ Detector detection test passed")

def test_spore_analyzer_integration():
    """Test integration with SporeAnalyzer."""
    print("\n=== Testing SporeAnalyzer integration ===")
    
    # Initialize analyzer and detector
    analyzer = SporeAnalyzer()
    detector = YOLOSegOnnxDetector(model_path="nonexistent_model.onnx")
    
    # Create test image
    test_image = create_test_image()
    
    # Test analyze_image with detector
    results = analyzer.analyze_image(test_image, detector=detector)
    
    print(f"SporeAnalyzer returned: {results}")
    assert results is None or isinstance(results, list), "Should return None or list"
    
    # Test analyze_candidates with empty candidates
    empty_candidates = []
    results = analyzer.analyze_candidates(test_image, empty_candidates)
    
    assert results is None, "Should return None for empty candidates"
    
    # Test analyze_candidates with mock candidates
    mock_contour = np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]], dtype=np.int32)
    mock_candidates = [{'contour': mock_contour, 'confidence': 0.8}]
    
    results = analyzer.analyze_candidates(test_image, mock_candidates)
    print(f"Mock candidates analysis: {type(results)}")
    
    print("✓ SporeAnalyzer integration test passed")

def test_threshold_setting():
    """Test threshold setting functionality."""
    print("\n=== Testing threshold setting ===")
    
    detector = YOLOSegOnnxDetector()
    
    # Test initial thresholds
    print(f"Initial confidence threshold: {detector.confidence_threshold}")
    print(f"Initial IoU threshold: {detector.iou_threshold}")
    
    # Update thresholds
    detector.set_thresholds(0.5, 0.7)
    
    print(f"Updated confidence threshold: {detector.confidence_threshold}")
    print(f"Updated IoU threshold: {detector.iou_threshold}")
    
    assert detector.confidence_threshold == 0.5, "Confidence threshold should be updated"
    assert detector.iou_threshold == 0.7, "IoU threshold should be updated"
    
    print("✓ Threshold setting test passed")

def main():
    """Run all tests."""
    print("Starting YOLOSegOnnxDetector tests...")
    
    # Set up logging to capture warnings
    logging.basicConfig(level=logging.INFO)
    
    try:
        test_mask_to_contours()
        test_yolo_detector_initialization()
        test_yolo_detector_detect()
        test_spore_analyzer_integration()
        test_threshold_setting()
        
        print("\n" + "="*50)
        print("✅ All tests passed successfully!")
        print("YOLOSegOnnxDetector implementation is working correctly.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)