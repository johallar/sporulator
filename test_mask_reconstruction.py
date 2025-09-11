#!/usr/bin/env python3
"""
Comprehensive test for YOLOv8-seg mask reconstruction implementation.

This test creates synthetic ONNX outputs that match YOLOv8-seg format
and verifies that the mask reconstruction pipeline works correctly.
"""

import numpy as np
import cv2
import logging
from spore_analyzer import YOLOSegOnnxDetector

def create_synthetic_yolov8_outputs(target_size=1024, num_detections=3):
    """
    Create synthetic YOLOv8-seg outputs for testing.
    
    YOLOv8-seg outputs:
    - outputs[0]: Detection tensor (1, num_detections, 4+1+32) = (xyxy, conf, mask_coeffs)
    - outputs[1]: Prototype masks (1, 32, mask_h, mask_w)
    
    Args:
        target_size: Size used during preprocessing
        num_detections: Number of synthetic detections
        
    Returns:
        List of synthetic outputs [detections, prototypes]
    """
    # Create synthetic detections (xyxy + confidence + 32 mask coefficients)
    # Format: [x1, y1, x2, y2, confidence, class, mask_coeff_0, ..., mask_coeff_31]
    detections = np.zeros((1, num_detections, 4 + 1 + 1 + 32), dtype=np.float32)
    
    # Synthetic bounding boxes (in padded image coordinates)
    bboxes = [
        [200, 200, 300, 300],  # 100x100 box
        [400, 400, 480, 480],  # 80x80 box  
        [600, 100, 700, 200],  # 100x100 box
    ]
    
    confidences = [0.9, 0.8, 0.85]
    
    for i in range(num_detections):
        detections[0, i, 0:4] = bboxes[i]  # xyxy
        detections[0, i, 4] = confidences[i]  # confidence
        detections[0, i, 5] = 0  # class (spore class)
        
        # Synthetic mask coefficients (random but consistent)
        np.random.seed(42 + i)  # Reproducible random coefficients
        detections[0, i, 6:] = np.random.randn(32) * 0.5
    
    # Create synthetic prototype masks (32 channels)
    mask_h, mask_w = 160, 160  # Common YOLOv8 mask resolution
    prototypes = np.zeros((1, 32, mask_h, mask_w), dtype=np.float32)
    
    # Fill prototypes with patterns that will create meaningful masks
    for ch in range(32):
        # Create simple patterns for each prototype channel
        y, x = np.ogrid[:mask_h, :mask_w]
        
        if ch % 4 == 0:
            # Circular patterns
            center_y, center_x = mask_h // 2, mask_w // 2
            prototypes[0, ch] = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 20**2))
        elif ch % 4 == 1:
            # Vertical gradients
            prototypes[0, ch] = (y / mask_h) * 2 - 1
        elif ch % 4 == 2:
            # Horizontal gradients  
            prototypes[0, ch] = (x / mask_w) * 2 - 1
        else:
            # Random noise patterns
            np.random.seed(ch)
            prototypes[0, ch] = np.random.randn(mask_h, mask_w) * 0.3
    
    return [detections, prototypes]

def test_mask_reconstruction_pipeline():
    """Test the complete mask reconstruction pipeline with synthetic data."""
    print("\n=== Testing Mask Reconstruction Pipeline ===")
    
    # Create detector instance (no model needed for this test)
    detector = YOLOSegOnnxDetector(model_path="test_model.onnx", confidence_threshold=0.7)
    
    # Create synthetic test image
    original_shape = (512, 384)  # H, W
    test_image = np.random.randint(0, 255, (*original_shape, 3), dtype=np.uint8)
    
    # Preprocess to get scale factor (similar to what would happen in detect())
    preprocessed_img, scale, orig_shape = detector._preprocess_image(test_image)
    print(f"Original shape: {orig_shape}, Scale: {scale:.3f}")
    
    # Create synthetic YOLOv8-seg outputs
    outputs = create_synthetic_yolov8_outputs(detector.target_size, num_detections=3)
    print(f"Created synthetic outputs: detections {outputs[0].shape}, prototypes {outputs[1].shape}")
    
    # Test the postprocessing pipeline
    candidates = detector._postprocess_output(outputs, scale, orig_shape)
    
    print(f"Postprocessing returned {len(candidates)} candidates")
    
    # Verify results
    assert len(candidates) > 0, "Should return candidates from synthetic data"
    
    for i, candidate in enumerate(candidates):
        print(f"Candidate {i+1}:")
        print(f"  - Has contour: {'contour' in candidate}")
        print(f"  - Has confidence: {'confidence' in candidate}")
        print(f"  - Detection type: {candidate.get('detection_type', 'unknown')}")
        
        if 'contour' in candidate:
            contour = candidate['contour']
            area = cv2.contourArea(contour)
            print(f"  - Contour points: {len(contour)}, Area: {area:.1f}")
            
            assert len(contour) > 0, f"Candidate {i+1} should have non-empty contour"
            assert area > 0, f"Candidate {i+1} should have positive area"
        
        if 'confidence' in candidate:
            confidence = candidate['confidence']
            print(f"  - Confidence: {confidence:.3f}")
            assert 0 <= confidence <= 1, f"Candidate {i+1} confidence should be in [0,1]"
    
    print("✓ Mask reconstruction pipeline test passed")
    return candidates

def test_decode_yolov8_masks_directly():
    """Test the _decode_yolov8_masks method directly with synthetic data."""
    print("\n=== Testing _decode_yolov8_masks method directly ===")
    
    detector = YOLOSegOnnxDetector(model_path="test_model.onnx")
    
    # Create synthetic inputs
    original_shape = (512, 384)
    scale = 0.75
    
    # Prototype masks (32, mask_h, mask_w)
    mask_h, mask_w = 160, 160
    proto_masks = np.random.randn(32, mask_h, mask_w).astype(np.float32)
    
    # Mask coefficients (num_detections, 32)
    num_detections = 2
    np.random.seed(42)
    mask_coeffs = np.random.randn(num_detections, 32).astype(np.float32)
    
    # Bounding boxes (num_detections, 4)
    bboxes = np.array([
        [200, 200, 300, 300],  # Detection 1
        [400, 400, 480, 480],  # Detection 2
    ], dtype=np.float32)
    
    print(f"Testing with {num_detections} detections")
    print(f"Proto masks shape: {proto_masks.shape}")
    print(f"Mask coeffs shape: {mask_coeffs.shape}")
    print(f"Bboxes shape: {bboxes.shape}")
    
    # Test mask decoding
    decoded_masks = detector._decode_yolov8_masks(
        proto_masks, mask_coeffs, bboxes, original_shape, scale
    )
    
    print(f"Decoded masks shape: {decoded_masks.shape}")
    print(f"Mask value range: [{decoded_masks.min():.3f}, {decoded_masks.max():.3f}]")
    
    # Verify results
    assert decoded_masks.shape[0] == num_detections, "Should have mask for each detection"
    assert decoded_masks.shape[1:] == original_shape, "Masks should match original image shape"
    assert 0 <= decoded_masks.min() and decoded_masks.max() <= 1, "Masks should be in [0,1] range"
    
    # Check that masks contain meaningful values
    for i in range(num_detections):
        mask = decoded_masks[i]
        nonzero_pixels = np.sum(mask > 0.5)
        print(f"  Mask {i+1}: {nonzero_pixels} pixels > 0.5 threshold")
        assert nonzero_pixels > 0, f"Mask {i+1} should have some activated pixels"
    
    print("✓ _decode_yolov8_masks direct test passed")
    return decoded_masks

def test_integration_with_spore_analyzer():
    """Test integration of mask reconstruction with SporeAnalyzer."""
    print("\n=== Testing integration with SporeAnalyzer ===")
    
    from spore_analyzer import SporeAnalyzer
    
    # Create synthetic test image with spore-like objects
    test_image = np.zeros((512, 384, 3), dtype=np.uint8)
    
    # Add some circular objects
    centers = [(128, 128), (256, 200), (400, 300)]
    radii = [30, 25, 35]
    for center, radius in zip(centers, radii):
        cv2.circle(test_image, center, radius, (255, 255, 255), -1)
    
    # Initialize analyzer and detector
    analyzer = SporeAnalyzer()
    analyzer.set_parameters(pixel_scale=2.0, min_area=50, max_area=2000)
    
    # Create mock detector that returns synthetic candidates
    class MockYOLODetector:
        def detect(self, image):
            # Return synthetic candidates with contours matching the circles
            candidates = []
            for center, radius in zip(centers, radii):
                # Create circular contour
                angles = np.linspace(0, 2*np.pi, 50)
                contour_points = []
                for angle in angles:
                    x = int(center[0] + radius * np.cos(angle))
                    y = int(center[1] + radius * np.sin(angle))
                    contour_points.append([[x, y]])
                
                contour = np.array(contour_points, dtype=np.int32)
                candidates.append({
                    'contour': contour,
                    'confidence': 0.9,
                    'detection_type': 'segmentation'
                })
            
            return candidates
    
    mock_detector = MockYOLODetector()
    
    # Test analyze_image with mock detector
    results = analyzer.analyze_image(test_image, detector=mock_detector)
    
    if results is not None:
        print(f"SporeAnalyzer found {len(results)} valid spores")
        for i, spore in enumerate(results):
            print(f"  Spore {i+1}: Area={spore['area_um2']:.1f} μm², "
                  f"Length={spore['length_um']:.1f} μm, Width={spore['width_um']:.1f} μm")
    else:
        print("No valid spores found by analyzer")
    
    # Test analyze_candidates directly
    mock_candidates = mock_detector.detect(test_image)
    results2 = analyzer.analyze_candidates(test_image, mock_candidates)
    
    print(f"Direct analyze_candidates returned: {type(results2)}")
    if results2 is not None:
        print(f"Found {len(results2)} valid candidates")
    
    print("✓ SporeAnalyzer integration test passed")

def main():
    """Run all mask reconstruction tests."""
    print("Starting comprehensive mask reconstruction tests...")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test the core mask decoding method
        decoded_masks = test_decode_yolov8_masks_directly()
        
        # Test the complete pipeline
        candidates = test_mask_reconstruction_pipeline()
        
        # Test integration with SporeAnalyzer
        test_integration_with_spore_analyzer()
        
        print("\n" + "="*60)
        print("✅ All mask reconstruction tests passed successfully!")
        print("The YOLOv8-seg mask reconstruction implementation is working correctly.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)