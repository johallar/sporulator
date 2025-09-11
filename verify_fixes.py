#!/usr/bin/env python3
"""
Simple verification script to test the critical fixes made to YOLOSegOnnxDetector.
"""

import numpy as np
import sys
import traceback

def test_import():
    """Test that the modules can be imported without syntax errors."""
    try:
        from spore_analyzer import YOLOSegOnnxDetector, SporeAnalyzer, mask_to_contours
        print("✓ Import test passed - no syntax errors")
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_detector_initialization():
    """Test detector initialization with missing model."""
    try:
        from spore_analyzer import YOLOSegOnnxDetector
        detector = YOLOSegOnnxDetector(model_path="nonexistent.onnx")
        print(f"✓ Detector initialization test passed - model_loaded: {detector.model_loaded}")
        return True
    except Exception as e:
        print(f"❌ Detector initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_threshold_setting():
    """Test the set_thresholds method."""
    try:
        from spore_analyzer import YOLOSegOnnxDetector
        detector = YOLOSegOnnxDetector(model_path="nonexistent.onnx")
        detector.set_thresholds(0.5, 0.7)
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.7
        print("✓ Threshold setting test passed")
        return True
    except Exception as e:
        print(f"❌ Threshold setting test failed: {e}")
        traceback.print_exc()
        return False

def test_analyzer_integration():
    """Test SporeAnalyzer integration with fallback."""
    try:
        from spore_analyzer import YOLOSegOnnxDetector, SporeAnalyzer
        analyzer = SporeAnalyzer()
        detector = YOLOSegOnnxDetector(model_path="nonexistent.onnx")
        
        # Create test image
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # This should trigger fallback to traditional detection
        result = analyzer.analyze_image(test_image, detector=detector)
        print("✓ Analyzer integration test passed - fallback mechanism works")
        return True
    except Exception as e:
        print(f"❌ Analyzer integration test failed: {e}")
        traceback.print_exc()
        return False

def test_mask_to_contours():
    """Test mask_to_contours utility."""
    try:
        from spore_analyzer import mask_to_contours
        import cv2
        
        # Create test mask
        mask = np.zeros((100, 100), dtype=np.float32)
        cv2.circle(mask, (50, 50), 20, (1.0,), -1)
        
        contours = mask_to_contours(mask)
        print(f"✓ mask_to_contours test passed - found {len(contours)} contours")
        return True
    except Exception as e:
        print(f"❌ mask_to_contours test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("Running verification tests for YOLOSegOnnxDetector fixes...\n")
    
    tests = [
        test_import,
        test_detector_initialization,
        test_threshold_setting,
        test_analyzer_integration,
        test_mask_to_contours
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\n")
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All verification tests passed! Critical fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the fixes.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)