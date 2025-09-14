#!/usr/bin/env python3
"""
Test script for watershed spore separation functionality.
This script tests the new watershed segmentation feature in SporeAnalyzer.
"""

import cv2
import numpy as np
from spore_analyzer import SporeAnalyzer

def create_touching_spores_image():
    """Create a synthetic image with touching spores for testing."""
    # Create a blank image
    img = np.zeros((200, 300), dtype=np.uint8)
    
    # Draw two touching circles (spores)
    cv2.circle(img, (80, 100), 30, 255, -1)    # First spore
    cv2.circle(img, (120, 100), 25, 255, -1)   # Second spore (touching)
    
    # Draw two more touching ellipses
    cv2.ellipse(img, (220, 80), (25, 15), 0, 0, 360, 255, -1)    # Third spore
    cv2.ellipse(img, (220, 120), (20, 18), 45, 0, 360, 255, -1)  # Fourth spore (touching)
    
    return img

def test_watershed_separation():
    """Test the watershed separation functionality."""
    print("Testing watershed spore separation...")
    
    # Create test image
    test_img = create_touching_spores_image()
    
    # Create analyzer with separation enabled
    analyzer = SporeAnalyzer()
    analyzer.set_parameters(
        pixel_scale=1.0,
        min_area=100,
        max_area=5000,
        exclude_touching=False,  # We want to enable separation, not exclusion
        separate_touching=True,  # Enable watershed separation
        separation_min_distance=10,
        separation_sigma=1.0,
        separation_erosion_iterations=1
    )
    
    print(f"Analyzer settings:")
    print(f"  - exclude_touching: {analyzer.exclude_touching}")
    print(f"  - separate_touching: {analyzer.separate_touching}")
    print(f"  - separation_min_distance: {analyzer.separation_min_distance}")
    
    # Analyze image
    results = analyzer.analyze_image(test_img)
    
    if results is None:
        print("❌ No spores detected!")
        return False
    
    print(f"\n✅ Analysis completed!")
    print(f"Number of spores detected: {len(results)}")
    
    # Check results
    separated_count = 0
    failed_separation_count = 0
    normal_count = 0
    
    for i, spore in enumerate(results):
        print(f"\nSpore {i+1}:")
        print(f"  - Area: {spore['area_um2']:.1f} μm²")
        print(f"  - Length: {spore['length_um']:.1f} μm")
        print(f"  - Width: {spore['width_um']:.1f} μm")
        print(f"  - Circularity: {spore['circularity']:.3f}")
        print(f"  - Touching detected: {spore['touching_detected']}")
        
        if 'separated_from_touching' in spore:
            print(f"  - 🔄 Separated from touching spores")
            separated_count += 1
        elif 'separation_failed' in spore:
            print(f"  - ⚠️ Separation failed")
            failed_separation_count += 1
        else:
            print(f"  - ✓ Normal detection")
            normal_count += 1
    
    print(f"\nSummary:")
    print(f"  - Successfully separated spores: {separated_count}")
    print(f"  - Failed separations: {failed_separation_count}")
    print(f"  - Normal detections: {normal_count}")
    
    # Test should pass if we got multiple spores from the touching regions
    return len(results) > 2  # We expect at least 3-4 spores from the touching pairs

def test_separation_fallback():
    """Test that separation gracefully falls back when it fails."""
    print("\n" + "="*50)
    print("Testing separation fallback behavior...")
    
    # Create a simple single circle (no touching)
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 255, -1)
    
    analyzer = SporeAnalyzer()
    analyzer.set_parameters(
        pixel_scale=1.0,
        min_area=100,
        max_area=2000,
        exclude_touching=False,  # Don't exclude
        separate_touching=True   # Try to separate
    )
    
    results = analyzer.analyze_image(img)
    
    if results is None or len(results) == 0:
        print("❌ Failed to detect single spore!")
        return False
    
    print(f"✅ Single spore detected correctly")
    print(f"Touching detected: {results[0]['touching_detected']}")
    
    return True

def test_exclusion_vs_separation():
    """Compare exclusion vs separation behavior."""
    print("\n" + "="*50)
    print("Testing exclusion vs separation comparison...")
    
    test_img = create_touching_spores_image()
    
    # Test with exclusion enabled
    analyzer_exclude = SporeAnalyzer()
    analyzer_exclude.set_parameters(
        pixel_scale=1.0,
        min_area=100,
        max_area=5000,
        exclude_touching=True,   # Exclude touching spores
        separate_touching=False
    )
    
    results_exclude = analyzer_exclude.analyze_image(test_img)
    exclude_count = len(results_exclude) if results_exclude else 0
    
    # Test with separation enabled
    analyzer_separate = SporeAnalyzer()
    analyzer_separate.set_parameters(
        pixel_scale=1.0,
        min_area=100,
        max_area=5000,
        exclude_touching=False,  # Don't exclude
        separate_touching=True   # Separate instead
    )
    
    results_separate = analyzer_separate.analyze_image(test_img)
    separate_count = len(results_separate) if results_separate else 0
    
    print(f"Results with exclusion: {exclude_count} spores")
    print(f"Results with separation: {separate_count} spores")
    
    # We expect more spores with separation than exclusion
    success = separate_count > exclude_count
    
    if success:
        print("✅ Separation detected more spores than exclusion")
    else:
        print("❌ Separation did not improve spore count")
        
    return success

if __name__ == "__main__":
    print("Testing Watershed Spore Separation Implementation")
    print("="*60)
    
    # Run tests
    test_results = []
    
    try:
        test_results.append(("Basic separation", test_watershed_separation()))
        test_results.append(("Fallback behavior", test_separation_fallback()))
        test_results.append(("Exclusion vs separation", test_exclusion_vs_separation()))
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY:")
        print("="*60)
        
        passed = 0
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(test_results)} tests passed")
        
        if passed == len(test_results):
            print("\n🎉 All tests passed! Watershed separation is working correctly.")
        else:
            print("\n⚠️ Some tests failed. Check the implementation.")
            
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        import traceback
        traceback.print_exc()