# Fungal Spore Analyzer - Technical Documentation

## Overview

The Fungal Spore Analyzer is a sophisticated computer vision application designed for automated detection, measurement, and analysis of fungal spores in microscopy images. The system combines traditional image processing techniques with advanced segmentation algorithms to provide accurate morphological measurements for mycological research.

## Core Technologies

### Computer Vision Pipeline

#### 1. Image Preprocessing
- **Gaussian Blur**: Reduces noise while preserving edge information using configurable kernel sizes (3x3 to 15x15)
- **Adaptive Thresholding**: Multiple methods available:
  - Otsu's method for automatic threshold selection
  - Manual threshold control for challenging samples
  - Binary image conversion optimized for microscopy images

#### 2. Contour Detection and Analysis
- **OpenCV Contour Finding**: Uses `cv2.findContours()` with optimized parameters
- **Hierarchical Analysis**: Employs `cv2.RETR_EXTERNAL` to focus on outer boundaries
- **Contour Approximation**: `cv2.CHAIN_APPROX_SIMPLE` for efficient boundary representation

### Advanced Spore Detection

#### Traditional Computer Vision Method
- **Edge Detection**: Canny edge detection with adaptive thresholds
- **Morphological Operations**: 
  - Opening operations to separate touching objects
  - Closing operations to fill small gaps
  - Erosion and dilation for noise reduction

#### Deep Learning Integration (Optional)
- **YOLO Segmentation**: Support for ONNX-format YOLO models
- **Instance Segmentation**: Pixel-level spore identification
- **Confidence Scoring**: Automated quality assessment of detections

### Watershed Segmentation for Touching Spores

One of the system's most advanced features is the automatic separation of touching or overlapping spores using watershed segmentation:

#### Algorithm Implementation
1. **Distance Transform**: Uses `cv2.distanceTransform()` to create a distance map
2. **Local Maxima Detection**: Employs `scipy.ndimage.maximum_filter` to find watershed seeds
3. **Gaussian Smoothing**: Optional smoothing parameter for improved separation
4. **Morphological Preprocessing**: Configurable erosion to thin connected regions

#### Touching Spore Detection Criteria
The system uses a multi-metric approach to identify potentially touching spores:

**Hard Rules:**
- Solidity < 0.85 with deep convexity defects
- Multiple deep convexity defects (≥2)
- Low convexity (<0.90) combined with poor ellipse fit
- Statistical area outliers using robust z-scores

**Score-Based Detection:**
```
Composite Score = 0.4 × max(0, 0.9 - solidity) + 
                  0.2 × max(0, 0.95 - convexity) + 
                  0.3 × min(1, deep_defects / 2) + 
                  0.1 × ellipse_residual
```

**Aggressiveness Levels:**
- Conservative: threshold = 0.6 (fewer separations)
- Balanced: threshold = 0.5 (default)
- Aggressive: threshold = 0.4 (more separations)

### Morphological Measurements

#### Primary Measurements
- **Length and Width**: Fitted ellipse major and minor axes
- **Area**: Pixel count converted to square micrometers
- **Perimeter**: Contour boundary length
- **Aspect Ratio**: Length-to-width ratio for shape classification

#### Shape Descriptors
- **Circularity**: `4π × Area / Perimeter²` (0-1, where 1 = perfect circle)
- **Solidity**: `Contour Area / Convex Hull Area` (measures concavity)
- **Convexity**: `Convex Hull Perimeter / Contour Perimeter` (boundary smoothness)
- **Extent**: `Contour Area / Bounding Rectangle Area` (space efficiency)

#### Advanced Shape Analysis
- **Convexity Defects**: Identifies indentations suggesting merged spores
- **Ellipse Fitting**: Least squares ellipse fitting with residual analysis
- **Moment Analysis**: Statistical moments for shape characterization

### Calibration Systems

#### Automatic Scale Detection
- **Scale Bar Recognition**: Computer vision detection of embedded scale bars
- **Known Object Calibration**: Reference objects of known dimensions
- **Hough Transform**: Line detection for scale bar identification
- **Template Matching**: Pattern recognition for standard scale markers

#### Manual Calibration
- **Interactive Point Selection**: User-defined reference measurements
- **Validation Systems**: Automatic verification of calibration accuracy
- **Multiple Unit Support**: Micrometers, nanometers, millimeters

### Quality Filtering

#### Multi-Parameter Filtering
- **Area Range**: Configurable minimum and maximum spore sizes
- **Shape Constraints**: Circularity, aspect ratio, and solidity limits
- **Edge Exclusion**: Removes partially visible spores at image boundaries
- **Statistical Outliers**: Robust z-score filtering for anomalous measurements

#### Touching Spore Handling
- **Detection**: Multi-metric identification of merged spores
- **Separation**: Watershed segmentation to split touching objects
- **Validation**: Quality assessment of separated components

### Statistical Analysis

#### Descriptive Statistics
- Mean, median, standard deviation for all measurements
- Minimum, maximum, and percentile calculations
- Sample size and measurement confidence intervals

#### Mycological Reporting
- **Standardized Format**: Research-grade statistical summaries
- **Dimensional Analysis**: Length × width format common in taxonomy
- **Population Metrics**: Distribution analysis and outlier identification

### Visualization System

#### Overlay Generation
- **Measurement Lines**: Visual representation of length and width
- **Multi-line Labels**: Spore number with measurements on separate lines
- **Connecting Lines**: Links between labels and corresponding spores
- **Color Coding**: Selected vs. unselected spore differentiation

#### Anti-Collision System
- **Global Text Positioning**: Prevents overlap between all text elements
- **Smart Positioning**: Automatic placement around spore boundaries
- **Boundary Detection**: Ensures text remains within image bounds
- **Background Styling**: Semi-transparent backgrounds for readability

## Technical Implementation

### Software Architecture
- **Modular Design**: Separate detector backends for extensibility
- **Abstract Interfaces**: Consistent API across detection methods
- **Session Management**: Streamlit state handling for user interactions
- **Error Handling**: Graceful degradation for edge cases

### Performance Optimizations
- **Vectorized Operations**: NumPy and OpenCV optimizations
- **Memory Management**: Efficient image processing pipelines
- **Caching**: Reuse of expensive computations
- **Batch Processing**: Efficient handling of multiple spores

### Dependencies
- **Core**: OpenCV, NumPy, SciPy, scikit-image
- **Visualization**: Matplotlib, Plotly, Streamlit
- **Data Processing**: Pandas for statistical analysis
- **Optional**: ONNX Runtime for deep learning models

## Research Applications

### Mycological Studies
- **Species Identification**: Morphometric analysis for taxonomy
- **Population Studies**: Statistical analysis of spore populations
- **Quality Control**: Standardized measurement protocols
- **Comparative Analysis**: Cross-sample statistical comparisons

### Measurement Accuracy
- **Pixel-Level Precision**: Sub-pixel accuracy through interpolation
- **Calibration Validation**: Automatic verification systems
- **Reproducibility**: Consistent measurements across sessions
- **Error Quantification**: Statistical confidence intervals

## Usage Guidelines

### Best Practices
1. **Image Quality**: High-contrast, well-focused microscopy images
2. **Calibration**: Always verify scale accuracy before analysis
3. **Parameter Tuning**: Adjust filters based on sample characteristics
4. **Quality Review**: Manual verification of automated detections

### Troubleshooting
- **Under-detection**: Reduce minimum area, adjust circularity filters
- **Over-detection**: Increase filtering stringency, enable edge exclusion
- **Poor Separation**: Adjust watershed parameters, try different aggressiveness
- **Calibration Issues**: Verify scale bar visibility, check reference dimensions

## Future Enhancements

### Planned Features
- **3D Analysis**: Support for z-stack image analysis
- **Machine Learning**: Custom model training for specific spore types
- **Batch Processing**: Automated analysis of image series
- **Advanced Statistics**: Multivariate analysis and clustering

### Research Integration
- **Database Connectivity**: Direct integration with specimen databases
- **Export Formats**: Additional formats for research software compatibility
- **API Development**: Programmatic access for research pipelines