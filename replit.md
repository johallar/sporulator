# Overview

The Fungal Spore Analyzer is a Streamlit-based web application for automated detection and measurement of fungal spores in microscopy images. The system provides computer vision capabilities to analyze spore morphology, calculate statistical measurements, and export results for research purposes. It supports multiple detection backends including traditional computer vision methods and YOLO-based deep learning models.

# Recent Changes

**2025-09-14**: Simplified calibration system to only include manual entry and auto-detect micrometer divisions methods. Removed auto-detect scale bar, auto-detect circular object, and upload calibration image options. Modified auto-detect micrometer divisions to use the same uploaded image instead of requiring separate image upload.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Interface**: Single-page application providing an intuitive user interface for image upload, parameter configuration, and results visualization
- **Interactive Visualization**: Plotly integration for statistical charts and graphs showing spore measurements and distributions
- **Session State Management**: Streamlit session state for maintaining analysis results, calibration settings, and user selections across interactions
- **Badge-Style Spore Selection**: Modern collapsible interface with badge buttons for individual spore selection/deselection

## Backend Architecture
- **Modular Detection System**: Abstract detector interface allowing multiple computer vision backends (traditional CV methods and YOLO deep learning models)
- **SporeAnalyzer Core**: Central analysis engine that processes images, applies filters, and calculates morphological measurements
- **Simplified Calibration System**: Two-method calibration system with manual entry and auto-detect micrometer divisions using the main uploaded image
- **Statistical Analysis**: Comprehensive morphological measurements including length, width, area, circularity, aspect ratio, and other shape descriptors

## Computer Vision Pipeline
- **Image Preprocessing**: OpenCV-based image enhancement and normalization
- **Detection Backends**: 
  - Traditional computer vision using edge detection and contour analysis
  - YOLO segmentation model (ONNX format) for deep learning-based detection
- **Morphological Analysis**: scikit-image integration for advanced shape analysis and feature extraction
- **Quality Filtering**: Configurable double-slider range filters for area, circularity, aspect ratio to remove false positives

## Data Processing
- **Results Export**: CSV and Excel export functionality for statistical analysis
- **Image Overlay Generation**: Visual feedback showing detected spores with measurements overlaid on original images
- **Statistical Calculations**: NumPy and Pandas-based statistical analysis with mean, standard deviation, min/max, and median calculations
- **Mycological Format Statistics**: Professional mycological dimensional summaries with percentile ranges and Qm values

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **OpenCV**: Computer vision library for image processing and analysis
- **NumPy**: Numerical computing for array operations and mathematical calculations
- **Pandas**: Data manipulation and analysis for statistical calculations
- **Plotly**: Interactive visualization library for charts and graphs
- **PIL (Pillow)**: Image processing library for format conversion and basic operations

## Scientific Computing
- **scikit-image**: Advanced image processing algorithms for morphological analysis
- **SciPy**: Scientific computing library for spatial distance calculations and advanced algorithms

## Optional Deep Learning
- **ONNX Runtime**: Optional dependency for YOLO model inference (graceful degradation when unavailable)
- **YOLO Models**: Support for ONNX-format YOLO segmentation models for enhanced detection accuracy

## Image Processing Stack
- **Morphological Operations**: Connected component analysis, watershed segmentation, and shape filtering
- **Edge Detection**: Canny edge detection and Hough transforms for feature extraction
- **Contour Analysis**: OpenCV contour detection and analysis for shape measurements