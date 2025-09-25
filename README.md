# Fungal Spore Analyzer

A comprehensive web-based application for automated detection and measurement of fungal spores in microscopy images. This tool provides computer vision capabilities to analyze spore morphology, calculate statistical measurements, and export results for research purposes.

![Fungal Spore Analyzer](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-green.svg)

## 🔬 Features

- **Automated Spore Detection**: Advanced computer vision algorithms with optional YOLO deep learning models
- **Interactive Calibration**: Multiple calibration methods including manual entry and auto-detect micrometer divisions
- **Watershed Segmentation**: Automatically separates touching or overlapping spores
- **Comprehensive Measurements**: Length, width, area, circularity, aspect ratio, and shape descriptors
- **iNaturalist Integration**: Direct image loading from iNaturalist observations
- **Interactive Visualization**: Plotly-based charts and overlay generation
- **Export Capabilities**: CSV and Excel export with mycological statistics format
- **Quality Filtering**: Configurable filters for area, circularity, and aspect ratio

## 🛠️ Prerequisites

Before running this application locally, ensure you have the following installed:

### Required Software

- **Python 3.11 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** - [Download Git](https://git-scm.com/downloads/)

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Memory**: Minimum 4GB RAM (8GB recommended for large images)
- **Storage**: At least 2GB free disk space

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fungal-spore-analyzer
```

### 2. Install Dependencies

#### UV

Install UV

#### Install Deps

```bash
uv sync
```

### 4. Create Configuration Directory

Note: This should already exist, make any edits needed
Create the Streamlit configuration directory and file:

```bash
mkdir -p .streamlit
```

Create `.streamlit/config.toml` with the following content:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

[theme]
base = "light"
primaryColor = "#b67feb"
```

## 🚀 Running the Application

### Start the Application

```bash
uv run streamlit run app.py --server.port 5000
```

### Access the Application

Once started, the application will be available at:

- **Local URL**: http://localhost:5000

The terminal will display both URLs when the application starts successfully.

## 📋 Usage Instructions

### 1. Image Source (Step 1)

- **Upload Image**: Use the file uploader to select microscopy images (JPG, PNG, TIFF)
- **iNaturalist Integration**: Enter an observation ID to load images directly from iNaturalist
- **Image Requirements**: High-contrast, well-focused microscopy images work best

### 2. Calibration (Step 2)

Choose one of the calibration methods:

#### Manual Entry

- Enter the pixel-to-micrometer conversion ratio manually
- Useful when you know the exact scale from microscope settings

#### Auto-detect Micrometer Divisions

- Upload an image with visible micrometer scale divisions
- The system will automatically detect and calibrate the scale
- Works best with clear, high-contrast scale markings

### 3. Analysis (Step 3)

- **Configure Detection Parameters**: Adjust area filters, circularity, and shape constraints
- **Choose Detection Method**: Traditional computer vision or YOLO deep learning (if model available)
- **Enable Watershed Separation**: Automatically separate touching spores
- **Run Analysis**: Process the image and view detected spores
- **Review Results**: Use the interactive spore legend to select/deselect individual spores
- **Export Data**: Download results as CSV or Excel files

## 🔧 Configuration Options

### Detection Parameters

- **Area Range**: Minimum and maximum spore size (in pixels or micrometers)
- **Circularity Filter**: Shape constraint (0.0 = any shape, 1.0 = perfect circle)
- **Aspect Ratio**: Length-to-width ratio limits
- **Edge Exclusion**: Remove partially visible spores at image borders

### Watershed Separation

- **Aggressiveness**: Conservative, Balanced, or Aggressive separation
- **Smoothing**: Gaussian smoothing for improved separation
- **Erosion Cycles**: Morphological preprocessing for connected regions

### Export Settings

- **Measurement Units**: Micrometers, millimeters, or pixels
- **Statistical Format**: Include mycological summary statistics
- **Image Overlays**: Export annotated images with measurements

## 🐛 Troubleshooting

### Common Installation Issues

#### OpenCV Installation Problems:

```bash
# If opencv-python fails to install:
pip install --upgrade pip setuptools wheel
pip install opencv-python-headless opencv-python
```

#### Memory Issues with Large Images:

- Resize images to maximum 2048x2048 pixels before upload
- Close other applications to free memory
- Consider using a machine with more RAM

#### Port Already in Use:

```bash
# Try a different port:
streamlit run app.py --server.port 8080
```

### Application Issues

#### No Spores Detected:

- Check image contrast and focus quality
- Adjust area range filters (try wider ranges)
- Reduce circularity filter threshold
- Ensure proper calibration

#### Too Many False Detections:

- Increase minimum area threshold
- Tighten circularity filter (closer to 1.0)
- Enable edge exclusion
- Adjust aspect ratio limits

#### Calibration Problems:

- Verify scale bar visibility and contrast
- Check reference distance accuracy
- Try manual calibration as fallback
- Ensure image contains clear scale markers

### Performance Issues

#### Slow Processing:

- Reduce image size before upload
- Disable watershed separation for faster processing
- Use traditional CV method instead of YOLO
- Close browser tabs and other applications

## 📁 Project Structure

```
fungal-spore-analyzer/
├── app.py                  # Main Streamlit application
├── spore_analyzer.py       # Core analysis engine
├── calibration.py          # Calibration system
├── utils.py               # Utility functions
├── pyproject.toml         # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── attached_assets/       # Sample images and assets
└── README.md             # This file
```

## 🔬 Technical Details

### Core Technologies

- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and image processing
- **NumPy & SciPy**: Numerical computing and scientific algorithms
- **Plotly**: Interactive visualizations and charts
- **Pandas**: Data manipulation and statistical analysis
- **scikit-image**: Advanced image processing algorithms

### Detection Methods

- **Traditional CV**: Edge detection, morphological operations, and contour analysis
- **YOLO Segmentation**: Deep learning models for enhanced detection (optional)
- **Watershed Segmentation**: Separation of touching or overlapping spores

### Measurement Accuracy

- Sub-pixel precision through interpolation and ellipse fitting
- Automatic calibration validation
- Statistical confidence intervals for measurements

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check this README for troubleshooting steps
2. Review the terminal output for error messages
3. Ensure all dependencies are properly installed
4. Try with different image samples to isolate issues

## 🔮 Future Enhancements

- 3D analysis for z-stack images
- Custom model training for specific spore types
- Batch processing for multiple images
- Advanced statistical analysis and clustering
- Database integration for specimen management

---

**Happy Spore Analysis!** 🔬✨
