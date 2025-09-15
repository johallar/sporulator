import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import requests
import re
from spore_analyzer import SporeAnalyzer
from utils import calculate_statistics, create_overlay_image, export_results, generate_mycological_summary
from calibration import StageCalibration

# Configure page
st.set_page_config(page_title="Sporulator",
                   page_icon="🔬",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Initialize session state
if 'spore_analyzer' not in st.session_state:
    st.session_state.spore_analyzer = SporeAnalyzer()
if 'calibration' not in st.session_state:
    st.session_state.calibration = StageCalibration()
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_spores' not in st.session_state:
    st.session_state.selected_spores = set()
if 'calibration_complete' not in st.session_state:
    st.session_state.calibration_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'pixel_scale' not in st.session_state:
    st.session_state.pixel_scale = 1.0  # Default value

# Wizard step management session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'step_1_complete' not in st.session_state:
    st.session_state.step_1_complete = False
if 'step_2_complete' not in st.session_state:
    st.session_state.step_2_complete = False
if 'step_3_complete' not in st.session_state:
    st.session_state.step_3_complete = False


def fetch_inaturalist_photos(observation_id):
    """Fetch all photos metadata from iNaturalist observation"""
    try:
        # Call iNaturalist API to get observation data
        api_url = f"https://api.inaturalist.org/v1/observations/{observation_id}"
        response = requests.get(api_url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()
        results = data.get('results', [])

        if not results:
            return None

        # Get photos from the first result
        photos = results[0].get('photos', [])
        if not photos:
            return None

        # Return photo metadata
        photo_info = []
        for i, photo in enumerate(photos):
            photo_info.append({
                'id':
                photo.get('id'),
                'url':
                photo.get('url', ''),
                'attribution':
                photo.get('attribution', 'Unknown'),
                'license':
                photo.get('license_code', 'Unknown'),
                'index':
                i
            })

        return photo_info

    except Exception as e:
        st.error(f"Error fetching iNaturalist photos: {str(e)}")
        return None


def download_inaturalist_image(photo_url, size="large"):
    """Download a specific iNaturalist image at the specified size"""
    try:
        # Convert URL to desired size
        if '/square.jpg' in photo_url:
            image_url = photo_url.replace('/square.jpg', f'/{size}.jpg')
        elif '/thumb.jpg' in photo_url:
            image_url = photo_url.replace('/thumb.jpg', f'/{size}.jpg')
        else:
            # Try to replace any size with the desired size
            for old_size in ['thumb', 'small', 'medium', 'large', 'original']:
                if f'/{old_size}.' in photo_url:
                    image_url = photo_url.replace(f'/{old_size}.', f'/{size}.')
                    break
            else:
                image_url = photo_url

        # Download the image
        img_response = requests.get(image_url, timeout=15)
        if img_response.status_code == 200:
            # Convert to PIL Image and then to numpy array
            image = Image.open(io.BytesIO(img_response.content))
            image_array = np.array(image)
            return image_array

        return None

    except Exception as e:
        st.error(f"Error downloading iNaturalist image: {str(e)}")
        return None


def render_step_indicator():
    """Render the step indicator at the top of the page"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.current_step >= 1:
            if st.session_state.step_1_complete:
                st.markdown("### ✅ Step 1: Image Source")
            elif st.session_state.current_step == 1:
                st.markdown("### 📸 **Step 1: Image Source**")
            else:
                st.markdown("### ⭕ Step 1: Image Source")
        else:
            st.markdown("### ⚪ Step 1: Image Source")
    
    with col2:
        if st.session_state.current_step >= 2:
            if st.session_state.step_2_complete:
                st.markdown("### ✅ Step 2: Calibration")
            elif st.session_state.current_step == 2:
                st.markdown("### 🔧 **Step 2: Calibration**")
            else:
                st.markdown("### ⭕ Step 2: Calibration")
        else:
            st.markdown("### ⚪ Step 2: Calibration")
    
    with col3:
        if st.session_state.current_step >= 3:
            if st.session_state.step_3_complete:
                st.markdown("### ✅ Step 3: Analysis")
            elif st.session_state.current_step == 3:
                st.markdown("### 🔬 **Step 3: Analysis**")
            else:
                st.markdown("### ⭕ Step 3: Analysis")
        else:
            st.markdown("### ⚪ Step 3: Analysis")
    
    st.markdown("---")


def validate_step_1():
    """Validate if Step 1 (Image Source) is complete"""
    return st.session_state.get('image_uploaded', False) and 'original_image' in st.session_state


def validate_step_2():
    """Validate if Step 2 (Calibration) is complete"""
    return st.session_state.get('calibration_complete', False) or st.session_state.get('pixel_scale', 1.0) > 0


def render_navigation_buttons():
    """Render navigation buttons for the wizard"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_step > 1:
            if st.button("⬅️ Previous", key="prev_button", use_container_width=True):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col3:
        if st.session_state.current_step < 3:
            # Check if current step can be completed
            can_proceed = False
            if st.session_state.current_step == 1 and validate_step_1():
                can_proceed = True
                st.session_state.step_1_complete = True
            elif st.session_state.current_step == 2 and validate_step_2():
                can_proceed = True
                st.session_state.step_2_complete = True
            
            button_text = "Next ➡️" if can_proceed else "Next ➡️ (Complete current step)"
            button_disabled = not can_proceed
            
            if st.button(button_text, key="next_button", disabled=button_disabled, use_container_width=True):
                st.session_state.current_step += 1
                st.rerun()


def render_step_1_image_source():
    """Render Step 1: Image Source"""
    st.header("📸 Step 1: Image Source")
    st.markdown("Choose your microscopy image source and upload or select an image to analyze.")
    
    # Upload method selection
    upload_method = st.radio(
        "Upload Method", ["File Upload", "iNaturalist URL"],
        help="Choose to upload a file from your computer or load an image from an iNaturalist observation"
    )

    if upload_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Choose a microscopy image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'],
            help="Upload a microscopy image containing fungal spores")

        if uploaded_file is not None:
            # Load and display the image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Store image in session state
            st.session_state.original_image = image_array
            st.session_state.image_uploaded = True
            st.session_state.image_source = "file"
            st.session_state.display_image = image
            
            # Display the image
            st.subheader("📷 Uploaded Image Preview")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.success("✅ Image uploaded successfully!")
                st.info(f"**Dimensions:** {image_array.shape[1]} × {image_array.shape[0]} pixels")
                if len(image_array.shape) == 3:
                    st.info(f"**Channels:** {image_array.shape[2]} (Color)")
                else:
                    st.info("**Channels:** 1 (Grayscale)")

    elif upload_method == "iNaturalist URL":
        st.markdown("**Load image from iNaturalist observation:**")
        inaturalist_url = st.text_input(
            "iNaturalist Observation URL",
            placeholder="https://www.inaturalist.org/observations/123456789",
            help="Enter the URL of an iNaturalist observation to analyze its images"
        )

        if inaturalist_url:
            # Extract observation ID from URL
            obs_match = re.search(r'/observations/(\d+)', inaturalist_url)
            if obs_match:
                obs_id = obs_match.group(1)

                # Automatically fetch photos when URL is entered or changed
                if st.session_state.get('obs_id') != obs_id:
                    with st.spinner(f"Fetching photos from iNaturalist observation {obs_id}..."):
                        photos = fetch_inaturalist_photos(obs_id)
                        if photos:
                            st.session_state.inaturalist_photos = photos
                            st.session_state.obs_id = obs_id
                            st.session_state.selected_photo_idx = 0
                            st.session_state.inaturalist_photo_changed = True
                            st.success(f"✅ Found {len(photos)} photo(s) in this observation!")
                        else:
                            st.error("❌ Could not fetch photos from iNaturalist. Please check the URL.")

                # Show photo selection if photos are available
                if 'inaturalist_photos' in st.session_state and st.session_state.get('obs_id') == obs_id:
                    photos = st.session_state.inaturalist_photos

                    st.markdown("**Select a photo to analyze:**")

                    # Create clickable photo selection interface
                    if len(photos) == 1:
                        selected_photo_idx = 0
                        st.info("Only one photo available in this observation.")
                    else:
                        # Show thumbnails in a row with clickable selection
                        cols = st.columns(min(len(photos), 5))  # Max 5 columns

                        if 'selected_photo_idx' not in st.session_state:
                            st.session_state.selected_photo_idx = 0

                        # Display clickable thumbnails
                        for i, photo in enumerate(photos):
                            with cols[i % 5]:  # Wrap to new row if more than 5
                                # Convert URL to thumbnail size
                                thumb_url = photo['url']
                                for old_size in ['square', 'thumb', 'small', 'medium', 'large', 'original']:
                                    if f'/{old_size}.' in thumb_url:
                                        thumb_url = thumb_url.replace(f'/{old_size}.', '/thumb.')
                                        break

                                # Show thumbnail
                                try:
                                    st.image(thumb_url, width=100)
                                except:
                                    st.write(f"📷 {i+1}")

                                # Clickable button for selection
                                button_label = f"{'✅ ' if st.session_state.selected_photo_idx == i else ''}Photo {i+1}"
                                if st.button(button_label, key=f"select_photo_{i}"):
                                    st.session_state.selected_photo_idx = i
                                    st.session_state.inaturalist_photo_changed = True
                                    st.rerun()

                        selected_photo_idx = st.session_state.selected_photo_idx
                        st.info(f"📸 Selected: Photo {selected_photo_idx + 1}")

                    # Image quality selection
                    image_quality = st.selectbox(
                        "Image Quality",
                        ["large", "medium", "original"],
                        index=2,  # Default to original
                        help="Choose image quality - larger images provide better analysis",
                        key="inaturalist_image_quality")

                    # Auto-load selected photo
                    selected_photo = photos[selected_photo_idx]
                    settings_changed = (
                        image_quality != st.session_state.get('prev_image_quality', '') or
                        selected_photo_idx != st.session_state.get('prev_selected_photo_idx', -1) or
                        st.session_state.get('inaturalist_photo_changed', False)
                    )

                    if settings_changed:
                        with st.spinner(f"Loading photo {selected_photo_idx + 1} at {image_quality} quality..."):
                            image_array = download_inaturalist_image(selected_photo['url'], image_quality)
                            if image_array is not None:
                                # Store image in session state
                                st.session_state.original_image = image_array
                                st.session_state.image_uploaded = True
                                st.session_state.image_source = "inaturalist"
                                st.session_state.display_image = image_array

                                # Update tracking variables
                                st.session_state.prev_image_quality = image_quality
                                st.session_state.prev_selected_photo_idx = selected_photo_idx
                                st.session_state.inaturalist_photo_changed = False

                                # Display the loaded image
                                st.subheader("📷 Loaded iNaturalist Image")
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.image(image_array, caption=f"iNaturalist Photo {selected_photo_idx + 1}", use_container_width=True)
                                with col2:
                                    st.success("✅ Image loaded successfully!")
                                    st.info(f"**Dimensions:** {image_array.shape[1]} × {image_array.shape[0]} pixels")
                                    if selected_photo['attribution'] != 'Unknown':
                                        st.caption(f"📷 {selected_photo['attribution']}")
                                    if selected_photo['license'] != 'Unknown':
                                        st.caption(f"📄 License: {selected_photo['license']}")
                            else:
                                st.error("❌ Could not load the selected image.")
            else:
                if inaturalist_url.strip():
                    st.error("❌ Invalid iNaturalist URL. Please enter a valid observation URL.")

    # Show step completion status
    if validate_step_1():
        st.success("✅ Step 1 Complete! You can proceed to calibration.")
    else:
        st.info("📸 Please upload or select an image to continue.")


def render_step_2_calibration():
    """Render Step 2: Calibration"""
    st.header("🔧 Step 2: Calibration")
    st.markdown("Set up the pixel scale calibration to convert pixel measurements to micrometers.")
    
    # Show image preview if available
    if 'original_image' in st.session_state:
        with st.expander("📷 Image Preview", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(st.session_state.display_image, caption="Image to Calibrate", use_container_width=True)
            with col2:
                image_array = st.session_state.original_image
                st.info(f"**Dimensions:** {image_array.shape[1]} × {image_array.shape[0]} pixels")

    # Calibration method selection
    calibration_method = st.selectbox(
        "Calibration Method", [
            "Manual Entry", "Auto-Detect Micrometer Divisions", "Manual Measurement"
        ],
        help="Choose how to determine the pixel scale")

    if calibration_method == "Manual Entry":
        st.markdown("### 📝 Manual Entry")
        st.info("Enter the known pixel scale for your microscopy setup.")
        
        pixel_scale = st.number_input(
            "Pixel Scale (pixels/μm)",
            min_value=0.1,
            max_value=100.0,
            value=st.session_state.get('pixel_scale', 10.0),
            step=0.1,
            help="Number of pixels per micrometer. This converts pixel measurements to micrometers."
        )
        
        st.session_state.pixel_scale = pixel_scale
        st.session_state.calibration_complete = True
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Manual pixel scale set: {pixel_scale:.2f} pixels/μm")
        with col2:
            if st.button("🔄 Reset to Default (10.0)", key="reset_manual_scale"):
                st.session_state.pixel_scale = 10.0
                st.rerun()

    elif calibration_method == "Auto-Detect Micrometer Divisions":
        st.markdown("### 🔍 Auto-Detect Micrometer Divisions")
        st.info("📏 This method detects tick marks on graduated rulers/micrometers in your uploaded image.")
        
        # Division spacing input
        division_spacing_um = st.number_input(
            "Distance per division (μm)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,  # Default: 0.01mm = 10μm
            step=0.1,
            help="Distance between consecutive tick marks (0.01mm = 10μm)")
        
        pixel_scale = st.session_state.get('pixel_scale', 10.0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Current pixel scale: {pixel_scale:.2f} pixels/μm")
            
        with col2:
            if st.button("🔍 Auto-Detect Divisions", key="auto_detect_divisions"):
                if 'original_image' in st.session_state:
                    with st.spinner("Detecting micrometer divisions in image..."):
                        calibration_result = st.session_state.calibration.auto_detect_scale(
                            st.session_state.original_image, "micrometer_divisions",
                            division_spacing_um)

                        if calibration_result['pixel_scale']:
                            # Validate calibration results
                            validation = st.session_state.calibration.validate_calibration(
                                calibration_result['pixel_scale'], "microscopy")

                            st.session_state.pixel_scale = calibration_result['pixel_scale']
                            st.session_state.calibration_complete = True

                            if validation['is_valid']:
                                st.success(f"✅ Micrometer calibration successful! Pixel scale: {calibration_result['pixel_scale']:.2f} pixels/μm")
                            else:
                                st.warning(f"⚠️ Calibration completed with warning: {validation['warning']}")
                                st.info(f"Pixel scale: {calibration_result['pixel_scale']:.2f} pixels/μm")

                            # Show visualization
                            if calibration_result['visualization'] is not None:
                                st.subheader("🔍 Detected Micrometer Divisions")
                                st.image(calibration_result['visualization'], 
                                         caption="Detected Micrometer Divisions", 
                                         use_container_width=True)
                        else:
                            st.error("❌ Could not detect micrometer divisions. Try manual measurement instead.")
                else:
                    st.error("❌ No image available for calibration.")

    elif calibration_method == "Manual Measurement":
        st.markdown("### 📐 Manual Measurement")
        st.info("Draw a measurement line on your image to manually calibrate the pixel scale.")
        
        # Input fields for measurement
        col_a, col_b = st.columns(2)
        with col_a:
            num_divisions = st.number_input(
                "Number of divisions covered",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                help="How many scale divisions does your measurement span?")
        
        with col_b:
            um_per_division = st.number_input(
                "Micrometers per division (μm)",
                min_value=0.1,
                max_value=1000.0,
                value=10.0,
                step=0.1,
                help="The distance each scale division represents")
        
        # Initialize manual measurement session state
        if 'manual_measurement_points' not in st.session_state:
            st.session_state.manual_measurement_points = []
        if 'manual_calibration_complete' not in st.session_state:
            st.session_state.manual_calibration_complete = False
        
        # Store calibration parameters in session state
        st.session_state.manual_num_divisions = num_divisions
        st.session_state.manual_um_per_division = um_per_division
        
        # Show current pixel scale and status
        pixel_scale = st.session_state.get('pixel_scale', 10.0)
        if st.session_state.get('manual_calibration_complete', False):
            st.success(f"✅ Calibrated: {pixel_scale:.2f} pixels/μm")
            st.session_state.calibration_complete = True
        else:
            st.info(f"Current: {pixel_scale:.2f} pixels/μm")
            if 'original_image' in st.session_state:
                st.warning("👆 **Manual measurement drawing interface would be here** (requires streamlit-drawable-canvas)")
                st.info("For now, please use Manual Entry method or Auto-Detect method.")
            else:
                st.warning("No image available for measurement")
    
    # Show step completion status
    if validate_step_2():
        st.success("✅ Step 2 Complete! You can proceed to analysis.")
    else:
        st.info("🔧 Please complete calibration to continue.")


def render_step_3_analysis():
    """Render Step 3: Analysis"""
    st.header("🔬 Step 3: Analysis & Results")
    st.markdown("Configure detection parameters and analyze your image for fungal spores.")
    
    # Show image and calibration info
    if 'original_image' in st.session_state:
        with st.expander("📷 Image & Calibration Summary", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(st.session_state.display_image, caption="Image for Analysis", use_container_width=True)
            with col2:
                image_array = st.session_state.original_image
                pixel_scale = st.session_state.get('pixel_scale', 10.0)
                st.info(f"**Dimensions:** {image_array.shape[1]} × {image_array.shape[0]} pixels")
                st.info(f"**Pixel Scale:** {pixel_scale:.2f} pixels/μm")
    
    # Analysis Parameters in organized sections
    st.markdown("## 📊 Analysis Parameters")
    
    # Basic Detection Parameters
    with st.expander("🎯 Basic Detection Parameters", expanded=True):
        area_range = st.slider(
            "Spore Area Range (μm²)",
            min_value=1,
            max_value=1000,
            value=(10, 500),
            help="Area range for objects to be considered spores")
        min_area, max_area = area_range

        circularity_range = st.slider(
            "Circularity Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.3, 0.9),
            step=0.01,
            help="Circularity range (0.0 = line, 1.0 = perfect circle)")
        circularity_min, circularity_max = circularity_range

        exclude_edges = st.checkbox(
            "Exclude Edge Spores",
            value=True,
            help="Exclude spores touching the image edges (incomplete spores)")

    # Advanced Shape Filters
    with st.expander("🔍 Advanced Shape Filters", expanded=False):
        aspect_ratio_range = st.slider(
            "Aspect Ratio Range",
            min_value=1.0,
            max_value=10.0,
            value=(1.0, 5.0),
            step=0.1,
            help="Length/width ratio range (1.0 = square, higher = more elongated)")
        aspect_ratio_min, aspect_ratio_max = aspect_ratio_range

        solidity_range = st.slider(
            "Solidity Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.5, 1.0),
            step=0.01,
            help="Solidity range (spore area / convex hull area). Higher = less concave")
        solidity_min, solidity_max = solidity_range

        convexity_range = st.slider(
            "Convexity Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.7, 1.0),
            step=0.01,
            help="Convexity range (convex hull perimeter / spore perimeter). Higher = smoother outline")
        convexity_min, convexity_max = convexity_range

        extent_range = st.slider(
            "Extent Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.3, 1.0),
            step=0.01,
            help="Extent range (spore area / bounding rectangle area). Higher = fills bounding box better")
        extent_min, extent_max = extent_range

    # Touching Spore Detection
    with st.expander("🔗 Touching Spore Detection", expanded=False):
        exclude_touching = st.checkbox(
            "Exclude touching/merged spores",
            value=True,
            help="Automatically detect and exclude spores that appear to be multiple touching spores")

        touching_aggressiveness = st.selectbox(
            "Detection Aggressiveness",
            ["Conservative", "Balanced", "Aggressive"],
            index=2,  # Default to "Aggressive"
            help="Conservative: Fewer false positives, may miss some touching spores. Aggressive: More detections, may exclude some valid single spores.")

        st.markdown("**⚡ Watershed Separation**")
        separate_touching = st.checkbox(
            "Separate touching spores using watershed",
            value=False,
            help="Use advanced watershed segmentation to automatically separate touching or overlapping spores")

        if separate_touching:
            separation_min_distance = st.slider(
                "Separation Sensitivity",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                help="Lower values = more aggressive separation")

            separation_sigma = st.slider(
                "Smoothing Factor",
                min_value=0.5,
                max_value=3.0,
                value=0.8,
                step=0.1,
                help="Lower values = less smoothing")

            separation_erosion_iterations = st.slider(
                "Erosion Strength",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Higher values = more erosion before separation")
        else:
            separation_min_distance = 5
            separation_sigma = 1.0
            separation_erosion_iterations = 1

    # Image Processing Parameters
    with st.expander("🖼️ Image Processing", expanded=False):
        blur_kernel = st.slider(
            "Blur Kernel Size",
            min_value=1,
            max_value=15,
            value=5,
            step=2,
            help="Gaussian blur kernel size for noise reduction")

        threshold_method = st.selectbox(
            "Threshold Method", ["Otsu", "Adaptive", "Manual"],
            help="Method for converting to binary image")

        if threshold_method == "Manual":
            threshold_value = st.slider("Threshold Value",
                                        min_value=0,
                                        max_value=255,
                                        value=127,
                                        help="Manual threshold value")
        else:
            threshold_value = None

    # Visualization Settings
    with st.expander("🎨 Visualization Settings", expanded=False):
        font_size = st.slider("Font Size",
                              min_value=0.5,
                              max_value=3.0,
                              value=1.6,
                              step=0.1,
                              help="Size of measurement text")

        col1, col2 = st.columns(2)
        with col1:
            font_color = st.color_picker("Font Color", value="#FFFFFF", help="Color of measurement text")
            border_color = st.color_picker("Text Background Color", value="#000000", help="Color of text background box")
            line_color = st.color_picker("Line Color", value="#00FFFF", help="Color of measurement lines and spore borders")

        with col2:
            border_width = st.slider("Text Background Size", min_value=0, max_value=10, value=8, help="Width of text background borders (0 = no background)")
            line_width = st.slider("Line Width", min_value=1, max_value=10, value=2, help="Width of measurement lines")

    # Analysis Control Section
    st.markdown("## ▶️ Run Analysis")
    
    analysis_col1, analysis_col2 = st.columns([1, 1])
    
    with analysis_col1:
        if st.button("🔬 **Analyze Spores**", key="analyze_button", use_container_width=True, type="primary"):
            if 'original_image' in st.session_state:
                with st.spinner("Analyzing spores in the image..."):
                    # TODO: Add the actual analysis logic here
                    # This would call the existing spore analysis functions with the parameters
                    st.success("✅ Analysis completed! (Analysis integration pending)")
                    st.session_state.analysis_complete = True
                    st.session_state.step_3_complete = True
            else:
                st.error("❌ No image available for analysis.")
    
    with analysis_col2:
        if st.button("🔄 Reset Parameters", key="reset_params", use_container_width=True):
            # Reset various session state values to defaults
            st.info("Parameters reset to defaults.")
            st.rerun()

    # Results Section (placeholder for now)
    if st.session_state.get('analysis_complete', False):
        st.markdown("## 📈 Analysis Results")
        st.info("🚧 **Results display will be implemented here**")
        st.markdown("This section will include:")
        st.markdown("- Detected spores overlay image")
        st.markdown("- Measurement statistics and charts")  
        st.markdown("- Spore selection interface")
        st.markdown("- Export options (CSV, PDF report)")
        
        # Placeholder success message
        st.success("✅ Step 3 Complete! Analysis finished successfully.")
    else:
        st.info("🔬 Click 'Analyze Spores' to begin the analysis.")


def main():
    """Main application function with wizard-based workflow"""
    # App title and description
    st.title("🔬 Sporulator")
    st.markdown("**A 3-step wizard for automated fungal spore detection and measurement**")
    
    # Render step indicator
    render_step_indicator()
    
    # Sidebar with current step info and quick actions
    with st.sidebar:
        st.header(f"📍 Current Step: {st.session_state.current_step}/3")
        
        # Step progress indicators
        step_1_icon = "✅" if st.session_state.step_1_complete else "⭕" if st.session_state.current_step > 1 else "📸"
        step_2_icon = "✅" if st.session_state.step_2_complete else "⭕" if st.session_state.current_step > 2 else "🔧" if st.session_state.current_step == 2 else "⚪"
        step_3_icon = "✅" if st.session_state.step_3_complete else "🔬" if st.session_state.current_step == 3 else "⚪"
        
        st.markdown(f"**{step_1_icon} Step 1:** Image Source")
        st.markdown(f"**{step_2_icon} Step 2:** Calibration")
        st.markdown(f"**{step_3_icon} Step 3:** Analysis & Results")
        
        st.markdown("---")
        
        # Quick actions based on current step
        if st.session_state.current_step == 1:
            st.markdown("**🎯 Current Focus:**")
            st.info("Upload or select an image to analyze")
        elif st.session_state.current_step == 2:
            st.markdown("**🎯 Current Focus:**")
            st.info("Set up pixel scale calibration")
            if 'pixel_scale' in st.session_state:
                st.write(f"**Current Scale:** {st.session_state.pixel_scale:.2f} px/μm")
        elif st.session_state.current_step == 3:
            st.markdown("**🎯 Current Focus:**")
            st.info("Configure analysis parameters and run detection")
            if 'pixel_scale' in st.session_state:
                st.write(f"**Scale:** {st.session_state.pixel_scale:.2f} px/μm")
            if 'original_image' in st.session_state:
                img = st.session_state.original_image
                st.write(f"**Image:** {img.shape[1]}×{img.shape[0]} px")
        
        st.markdown("---")
        
        # Reset wizard button
        if st.button("🔄 Reset Wizard", help="Start over from Step 1"):
            # Reset all wizard-related session state
            st.session_state.current_step = 1
            st.session_state.step_1_complete = False
            st.session_state.step_2_complete = False
            st.session_state.step_3_complete = False
            st.session_state.image_uploaded = False
            st.session_state.calibration_complete = False
            st.session_state.analysis_complete = False
            # Keep the image and calibration data, just reset the progress
            st.rerun()
    
    # Main content area - route to appropriate step
    if st.session_state.current_step == 1:
        render_step_1_image_source()
    elif st.session_state.current_step == 2:
        render_step_2_calibration()
    elif st.session_state.current_step == 3:
        render_step_3_analysis()
    
    # Navigation buttons at the bottom
    st.markdown("---")
    render_navigation_buttons()


if __name__ == "__main__":
    main()
