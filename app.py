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
# Canvas functionality will be implemented with Plotly
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
    st.session_state.pixel_scale = None  # Requires explicit calibration

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
    pixel_scale = st.session_state.get('pixel_scale')
    return (st.session_state.get('calibration_complete', False) and 
            pixel_scale is not None and 
            pixel_scale > 0)


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
        
        current_value = st.session_state.get('pixel_scale', 10.0)
        if current_value is None:
            current_value = 10.0
            
        pixel_scale = st.number_input(
            "Pixel Scale (pixels/μm)",
            min_value=0.1,
            max_value=100.0,
            value=current_value,
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
        st.info("🖱️ **Instructions:** Use the drawing tool below to drag a line across a known distance on your image.")
        
        if 'original_image' in st.session_state:
            # Initialize session state for manual measurement
            if 'manual_line_drawn' not in st.session_state:
                st.session_state.manual_line_drawn = False
            if 'manual_line_coords' not in st.session_state:
                st.session_state.manual_line_coords = None
            
            # Create drawable canvas for line drawing
            display_image = np.array(st.session_state.display_image.copy())
            
            # Create the canvas
            # Create interactive Plotly figure for line drawing
            import plotly.graph_objects as go
            
            # Setup figure with image background
            fig = go.Figure()
            
            # Convert PIL image to base64 for Plotly
            import io
            import base64
            img_buffer = io.BytesIO()
            st.session_state.display_image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Add background image
            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{img_base64}",
                    xref="x",
                    yref="y",
                    x=0,
                    y=0,
                    sizex=display_image.shape[1],
                    sizey=display_image.shape[0],
                    sizing="stretch",
                    opacity=1,
                    layer="below"
                )
            )
            
            # Add existing line if it exists
            if st.session_state.manual_line_coords:
                x1, y1, x2, y2 = st.session_state.manual_line_coords
                fig.add_trace(go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode='lines+markers',
                    line=dict(color='lime', width=4),
                    marker=dict(size=10, color='lime'),
                    name='Measurement Line',
                    showlegend=False
                ))
            
            # Configure layout for drawing
            fig.update_layout(
                title="🖱️ Use the drawing tools above to draw a measurement line",
                xaxis=dict(
                    range=[0, display_image.shape[1]],
                    title="X coordinate (pixels)",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.3)'
                ),
                yaxis=dict(
                    range=[display_image.shape[0], 0],  # Flip Y axis for image coordinates
                    title="Y coordinate (pixels)",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.3)',
                    scaleanchor="x",
                    scaleratio=1
                ),
                width=min(700, display_image.shape[1]),
                height=min(600, display_image.shape[0]),
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50),
                # Enable drawing mode
                dragmode="drawline",
                newshape=dict(
                    line=dict(color="lime", width=4),
                    opacity=0.8
                )
            )
            
            # Configuration for drawing tools
            config = {
                'modeBarButtonsToAdd': [
                    'drawline',
                    'eraseshape'
                ],
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                'displaylogo': False,
                'displayModeBar': True
            }
            
            # Display the interactive plot with drawing tools and capture events
            from streamlit_plotly_events import plotly_events
            
            selected_data = plotly_events(
                fig, 
                click_event=False, 
                hover_event=False, 
                select_event=False,
                relayout_event=True,  # This captures drawing events
                key="measurement_plot",
                override_height=min(600, display_image.shape[0]),
                override_width=min(700, display_image.shape[1])
            )
            
            # Process drawing events to capture line coordinates
            if selected_data and 'relayout' in selected_data and selected_data['relayout']:
                relayout_data = selected_data['relayout']
                
                # Check for new shapes (drawn lines)
                if 'shapes' in relayout_data:
                    shapes = relayout_data['shapes']
                    if shapes and len(shapes) > 0:
                        # Get the most recent shape (last drawn line)
                        latest_shape = shapes[-1]
                        
                        if latest_shape.get('type') == 'line':
                            # Extract line coordinates
                            x1 = latest_shape.get('x0', 0)
                            y1 = latest_shape.get('y0', 0)
                            x2 = latest_shape.get('x1', 0)
                            y2 = latest_shape.get('y1', 0)
                            
                            # Store coordinates in session state
                            new_coords = (int(x1), int(y1), int(x2), int(y2))
                            if st.session_state.manual_line_coords != new_coords:
                                st.session_state.manual_line_coords = new_coords
                                st.success(f"✅ **Line captured!** Coordinates: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
                                st.rerun()
            
            # Show current line status
            if st.session_state.manual_line_coords:
                x1, y1, x2, y2 = st.session_state.manual_line_coords
                pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                st.info(f"📏 **Current line:** ({x1}, {y1}) to ({x2}, {y2}) - Length: {pixel_distance:.1f} pixels")
            else:
                st.info("📌 **Draw a measurement line using the drawing tools above**")
            
            # Initialize click points if not exist
            if 'manual_click_points' not in st.session_state:
                st.session_state.manual_click_points = []
            
            # Manual coordinate input section (since Plotly click capture in Streamlit is limited)
            with st.expander("✏️ Manual Coordinate Entry", expanded=not st.session_state.manual_line_coords):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📍 Start Point:**")
                    x1_input = st.number_input("X1 (pixels)", min_value=0, max_value=display_image.shape[1], value=100, key="manual_x1_new")
                    y1_input = st.number_input("Y1 (pixels)", min_value=0, max_value=display_image.shape[0], value=100, key="manual_y1_new")
                
                with col2:
                    st.markdown("**📍 End Point:**")
                    x2_input = st.number_input("X2 (pixels)", min_value=0, max_value=display_image.shape[1], value=300, key="manual_x2_new")
                    y2_input = st.number_input("Y2 (pixels)", min_value=0, max_value=display_image.shape[0], value=200, key="manual_y2_new")
                
                col_draw, col_clear = st.columns(2)
                with col_draw:
                    if st.button("📏 Draw Line", type="primary", key="draw_line_btn"):
                        st.session_state.manual_line_coords = (int(x1_input), int(y1_input), int(x2_input), int(y2_input))
                        st.rerun()
                
                with col_clear:
                    if st.button("🗑️ Clear Line", key="clear_line_btn"):
                        st.session_state.manual_line_coords = None
                        st.rerun()
            
            # Canvas drawing instructions
            st.info("🖱️ **Instructions:** Use the line tool from the canvas toolbar above to click and drag a measurement line across a known distance on your image.")
            
            # Show current line info if available
            if st.session_state.manual_line_coords:
                x1, y1, x2, y2 = st.session_state.manual_line_coords
                pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                col_info, col_clear = st.columns([3, 1])
                with col_info:
                    st.success(f"📏 Line drawn: {pixel_distance:.1f} pixels from ({x1}, {y1}) to ({x2}, {y2})")
                with col_clear:
                    if st.button("🔄 Clear Line", key="clear_canvas_line"):
                        st.session_state.manual_line_coords = None
                        st.session_state.manual_line_drawn = False
                        st.rerun()
            
            # Alternative input method for precise coordinates
            st.markdown("---")
            st.markdown("**Alternative: Enter coordinates manually**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Start Point:**")
                x1 = st.number_input("X1", min_value=0, max_value=display_image.shape[1], value=100, key="manual_x1")
                y1 = st.number_input("Y1", min_value=0, max_value=display_image.shape[0], value=100, key="manual_y1")
            
            with col2:
                st.markdown("**End Point:**")
                x2 = st.number_input("X2", min_value=0, max_value=display_image.shape[1], value=300, key="manual_x2")
                y2 = st.number_input("Y2", min_value=0, max_value=display_image.shape[0], value=200, key="manual_y2")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📏 Set Line Coordinates", key="set_manual_line"):
                    st.session_state.manual_line_coords = (int(x1), int(y1), int(x2), int(y2))
                    st.session_state.manual_line_drawn = True
                    st.rerun()
            
            with col_btn2:
                if st.button("🔄 Clear All Lines", key="clear_manual_line"):
                    st.session_state.manual_line_coords = None
                    st.session_state.manual_line_drawn = False
                    st.rerun()
                
            
            # Measurement input section (show if line is drawn)
            if st.session_state.manual_line_coords:
                st.markdown("---")
                st.markdown("### 📏 Set Known Distance")
                
                x1, y1, x2, y2 = st.session_state.manual_line_coords
                pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                st.info(f"**Line drawn:** {pixel_distance:.1f} pixels long")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Initialize measurement values
                    if 'manual_known_distance' not in st.session_state:
                        st.session_state.manual_known_distance = 50.0
                    if 'manual_measurement_unit' not in st.session_state:
                        st.session_state.manual_measurement_unit = "micrometers"
                    
                    # Unit selection
                    unit = st.selectbox(
                        "Measurement unit:",
                        ["micrometers (μm)", "millimeters (mm)"],
                        index=0 if st.session_state.manual_measurement_unit == "micrometers" else 1,
                        key="manual_unit_select"
                    )
                    st.session_state.manual_measurement_unit = "micrometers" if "micrometers" in unit else "millimeters"
                    
                    # Distance input
                    unit_symbol = "μm" if st.session_state.manual_measurement_unit == "micrometers" else "mm"
                    max_val = 10000.0 if st.session_state.manual_measurement_unit == "micrometers" else 100.0
                    step_val = 0.1 if st.session_state.manual_measurement_unit == "micrometers" else 0.01
                    
                    known_distance = st.number_input(
                        f"Known distance ({unit_symbol}):",
                        min_value=0.001,
                        max_value=max_val,
                        value=st.session_state.manual_known_distance,
                        step=step_val,
                        help=f"The actual distance of your drawn line in {st.session_state.manual_measurement_unit}"
                    )
                    st.session_state.manual_known_distance = known_distance
                
                with col2:
                    st.markdown("**Line Info:**")
                    st.write(f"Start: ({x1}, {y1})")
                    st.write(f"End: ({x2}, {y2})")
                    st.write(f"Length: {pixel_distance:.1f}px")
                
                # Apply calibration button
                if st.button("🎯 Apply Calibration", type="primary", key="apply_manual_calibration"):
                    if pixel_distance > 0 and known_distance > 0:
                        # Convert to micrometers if needed
                        distance_um = known_distance if st.session_state.manual_measurement_unit == "micrometers" else known_distance * 1000
                        
                        # Use calibration.py manual_calibration method
                        point1 = (x1, y1)
                        point2 = (x2, y2)
                        calibration_result = st.session_state.calibration.manual_calibration(
                            st.session_state.original_image, point1, point2, distance_um
                        )
                        
                        if calibration_result:
                            st.session_state.pixel_scale = calibration_result['pixel_scale']
                            st.session_state.calibration_complete = True
                            st.session_state.manual_calibration_complete = True
                            st.session_state.manual_calibration_visualization = calibration_result['visualization']
                            
                            unit_display = f"{known_distance} {unit_symbol}"
                            st.success(f"✅ Manual calibration successful!")
                            st.success(f"📏 {unit_display} = {pixel_distance:.1f} pixels")
                            st.success(f"🔬 Pixel scale: {calibration_result['pixel_scale']:.2f} μm/pixel")
                            st.rerun()
                        else:
                            st.error("❌ Calibration failed. Please try again.")
                    else:
                        st.error("❌ Invalid measurements. Please ensure both distance values are positive.")
                        
            # Display calibration results if available
            if st.session_state.get('manual_calibration_complete', False) and 'manual_calibration_visualization' in st.session_state:
                st.markdown("---")
                st.markdown("### ✅ Calibration Results")
                st.image(st.session_state.manual_calibration_visualization, 
                        caption=f"Manual measurement calibration - Scale: {st.session_state.pixel_scale:.2f} μm/pixel", 
                        use_container_width=True)
        else:
            st.warning("⚠️ No image available for measurement. Please complete Step 1 first.")
    
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
        # Display Options
        col1, col2 = st.columns(2)
        with col1:
            show_labels = st.checkbox("Show Spore Labels", value=True, help="Show spore ID numbers")
            show_measurements = st.checkbox("Show Measurements", value=True, help="Show length/width measurements")
        with col2:
            overlay_color_map = st.selectbox("Color Scheme", ["rainbow", "viridis", "plasma", "cool", "hot"], 
                                           index=0, help="Color scheme for spore visualization")
            background_alpha = st.slider("Background Transparency", min_value=0.0, max_value=1.0, 
                                       value=0.7, step=0.1, help="Text background transparency")
        
        # Font Settings
        label_fontsize = st.slider("Label Font Size", min_value=0.5, max_value=3.0, value=1.6, step=0.1,
                                 help="Size of spore ID numbers")
        measurement_fontsize = st.slider("Measurement Font Size", min_value=0.5, max_value=3.0, value=1.2, step=0.1,
                                        help="Size of measurement text")

        # Colors and Style
        col3, col4 = st.columns(2)
        with col3:
            font_color = st.color_picker("Font Color", value="#FFFFFF", help="Color of measurement text")
            border_color = st.color_picker("Text Background Color", value="#000000", help="Color of text background box")
            line_color = st.color_picker("Line Color", value="#00FFFF", help="Color of measurement lines and spore borders")

        with col4:
            border_width = st.slider("Text Background Size", min_value=0, max_value=10, value=8, help="Width of text background borders (0 = no background)")
            line_width = st.slider("Line Width", min_value=1, max_value=10, value=2, help="Width of measurement lines")

    # Analysis Control Section
    st.markdown("## ▶️ Run Analysis")
    
    analysis_col1, analysis_col2 = st.columns([1, 1])
    
    with analysis_col1:
        if st.button("🔬 **Analyze Spores**", key="analyze_button", use_container_width=True, type="primary"):
            if 'original_image' in st.session_state:
                with st.spinner("Analyzing spores in the image..."):
                    try:
                        # Get analysis parameters from UI
                        analyzer = st.session_state.spore_analyzer
                        pixel_scale = st.session_state.get('pixel_scale', 10.0)
                        
                        # Set parameters on analyzer
                        analyzer.set_parameters(
                            pixel_scale=pixel_scale,
                            min_area=min_area,
                            max_area=max_area,
                            circularity_range=(circularity_min, circularity_max),
                            aspect_ratio_range=(aspect_ratio_min, aspect_ratio_max),
                            solidity_range=(solidity_min, solidity_max),
                            convexity_range=(convexity_min, convexity_max),
                            extent_range=(extent_min, extent_max),
                            exclude_edges=exclude_edges,
                            blur_kernel=blur_kernel,
                            threshold_method=threshold_method,
                            threshold_value=threshold_value if threshold_method == "Manual" else None,
                            exclude_touching=exclude_touching,
                            touching_aggressiveness=touching_aggressiveness,
                            separate_touching=separate_touching,
                            separation_min_distance=separation_min_distance,
                            separation_sigma=separation_sigma,
                            separation_erosion_iterations=separation_erosion_iterations
                        )
                        
                        # Run analysis
                        results = analyzer.analyze_image(st.session_state.original_image)
                        
                        if results and len(results) > 0:
                            # Store results in session state
                            st.session_state.analysis_results = results
                            st.session_state.analysis_complete = True
                            st.session_state.step_3_complete = True
                            
                            # Convert hex colors to BGR tuples (OpenCV format)
                            def hex_to_bgr(hex_color):
                                hex_color = hex_color.lstrip('#')
                                # Convert to RGB first, then reverse to BGR for OpenCV
                                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                return (rgb[2], rgb[1], rgb[0])  # BGR format
                            
                            # Create overlay image for visualization  
                            vis_settings = {
                                'color_map': overlay_color_map,
                                'show_labels': show_labels,
                                'show_measurements': show_measurements,
                                'font_size': label_fontsize,  # Use label_fontsize as the main font_size
                                'measurement_fontsize': measurement_fontsize,
                                'background_alpha': background_alpha,
                                'border_width': border_width,
                                'line_width': line_width,
                                'font_color': hex_to_bgr(font_color),
                                'border_color': hex_to_bgr(border_color),
                                'line_color': hex_to_bgr(line_color)
                            }
                            # Initialize selected spores (all spores by default)
                            st.session_state.selected_spores = set(range(len(results)))
                            
                            overlay_image = create_overlay_image(
                                st.session_state.original_image,
                                results,
                                list(st.session_state.selected_spores),  # Use selected spores
                                pixel_scale,
                                vis_settings,
                                True  # include_stats
                            )
                            st.session_state.overlay_image = overlay_image
                            
                            st.success(f"✅ Analysis completed! Found {len(results)} spores.")
                            st.rerun()
                        else:
                            st.warning("⚠️ No spores detected with current parameters. Try adjusting the detection settings.")
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
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
        
        # Get results from session state
        results = st.session_state.get('analysis_results', [])
        overlay_image = st.session_state.get('overlay_image', None)
        
        if results and len(results) > 0:
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Spores", len(results))
            with col2:
                avg_length = np.mean([spore['length_um'] for spore in results])
                st.metric("Avg Length", f"{avg_length:.1f} μm")
            with col3:
                avg_width = np.mean([spore['width_um'] for spore in results])
                st.metric("Avg Width", f"{avg_width:.1f} μm")
            with col4:
                avg_area = np.mean([spore['area_um2'] for spore in results])
                st.metric("Avg Area", f"{avg_area:.1f} μm²")
            
            # Display overlay image
            st.markdown("### 🔬 Detected Spores Visualization")
            if overlay_image is not None:
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_count = len(st.session_state.get('selected_spores', []))
                    st.image(overlay_image, caption=f"Showing {selected_count} of {len(results)} detected spores", 
                            use_container_width=True)
                with col2:
                    st.markdown("**Legend:**")
                    st.markdown("*Select spores to display:*")
                    
                    # Initialize selected spores if not exists
                    if 'selected_spores' not in st.session_state:
                        st.session_state.selected_spores = set(range(len(results)))
                    
                    # Spore selection checkboxes
                    for i in range(len(results)):
                        spore = results[i]
                        is_selected = i in st.session_state.selected_spores
                        checkbox_label = f"Spore {i+1} ({spore['length_um']:.1f}×{spore['width_um']:.1f} μm)"
                        
                        if st.checkbox(checkbox_label, value=is_selected, key=f"spore_checkbox_{i}"):
                            st.session_state.selected_spores.add(i)
                        else:
                            st.session_state.selected_spores.discard(i)
                    
                    st.markdown("---")
                    st.markdown("**Controls:**")
                    
                    # Select/Deselect all buttons
                    col_all, col_none = st.columns(2)
                    with col_all:
                        if st.button("Select All", key="select_all_spores", use_container_width=True):
                            st.session_state.selected_spores = set(range(len(results)))
                            st.rerun()
                    with col_none:
                        if st.button("None", key="deselect_all_spores", use_container_width=True):
                            st.session_state.selected_spores = set()
                            st.rerun()
                    
                    if st.button("🔄 Regenerate Overlay", key="regenerate_overlay", use_container_width=True):
                        # Regenerate overlay with current settings
                        # Convert hex colors to BGR tuples (OpenCV format)
                        def hex_to_bgr(hex_color):
                            hex_color = hex_color.lstrip('#')
                            # Convert to RGB first, then reverse to BGR for OpenCV
                            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            return (rgb[2], rgb[1], rgb[0])  # BGR format
                        
                        vis_settings = {
                            'color_map': overlay_color_map,
                            'show_labels': show_labels,
                            'show_measurements': show_measurements,
                            'font_size': label_fontsize,  # Use label_fontsize as the main font_size
                            'measurement_fontsize': measurement_fontsize,
                            'background_alpha': background_alpha,
                            'border_width': border_width,
                            'line_width': line_width,
                            'font_color': hex_to_bgr(font_color),
                            'border_color': hex_to_bgr(border_color),
                            'line_color': hex_to_bgr(line_color)
                        }
                        new_overlay = create_overlay_image(
                            st.session_state.original_image,
                            results,
                            list(st.session_state.selected_spores),  # Use selected spores
                            st.session_state.get('pixel_scale', 10.0),
                            vis_settings,
                            True  # include_stats
                        )
                        st.session_state.overlay_image = new_overlay
                        st.rerun()
            
            # Results table and statistics
            st.markdown("### 📊 Measurement Data & Statistics")
            
            # Create dataframe for display
            df_data = []
            for i, spore in enumerate(results):
                df_data.append({
                    'ID': i + 1,
                    'Length (μm)': round(spore['length_um'], 2),
                    'Width (μm)': round(spore['width_um'], 2),
                    'Area (μm²)': round(spore['area_um2'], 2),
                    'Aspect Ratio': round(spore['aspect_ratio'], 2),
                    'Circularity': round(spore['circularity'], 3),
                    'Solidity': round(spore['solidity'], 3),
                    'Convexity': round(spore['convexity'], 3),
                    'Extent': round(spore['extent'], 3)
                })
            
            df = pd.DataFrame(df_data)
            
            # Display table with selection capability
            st.markdown("**Individual Spore Measurements:**")
            st.dataframe(df, use_container_width=True, height=300)
            
            # Calculate and display statistics
            stats = calculate_statistics(results)
            if stats:
                st.markdown("**Summary Statistics:**")
                
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.markdown("**Length (μm):**")
                    st.markdown(f"- Mean: {stats['length_mean']:.2f} ± {stats['length_std']:.2f}")
                    st.markdown(f"- Range: {stats['length_min']:.2f} - {stats['length_max']:.2f}")
                    st.markdown(f"- Median: {stats['length_median']:.2f}")
                    
                    st.markdown("**Area (μm²):**")
                    st.markdown(f"- Mean: {stats['area_mean']:.2f} ± {stats['area_std']:.2f}")
                    st.markdown(f"- Range: {stats['area_min']:.2f} - {stats['area_max']:.2f}")
                    st.markdown(f"- Median: {stats['area_median']:.2f}")
                
                with stats_col2:
                    st.markdown("**Width (μm):**")
                    st.markdown(f"- Mean: {stats['width_mean']:.2f} ± {stats['width_std']:.2f}")
                    st.markdown(f"- Range: {stats['width_min']:.2f} - {stats['width_max']:.2f}")
                    st.markdown(f"- Median: {stats['width_median']:.2f}")
                    
                    st.markdown("**Aspect Ratio:**")
                    st.markdown(f"- Mean: {stats['aspect_ratio_mean']:.2f} ± {stats['aspect_ratio_std']:.2f}")
                    st.markdown(f"- Range: {stats['aspect_ratio_min']:.2f} - {stats['aspect_ratio_max']:.2f}")
            
            # Generate mycological summary
            st.markdown("### 📋 Mycological Summary")
            with st.expander("📝 Standard Format Summary", expanded=False):
                mycological_summary = generate_mycological_summary(results)
                st.markdown(mycological_summary)
                
                # Copy to clipboard button
                if st.button("📋 Copy Summary to Clipboard", key="copy_summary"):
                    st.code(mycological_summary, language=None)
                    st.success("Summary displayed above - you can copy it manually.")
            
            # Export functionality
            st.markdown("### 📥 Export Results")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📊 Export CSV", key="export_csv", use_container_width=True):
                    try:
                        # Convert results to dataframe for export
                        df_export = pd.DataFrame([{
                            'Length_um': spore['length_um'],
                            'Width_um': spore['width_um'], 
                            'Area_um2': spore['area_um2'],
                            'Aspect_Ratio': spore['aspect_ratio'],
                            'Circularity': spore['circularity'],
                            'Solidity': spore['solidity'],
                            'Convexity': spore['convexity'],
                            'Extent': spore['extent']
                        } for spore in results])
                        csv_data = export_results(df_export, format_type='csv')
                        if csv_data:
                            st.download_button(
                                label="💾 Download CSV File",
                                data=csv_data,
                                file_name=f"spore_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to generate CSV export")
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
            
            with export_col2:
                if st.button("📈 Export Excel", key="export_excel", use_container_width=True):
                    try:
                        # Convert results to dataframe for export
                        df_export = pd.DataFrame([{
                            'Length_um': spore['length_um'],
                            'Width_um': spore['width_um'], 
                            'Area_um2': spore['area_um2'],
                            'Aspect_Ratio': spore['aspect_ratio'],
                            'Circularity': spore['circularity'],
                            'Solidity': spore['solidity'],
                            'Convexity': spore['convexity'],
                            'Extent': spore['extent']
                        } for spore in results])
                        excel_data = export_results(df_export, format_type='excel')
                        if excel_data:
                            st.download_button(
                                label="💾 Download Excel File",
                                data=excel_data,
                                file_name=f"spore_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to generate Excel export")
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
            
            with export_col3:
                if st.button("🖼️ Save Overlay", key="export_overlay", use_container_width=True):
                    if overlay_image is not None:
                        try:
                            # Convert numpy array to PIL Image and then to bytes
                            pil_image = Image.fromarray(overlay_image)
                            img_buffer = io.BytesIO()
                            pil_image.save(img_buffer, format='PNG')
                            
                            st.download_button(
                                label="💾 Download Overlay Image",
                                data=img_buffer.getvalue(),
                                file_name=f"spore_overlay_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Failed to save overlay image: {str(e)}")
                    else:
                        st.error("No overlay image available")
            
            # Mark step 3 as complete
            st.session_state.step_3_complete = True
            st.success("✅ Step 3 Complete! Analysis finished successfully.")
            
        else:
            st.warning("⚠️ No analysis results available. Please run the analysis first.")
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
            if 'pixel_scale' in st.session_state and st.session_state.pixel_scale is not None:
                st.write(f"**Current Scale:** {st.session_state.pixel_scale:.2f} px/μm")
            else:
                st.write("**Current Scale:** Not calibrated yet")
        elif st.session_state.current_step == 3:
            st.markdown("**🎯 Current Focus:**")
            st.info("Configure analysis parameters and run detection")
            if 'pixel_scale' in st.session_state and st.session_state.pixel_scale is not None:
                st.write(f"**Scale:** {st.session_state.pixel_scale:.2f} px/μm")
            else:
                st.write("**Scale:** Not calibrated")
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
