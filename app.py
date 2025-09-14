import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
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


def main():
    st.title("🔬 Sporulator")
    st.markdown(
        "Upload microscopy images to automatically detect and measure fungal spores"
    )

    # Sidebar controls
    with st.sidebar:
        st.header("📊 Analysis Settings")

        # Pixel scale calibration
        st.subheader("Calibration")

        # Calibration method selection
        calibration_method = st.selectbox(
            "Calibration Method", [
                "Manual Entry", "Auto-Detect Scale Bar",
                "Auto-Detect Circular Object", "Upload Calibration Image"
            ],
            help="Choose how to determine the pixel scale")

        if calibration_method == "Manual Entry":
            pixel_scale = st.number_input(
                "Pixel Scale (pixels/μm)",
                min_value=0.1,
                max_value=100.0,
                value=10.0,
                step=0.1,
                help=
                "Number of pixels per micrometer. This converts pixel measurements to micrometers."
            )

        elif calibration_method == "Upload Calibration Image":
            st.markdown(
                "**Upload a calibration image with known measurements:**")
            calibration_file = st.file_uploader(
                "Choose calibration image",
                type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
                key="calibration_upload",
                help=
                "Upload an image containing a scale bar or known reference object"
            )

            if calibration_file is not None:
                # Load calibration image
                cal_image = Image.open(calibration_file)
                cal_array = np.array(cal_image)

                # Show calibration image
                st.image(cal_image, caption="Calibration Image", width=300)

                # Reference type selection
                ref_type = st.selectbox(
                    "Reference Type", ["scale_bar", "circular_object"],
                    format_func=lambda x: "Scale Bar"
                    if x == "scale_bar" else "Circular Object")

                # Known measurement input
                known_length = st.number_input(
                    f"Known {'Length' if ref_type == 'scale_bar' else 'Diameter'} (μm)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=10.0,
                    step=0.1)

                if st.button("🔍 Auto-Detect Calibration",
                             key="auto_calibrate"):
                    with st.spinner("Detecting calibration reference..."):
                        calibration_result = st.session_state.calibration.auto_detect_scale(
                            cal_array, ref_type, known_length)

                        if calibration_result['pixel_scale']:
                            # Validate calibration results
                            validation = st.session_state.calibration.validate_calibration(
                                calibration_result['pixel_scale'],
                                "microscopy")

                            st.session_state.pixel_scale = calibration_result[
                                'pixel_scale']
                            st.session_state.calibration_complete = True

                            # Show results
                            if validation['is_valid']:
                                st.success(f"✅ Calibration successful!")
                            else:
                                st.warning(
                                    f"⚠️ Calibration completed with warning: {validation['warning']}"
                                )

                            st.metric(
                                "Detected Pixel Scale",
                                f"{calibration_result['pixel_scale']:.2f} pixels/μm"
                            )
                            st.metric(
                                "Detection Confidence",
                                f"{calibration_result['confidence']*100:.0f}%")
                            st.caption(
                                f"Expected range: {validation['expected_range'][0]}-{validation['expected_range'][1]} pixels/μm"
                            )

                            # Show visualization
                            if calibration_result['visualization'] is not None:
                                st.image(
                                    calibration_result['visualization'],
                                    caption="Detected Calibration Objects",
                                    width=300)
                        else:
                            st.error(
                                "❌ Could not detect calibration reference. Try manual entry or different image."
                            )

            # Use detected or default pixel scale
            pixel_scale = st.session_state.get('pixel_scale', 10.0)
            st.info(f"Current pixel scale: {pixel_scale:.2f} pixels/μm")

        else:
            # For auto-detect methods, show current pixel scale
            pixel_scale = st.session_state.get('pixel_scale', 10.0)
            st.info(f"Auto-detection will be performed on uploaded images")
            st.info(f"Current pixel scale: {pixel_scale:.2f} pixels/μm")

        # Detection parameters
        st.subheader("Detection Parameters")
        min_area = st.slider(
            "Minimum Spore Area (μm²)",
            min_value=1,
            max_value=100,
            value=10,
            help="Minimum area for a detected object to be considered a spore")

        max_area = st.slider(
            "Maximum Spore Area (μm²)",
            min_value=100,
            max_value=1000,
            value=500,
            help="Maximum area for a detected object to be considered a spore")

        circularity_min = st.slider(
            "Minimum Circularity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Filter out linear objects (0.0 = line, 1.0 = perfect circle)"
        )

        circularity_max = st.slider("Maximum Circularity",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.9,
                                    step=0.01,
                                    help="Filter out overly round objects")

        exclude_edges = st.checkbox(
            "Exclude Edge Spores",
            value=True,
            help="Exclude spores touching the image edges (incomplete spores)")

        # Enhanced shape filters - collapsible and collapsed by default
        with st.expander("Advanced Shape Filters", expanded=False):
            aspect_ratio_min = st.slider(
                "Minimum Aspect Ratio",
                min_value=1.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help=
                "Minimum length/width ratio (1.0 = square, higher = more elongated)"
            )

            aspect_ratio_max = st.slider("Maximum Aspect Ratio",
                                         min_value=1.0,
                                         max_value=10.0,
                                         value=5.0,
                                         step=0.1,
                                         help="Maximum length/width ratio")

            solidity_min = st.slider(
                "Minimum Solidity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help=
                "Minimum solidity (spore area / convex hull area). Higher = less concave"
            )

            solidity_max = st.slider("Maximum Solidity",
                                     min_value=0.0,
                                     max_value=1.0,
                                     value=1.0,
                                     step=0.01,
                                     help="Maximum solidity")

            convexity_min = st.slider(
                "Minimum Convexity",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.01,
                help=
                "Minimum convexity (convex hull perimeter / spore perimeter). Higher = smoother outline"
            )

            convexity_max = st.slider("Maximum Convexity",
                                      min_value=0.0,
                                      max_value=1.0,
                                      value=1.0,
                                      step=0.01,
                                      help="Maximum convexity")

            extent_min = st.slider(
                "Minimum Extent",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.01,
                help=
                "Minimum extent (spore area / bounding rectangle area). Higher = fills bounding box better"
            )

            extent_max = st.slider("Maximum Extent",
                                   min_value=0.0,
                                   max_value=1.0,
                                   value=1.0,
                                   step=0.01,
                                   help="Maximum extent")

        # Image processing parameters
        st.subheader("Image Processing")
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

        # Visualization settings
        st.subheader("Visualization Settings")

        font_size = st.slider("Font Size",
                              min_value=0.5,
                              max_value=3.0,
                              value=1.6,
                              step=0.1,
                              help="Size of measurement text")

        col1, col2 = st.columns(2)
        with col1:
            font_color = st.color_picker("Font Color",
                                         value="#FFFFFF",
                                         help="Color of measurement text")

            border_color = st.color_picker("Text Background Color",
                                           value="#000000",
                                           help="Color of text background box")

            line_color = st.color_picker(
                "Line Color",
                value="#00FFFF",
                help="Color of measurement lines and spore borders")

        with col2:
            border_width = st.slider(
                "Text Background Size",
                min_value=0,
                max_value=10,
                value=8,
                help="Width of text background borders (0 = no background)")

            line_width = st.slider("Line Width",
                                   min_value=1,
                                   max_value=10,
                                   value=2,
                                   help="Width of measurement lines")

    # Image Upload Section (consolidated)
    st.header("📁 Image Upload")
    uploaded_file = st.file_uploader(
        "Choose a microscopy image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'],
        help="Upload a microscopy image containing fungal spores")

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        # Store image in session state
        st.session_state.original_image = image_array
        st.session_state.image_uploaded = True

        # Main content area - two columns for image display
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Original Image", width='stretch')
        # Auto-calibration and analysis controls
        if calibration_method in [
                "Auto-Detect Scale Bar", "Auto-Detect Circular Object"
        ]:
            st.subheader("🎯 Auto-Calibration")

            ref_type = "scale_bar" if calibration_method == "Auto-Detect Scale Bar" else "circular_object"

            col_a, col_b = st.columns(2)
            with col_a:
                known_ref_length = st.number_input(
                    f"Known {'Scale Bar Length' if ref_type == 'scale_bar' else 'Object Diameter'} (μm)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=10.0,
                    step=0.1,
                    key="main_calibration_length")

            with col_b:
                if st.button("🔍 Auto-Calibrate from Image",
                             key="main_auto_calibrate"):
                    with st.spinner(
                            "Detecting calibration reference in image..."):
                        calibration_result = st.session_state.calibration.auto_detect_scale(
                            image_array, ref_type, known_ref_length)

                        if calibration_result['pixel_scale']:
                            # Validate calibration results
                            validation = st.session_state.calibration.validate_calibration(
                                calibration_result['pixel_scale'],
                                "microscopy")

                            st.session_state.pixel_scale = calibration_result[
                                'pixel_scale']
                            pixel_scale = calibration_result['pixel_scale']
                            st.session_state.calibration_complete = True

                            if validation['is_valid']:
                                st.success(
                                    f"✅ Auto-calibration successful! Pixel scale: {pixel_scale:.2f} pixels/μm"
                                )
                            else:
                                st.warning(
                                    f"⚠️ Auto-calibration completed with warning: {validation['warning']}"
                                )
                                st.info(
                                    f"Pixel scale: {pixel_scale:.2f} pixels/μm (expected: {validation['expected_range'][0]}-{validation['expected_range'][1]})"
                                )

                            # Show visualization
                            if calibration_result['visualization'] is not None:
                                st.image(
                                    calibration_result['visualization'],
                                    caption="Auto-Detected Calibration Objects",
                                    width='stretch')
                        else:
                            st.error(
                                "❌ Could not auto-detect calibration. Using manual pixel scale."
                            )

        # Analysis button
        if st.button("🔍 Analyze Spores", type="primary"):
            with st.spinner("Analyzing spores..."):
                # Configure analyzer
                analyzer = st.session_state.spore_analyzer
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
                    threshold_value=threshold_value)

                # Perform analysis
                results = analyzer.analyze_image(image_array)

                if results is not None:
                    st.session_state.analysis_results = results
                    st.session_state.analysis_complete = True
                    st.session_state.selected_spores = set(range(len(results)))
                    st.success(
                        f"Analysis complete! Detected {len(results)} spores.")
                    st.rerun()
                else:
                    st.error(
                        "No spores detected. Try adjusting the parameters.")

        with col2:
            if st.session_state.analysis_complete:
                st.subheader("Detection Results")

                # Convert color picker hex values to BGR tuples for OpenCV
                def hex_to_bgr(hex_color):
                    hex_color = hex_color.lstrip('#')
                    # Extract RGB values and convert to BGR order
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16) 
                    b = int(hex_color[4:6], 16)
                    return (b, g, r)  # BGR order for OpenCV

                visualization_settings = {
                    'font_size': font_size,
                    'font_color': hex_to_bgr(font_color),
                    'border_color': hex_to_bgr(border_color),
                    'border_width': border_width,
                    'line_color': hex_to_bgr(line_color),
                    'line_width': line_width
                }

                # Create overlay image
                overlay_image = create_overlay_image(
                    st.session_state.original_image,
                    st.session_state.analysis_results,
                    st.session_state.selected_spores, pixel_scale,
                    visualization_settings)

                st.image(overlay_image,
                         caption="Detected Spores with Measurements",
                         width='stretch')
            else:
                st.subheader("Analysis Results")
                st.write("Processed image will appear here after analysis")
    else:
        st.info("👆 Upload an image and click 'Analyze Spores' to begin")

    if st.session_state.analysis_complete and st.session_state.selected_spores:
        # Spore selection interface
        st.subheader("✅ Spore Selection")
        st.write("Click checkboxes to include/exclude spores from analysis:")

        # Create a grid of checkboxes for spore selection
        results = st.session_state.analysis_results
        cols = st.columns(5)

        for i, spore in enumerate(results):
            col_idx = i % 5
            with cols[col_idx]:
                is_selected = st.checkbox(f"Spore {i+1}",
                                          value=i
                                          in st.session_state.selected_spores,
                                          key=f"spore_{i}")

                if is_selected and i not in st.session_state.selected_spores:
                    st.session_state.selected_spores.add(i)
                    st.rerun()
                elif not is_selected and i in st.session_state.selected_spores:
                    st.session_state.selected_spores.remove(i)
                    st.rerun()

    # Statistics and results section
    if st.session_state.analysis_complete and st.session_state.selected_spores:
        st.header("📈 Statistical Analysis")

        # Filter selected spores
        selected_results = [
            st.session_state.analysis_results[i]
            for i in st.session_state.selected_spores
        ]

        # Calculate statistics
        stats = calculate_statistics(selected_results)

        # Summary section
        st.subheader("📄 Summary")
        mycological_summary = generate_mycological_summary(selected_results)
        st.markdown(mycological_summary)

        # Create DataFrame for plotting
        df_results = pd.DataFrame([{
            'Spore_ID': i + 1,
            'Length_um': spore['length_um'],
            'Width_um': spore['width_um'],
            'Area_um2': spore['area_um2'],
            'Aspect_Ratio': spore['aspect_ratio'],
            'Circularity': spore['circularity'],
            'Solidity': spore['solidity'],
            'Convexity': spore['convexity'],
            'Extent': spore['extent']
        } for i, spore in enumerate(selected_results)])

        # Data table
        st.subheader("📋 Detailed Measurements")
        st.dataframe(df_results, width='stretch')

        # Display statistics in columns
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Spores", len(selected_results))
            st.metric("Mean Length (μm)", f"{stats['length_mean']:.2f}")
            st.metric("Mean Width (μm)", f"{stats['width_mean']:.2f}")

        with col2:
            st.metric("Std Dev Length", f"{stats['length_std']:.2f}")
            st.metric("Std Dev Width", f"{stats['width_std']:.2f}")
            st.metric("Mean Aspect Ratio", f"{stats['aspect_ratio_mean']:.2f}")

        with col3:
            st.metric("Min Length (μm)", f"{stats['length_min']:.2f}")
            st.metric("Max Length (μm)", f"{stats['length_max']:.2f}")
            st.metric("Mean Area (μm²)", f"{stats['area_mean']:.2f}")

        with col4:
            st.metric("Min Width (μm)", f"{stats['width_min']:.2f}")
            st.metric("Max Width (μm)", f"{stats['width_max']:.2f}")
            st.metric("Mean Circularity", f"{stats['circularity_mean']:.3f}")

        with col5:
            st.metric("Mean Solidity", f"{stats['solidity_mean']:.3f}")
            st.metric("Mean Convexity", f"{stats['convexity_mean']:.3f}")
            st.metric("Mean Extent", f"{stats['extent_mean']:.3f}")

        # Export functionality
        st.subheader("💾 Export Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = export_results(df_results, 'csv')
            st.download_button(
                "📄 Download CSV",
                data=csv_data,
                file_name=
                f"spore_measurements_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv")

        with col2:
            excel_data = export_results(df_results, 'excel')
            st.download_button(
                "📊 Download Excel",
                data=excel_data,
                file_name=
                f"spore_measurements_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime=
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col3:
            # Export overlay image
            if 'analysis_results' in st.session_state:
                overlay_image = create_overlay_image(
                    st.session_state.original_image,
                    st.session_state.analysis_results,
                    st.session_state.selected_spores, pixel_scale)
                overlay_bytes = io.BytesIO()
                overlay_pil = Image.fromarray(overlay_image)
                overlay_pil.save(overlay_bytes, format='PNG')
                st.download_button(
                    "🖼️ Download Overlay Image",
                    data=overlay_bytes.getvalue(),
                    file_name=
                    f"spore_overlay_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png")

        # Histograms
        st.subheader("📊 Distribution Plots")

        col1, col2 = st.columns(2)

        with col1:
            # Length distribution
            fig_length = px.histogram(df_results,
                                      x='Length_um',
                                      title='Spore Length Distribution',
                                      labels={
                                          'Length_um': 'Length (μm)',
                                          'count': 'Frequency'
                                      },
                                      nbins=20)
            st.plotly_chart(fig_length, width='stretch')

            # Area distribution
            fig_area = px.histogram(df_results,
                                    x='Area_um2',
                                    title='Spore Area Distribution',
                                    labels={
                                        'Area_um2': 'Area (μm²)',
                                        'count': 'Frequency'
                                    },
                                    nbins=20)
            st.plotly_chart(fig_area, width='stretch')

        with col2:
            # Width distribution
            fig_width = px.histogram(df_results,
                                     x='Width_um',
                                     title='Spore Width Distribution',
                                     labels={
                                         'Width_um': 'Width (μm)',
                                         'count': 'Frequency'
                                     },
                                     nbins=20)
            st.plotly_chart(fig_width, width='stretch')

            # Aspect ratio distribution
            fig_aspect = px.histogram(df_results,
                                      x='Aspect_Ratio',
                                      title='Aspect Ratio Distribution',
                                      labels={
                                          'Aspect_Ratio': 'Length/Width Ratio',
                                          'count': 'Frequency'
                                      },
                                      nbins=20)
            st.plotly_chart(fig_aspect, width='stretch')

        # Shape metrics distributions
        st.subheader("🔸 Enhanced Shape Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Solidity distribution
            fig_solidity = px.histogram(df_results,
                                        x='Solidity',
                                        title='Solidity Distribution',
                                        labels={
                                            'Solidity':
                                            'Solidity (Area/Convex Hull Area)',
                                            'count': 'Frequency'
                                        },
                                        nbins=20)
            st.plotly_chart(fig_solidity, width='stretch')

        with col2:
            # Convexity distribution
            fig_convexity = px.histogram(
                df_results,
                x='Convexity',
                title='Convexity Distribution',
                labels={
                    'Convexity': 'Convexity (Hull Perimeter/Perimeter)',
                    'count': 'Frequency'
                },
                nbins=20)
            st.plotly_chart(fig_convexity, width='stretch')

        with col3:
            # Extent distribution
            fig_extent = px.histogram(df_results,
                                      x='Extent',
                                      title='Extent Distribution',
                                      labels={
                                          'Extent':
                                          'Extent (Area/Bounding Rect Area)',
                                          'count': 'Frequency'
                                      },
                                      nbins=20)
            st.plotly_chart(fig_extent, width='stretch')

        # Scatter plot: Length vs Width
        fig_scatter = px.scatter(df_results,
                                 x='Width_um',
                                 y='Length_um',
                                 title='Spore Dimensions Scatter Plot',
                                 labels={
                                     'Width_um': 'Width (μm)',
                                     'Length_um': 'Length (μm)'
                                 },
                                 hover_data=[
                                     'Spore_ID', 'Area_um2', 'Aspect_Ratio',
                                     'Solidity', 'Convexity', 'Extent'
                                 ])
        st.plotly_chart(fig_scatter, width='stretch')


if __name__ == "__main__":
    main()
