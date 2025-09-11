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
from utils import calculate_statistics, create_overlay_image, export_results

# Configure page
st.set_page_config(
    page_title="Fungal Spore Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'spore_analyzer' not in st.session_state:
    st.session_state.spore_analyzer = SporeAnalyzer()
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_spores' not in st.session_state:
    st.session_state.selected_spores = set()

def main():
    st.title("🔬 Fungal Spore Analyzer")
    st.markdown("Upload microscopy images to automatically detect and measure fungal spores")
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Analysis Settings")
        
        # Pixel scale calibration
        st.subheader("Calibration")
        pixel_scale = st.number_input(
            "Pixel Scale (pixels/μm)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            help="Number of pixels per micrometer. This converts pixel measurements to micrometers."
        )
        
        # Detection parameters
        st.subheader("Detection Parameters")
        min_area = st.slider(
            "Minimum Spore Area (μm²)",
            min_value=1,
            max_value=100,
            value=10,
            help="Minimum area for a detected object to be considered a spore"
        )
        
        max_area = st.slider(
            "Maximum Spore Area (μm²)",
            min_value=100,
            max_value=1000,
            value=500,
            help="Maximum area for a detected object to be considered a spore"
        )
        
        circularity_min = st.slider(
            "Minimum Circularity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Filter out linear objects (0.0 = line, 1.0 = perfect circle)"
        )
        
        circularity_max = st.slider(
            "Maximum Circularity",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.01,
            help="Filter out overly round objects"
        )
        
        exclude_edges = st.checkbox(
            "Exclude Edge Spores",
            value=True,
            help="Exclude spores touching the image edges (incomplete spores)"
        )
        
        # Image processing parameters
        st.subheader("Image Processing")
        blur_kernel = st.slider(
            "Blur Kernel Size",
            min_value=1,
            max_value=15,
            value=5,
            step=2,
            help="Gaussian blur kernel size for noise reduction"
        )
        
        threshold_method = st.selectbox(
            "Threshold Method",
            ["Otsu", "Adaptive", "Manual"],
            help="Method for converting to binary image"
        )
        
        if threshold_method == "Manual":
            threshold_value = st.slider(
                "Threshold Value",
                min_value=0,
                max_value=255,
                value=127,
                help="Manual threshold value"
            )
        else:
            threshold_value = None
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose a microscopy image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'],
            help="Upload a microscopy image containing fungal spores"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Store image in session state
            st.session_state.original_image = image_array
            st.session_state.image_uploaded = True
            
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
                        exclude_edges=exclude_edges,
                        blur_kernel=blur_kernel,
                        threshold_method=threshold_method,
                        threshold_value=threshold_value
                    )
                    
                    # Perform analysis
                    results = analyzer.analyze_image(image_array)
                    
                    if results is not None:
                        st.session_state.analysis_results = results
                        st.session_state.analysis_complete = True
                        st.session_state.selected_spores = set(range(len(results)))
                        st.success(f"Analysis complete! Detected {len(results)} spores.")
                        st.rerun()
                    else:
                        st.error("No spores detected. Try adjusting the parameters.")
    
    with col2:
        if st.session_state.analysis_complete:
            st.header("🎯 Detection Results")
            
            # Create overlay image
            overlay_image = create_overlay_image(
                st.session_state.original_image,
                st.session_state.analysis_results,
                st.session_state.selected_spores,
                pixel_scale
            )
            
            st.image(overlay_image, caption="Detected Spores with Measurements", use_column_width=True)
            
            # Spore selection interface
            st.subheader("✅ Spore Selection")
            st.write("Click checkboxes to include/exclude spores from analysis:")
            
            # Create a grid of checkboxes for spore selection
            results = st.session_state.analysis_results
            cols = st.columns(5)
            
            for i, spore in enumerate(results):
                col_idx = i % 5
                with cols[col_idx]:
                    is_selected = st.checkbox(
                        f"Spore {i+1}",
                        value=i in st.session_state.selected_spores,
                        key=f"spore_{i}"
                    )
                    
                    if is_selected and i not in st.session_state.selected_spores:
                        st.session_state.selected_spores.add(i)
                        st.rerun()
                    elif not is_selected and i in st.session_state.selected_spores:
                        st.session_state.selected_spores.remove(i)
                        st.rerun()
        else:
            st.info("👆 Upload an image and click 'Analyze Spores' to begin")
    
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
        
        # Display statistics in columns
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        # Histograms
        st.subheader("📊 Distribution Plots")
        
        # Create DataFrame for plotting
        df_results = pd.DataFrame([{
            'Spore_ID': i+1,
            'Length_um': spore['length_um'],
            'Width_um': spore['width_um'],
            'Area_um2': spore['area_um2'],
            'Aspect_Ratio': spore['aspect_ratio'],
            'Circularity': spore['circularity']
        } for i, spore in enumerate(selected_results)])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Length distribution
            fig_length = px.histogram(
                df_results, 
                x='Length_um',
                title='Spore Length Distribution',
                labels={'Length_um': 'Length (μm)', 'count': 'Frequency'},
                nbins=20
            )
            st.plotly_chart(fig_length, use_container_width=True)
            
            # Area distribution
            fig_area = px.histogram(
                df_results,
                x='Area_um2',
                title='Spore Area Distribution',
                labels={'Area_um2': 'Area (μm²)', 'count': 'Frequency'},
                nbins=20
            )
            st.plotly_chart(fig_area, use_container_width=True)
        
        with col2:
            # Width distribution
            fig_width = px.histogram(
                df_results,
                x='Width_um',
                title='Spore Width Distribution',
                labels={'Width_um': 'Width (μm)', 'count': 'Frequency'},
                nbins=20
            )
            st.plotly_chart(fig_width, use_container_width=True)
            
            # Aspect ratio distribution
            fig_aspect = px.histogram(
                df_results,
                x='Aspect_Ratio',
                title='Aspect Ratio Distribution',
                labels={'Aspect_Ratio': 'Length/Width Ratio', 'count': 'Frequency'},
                nbins=20
            )
            st.plotly_chart(fig_aspect, use_container_width=True)
        
        # Scatter plot: Length vs Width
        fig_scatter = px.scatter(
            df_results,
            x='Width_um',
            y='Length_um',
            title='Spore Dimensions Scatter Plot',
            labels={'Width_um': 'Width (μm)', 'Length_um': 'Length (μm)'},
            hover_data=['Spore_ID', 'Area_um2', 'Aspect_Ratio']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Data table
        st.subheader("📋 Detailed Measurements")
        st.dataframe(df_results, use_container_width=True)
        
        # Export functionality
        st.subheader("💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = export_results(df_results, 'csv')
            st.download_button(
                "📄 Download CSV",
                data=csv_data,
                file_name=f"spore_measurements_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_data = export_results(df_results, 'excel')
            st.download_button(
                "📊 Download Excel",
                data=excel_data,
                file_name=f"spore_measurements_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # Export overlay image
            if 'analysis_results' in st.session_state:
                overlay_image = create_overlay_image(
                    st.session_state.original_image,
                    st.session_state.analysis_results,
                    st.session_state.selected_spores,
                    pixel_scale
                )
                overlay_bytes = io.BytesIO()
                overlay_pil = Image.fromarray(overlay_image)
                overlay_pil.save(overlay_bytes, format='PNG')
                st.download_button(
                    "🖼️ Download Overlay Image",
                    data=overlay_bytes.getvalue(),
                    file_name=f"spore_overlay_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
