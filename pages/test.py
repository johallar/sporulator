import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from scipy import ndimage
from skimage.segmentation import watershed

st.set_page_config(page_title="Image Processing Test", page_icon="🔬", layout="wide")

st.title("🔬 Image Processing Pipeline Test")
st.markdown("Upload an image and see all the intermediate processing steps")

uploaded_file = st.file_uploader("Upload a microscopy image", type=['png', 'jpg', 'jpeg', 'tiff'])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    
    st.markdown("---")
    st.markdown("## 📊 Processing Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Basic Detection Parameters")
        
        pixel_scale = st.number_input(
            "Pixel Scale (pixels/μm)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            help="Number of pixels per micrometer for area conversion")
        
        area_range = st.slider(
            "Spore Area Range (μm²)",
            min_value=1,
            max_value=10000,
            value=(10, 500),
            help="Area range for objects to be considered spores")
        min_area, max_area = area_range
        
        # Convert area from μm² to pixels²
        min_area_pixels = int(min_area * (pixel_scale ** 2))
        max_area_pixels = int(max_area * (pixel_scale ** 2))
        st.caption(f"📐 Area in pixels: {min_area_pixels} - {max_area_pixels} px²")

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
            help="Exclude spores touching the image edges")
    
    with col2:
        st.markdown("### 🔍 Advanced Shape Filters")
        aspect_ratio_range = st.slider(
            "Aspect Ratio Range",
            min_value=1.0,
            max_value=10.0,
            value=(1.0, 5.0),
            step=0.1,
            help="Length/width ratio range")
        
        solidity_range = st.slider(
            "Solidity Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.5, 1.0),
            step=0.01,
            help="Solidity (area/convex hull area)")
        
        convexity_range = st.slider(
            "Convexity Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.7, 1.0),
            step=0.01,
            help="Convexity (hull perimeter/contour perimeter)")
    
    st.markdown("### 🖼️ Image Processing Settings")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        contrast_alpha = st.slider(
            "Contrast Enhancement",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Contrast multiplier (1.0 = no change, >1.0 = more contrast)")
        
        blur_kernel = st.slider(
            "Blur Kernel Size",
            min_value=1,
            max_value=15,
            value=5,
            step=2,
            help="Gaussian blur kernel (odd numbers only)")
    
    with col4:
        threshold_method = st.selectbox(
            "Threshold Method",
            ["Otsu", "Adaptive", "Manual"],
            help="Method for binary threshold")
        
        threshold_value = None
        if threshold_method == "Manual":
            threshold_value = st.slider("Threshold Value", 0, 255, 127)
    
    with col5:
        close_kernel = st.slider(
            "Outline Closing Kernel",
            min_value=3,
            max_value=15,
            value=5,
            step=2,
            help="Morphological closing kernel size")
        
        close_iterations = st.slider(
            "Closing Iterations",
            min_value=1,
            max_value=5,
            value=1,
            help="Number of closing operations")
        
        dilate_iters = st.slider(
            "Dilation Iterations",
            min_value=0,
            max_value=5,
            value=1,
            help="Number of dilation operations")
    
    st.markdown("### 💧 Watershed Separation Settings")
    
    col6, col7, col8 = st.columns(3)
    
    with col6:
        enable_watershed = st.checkbox(
            "Enable Watershed Separation",
            value=False,
            help="Apply watershed to separate touching spores")
    
    with col7:
        if enable_watershed:
            erosion_iterations = st.slider(
                "Erosion Iterations",
                min_value=0,
                max_value=5,
                value=1,
                help="Erosion before distance transform")
    
    with col8:
        if enable_watershed:
            watershed_sigma = st.slider(
                "Smoothing Sigma",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.5,
                help="Gaussian smoothing for distance transform")
            
            min_distance = st.slider(
                "Min Distance Between Seeds",
                min_value=3,
                max_value=21,
                value=7,
                step=2,
                help="Minimum distance between watershed seeds")
    
    st.markdown("---")
    st.markdown("## 🔍 Processing Pipeline Visualization")
    
    if st.button("🚀 Run Processing Pipeline", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = {}
        
        status_text.text("Step 1/10: Loading original image...")
        progress_bar.progress(1/10)
        steps['1_original'] = image_array.copy()
        
        status_text.text("Step 2/10: Applying contrast enhancement...")
        progress_bar.progress(2/10)
        contrasted = cv2.convertScaleAbs(image_array, alpha=contrast_alpha, beta=0)
        steps['2_contrasted'] = contrasted.copy()
        
        status_text.text("Step 3/10: Converting to grayscale...")
        progress_bar.progress(3/10)
        if len(contrasted.shape) == 3:
            gray = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)
        else:
            gray = contrasted.copy()
        steps['3_grayscale'] = gray
        
        status_text.text("Step 4/10: Applying Gaussian blur...")
        progress_bar.progress(4/10)
        if blur_kernel > 1:
            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        else:
            blurred = gray.copy()
        steps['4_blurred'] = blurred
        
        status_text.text("Step 5/10: Applying threshold...")
        progress_bar.progress(5/10)
        if threshold_method == "Otsu":
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_method == "Adaptive":
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        else:
            thresh_val = threshold_value if threshold_value is not None else 127
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        steps['5_threshold'] = binary.copy()
        
        status_text.text("Step 6/10: Inverting binary (if needed)...")
        progress_bar.progress(6/10)
        if np.sum(binary == 255) > np.sum(binary == 0):
            binary = cv2.bitwise_not(binary)
        steps['6_inverted'] = binary.copy()
        
        status_text.text("Step 7/10: Applying morphological closing...")
        progress_bar.progress(7/10)
        k = close_kernel if close_kernel % 2 == 1 else close_kernel + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
        steps['7_closed'] = closed.copy()
        
        status_text.text("Step 8/10: Dilating outlines...")
        progress_bar.progress(8/10)
        if dilate_iters > 0:
            dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(closed, dkernel, iterations=dilate_iters)
        else:
            dilated = closed.copy()
        steps['8_dilated'] = dilated.copy()
        
        status_text.text("Step 9/10: Finding and filling contours (with area filtering)...")
        progress_bar.progress(9/10)
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(dilated)
        filtered_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area_pixels <= area <= max_area_pixels:
                cv2.drawContours(filled, [cnt], -1, (255,), thickness=cv2.FILLED)
                filtered_count += 1
        steps['9_filled'] = filled.copy()
        
        status_text.text("Step 10/11: Drawing final contours (area filtered)...")
        progress_bar.progress(10/11)
        final_image = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2RGB)
        contours_final, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Apply area filter again for final contours
        contours_filtered = []
        for cnt in contours_final:
            area = cv2.contourArea(cnt)
            if min_area_pixels <= area <= max_area_pixels:
                contours_filtered.append(cnt)
        contours_final = contours_filtered
        cv2.drawContours(final_image, contours_final, -1, (0, 255, 0), 2)
        steps['10_final'] = final_image
        
        status_text.text("Step 11/11: Adding measurement lines (length & width)...")
        progress_bar.progress(11/11)
        measured_image = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2RGB)
        
        # Calculate and draw measurements for each contour
        measurements = []
        for cnt in contours_final:
            # Fit ellipse to get dimensions
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    center, axes, angle = ellipse
                    cx, cy = int(center[0]), int(center[1])
                    
                    # Get major and minor axes
                    width_px, height_px = axes
                    if width_px >= height_px:
                        major_axis = width_px
                        minor_axis = height_px
                        major_angle = angle
                    else:
                        major_axis = height_px
                        minor_axis = width_px
                        major_angle = angle + 90
                    
                    # Convert to micrometers
                    length_um = major_axis / pixel_scale
                    width_um = minor_axis / pixel_scale
                    
                    # Calculate line endpoints
                    angle_rad = np.radians(major_angle)
                    half_length = major_axis / 2
                    length_x1 = int(cx - half_length * np.cos(angle_rad))
                    length_y1 = int(cy - half_length * np.sin(angle_rad))
                    length_x2 = int(cx + half_length * np.cos(angle_rad))
                    length_y2 = int(cy + half_length * np.sin(angle_rad))
                    
                    width_angle_rad = angle_rad + np.pi/2
                    half_width = minor_axis / 2
                    width_x1 = int(cx - half_width * np.cos(width_angle_rad))
                    width_y1 = int(cy - half_width * np.sin(width_angle_rad))
                    width_x2 = int(cx + half_width * np.cos(width_angle_rad))
                    width_y2 = int(cy + half_width * np.sin(width_angle_rad))
                    
                    # Draw contour
                    cv2.drawContours(measured_image, [cnt], -1, (0, 255, 0), 2)
                    
                    # Draw measurement lines (cyan for length, yellow for width)
                    cv2.line(measured_image, (length_x1, length_y1), (length_x2, length_y2), (255, 255, 0), 2)
                    cv2.line(measured_image, (width_x1, width_y1), (width_x2, width_y2), (255, 0, 255), 2)
                    
                    # Draw centroid
                    cv2.circle(measured_image, (cx, cy), 4, (255, 0, 0), -1)
                    
                    # Add text with measurements
                    text = f"L:{length_um:.1f} W:{width_um:.1f}"
                    cv2.putText(measured_image, text, (cx + 10, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    measurements.append({
                        'length_um': length_um,
                        'width_um': width_um,
                        'area_um2': (major_axis * minor_axis * np.pi / 4) / (pixel_scale ** 2)
                    })
                except:
                    pass
        
        steps['11_measured'] = measured_image
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(1.0)
        
        st.markdown("---")
        st.markdown("### 📸 Processing Steps")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("#### 1️⃣ Original Image")
            st.image(steps['1_original'], use_container_width=True)
            st.caption(f"Shape: {steps['1_original'].shape}")
            
            st.markdown("#### 4️⃣ Blurred")
            st.image(steps['4_blurred'], use_container_width=True, clamp=True)
            st.caption(f"Kernel: {blur_kernel}×{blur_kernel}")
            
            st.markdown("#### 7️⃣ Morphological Closing")
            st.image(steps['7_closed'], use_container_width=True, clamp=True)
            st.caption(f"Kernel: {k}×{k}, Iterations: {close_iterations}")
            
            st.markdown("#### 🔟 Final Contours")
            st.image(steps['10_final'], use_container_width=True)
            st.caption(f"Detected contours: {len(contours_final)}")
            
            st.markdown("#### 1️⃣1️⃣ Measured Spores")
            st.image(steps['11_measured'], use_container_width=True)
            st.caption(f"With length (cyan) & width (magenta) lines")
        
        with col_b:
            st.markdown("#### 2️⃣ Contrast Enhanced")
            st.image(steps['2_contrasted'], use_container_width=True)
            st.caption(f"Alpha: {contrast_alpha}")
            
            st.markdown("#### 5️⃣ Threshold")
            st.image(steps['5_threshold'], use_container_width=True, clamp=True)
            st.caption(f"Method: {threshold_method}")
            
            st.markdown("#### 8️⃣ Dilated")
            st.image(steps['8_dilated'], use_container_width=True, clamp=True)
            st.caption(f"Iterations: {dilate_iters}")
        
        with col_c:
            st.markdown("#### 3️⃣ Grayscale")
            st.image(steps['3_grayscale'], use_container_width=True, clamp=True)
            st.caption(f"Shape: {steps['3_grayscale'].shape}")
            
            st.markdown("#### 6️⃣ Inverted Binary")
            st.image(steps['6_inverted'], use_container_width=True, clamp=True)
            st.caption("White spores on black background")
            
            st.markdown("#### 9️⃣ Filled Contours")
            st.image(steps['9_filled'], use_container_width=True, clamp=True)
            st.caption(f"Area range: {min_area_pixels}-{max_area_pixels} px² ({min_area}-{max_area} μm²)")
        
        st.markdown("---")
        st.markdown("### 📊 Detection Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Total Contours", len(contours_final))
        
        with stat_col2:
            total_white_pixels = np.sum(filled == 255)
            st.metric("White Pixels (filled)", f"{total_white_pixels:,}")
        
        with stat_col3:
            white_percentage = (total_white_pixels / (filled.shape[0] * filled.shape[1])) * 100
            st.metric("Coverage", f"{white_percentage:.2f}%")
        
        with stat_col4:
            if len(contours_final) > 0:
                avg_area = np.mean([cv2.contourArea(c) for c in contours_final])
                st.metric("Avg Contour Area", f"{avg_area:.1f} px²")
            else:
                st.metric("Avg Contour Area", "N/A")
        
        if measurements:
            st.markdown("---")
            st.markdown("### 📏 Spore Measurements")
            
            meas_col1, meas_col2, meas_col3, meas_col4 = st.columns(4)
            
            lengths = [m['length_um'] for m in measurements]
            widths = [m['width_um'] for m in measurements]
            areas = [m['area_um2'] for m in measurements]
            
            with meas_col1:
                st.metric("Avg Length", f"{np.mean(lengths):.2f} μm")
                st.caption(f"Range: {np.min(lengths):.1f} - {np.max(lengths):.1f}")
            
            with meas_col2:
                st.metric("Avg Width", f"{np.mean(widths):.2f} μm")
                st.caption(f"Range: {np.min(widths):.1f} - {np.max(widths):.1f}")
            
            with meas_col3:
                st.metric("Avg Area", f"{np.mean(areas):.1f} μm²")
                st.caption(f"Range: {np.min(areas):.1f} - {np.max(areas):.1f}")
            
            with meas_col4:
                aspect_ratios = [l/w for l, w in zip(lengths, widths)]
                st.metric("Avg Aspect Ratio", f"{np.mean(aspect_ratios):.2f}")
                st.caption(f"Range: {np.min(aspect_ratios):.1f} - {np.max(aspect_ratios):.1f}")
        
        if enable_watershed and len(contours_final) > 0:
            st.markdown("---")
            st.markdown("### 💧 Watershed Separation Debug")
            
            st.info("🔬 **Demonstrating watershed on the first detected contour**")
            
            watershed_steps = {}
            
            test_contour = contours_final[0]
            
            ws_status = st.empty()
            ws_progress = st.progress(0)
            
            ws_status.text("Watershed Step 1/6: Creating mask from contour...")
            ws_progress.progress(1/6)
            mask = np.zeros(filled.shape, dtype=np.uint8)
            cv2.drawContours(mask, [test_contour], -1, 255, thickness=cv2.FILLED)
            watershed_steps['1_mask'] = mask.copy()
            
            ws_status.text("Watershed Step 2/6: Applying erosion...")
            ws_progress.progress(2/6)
            if erosion_iterations > 0:
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(mask, kernel, iterations=erosion_iterations)
            else:
                eroded = mask.copy()
            watershed_steps['2_eroded'] = eroded.copy()
            
            ws_status.text("Watershed Step 3/6: Computing distance transform...")
            ws_progress.progress(3/6)
            distance = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
            distance_normalized = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            watershed_steps['3_distance'] = distance_normalized
            
            ws_status.text("Watershed Step 4/6: Smoothing distance transform...")
            ws_progress.progress(4/6)
            if watershed_sigma > 0:
                distance_smooth = ndimage.gaussian_filter(distance, sigma=watershed_sigma)
            else:
                distance_smooth = distance.copy()
            distance_smooth_normalized = cv2.normalize(distance_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            watershed_steps['4_smoothed'] = distance_smooth_normalized
            
            ws_status.text("Watershed Step 5/6: Finding local maxima (seeds)...")
            ws_progress.progress(5/6)
            footprint = np.ones((min_distance, min_distance))
            local_maxima = (distance_smooth == ndimage.maximum_filter(distance_smooth, footprint=footprint))
            threshold = 0.3 * distance_smooth.max()
            local_maxima = local_maxima & (distance_smooth >= threshold)
            coordinates = np.argwhere(local_maxima)
            
            markers = np.zeros(distance_smooth.shape, dtype=np.int32)
            for i, (y, x) in enumerate(coordinates):
                markers[y, x] = i + 1
            
            markers_visual = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            markers_colored = cv2.applyColorMap(markers_visual, cv2.COLORMAP_JET)
            watershed_steps['5_markers'] = markers_colored
            
            ws_status.text("Watershed Step 6/6: Applying watershed segmentation...")
            ws_progress.progress(6/6)
            if len(coordinates) > 0:
                labels = watershed(-distance_smooth, markers, mask=eroded)
                labels_normalized = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                labels_colored = cv2.applyColorMap(labels_normalized, cv2.COLORMAP_JET)
                watershed_steps['6_labels'] = labels_colored
            else:
                watershed_steps['6_labels'] = np.zeros_like(mask)
            
            ws_status.text("✅ Watershed processing complete!")
            ws_progress.progress(1.0)
            
            st.markdown("#### 🔬 Watershed Processing Steps")
            
            ws_col1, ws_col2, ws_col3 = st.columns(3)
            
            with ws_col1:
                st.markdown("**Step 1: Contour Mask**")
                st.image(watershed_steps['1_mask'], use_container_width=True, clamp=True)
                st.caption("Binary mask from contour")
                
                st.markdown("**Step 4: Smoothed Distance**")
                st.image(watershed_steps['4_smoothed'], use_container_width=True, clamp=True)
                st.caption(f"Sigma: {watershed_sigma}")
            
            with ws_col2:
                st.markdown("**Step 2: Eroded Mask**")
                st.image(watershed_steps['2_eroded'], use_container_width=True, clamp=True)
                st.caption(f"Iterations: {erosion_iterations}")
                
                st.markdown("**Step 5: Watershed Seeds**")
                st.image(watershed_steps['5_markers'], use_container_width=True)
                st.caption(f"Found {len(coordinates)} seeds (min dist: {min_distance})")
            
            with ws_col3:
                st.markdown("**Step 3: Distance Transform**")
                st.image(watershed_steps['3_distance'], use_container_width=True, clamp=True)
                st.caption("Distance to background")
                
                st.markdown("**Step 6: Watershed Labels**")
                st.image(watershed_steps['6_labels'], use_container_width=True)
                st.caption("Separated regions")
        
        st.success("✅ **Pipeline visualization complete!** Use the settings above to adjust parameters and re-run.")

else:
    st.info("👆 **Upload an image above to begin processing**")
    st.markdown("""
    ### How to use this test app:
    
    1. **Upload an image** using the file uploader at the top
    2. **Adjust processing parameters** in the settings sections
    3. **Click 'Run Processing Pipeline'** to see all intermediate steps
    4. **Review each step** to understand how the image is processed
    
    This tool helps you:
    - 🔍 Debug detection issues by seeing each processing step
    - ⚙️ Fine-tune parameters for optimal detection
    - 📊 Understand the complete image processing pipeline
    - 🎯 Identify which step may need adjustment
    """)
