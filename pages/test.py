import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

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
        area_range = st.slider(
            "Spore Area Range (μm²)",
            min_value=1,
            max_value=10000,
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
        
        status_text.text("Step 9/10: Finding and filling contours...")
        progress_bar.progress(9/10)
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(dilated)
        min_area_pixels = 20
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area_pixels:
                cv2.drawContours(filled, [cnt], -1, (255,), thickness=cv2.FILLED)
        steps['9_filled'] = filled.copy()
        
        status_text.text("Step 10/10: Drawing final contours...")
        progress_bar.progress(10/10)
        final_image = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2RGB)
        contours_final, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_image, contours_final, -1, (0, 255, 0), 2)
        steps['10_final'] = final_image
        
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
            
            st.markdown("#### 🔟 Final Result")
            st.image(steps['10_final'], use_container_width=True)
            st.caption(f"Detected contours: {len(contours_final)}")
        
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
            st.caption(f"Min area: {min_area_pixels} pixels")
        
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
