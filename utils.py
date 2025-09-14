import numpy as np
import pandas as pd
import cv2
import io
from PIL import Image, ImageDraw, ImageFont

def calculate_statistics(spore_results):
    """Calculate statistical measures for spore measurements"""
    if not spore_results:
        return {}
    
    # Extract measurements
    lengths = [spore['length_um'] for spore in spore_results]
    widths = [spore['width_um'] for spore in spore_results]
    areas = [spore['area_um2'] for spore in spore_results]
    aspect_ratios = [spore['aspect_ratio'] for spore in spore_results]
    circularities = [spore['circularity'] for spore in spore_results]
    solidities = [spore['solidity'] for spore in spore_results]
    convexities = [spore['convexity'] for spore in spore_results]
    extents = [spore['extent'] for spore in spore_results]
    
    # Calculate statistics
    stats = {
        'count': len(spore_results),
        'length_mean': np.mean(lengths),
        'length_std': np.std(lengths),
        'length_min': np.min(lengths),
        'length_max': np.max(lengths),
        'length_median': np.median(lengths),
        'width_mean': np.mean(widths),
        'width_std': np.std(widths),
        'width_min': np.min(widths),
        'width_max': np.max(widths),
        'width_median': np.median(widths),
        'area_mean': np.mean(areas),
        'area_std': np.std(areas),
        'area_min': np.min(areas),
        'area_max': np.max(areas),
        'area_median': np.median(areas),
        'aspect_ratio_mean': np.mean(aspect_ratios),
        'aspect_ratio_std': np.std(aspect_ratios),
        'aspect_ratio_min': np.min(aspect_ratios),
        'aspect_ratio_max': np.max(aspect_ratios),
        'circularity_mean': np.mean(circularities),
        'circularity_std': np.std(circularities),
        'circularity_min': np.min(circularities),
        'circularity_max': np.max(circularities),
        'solidity_mean': np.mean(solidities),
        'solidity_std': np.std(solidities),
        'solidity_min': np.min(solidities),
        'solidity_max': np.max(solidities),
        'convexity_mean': np.mean(convexities),
        'convexity_std': np.std(convexities),
        'convexity_min': np.min(convexities),
        'convexity_max': np.max(convexities),
        'extent_mean': np.mean(extents),
        'extent_std': np.std(extents),
        'extent_min': np.min(extents),
        'extent_max': np.max(extents)
    }
    
    return stats

def create_overlay_image(original_image, spore_results, selected_spores, pixel_scale, vis_settings=None):
    """Create an overlay image showing detected spores with measurement lines"""
    # Default visualization settings
    default_settings = {
        'font_size': 1.6,
        'font_color': (255, 255, 255),  # White
        'border_color': (0, 0, 0),      # Black
        'border_width': 8,
        'line_color': (0, 255, 255),    # Yellow
        'line_width': 2
    }
    
    # Use provided settings or defaults
    settings = vis_settings if vis_settings else default_settings
    
    # Convert to RGB if needed
    if len(original_image.shape) == 3:
        overlay = original_image.copy()
    else:
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Ensure RGB format
    if overlay.dtype != np.uint8:
        overlay = (overlay * 255).astype(np.uint8)
    
    # No need for a separate border overlay - we'll use temporary overlays per spore
    
    from spore_analyzer import SporeAnalyzer
    analyzer = SporeAnalyzer()
    analyzer.pixel_scale = pixel_scale
    
    # Draw each spore
    for i, spore in enumerate(spore_results):
        if i in selected_spores:
            # Selected spores - green
            contour_color = (0, 255, 0)  # Green
            text_color = (0, 255, 0)     # Green
        else:
            # Unselected spores - red
            contour_color = (255, 0, 0)  # Red
            text_color = (255, 0, 0)     # Red
        
        # Draw contour with 0.5 opacity  
        contour_temp = overlay.copy()
        cv2.drawContours(contour_temp, [spore['contour']], -1, contour_color, 2)
        cv2.addWeighted(overlay, 0.5, contour_temp, 0.5, 0, overlay)
        
        # Get measurement lines
        lines = analyzer.get_measurement_lines(spore)
        
        # Draw lines at full opacity with user settings
        cv2.line(overlay, lines['length_line'][0], lines['length_line'][1], settings['line_color'], settings['line_width'])
        cv2.line(overlay, lines['width_line'][0], lines['width_line'][1], settings['line_color'], settings['line_width'])
        
        # Draw centroid
        cv2.circle(overlay, lines['centroid'], 3, text_color, -1)
        
        # Add spore number
        cv2.putText(overlay, str(i+1), 
                   (lines['centroid'][0] + 10, lines['centroid'][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Add measurement text with L/W prefixes, no units
        length_text = f"L: {spore['length_um']:.2f}"
        width_text = f"W: {spore['width_um']:.2f}"
        
        # Position length text at the end of the length line
        length_end_x = lines['length_line'][1][0] + 10
        length_end_y = lines['length_line'][1][1] - 5
        
        # Position width text at the end of the width line  
        width_end_x = lines['width_line'][1][0] + 10
        width_end_y = lines['width_line'][1][1] - 5
        
        # Calculate text size for background rectangles
        font_scale = settings['font_size']
        thickness = max(1, int(font_scale * 2))
        text_size_length = cv2.getTextSize(length_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_size_width = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Add background rectangles with transparency (0.5 opacity) if border_width > 0
        if settings['border_width'] > 0:
            # Create mask for background areas only
            background_mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
            
            # Define background rectangle areas
            length_rect = (
                max(0, length_end_x - settings['border_width']),
                max(0, length_end_y - text_size_length[1] - settings['border_width']),
                min(overlay.shape[1], length_end_x + text_size_length[0] + settings['border_width']),
                min(overlay.shape[0], length_end_y + settings['border_width'])
            )
            
            width_rect = (
                max(0, width_end_x - settings['border_width']),
                max(0, width_end_y - text_size_width[1] - settings['border_width']),
                min(overlay.shape[1], width_end_x + text_size_width[0] + settings['border_width']),
                min(overlay.shape[0], width_end_y + settings['border_width'])
            )
            
            # Fill mask areas where backgrounds should be
            cv2.rectangle(background_mask, (length_rect[0], length_rect[1]), (length_rect[2], length_rect[3]), 255, -1)
            cv2.rectangle(background_mask, (width_rect[0], width_rect[1]), (width_rect[2], width_rect[3]), 255, -1)
            
            # Create background overlay
            background_overlay = overlay.copy()
            cv2.rectangle(background_overlay, (length_rect[0], length_rect[1]), (length_rect[2], length_rect[3]), settings['border_color'], -1)
            cv2.rectangle(background_overlay, (width_rect[0], width_rect[1]), (width_rect[2], width_rect[3]), settings['border_color'], -1)
            
            # Apply 0.5 opacity only to background areas using the mask
            mask_3d = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR) / 255.0
            overlay = overlay * (1 - mask_3d * 0.5) + background_overlay * (mask_3d * 0.5)
            overlay = overlay.astype(np.uint8)
        
        # Draw the measurement text at user-defined size (full opacity)
        cv2.putText(overlay, length_text, (length_end_x, length_end_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, settings['font_color'], thickness)
        cv2.putText(overlay, width_text, (width_end_x, width_end_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, settings['font_color'], thickness)
    
    # Add units legend in top-right corner
    legend_text = "Units: micrometers (μm)"
    legend_font_scale = 0.7
    legend_thickness = 2
    legend_size = cv2.getTextSize(legend_text, cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, legend_thickness)[0]
    
    # Position legend in top-right corner with full bounds checking
    legend_x = max(10, min(overlay.shape[1] - legend_size[0] - 20, overlay.shape[1] - 30))
    legend_y = max(legend_size[1] + 10, min(30, overlay.shape[0] - 20))
    
    # Create legend background with 0.5 opacity
    legend_background = overlay.copy()
    cv2.rectangle(legend_background, 
                 (legend_x - 10, legend_y - legend_size[1] - 10), 
                 (legend_x + legend_size[0] + 10, legend_y + 10), 
                 (0, 0, 0), -1)
    
    # Blend legend background at 0.5 opacity
    cv2.addWeighted(overlay, 0.5, legend_background, 0.5, 0, overlay)
    
    # Draw legend text at full opacity
    cv2.putText(overlay, legend_text, (legend_x, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (255, 255, 255), legend_thickness)
    
    return overlay

def export_results(df, format_type='csv'):
    """Export results to different formats"""
    if format_type == 'csv':
        return df.to_csv(index=False)
    elif format_type == 'excel':
        output = io.BytesIO()
        # Use the output buffer directly
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Spore_Measurements', index=False)
            
            # Add statistics sheet
            stats_data = {
                'Metric': ['Count', 'Mean Length (μm)', 'Std Length (μm)', 'Mean Width (μm)', 
                          'Std Width (μm)', 'Mean Area (μm²)', 'Mean Aspect Ratio'],
                'Value': [
                    len(df),
                    df['Length_um'].mean(),
                    df['Length_um'].std(),
                    df['Width_um'].mean(),
                    df['Width_um'].std(),
                    df['Area_um2'].mean(),
                    df['Aspect_Ratio'].mean()
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        output.seek(0)  # Reset buffer position
        return output.getvalue()
    else:
        return df.to_csv(index=False)

def create_measurement_report(spore_results, selected_spores, pixel_scale):
    """Create a comprehensive measurement report"""
    selected_results = [spore_results[i] for i in selected_spores]
    stats = calculate_statistics(selected_results)
    
    report = f"""
    FUNGAL SPORE MEASUREMENT REPORT
    ================================
    
    Analysis Parameters:
    - Pixel Scale: {pixel_scale:.2f} pixels/μm
    - Total Detected Spores: {len(spore_results)}
    - Selected Spores: {len(selected_spores)}
    
    Statistical Summary:
    - Mean Length: {stats.get('length_mean', 0):.2f} ± {stats.get('length_std', 0):.2f} μm
    - Mean Width: {stats.get('width_mean', 0):.2f} ± {stats.get('width_std', 0):.2f} μm
    - Mean Area: {stats.get('area_mean', 0):.2f} μm²
    - Mean Aspect Ratio: {stats.get('aspect_ratio_mean', 0):.2f}
    - Mean Circularity: {stats.get('circularity_mean', 0):.3f}
    
    Length Range: {stats.get('length_min', 0):.2f} - {stats.get('length_max', 0):.2f} μm
    Width Range: {stats.get('width_min', 0):.2f} - {stats.get('width_max', 0):.2f} μm
    """
    
    return report

def validate_image(image_array):
    """Validate uploaded image for spore analysis"""
    errors = []
    warnings = []
    
    # Check image dimensions
    if len(image_array.shape) not in [2, 3]:
        errors.append("Invalid image format. Expected 2D or 3D array.")
    
    if image_array.shape[0] < 100 or image_array.shape[1] < 100:
        warnings.append("Image resolution is quite low. Consider using higher resolution images for better accuracy.")
    
    if image_array.shape[0] > 5000 or image_array.shape[1] > 5000:
        warnings.append("Very large image detected. Processing may take longer.")
    
    # Check image type
    if image_array.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        errors.append("Unsupported image data type.")
    
    # Check for completely black or white images
    if len(image_array.shape) == 2:
        unique_values = len(np.unique(image_array))
    else:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        unique_values = len(np.unique(gray))
    
    if unique_values < 10:
        warnings.append("Image appears to have very low contrast. Consider adjusting image settings.")
    
    return errors, warnings

def estimate_optimal_parameters(image_array):
    """Estimate optimal parameters for spore detection based on image characteristics"""
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Calculate image statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # Estimate blur kernel based on image noise
    blur_kernel = 3 if std_intensity < 20 else 5 if std_intensity < 50 else 7
    
    # Ensure odd kernel size
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    
    # Recommend threshold method based on intensity distribution
    if std_intensity > 40:
        threshold_method = "Adaptive"
    else:
        threshold_method = "Otsu"
    
    recommendations = {
        'blur_kernel': blur_kernel,
        'threshold_method': threshold_method,
        'notes': f"Image statistics - Mean: {mean_intensity:.1f}, Std: {std_intensity:.1f}"
    }
    
    return recommendations
