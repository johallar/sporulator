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

def generate_q_value_summary(selected_results):
    """Generate mycological summary in the standard format"""
    if not selected_results:
        return "No spores selected for analysis."

    lengths = [spore['length_um'] for spore in selected_results]
    aspect_ratios = [spore['aspect_ratio'] for spore in selected_results]

    # Sort to get quartiles and extremes
    lengths_sorted = sorted(lengths)
    aspect_ratios_sorted = sorted(aspect_ratios)

    n = len(lengths_sorted)

    # Calculate quartiles (25th and 75th percentile) for the main range
    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)

    # Aspect ratio statistics
    aspect_min = min(aspect_ratios)
    aspect_q1 = aspect_ratios_sorted[q1_idx]
    aspect_q3 = aspect_ratios_sorted[q3_idx]
    aspect_max = max(aspect_ratios)
    aspect_mean = sum(aspect_ratios) / len(aspect_ratios)
    aspect_std = (sum(
        (x - aspect_mean)**2 for x in aspect_ratios) / len(aspect_ratios))**0.5
    summary = f"**Q value (length/width ratio):**\n\n"
    summary += f"Q = ({aspect_min:.1f}–) {aspect_q1:.1f}–{aspect_q3:.1f} (–{aspect_max:.1f}), "
    summary += f"Qm = {aspect_mean:.1f} ± {aspect_std:.1f}\n\n"
    return summary

def generate_dimensions_summary(selected_results):
    """Generate spore dimension summary in the standard format"""
    if not selected_results:
        return "No spores selected for analysis."

    lengths = [spore['length_um'] for spore in selected_results]
    widths = [spore['width_um'] for spore in selected_results]

    # Sort to get quartiles and extremes
    lengths_sorted = sorted(lengths)
    widths_sorted = sorted(widths)

    n = len(lengths_sorted)

    # Calculate quartiles (25th and 75th percentile) for the main range
    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)

    # Length statistics
    length_min = min(lengths)
    length_q1 = lengths_sorted[q1_idx]
    length_q3 = lengths_sorted[q3_idx]
    length_max = max(lengths)

    # Width statistics
    width_min = min(widths)
    width_q1 = widths_sorted[q1_idx]
    width_q3 = widths_sorted[q3_idx]
    width_max = max(widths)

    # Format the mycological summary
    summary = f"**Spore dimensions:**\n\n"
    summary += f"({length_min:.1f}) {length_q1:.1f}–{length_q3:.1f} ({length_max:.1f}) × "
    summary += f"({width_min:.1f}) {width_q1:.1f}–{width_q3:.1f} ({width_max:.1f}) µm\n\n"
    return summary


def rectangles_overlap(rect1, rect2):
    """Check if two rectangles overlap.
    Each rectangle is (x, y, width, height)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Add small padding to prevent touching text
    padding = 5
    
    return not (x1 + w1 + padding < x2 or x2 + w2 + padding < x1 or 
                y1 + h1 + padding < y2 or y2 + h2 + padding < y1)

def adjust_text_position(x, y, text_size, existing_rects, image_shape):
    """Adjust text position to avoid overlapping with existing text rectangles.
    Returns (new_x, new_y)
    """
    width, height = text_size
    img_width, img_height = image_shape[1], image_shape[0]
    
    # Try different positions around the original position
    offsets = [
        (0, 0),          # Original position
        (0, -30),        # Above
        (0, 30),         # Below  
        (-30, 0),        # Left
        (30, 0),         # Right
        (-30, -30),      # Top-left
        (30, -30),       # Top-right
        (-30, 30),       # Bottom-left
        (30, 30),        # Bottom-right
    ]
    
    for offset_x, offset_y in offsets:
        new_x = x + offset_x
        new_y = y + offset_y
        
        # Ensure position is within image bounds
        new_x = max(0, min(img_width - width, new_x))
        new_y = max(height, min(img_height, new_y))
        
        # Check if this position overlaps with any existing rectangles
        new_rect = (new_x, new_y - height, width, height)
        overlap = False
        
        for existing_rect in existing_rects:
            if rectangles_overlap(new_rect, existing_rect):
                overlap = True
                break
                
        if not overlap:
            return new_x, new_y
    
    # If no non-overlapping position found, return adjusted original position
    return max(0, min(img_width - width, x)), max(height, min(img_height, y))

def create_overlay_image(original_image,
                         spore_results,
                         selected_spores,
                         pixel_scale,
                         vis_settings=None,
                         include_stats=True):
    """Create an overlay image showing detected spores with measurement lines"""
    # Default visualization settings
    default_settings = {
        'font_size': 1.6,
        'font_color': (255, 255, 255),  # White
        'border_color': (0, 0, 0),  # Black
        'border_width': 8,
        'line_color': (255, 255, 0),  # Cyan (matches #00FFFF)
        'line_width': 2
    }

    # Use provided settings or defaults
    settings = vis_settings if vis_settings else default_settings

    # Convert to BGR for OpenCV drawing operations
    if len(original_image.shape) == 3:
        overlay = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    else:
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # Ensure uint8 format
    if overlay.dtype != np.uint8:
        overlay = (overlay * 255).astype(np.uint8)

    from spore_analyzer import SporeAnalyzer
    analyzer = SporeAnalyzer()
    analyzer.pixel_scale = pixel_scale

    # Global text rectangles for collision detection across all spores
    global_text_rectangles = []

    # First pass: Draw spores and measurement lines
    for i, spore in enumerate(spore_results):
        if i in selected_spores:
            # Selected spores - use line color for both contours and measurement lines
            contour_color = settings['line_color']
            text_color = (0, 255, 0)  # Green for spore numbers
            
            # Draw contour with 0.5 opacity using the line color
            contour_temp = overlay.copy()
            cv2.drawContours(contour_temp, [spore['contour']], -1, contour_color,
                             2)
            cv2.addWeighted(overlay, 0.5, contour_temp, 0.5, 0, overlay)

            # Get measurement lines
            lines = analyzer.get_measurement_lines(spore)

            # Draw lines at full opacity with user settings
            cv2.line(overlay, lines['length_line'][0], lines['length_line'][1],
                     settings['line_color'], settings['line_width'])
            cv2.line(overlay, lines['width_line'][0], lines['width_line'][1],
                     settings['line_color'], settings['line_width'])

            # Draw centroid
            cv2.circle(overlay, lines['centroid'], 3, text_color, -1)
        else:
            # Unselected spores - draw dimmed contour only (no measurement lines)
            contour_color = tuple(
                int(c * 0.3) for c in settings['line_color'])  # Very dimmed version
            contour_temp = overlay.copy()
            cv2.drawContours(contour_temp, [spore['contour']], -1, contour_color,
                             1)  # Thinner line for unselected
            cv2.addWeighted(overlay, 0.7, contour_temp, 0.3, 0, overlay)

    # Second pass: Add multi-line text boxes with global collision detection (ONLY for selected spores)
    for i, spore in enumerate(spore_results):
        if i in selected_spores:  # Only draw text for selected spores
            text_color = (0, 255, 0)  # Green for selected spores

            # Get measurement lines for centroid
            lines = analyzer.get_measurement_lines(spore)
            centroid = lines['centroid']

            # Create multi-line text
            spore_number = str(i + 1)
            line1 = f"#{spore_number}"
            line2 = f"L: {spore['length_um']:.2f}"
            line3 = f"W: {spore['width_um']:.2f}"
            text_lines = [line1, line2, line3]

            font_scale = settings['font_size']
            thickness = max(1, int(font_scale * 2))
            
            # Calculate text box dimensions
            line_heights = []
            max_width = 0
            for line in text_lines:
                line_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                line_width, line_height = line_size[0]
                max_width = max(max_width, line_width)
                line_heights.append(line_height)
            
            # Calculate total text box size with proper baseline accounting
            line_spacing = 10  # Increased spacing for better readability
            total_height = sum(line_heights) + (len(text_lines) - 1) * line_spacing
            text_box_size = (max_width, total_height)
            
            # Calculate the actual text start position (accounting for baseline)
            text_start_offset = line_heights[0]  # First line height for proper baseline positioning

            # Initial positioning - try to place text box away from spore
            contour = spore['contour']
            x, y, w, h = cv2.boundingRect(contour)
        
            # Try different positions around the spore (adjusted for proper text baseline)
            candidate_positions = [
                (x + w + 20, y + text_start_offset),  # Right of spore
                (x - max_width - 20, y + text_start_offset),  # Left of spore
                (x, y - 20),  # Above spore
                (x, y + h + total_height + 20),  # Below spore
                (x + w + 20, y + h // 2 + text_start_offset),  # Right-center
                (x - max_width - 20, y + h // 2 + text_start_offset),  # Left-center
            ]
        
            # Find best position with global collision detection
            text_x, text_y = None, None
            for candidate_x, candidate_y in candidate_positions:
                # Ensure position is within image bounds
                candidate_x = max(0, min(overlay.shape[1] - max_width, candidate_x))
                candidate_y = max(total_height, min(overlay.shape[0], candidate_y))
                
                # Check if this position overlaps with any existing text rectangles
                # The rectangle should start from where the first line will be drawn
                candidate_rect = (candidate_x, candidate_y - text_start_offset, max_width, total_height)
                overlap = False
                
                for existing_rect in global_text_rectangles:
                    if rectangles_overlap(candidate_rect, existing_rect):
                        overlap = True
                        break
                        
                if not overlap:
                    text_x, text_y = candidate_x, candidate_y
                    break
        
            # If no non-overlapping position found, use adjust_text_position
            if text_x is None or text_y is None:
                text_x = x + w + 20
                text_y = y + text_start_offset
                text_x, text_y = adjust_text_position(
                    text_x, text_y, text_box_size, global_text_rectangles, overlay.shape)

            # Store this text box rectangle for future collision detection
            text_rect = (text_x, text_y - text_start_offset, max_width, total_height)
            global_text_rectangles.append(text_rect)

            # Draw connecting line from text box to centroid
            # Connect from the center of text box to centroid
            text_center_x = text_x + max_width // 2
            text_center_y = text_y - text_start_offset + total_height // 2
            
            cv2.line(overlay, (text_center_x, text_center_y), centroid,
                     settings['line_color'], 1)

            # Add background rectangle with transparency if border_width > 0
            if settings['border_width'] > 0:
                # Create mask for background area
                background_mask = np.zeros(overlay.shape[:2], dtype=np.uint8)

                # Define background rectangle area (properly aligned with text)
                bg_rect = (max(0, text_x - settings['border_width']),
                          max(0, text_y - text_start_offset - settings['border_width']),
                          min(overlay.shape[1], text_x + max_width + settings['border_width']),
                          min(overlay.shape[0], text_y - text_start_offset + total_height + settings['border_width']))

                # Fill mask area where background should be
                cv2.rectangle(background_mask, (bg_rect[0], bg_rect[1]),
                              (bg_rect[2], bg_rect[3]), 255, -1)

                # Create background overlay
                background_overlay = overlay.copy()
                cv2.rectangle(background_overlay, (bg_rect[0], bg_rect[1]),
                              (bg_rect[2], bg_rect[3]),
                              settings['border_color'], -1)

                # Apply 0.5 opacity only to background area using the mask
                mask_3d = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR) / 255.0
                overlay = overlay * (1 - mask_3d * 0.5) + background_overlay * (mask_3d * 0.5)
                overlay = overlay.astype(np.uint8)

            # Draw multi-line text
            current_y = text_y
            for j, line in enumerate(text_lines):
                cv2.putText(overlay, line, (text_x, current_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            settings['font_color'], thickness)
                current_y += line_heights[j] + 10  # 10px spacing between lines

    # Add enhanced legend with statistics in top-right corner
    if include_stats and selected_spores:
        # Calculate statistics for selected spores
        selected_results = [spore_results[i] for i in selected_spores if i < len(spore_results)]
        
        if selected_results:
            lengths = [spore['length_um'] for spore in selected_results]
            widths = [spore['width_um'] for spore in selected_results]
            
            # Calculate Qm value (length/width ratio)
            q_values = [l/w for l, w in zip(lengths, widths)]
            mean_q = np.mean(q_values)
            std_q = np.std(q_values)
            
            # Calculate mycological summary format ranges
            # Length ranges: (min–) 10th percentile–90th percentile (–max)
            length_min, length_max = min(lengths), max(lengths)
            length_10th = np.percentile(lengths, 10)
            length_90th = np.percentile(lengths, 90)
            
            # Width ranges: (min–) 10th percentile–90th percentile (–max)  
            width_min, width_max = min(widths), max(widths)
            width_10th = np.percentile(widths, 10)
            width_90th = np.percentile(widths, 90)
            
            # Create mycological format dimension string (using ASCII characters for OpenCV compatibility)
            dimension_str = f"({length_min:.1f}) {length_10th:.1f}-{length_90th:.1f} ({length_max:.1f}) x ({width_min:.1f}) {width_10th:.1f}-{width_90th:.1f} ({width_max:.1f}) um"
            
            # Create legend text lines with mycological format (ASCII only)
            legend_lines = [
                f"n = {len(selected_results)} spores",
                dimension_str,
                f"Qm = {mean_q:.1f} +/- {std_q:.1f}"
            ]
        else:
            legend_lines = ["No spores selected"]
    else:
        legend_lines = ["Units: micrometers (um)"]

    # Increased font size
    legend_font_scale = settings["measurement_fontsize"] if "measurement_fontsize" in settings else 1.4
    legend_thickness = 4
    line_spacing = 25
    
    # Calculate maximum text width and total height
    max_width = 0
    total_height = 0
    line_heights = []
    
    for line in legend_lines:
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 
                                   legend_font_scale, legend_thickness)[0]
        max_width = max(max_width, text_size[0])
        line_heights.append(text_size[1])
        total_height += text_size[1] + line_spacing
    
    total_height -= line_spacing  # Remove last line spacing

    # Position legend in top-right corner with full bounds checking
    legend_x = max(15, min(overlay.shape[1] - max_width - 30, overlay.shape[1] - 40))
    legend_y = max(line_heights[0] + 15, min(40, overlay.shape[0] - total_height - 30))

    # Create larger legend background with 0.5 opacity
    legend_background = overlay.copy()
    cv2.rectangle(legend_background,
                  (legend_x - 15, legend_y - line_heights[0] - 15),
                  (legend_x + max_width + 15, legend_y + total_height - line_heights[0] + 15), 
                  (0, 0, 0), -1)

    # Blend legend background at 0.5 opacity
    cv2.addWeighted(overlay, 0.5, legend_background, 0.5, 0, overlay)

    # Draw legend text lines at full opacity
    current_y = legend_y
    for i, line in enumerate(legend_lines):
        cv2.putText(overlay, line, (legend_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (255, 255, 255),
                    legend_thickness)
        if i < len(legend_lines) - 1:  # Don't add spacing after last line
            current_y += line_heights[i] + line_spacing

    # Convert back to RGB for Streamlit display
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay


def export_results(df, format_type='csv'):
    """Export results to different formats"""
    if format_type == 'csv':
        return df.to_csv(index=False)
    elif format_type == 'excel':
        try:
            output = io.BytesIO()
            # Use the output buffer directly
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Spore_Measurements', index=False)

                # Add statistics sheet
                stats_data = {
                    'Metric': [
                        'Count', 'Mean Length (μm)', 'Std Length (μm)',
                        'Mean Width (μm)', 'Std Width (μm)', 'Mean Area (μm²)',
                        'Mean Aspect Ratio'
                    ],
                    'Value': [
                        len(df), df['Length_um'].mean(), df['Length_um'].std(),
                        df['Width_um'].mean(), df['Width_um'].std(),
                        df['Area_um2'].mean(), df['Aspect_Ratio'].mean()
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)

            output.seek(0)  # Reset buffer position
            return output.getvalue()
        except (ImportError, Exception):
            # Fallback to CSV if openpyxl not available or other Excel errors
            return df.to_csv(index=False)
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
        warnings.append(
            "Image resolution is quite low. Consider using higher resolution images for better accuracy."
        )

    if image_array.shape[0] > 5000 or image_array.shape[1] > 5000:
        warnings.append(
            "Very large image detected. Processing may take longer.")

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
        warnings.append(
            "Image appears to have very low contrast. Consider adjusting image settings."
        )

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
        'blur_kernel':
        blur_kernel,
        'threshold_method':
        threshold_method,
        'notes':
        f"Image statistics - Mean: {mean_intensity:.1f}, Std: {std_intensity:.1f}"
    }

    return recommendations
