import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
import zipfile
from pathlib import Path
import datetime
from spore_analyzer import SporeAnalyzer
from utils import calculate_statistics, create_overlay_image, export_results

class BatchProcessor:
    """Handle batch processing of multiple microscopy images"""
    
    def __init__(self):
        self.results = []
        self.processed_images = []
        self.processing_stats = {
            'total_images': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_spores_detected': 0,
            'processing_time': 0
        }
    
    def process_images(self, uploaded_files, analyzer_params, progress_callback=None):
        """Process multiple images with the same analyzer parameters"""
        self.results = []
        self.processed_images = []
        
        analyzer = SporeAnalyzer()
        analyzer.set_parameters(**analyzer_params)
        
        self.processing_stats['total_images'] = len(uploaded_files)
        self.processing_stats['successful_analyses'] = 0
        self.processing_stats['failed_analyses'] = 0
        self.processing_stats['total_spores_detected'] = 0
        
        start_time = datetime.datetime.now()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            if progress_callback:
                progress_callback(idx + 1, len(uploaded_files), uploaded_file.name)
            
            try:
                # Load image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Analyze image
                spore_results = analyzer.analyze_image(image_array)
                
                if spore_results:
                    # Store results with metadata
                    image_result = {
                        'filename': uploaded_file.name,
                        'image_array': image_array,
                        'spore_results': spore_results,
                        'spore_count': len(spore_results),
                        'status': 'success',
                        'error': None
                    }
                    
                    self.processing_stats['successful_analyses'] += 1
                    self.processing_stats['total_spores_detected'] += len(spore_results)
                else:
                    # No spores detected
                    image_result = {
                        'filename': uploaded_file.name,
                        'image_array': image_array,
                        'spore_results': [],
                        'spore_count': 0,
                        'status': 'no_spores',
                        'error': 'No spores detected'
                    }
                    self.processing_stats['failed_analyses'] += 1
                
                self.results.append(image_result)
                self.processed_images.append(uploaded_file.name)
                
            except Exception as e:
                # Processing error
                image_result = {
                    'filename': uploaded_file.name,
                    'image_array': None,
                    'spore_results': [],
                    'spore_count': 0,
                    'status': 'error',
                    'error': str(e)
                }
                self.results.append(image_result)
                self.processing_stats['failed_analyses'] += 1
        
        end_time = datetime.datetime.now()
        self.processing_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        return self.results
    
    def get_batch_statistics(self):
        """Calculate comprehensive statistics across all processed images"""
        if not self.results:
            return None
        
        # Collect all spore data
        all_spores = []
        image_stats = []
        
        for result in self.results:
            if result['status'] == 'success' and result['spore_results']:
                # Image-level statistics
                spores = result['spore_results']
                lengths = [s['length_um'] for s in spores]
                widths = [s['width_um'] for s in spores]
                areas = [s['area_um2'] for s in spores]
                
                image_stat = {
                    'filename': result['filename'],
                    'spore_count': len(spores),
                    'mean_length': np.mean(lengths),
                    'mean_width': np.mean(widths),
                    'mean_area': np.mean(areas),
                    'std_length': np.std(lengths),
                    'std_width': np.std(widths),
                    'std_area': np.std(areas)
                }
                image_stats.append(image_stat)
                all_spores.extend(spores)
        
        if not all_spores:
            return None
        
        # Overall statistics
        overall_stats = calculate_statistics(all_spores)
        
        # Per-image statistics
        image_stats_df = pd.DataFrame(image_stats)
        
        return {
            'overall_stats': overall_stats,
            'image_stats': image_stats_df,
            'all_spores': all_spores,
            'processing_stats': self.processing_stats
        }
    
    def create_batch_report(self, batch_stats):
        """Create a comprehensive batch processing report"""
        if not batch_stats:
            return "No valid results to report."
        
        overall = batch_stats['overall_stats']
        processing = batch_stats['processing_stats']
        
        report = f"""
BATCH SPORE ANALYSIS REPORT
===========================
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROCESSING SUMMARY:
- Total Images Processed: {processing['total_images']}
- Successful Analyses: {processing['successful_analyses']}
- Failed Analyses: {processing['failed_analyses']}
- Total Spores Detected: {processing['total_spores_detected']}
- Processing Time: {processing['processing_time']:.2f} seconds
- Average per Image: {processing['processing_time']/processing['total_images']:.2f} seconds

OVERALL SPORE STATISTICS:
- Total Spores Analyzed: {overall['count']}
- Mean Length: {overall['length_mean']:.2f} ± {overall['length_std']:.2f} μm
- Mean Width: {overall['width_mean']:.2f} ± {overall['width_std']:.2f} μm
- Mean Area: {overall['area_mean']:.2f} ± {overall['area_std']:.2f} μm²
- Mean Aspect Ratio: {overall['aspect_ratio_mean']:.2f} ± {overall['aspect_ratio_std']:.2f}
- Mean Circularity: {overall['circularity_mean']:.3f} ± {overall['circularity_std']:.3f}

RANGE STATISTICS:
- Length Range: {overall['length_min']:.2f} - {overall['length_max']:.2f} μm
- Width Range: {overall['width_min']:.2f} - {overall['width_max']:.2f} μm
- Area Range: {overall['area_min']:.2f} - {overall['area_max']:.2f} μm²

IMAGE-WISE BREAKDOWN:
"""
        
        # Add per-image summary
        for _, row in batch_stats['image_stats'].iterrows():
            report += f"- {row['filename']}: {row['spore_count']} spores, "
            report += f"avg length {row['mean_length']:.1f}μm, avg width {row['mean_width']:.1f}μm\n"
        
        return report
    
    def export_batch_results(self, batch_stats, format_type='excel'):
        """Export comprehensive batch results to various formats"""
        if not batch_stats:
            return None
        
        if format_type == 'excel':
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Overall statistics sheet
                overall_df = pd.DataFrame([batch_stats['overall_stats']])
                overall_df.to_excel(writer, sheet_name='Overall_Statistics', index=False)
                
                # Image-wise statistics
                batch_stats['image_stats'].to_excel(writer, sheet_name='Image_Statistics', index=False)
                
                # Individual spore measurements
                all_spores_df = pd.DataFrame([{
                    'Image': 'Combined',  # We'll update this below
                    'Spore_ID': i+1,
                    'Length_um': spore['length_um'],
                    'Width_um': spore['width_um'],
                    'Area_um2': spore['area_um2'],
                    'Aspect_Ratio': spore['aspect_ratio'],
                    'Circularity': spore['circularity']
                } for i, spore in enumerate(batch_stats['all_spores'])])
                
                # Add image filename to each spore measurement
                spore_idx = 0
                for result in self.results:
                    if result['status'] == 'success' and result['spore_results']:
                        count = len(result['spore_results'])
                        all_spores_df.loc[spore_idx:spore_idx+count-1, 'Image'] = result['filename']
                        spore_idx += count
                
                all_spores_df.to_excel(writer, sheet_name='All_Spore_Measurements', index=False)
                
                # Processing summary
                processing_df = pd.DataFrame([batch_stats['processing_stats']])
                processing_df.to_excel(writer, sheet_name='Processing_Summary', index=False)
            
            output.seek(0)
            return output.getvalue()
        
        elif format_type == 'csv':
            # Return combined CSV of all measurements
            all_spores_df = pd.DataFrame([{
                'Image': 'Combined',
                'Spore_ID': i+1,
                'Length_um': spore['length_um'],
                'Width_um': spore['width_um'],
                'Area_um2': spore['area_um2'],
                'Aspect_Ratio': spore['aspect_ratio'],
                'Circularity': spore['circularity']
            } for i, spore in enumerate(batch_stats['all_spores'])])
            
            # Add image filename to each measurement
            spore_idx = 0
            for result in self.results:
                if result['status'] == 'success' and result['spore_results']:
                    count = len(result['spore_results'])
                    all_spores_df.loc[spore_idx:spore_idx+count-1, 'Image'] = result['filename']
                    spore_idx += count
            
            return all_spores_df.to_csv(index=False)
        
        return None
    
    def create_batch_overlay_images(self, pixel_scale):
        """Create overlay images for all successfully processed images"""
        overlay_images = []
        
        for result in self.results:
            if result['status'] == 'success' and result['spore_results']:
                # Create overlay for all spores (all selected)
                selected_spores = set(range(len(result['spore_results'])))
                
                overlay = create_overlay_image(
                    result['image_array'],
                    result['spore_results'],
                    selected_spores,
                    pixel_scale
                )
                
                overlay_images.append({
                    'filename': result['filename'],
                    'overlay': overlay
                })
        
        return overlay_images
    
    def export_overlay_zip(self, overlay_images):
        """Export all overlay images as a ZIP file"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for overlay_data in overlay_images:
                # Convert overlay to PNG bytes
                img_buffer = io.BytesIO()
                overlay_pil = Image.fromarray(overlay_data['overlay'])
                overlay_pil.save(img_buffer, format='PNG')
                
                # Add to ZIP with modified filename
                base_name = Path(overlay_data['filename']).stem
                zip_filename = f"{base_name}_overlay.png"
                zip_file.writestr(zip_filename, img_buffer.getvalue())
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def get_processing_summary(self):
        """Get a summary of the batch processing results"""
        if not self.results:
            return None
        
        successful = sum(1 for r in self.results if r['status'] == 'success')
        no_spores = sum(1 for r in self.results if r['status'] == 'no_spores')
        errors = sum(1 for r in self.results if r['status'] == 'error')
        total_spores = sum(r['spore_count'] for r in self.results)
        
        return {
            'total_images': len(self.results),
            'successful': successful,
            'no_spores': no_spores,
            'errors': errors,
            'total_spores': total_spores,
            'success_rate': successful / len(self.results) * 100 if self.results else 0
        }