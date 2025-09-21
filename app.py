import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import streamlit as st
import torch
import io
import base64
from datetime import datetime
import google.generativeai as genai

from terratrack import ChangeDetector, load_image_pair, get_dates, EnvironmentalChangeAnalyzer
from visualization import ChangeVisualization

# Constants
import os
DATASET_PATH = os.environ.get("TERRATRACK_DATASET", "./data/images")
LABELS_PATH = os.environ.get("TERRATRACK_LABELS", "./data/labels")
OUTPUT_DIR = os.environ.get("TERRATRACK_OUTPUT", "./output")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBYJYjMg5Bi8I8AdJBfpSkc3ZJPPIf-OQc")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set page config
st.set_page_config(page_title="TerraTrack - Environmental Change Detection", 
                  page_icon="🌍", 
                  layout="wide")

# Initialize model
@st.cache_resource
def load_model():
    return ChangeDetector(use_mock=True)

# Initialize visualization
@st.cache_resource
def load_visualization():
    return ChangeVisualization(output_dir=OUTPUT_DIR)

# Initialize analyzer
@st.cache_resource
def load_analyzer(_detector):
    return EnvironmentalChangeAnalyzer(_detector)

# Get available locations
def get_available_locations():
    locations = []
    
    # If dataset path doesn't exist, return a list with a demo location
    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        print(f"Warning: Dataset path {DATASET_PATH} does not exist")
        # Return a demo location for testing
        return ["Demo Location"]
    
    try:            
        for location in os.listdir(DATASET_PATH):
            try:
                loc_path = os.path.join(DATASET_PATH, location)
                if os.path.isdir(loc_path) and location not in ['_pycache_']:
                    # Check if pair folder exists and contains image files
                    pair_path = os.path.join(loc_path, 'pair')
                    if os.path.exists(pair_path) and any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')) 
                                                        for f in os.listdir(pair_path)):
                        locations.append(location)
            except Exception as e:
                print(f"Warning: Error processing location {location}: {str(e)}")
                continue
    except Exception as e:
        print(f"Warning: Error accessing dataset path: {str(e)}")
    
    # If no locations found, return a demo location
    if not locations:
        return ["Demo Location"]
    
    return sorted(locations)

# Load images for a location
def load_location_images(location):
    img1, img2 = load_image_pair(location)
    date1, date2 = get_dates(location)
    
    # Handle case when dates are None
    if date1 is None:
        date1 = "Unknown Date"
    if date2 is None:
        date2 = "Unknown Date"
        
    return img1, img2, date1, date2

# Convert image to base64 for display
def image_to_base64(img):
    img_pil = Image.fromarray(img)
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{img_base64}'

# Analyze images using Gemini
def analyze_images_with_gemini(img1, img2, date1, date2):
    """Analyze satellite images using Gemini AI
    
    Args:
        img1: Before image as numpy array
        img2: After image as numpy array
        date1: Date of before image
        date2: Date of after image
        
    Returns:
        str: Analysis text from Gemini AI
    """
    # Convert images to PIL format for Gemini
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
    
    # Save images to bytes
    img1_bytes = io.BytesIO()
    img2_bytes = io.BytesIO()
    img1_pil.save(img1_bytes, format='PNG')
    img2_pil.save(img2_bytes, format='PNG')
    img1_bytes.seek(0)
    img2_bytes.seek(0)
    
    # Create Gemini model
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    # Create prompt with dates
    prompt = f"""Analyze these two satellite images taken on {date1} and {date2} respectively.
    
    1. Describe what you see in the first (before) image in detail.
    2. Describe what you see in the second (after) image in detail.
    3. Identify and explain the key changes between the two images.
    4. Analyze the potential environmental impact of these changes.
    5. Suggest possible causes for these changes.
    6. Quantify the changes if possible (approximate percentage of area changed).
    7. Provide recommendations for environmental management based on these changes.
    
    Format your response with clear headings and bullet points for readability.
    """
    
    try:
        # Generate content with images
        response = model.generate_content(
            [
                prompt,
                {"mime_type": "image/png", "data": img1_bytes.getvalue()},
                {"mime_type": "image/png", "data": img2_bytes.getvalue()}
            ]
        )
        
        analysis = response.text
    except Exception as e:
        analysis = f"Error analyzing images with Gemini: {str(e)}\n\nThis could be due to API key issues, network connectivity, or image format problems. Please check your API key and try again."
    
    return analysis

# Main app
def main():
    # Load resources
    detector = load_model()
    visualizer = load_visualization()
    analyzer = load_analyzer(detector)
    
    # App title
    st.title("🌍 TerraTrack - Environmental Change Detection")
    st.markdown("""Detect and analyze environmental changes from satellite imagery using AI.""")
    
    # Sidebar
    st.sidebar.title("Options")
    
    # Upload Images section
    st.sidebar.subheader("Upload Satellite Images")
    
    # Upload images
    uploaded_img1 = st.sidebar.file_uploader("Upload Before Image", type=["png", "jpg", "jpeg"])
    uploaded_img2 = st.sidebar.file_uploader("Upload After Image", type=["png", "jpg", "jpeg"])
    
    # Input dates
    col1, col2 = st.sidebar.columns(2)
    with col1:
        date1 = st.date_input("Before Date")
    with col2:
        date2 = st.date_input("After Date")
    
    # Format dates
    date1_formatted = date1.strftime("%Y-%m-%d")
    date2_formatted = date2.strftime("%Y-%m-%d")
    
    # Check if images are uploaded
    if uploaded_img1 is not None and uploaded_img2 is not None:
        # Read images
        img1 = np.array(Image.open(uploaded_img1).convert('RGB'))
        img2 = np.array(Image.open(uploaded_img2).convert('RGB'))
        
        # Resize images if they are too large
        max_size = 1024
        if img1.shape[0] > max_size or img1.shape[1] > max_size:
            scale = max_size / max(img1.shape[0], img1.shape[1])
            img1 = cv2.resize(img1, (int(img1.shape[1] * scale), int(img1.shape[0] * scale)))
        
        if img2.shape[0] > max_size or img2.shape[1] > max_size:
            scale = max_size / max(img2.shape[0], img2.shape[1])
            img2 = cv2.resize(img2, (int(img2.shape[1] * scale), int(img2.shape[0] * scale)))
        
        # Display images
        st.subheader("Uploaded Satellite Images")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img1, caption=f"Before ({date1_formatted})")
        
        with col2:
            st.image(img2, caption=f"After ({date2_formatted})")
        
        selected_location = "uploaded"
    else:
        st.info("Please upload both before and after images.")
        return
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    threshold = st.sidebar.slider("Change Detection Threshold", 0.0, 1.0, 0.5, 0.05)
    use_gemini = st.sidebar.checkbox("Use AI for detailed analysis", value=True)
    
    # Run analysis button
    if st.sidebar.button("Run Analysis"):
        with st.spinner("Analyzing changes..."):
            # Analyze changes
            results = analyzer.analyze_changes(img1, img2, threshold)
            
            # Get Gemini analysis if enabled
            if use_gemini:
                with st.spinner("Getting AI analysis"):
                    gemini_analysis = analyze_images_with_gemini(img1, img2, date1_formatted, date2_formatted)
            
            # Display results
            st.subheader("Change Detection Results")
            
            # Create tabs for results - add AI Analysis tab if Gemini is enabled
            if use_gemini:
                tab1, tab2, tab3, tab4 = st.tabs(["Change Map", "AI Analysis", "Statistics", "Report"])
            else:
                tab1, tab2, tab3 = st.tabs(["Change Map", "Statistics",  "Report"])
            
            with tab1:
                # Display change map without subheader
                
                # Create a more visually appealing layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # Apply a colormap to make the probability map more visually informative
                    prob_map = results['change_probability'].copy()
                    # Convert to heatmap using cv2's COLORMAP_JET
                    prob_map_colored = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    # Convert back to RGB for display
                    prob_map_colored = cv2.cvtColor(prob_map_colored, cv2.COLOR_BGR2RGB)
                    st.image(prob_map_colored, caption="Change Probability Heatmap", use_container_width=True)
                
                with col2:
                    # Enhance the visualization with better highlighting
                    vis_enhanced = results['visualization'].copy()
                    # Make changes more prominent with a brighter color
                    st.image(vis_enhanced, caption="Changes Highlighted", use_container_width=True)
                    
                # Add a description
                st.markdown("""The *Change Probability Heatmap* shows the likelihood of change at each pixel, 
                with warmer colors (red, yellow) indicating higher probability of change and cooler colors (blue, green) 
                indicating lower probability. The *Changes Highlighted* image shows the detected changes overlaid on the original image.""")
            
            # AI Analysis tab (if Gemini is enabled)
            if use_gemini:
                with tab2:
                    st.subheader("AI Analysis ")
                    st.markdown(gemini_analysis)
                    
                with tab3:
                    # Display statistics
                    st.subheader("Change Statistics")
                    
                    # Use a more visually appealing layout for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Changed Area", f"{results['change_percentage']:.2f}%")
                    
                    with col2:
                        st.metric("Change Regions", results['num_change_regions'])
                    
                    with col3:
                        # Calculate average region size
                        if results['num_change_regions'] > 0:
                            avg_size = results['change_mask'].sum() / results['num_change_regions']
                            st.metric("Avg Region Size", f"{avg_size:.1f} pixels")
                        else:
                            st.metric("Avg Region Size", "0 pixels")
                    
                    # Display change types with better formatting
                    if results['num_change_regions'] > 0:
                        st.subheader("Detected Change Types")
                        for i, change_type in enumerate(results['change_types']):
                            st.markdown(f"*Region {i+1}*: {change_type}")
            else:
                with tab2:
                    # Display statistics
                    st.subheader("Change Statistics")
                    
                    # Use a more visually appealing layout for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Changed Area", f"{results['change_percentage']:.2f}%")
                    
                    with col2:
                        st.metric("Change Regions", results['num_change_regions'])
                    
                    with col3:
                        # Calculate average region size
                        if results['num_change_regions'] > 0:
                            avg_size = results['change_mask'].sum() / results['num_change_regions']
                            st.metric("Avg Region Size", f"{avg_size:.1f} pixels")
                        else:
                            st.metric("Avg Region Size", "0 pixels")
                    
                    # Display change types with better formatting
                    if results['num_change_regions'] > 0:
                        st.subheader("Detected Change Types")
                        for i, change_type in enumerate(results['change_types']):
                            st.markdown(f"*Region {i+1}*: {change_type}")
            
            # Adjust tab indices based on whether Gemini is enabled
            slider_tab = tab3 if use_gemini else tab2
            report_tab = tab4 if use_gemini else tab3
            
            
            with report_tab:
                # Generate report
                report = analyzer.generate_report("Custom Area", date1_formatted, date2_formatted, results)
                
                # If Gemini is enabled, include its analysis in the report
                if use_gemini:
                    report += "\n\n" + "=" * 50 + "\n\n"
                    report += "AI ANALYSIS \n"
                    report += "=" * 50 + "\n\n"
                    report += gemini_analysis
                
                # Display report
                st.text_area("Environmental Change Analysis Report", report, height=400)
                
                # Provide download link
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"change_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            # Save visualization
            fig = visualizer.plot_change_detection(
                img1, img2, results['change_mask'], results['change_probability'],
                titles=[f"Before ({date1_formatted})", f"After ({date2_formatted})", 
                        "Change Mask", "Changes Highlighted"]
            )
            
            vis_path = visualizer.save_visualization(
                fig, f"{selected_location}changes{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            
            # Provide download link for visualization
            with open(vis_path, 'rb') as f:
                vis_data = f.read()
            
            st.sidebar.download_button(
                label="Download Visualization",
                data=vis_data,
                file_name=os.path.basename(vis_path),
                mime="image/png"
            )

# Run the app
if __name__ == "__main__":
    main()