import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2
from PIL import Image
import io
import base64

# Make folium optional
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Warning: folium or its dependencies not found. Interactive maps will not be available.")

class ChangeVisualization:
    """
    Class for visualizing environmental changes detected in satellite images.
    """
    def __init__(self, output_dir='./output'):
        """
        Initialize the visualization module.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_image_pair(self, img1, img2, titles=None, figsize=(10, 5)):
        """
        Plot a pair of images side by side.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            titles (list, optional): Titles for the images
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].imshow(img1)
        axes[0].set_title(titles[0] if titles else 'Image 1')
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title(titles[1] if titles else 'Image 2')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_change_detection(self, img1, img2, change_mask, change_prob=None, 
                             titles=None, figsize=(15, 10)):
        """
        Plot change detection results.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            change_mask (numpy.ndarray): Binary change mask
            change_prob (numpy.ndarray, optional): Change probability map
            titles (list, optional): Titles for the plots
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if change_prob is None:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        else:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(2, 3, figure=fig)
            axes = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[0, 2]),
                fig.add_subplot(gs[1, :]),
            ]
        
        # Plot original images
        axes[0].imshow(img1)
        axes[0].set_title(titles[0] if titles else 'Before')
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title(titles[1] if titles else 'After')
        axes[1].axis('off')
        
        # Plot change mask
        axes[2].imshow(change_mask, cmap='gray')
        axes[2].set_title(titles[2] if titles else 'Change Mask')
        axes[2].axis('off')
        
        # Plot overlay
        overlay = self.create_change_overlay(img2, change_mask)
        axes[3].imshow(overlay)
        axes[3].set_title(titles[3] if titles else 'Changes Highlighted')
        axes[3].axis('off')
        
        # Add legend for overlay
        red_patch = mpatches.Patch(color='red', label='Changed Areas')
        axes[3].legend(handles=[red_patch], loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def create_change_overlay(self, img, mask, color=[255, 0, 0], alpha=0.5):
        """
        Create an overlay of changes on the image.
        
        Args:
            img (numpy.ndarray): Base image
            mask (numpy.ndarray): Binary mask of changes
            color (list): RGB color for highlighting changes
            alpha (float): Transparency of the overlay
            
        Returns:
            numpy.ndarray: Image with changes highlighted
        """
        # Create a copy of the image
        overlay = img.copy()
        
        # Create a color mask
        color_mask = np.zeros_like(img)
        for i in range(3):
            color_mask[:, :, i] = color[i]
        
        # Apply the mask with alpha blending
        mask_3d = np.stack([mask] * 3, axis=2)
        overlay = np.where(mask_3d, 
                          (1 - alpha) * overlay + alpha * color_mask, 
                          overlay)
        
        # Ensure values are in valid range
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay
    
    def create_change_heatmap(self, change_prob, cmap='jet'):
        """
        Create a heatmap visualization of change probabilities.
        
        Args:
            change_prob (numpy.ndarray): Change probability map
            cmap (str): Colormap name
            
        Returns:
            numpy.ndarray: Heatmap image
        """
        # Normalize probabilities to 0-1 range
        if change_prob.min() < 0 or change_prob.max() > 1:
            change_prob = (change_prob - change_prob.min()) / (change_prob.max() - change_prob.min())
        
        # Apply colormap
        cmap = plt.get_cmap(cmap)
        heatmap = cmap(change_prob)
        
        # Convert to uint8
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        
        return heatmap
    
    def blend_heatmap(self, img, heatmap, alpha=0.7):
        """
        Blend a heatmap with an image.
        
        Args:
            img (numpy.ndarray): Base image
            heatmap (numpy.ndarray): Heatmap image
            alpha (float): Blending factor
            
        Returns:
            numpy.ndarray: Blended image
        """
        # Resize heatmap to match image size if needed
        if img.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Blend images
        blended = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        
        return blended
    
    def create_before_after_slider(self, img1, img2, change_mask=None, output_path=None):
        """
        Create an HTML file with a before/after slider.
        
        Args:
            img1 (numpy.ndarray): Before image
            img2 (numpy.ndarray): After image
            change_mask (numpy.ndarray, optional): Binary change mask
            output_path (str, optional): Path to save the HTML file
            
        Returns:
            str: Path to the HTML file
        """
        # Convert images to PIL format
        img1_pil = Image.fromarray(img1)
        img2_pil = Image.fromarray(img2)
        
        # Create overlay if mask is provided
        if change_mask is not None:
            img2_overlay = self.create_change_overlay(img2, change_mask)
            img2_pil = Image.fromarray(img2_overlay)
        
        # Save images to temporary files
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'before_after_slider.html')
        
        img1_path = os.path.join(self.output_dir, 'before.jpg')
        img2_path = os.path.join(self.output_dir, 'after.jpg')
        
        img1_pil.save(img1_path)
        img2_pil.save(img2_path)
        
        # Create HTML with slider
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Before/After Slider</title>
            <style>
                .img-comp-container {{
                    position: relative;
                    height: 600px; /* Set this to your image height */
                }}
                .img-comp-img {{
                    position: absolute;
                    width: 100%;
                    height: auto;
                    overflow: hidden;
                }}
                .img-comp-img img {{
                    display: block;
                    vertical-align: middle;
                    max-height: 600px;
                    max-width: 100%;
                }}
                .img-comp-slider {{
                    position: absolute;
                    z-index: 9;
                    cursor: ew-resize;
                    width: 40px;
                    height: 40px;
                    background-color: #2196F3;
                    opacity: 0.7;
                    border-radius: 50%;
                }}
                .img-comp-slider:hover {{
                    opacity: 1;
                }}
            </style>
        </head>
        <body>
            <h1>Environmental Change Detection</h1>
            <div class="img-comp-container">
                <div class="img-comp-img">
                    <img src="after.jpg" width="100%">
                </div>
                <div class="img-comp-img">
                    <img src="before.jpg" width="100%">
                </div>
            </div>
            <div style="margin-top: 20px;">
                <p><strong>Instructions:</strong> Drag the slider to compare before and after images.</p>
            </div>

            <script>
                function initComparisons() {{
                    var x, i;
                    x = document.getElementsByClassName("img-comp-overlay");
                    for (i = 0; i < x.length; i++) {{
                        compareImages(x[i]);
                    }}
                    function compareImages(img) {{
                        var slider, img, clicked = 0, w, h;
                        w = img.offsetWidth;
                        h = img.offsetHeight;
                        img.style.width = (w / 2) + "px";
                        slider = document.createElement("DIV");
                        slider.setAttribute("class", "img-comp-slider");
                        img.parentElement.insertBefore(slider, img);
                        slider.style.top = (h / 2) - (slider.offsetHeight / 2) + "px";
                        slider.style.left = (w / 2) - (slider.offsetWidth / 2) + "px";
                        slider.addEventListener("mousedown", slideReady);
                        window.addEventListener("mouseup", slideFinish);
                        slider.addEventListener("touchstart", slideReady);
                        window.addEventListener("touchend", slideFinish);
                        function slideReady(e) {{
                            e.preventDefault();
                            clicked = 1;
                            window.addEventListener("mousemove", slideMove);
                            window.addEventListener("touchmove", slideMove);
                        }}
                        function slideFinish() {{
                            clicked = 0;
                        }}
                        function slideMove(e) {{
                            var pos;
                            if (clicked == 0) return false;
                            pos = getCursorPos(e)
                            if (pos < 0) pos = 0;
                            if (pos > w) pos = w;
                            slide(pos);
                        }}
                        function getCursorPos(e) {{
                            var a, x = 0;
                            e = e || window.event;
                            a = img.getBoundingClientRect();
                            x = e.pageX - a.left;
                            x = x - window.pageXOffset;
                            return x;
                        }}
                        function slide(x) {{
                            img.style.width = x + "px";
                            slider.style.left = img.offsetWidth - (slider.offsetWidth / 2) + "px";
                        }}
                    }}
                }}

                // Initialize comparison on page load
                window.onload = function() {{
                    // Set the initial width of the overlay to 50%
                    var imgContainers = document.getElementsByClassName("img-comp-container");
                    for (var i = 0; i < imgContainers.length; i++) {{
                        var images = imgContainers[i].getElementsByClassName("img-comp-img");
                        if (images.length > 1) {{
                            images[1].classList.add("img-comp-overlay");
                        }}
                    }}
                    initComparisons();
                }};
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def create_interactive_map(self, img1, img2, change_mask, geo_bounds=None, output_path=None):
        """
        Create an interactive map with change detection results.
        If folium is not available, creates a static visualization instead.
        
        Args:
            img1 (numpy.ndarray): Before image
            img2 (numpy.ndarray): After image
            change_mask (numpy.ndarray): Binary change mask
            geo_bounds (list, optional): Geographic bounds [min_lat, min_lon, max_lat, max_lon]
            output_path (str, optional): Path to save the HTML file
            
        Returns:
            str: Path to the HTML file or image
        """
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'interactive_map.html')
            
        # Check if folium is available
        if not FOLIUM_AVAILABLE:
            print("Folium not available. Creating static visualization instead.")
            # Create a static visualization as fallback
            fig = self.plot_change_detection(img1, img2, change_mask, 
                                           titles=['Before', 'After', 'Changes Detected', 'Changes Highlighted'])
            
            # Change extension to png for static image
            output_path = output_path.replace('.html', '.png')
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            return output_path
            
        # Default geo bounds if not provided (this is just a placeholder)
        if geo_bounds is None:
            # Default to Dubai coordinates as an example
            geo_bounds = [25.0, 55.0, 25.3, 55.3]  # [min_lat, min_lon, max_lat, max_lon]
        
        # Create a folium map centered on the area
        center_lat = (geo_bounds[0] + geo_bounds[2]) / 2
        center_lon = (geo_bounds[1] + geo_bounds[3]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Create overlay image with changes highlighted
        overlay = self.create_change_overlay(img2, change_mask)
        
        # Convert images to base64 for embedding in HTML
        img1_base64 = self._image_to_base64(img1)
        img2_base64 = self._image_to_base64(img2)
        overlay_base64 = self._image_to_base64(overlay)
        
        # Add image overlays to the map
        bounds = [[geo_bounds[0], geo_bounds[1]], [geo_bounds[2], geo_bounds[3]]]
        
        # Add a layer control panel
        folium.raster_layers.ImageOverlay(
            image=img1_base64,
            bounds=bounds,
            name='Before',
            opacity=0.7
        ).add_to(m)
        
        folium.raster_layers.ImageOverlay(
            image=img2_base64,
            bounds=bounds,
            name='After',
            opacity=0.7
        ).add_to(m)
        
        folium.raster_layers.ImageOverlay(
            image=overlay_base64,
            bounds=bounds,
            name='Changes Detected',
            opacity=0.7
        ).add_to(m)
        
        # Add heatmap of changes
        points = []
        rows, cols = change_mask.shape
        for i in range(rows):
            for j in range(cols):
                if change_mask[i, j] > 0:
                    # Convert pixel coordinates to geo coordinates
                    lat = geo_bounds[0] + (geo_bounds[2] - geo_bounds[0]) * i / rows
                    lon = geo_bounds[1] + (geo_bounds[3] - geo_bounds[1]) * j / cols
                    points.append([lat, lon])
        
        # Add heatmap if there are changes
        if points:
            HeatMap(points, name='Change Heatmap').add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save the map
        m.save(output_path)
        
        return output_path
    
    def _image_to_base64(self, img):
        """
        Convert an image to base64 encoding.
        
        Args:
            img (numpy.ndarray): Image to convert
            
        Returns:
            str: Base64 encoded image
        """
        # Convert to PIL image
        img_pil = Image.fromarray(img)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f'data:image/png;base64,{img_base64}'
    
    def save_visualization(self, fig, filename):
        """
        Save a matplotlib figure to the output directory.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to save
            filename (str): The filename to save as
            
        Returns:
            str: The path to the saved file
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return output_path

# Example usage
def main():
    # Create sample data
    img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    change_mask = np.zeros((256, 256), dtype=np.uint8)
    change_mask[100:150, 100:150] = 1  # Simulate a change region
    
    # Create visualization
    vis = ChangeVisualization(output_dir='./output')
    
    # Plot image pair
    fig1 = vis.plot_image_pair(img1, img2, titles=['Before', 'After'])
    vis.save_visualization(fig1, 'image_pair.png')
    
    # Plot change detection
    fig2 = vis.plot_change_detection(img1, img2, change_mask)
    vis.save_visualization(fig2, 'change_detection.png')
    
    # Create before/after slider
    slider_path = vis.create_before_after_slider(img1, img2, change_mask)
    print(f"Before/after slider saved to: {slider_path}")
    
    # Create interactive map
    map_path = vis.create_interactive_map(img1, img2, change_mask)
    print(f"Interactive map saved to: {map_path}")

if __name__ == "__main__":
    main()