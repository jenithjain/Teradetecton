import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
# Make rasterio optional
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio module not found. Some geospatial functionality will be limited.")
import warnings
warnings.filterwarnings('ignore')

# Constants
import os
DATASET_PATH = os.environ.get("TERRATRACK_DATASET", "./data/images")
LABELS_PATH = os.environ.get("TERRATRACK_LABELS", "./data/labels")

# Utility functions
def load_image_pair(location, use_rectified=True):
    """
    Load a pair of satellite images for a specific location.
    
    Args:
        location (str): Name of the location (e.g., 'dubai', 'paris')
        use_rectified (bool): Whether to use rectified images
        
    Returns:
        tuple: (image1, image2) as numpy arrays
    """
    # Check if dataset path exists
    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        print(f"Warning: Dataset path {DATASET_PATH} does not exist")
        # Return placeholder images for demonstration
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = np.zeros((256, 256, 3), dtype=np.uint8)
        # Add some text to the images
        cv2.putText(img1, "No Dataset", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img2, "Available", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img1, img2
        
    # Check if location exists
    if not os.path.exists(os.path.join(DATASET_PATH, location)):
        print(f"Warning: Location {location} not found in dataset")
        # Return placeholder images for demonstration
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = np.zeros((256, 256, 3), dtype=np.uint8)
        # Add some text to the images
        cv2.putText(img1, f"Location", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img2, f"{location} not found", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img1, img2
    
    if use_rectified:
        img1_path = os.path.join(DATASET_PATH, location, 'pair', 'img1.png')
        img2_path = os.path.join(DATASET_PATH, location, 'pair', 'img2.png')
    else:
        # For multispectral data, we would need to handle the 13 bands
        # This is simplified for the RGB case
        img1_path = os.path.join(DATASET_PATH, location, 'pair', 'img1.png')
        img2_path = os.path.join(DATASET_PATH, location, 'pair', 'img2.png')
    
    # Check if files exist
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Warning: Images for {location} not found at {img1_path} or {img2_path}")
        # Return placeholder images for demonstration
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = np.zeros((256, 256, 3), dtype=np.uint8)
        # Add some text to the images
        cv2.putText(img1, "Images", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img2, "Not Found", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img1, img2
    
    # Load images - try with rasterio first if available, then fall back to OpenCV
    if RASTERIO_AVAILABLE and img1_path.endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(img1_path) as src:
                img1 = src.read().transpose(1, 2, 0)  # CxHxW to HxWxC
            with rasterio.open(img2_path) as src:
                img2 = src.read().transpose(1, 2, 0)  # CxHxW to HxWxC
                
            # Convert to RGB if needed
            if img1.shape[2] > 3:
                img1 = img1[:, :, :3]
            if img2.shape[2] > 3:
                img2 = img2[:, :, :3]
        except Exception as e:
            print(f"Warning: Error reading with rasterio: {str(e)}. Falling back to OpenCV.")
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"Warning: Failed to load images for {location}")
                return None, None
                
            # Convert from BGR to RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    else:
        # Use OpenCV
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Warning: Failed to load images for {location}")
            return None, None
        
        # Convert from BGR to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    return img1, img2

def load_change_mask(location):
    """
    Load the change mask for a specific location.
    
    Args:
        location (str): Name of the location (e.g., 'dubai', 'paris')
        
    Returns:
        numpy.ndarray: Change mask as a binary image
    """
    mask_path = os.path.join(LABELS_PATH, location, 'cm', 'cm.png')
    
    if not os.path.exists(mask_path):
        print(f"Warning: Change mask for {location} not found at {mask_path}")
        return None
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Warning: Failed to load change mask for {location}")
        return None
    
    # Convert to binary (0 for no change, 1 for change)
    mask = (mask > 0).astype(np.uint8)
    
    return mask

def get_dates(location):
    """
    Get the acquisition dates for a pair of images.
    
    Args:
        location (str): Name of the location (e.g., 'dubai', 'paris')
        
    Returns:
        tuple: (date1, date2) as strings
    """
    # Check if dataset path exists
    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        print(f"Warning: Dataset path {DATASET_PATH} does not exist")
        return None, None
        
    # Check if location exists
    if not os.path.exists(os.path.join(DATASET_PATH, location)):
        print(f"Warning: Location {location} not found in dataset")
        return None, None
    
    dates_path = os.path.join(DATASET_PATH, location, 'dates.txt')
    
    if not os.path.exists(dates_path):
        print(f"Warning: Dates file for {location} not found at {dates_path}")
        return None, None
    
    with open(dates_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print(f"Warning: Dates file for {location} has incorrect format")
        return None, None
    
    date1 = lines[0].strip().split(': ')[1]
    date2 = lines[1].strip().split(': ')[1]
    
    return date1, date2

# Dataset class
class ChangeDetectionDataset(Dataset):
    """
    Dataset for satellite image change detection.
    """
    def __init__(self, locations, transform=None, img_size=(256, 256)):
        """
        Initialize the dataset.
        
        Args:
            locations (list): List of location names
            transform (callable, optional): Optional transform to be applied on images
            img_size (tuple): Size to resize images to (default: (256, 256))
        """
        self.locations = locations
        
        # Define transforms if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Define mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, idx):
        location = self.locations[idx]
        
        # Load images and mask
        img1, img2 = load_image_pair(location)
        mask = load_change_mask(location)
        
        # Handle missing data
        if img1 is None or img2 is None or mask is None:
            # Return zeros as a fallback
            img1 = np.zeros((256, 256, 3), dtype=np.uint8)
            img2 = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Convert to PIL images for transforms
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        mask = Image.fromarray(mask * 255)  # Scale to 0-255 for PIL
        
        # Apply transforms
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        # For mask, we use a different transform (no normalization)
        mask = self.mask_transform(mask)
        
        # Ensure mask is binary
        mask = (mask > 0.5).float()
        
        return {
            'image1': img1,
            'image2': img2,
            'mask': mask.squeeze(),
            'location': location
        }

# Change Detection Model
class SiameseUNet(nn.Module):
    """
    Siamese U-Net model for change detection using pretrained encoder.
    """
    def __init__(self, encoder_name='resnet18', pretrained=True):
        """
        Initialize the model.
        
        Args:
            encoder_name (str): Name of the encoder backbone
            pretrained (bool): Whether to use pretrained weights
        """
        super(SiameseUNet, self).__init__()
        
        # Load pretrained encoder
        if encoder_name == 'resnet18':
            encoder = models.resnet18(pretrained=pretrained)
            self.encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet34':
            encoder = models.resnet34(pretrained=pretrained)
            self.encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            encoder = models.resnet50(pretrained=pretrained)
            self.encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
        # Extract encoder layers
        self.encoder1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.encoder2 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.encoder3 = encoder.layer2
        self.encoder4 = encoder.layer3
        self.encoder5 = encoder.layer4
        
        # Decoder layers
        self.decoder4 = self._make_decoder_block(self.encoder_channels[4], self.encoder_channels[3])
        self.decoder3 = self._make_decoder_block(self.encoder_channels[3], self.encoder_channels[2])
        self.decoder2 = self._make_decoder_block(self.encoder_channels[2], self.encoder_channels[1])
        self.decoder1 = self._make_decoder_block(self.encoder_channels[1], self.encoder_channels[0])
        
        # Final layer
        self.final = nn.Conv2d(self.encoder_channels[0], 1, kernel_size=1)
        
        # Difference module
        self.diff_module = nn.Sequential(
            nn.Conv2d(self.encoder_channels[4] * 2, self.encoder_channels[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder_channels[4]),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def forward_single(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 1/2
        e2 = self.encoder2(e1)  # 1/4
        e3 = self.encoder3(e2)  # 1/8
        e4 = self.encoder4(e3)  # 1/16
        e5 = self.encoder5(e4)  # 1/32
        
        return e1, e2, e3, e4, e5
    
    def forward(self, x1, x2):
        # Encode both images
        e1_1, e2_1, e3_1, e4_1, e5_1 = self.forward_single(x1)
        e1_2, e2_2, e3_2, e4_2, e5_2 = self.forward_single(x2)
        
        # Compute difference features
        diff5 = torch.cat([e5_1, e5_2], dim=1)
        diff5 = self.diff_module(diff5)
        
        # Decoder with skip connections
        d4 = self.decoder4(diff5) + (e4_1 - e4_2).abs()
        d3 = self.decoder3(d4) + (e3_1 - e3_2).abs()
        d2 = self.decoder2(d3) + (e2_1 - e2_2).abs()
        d1 = self.decoder1(d2) + (e1_1 - e1_2).abs()
        
        # Final prediction
        out = self.final(d1)
        
        return torch.sigmoid(out)

# Change Detection Predictor
class ChangeDetector:
    """
    Class for detecting changes between satellite images.
    """
    def __init__(self, model_path=None, use_mock=False, img_size=(256, 256)):
        """
        Initialize the change detector.
        
        Args:
            model_path (str, optional): Path to a pretrained model
            use_mock (bool): Whether to use a mock model for demonstration
            img_size (tuple): Size to resize images to for model input
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # For Streamlit compatibility, allow mock model option
        self.use_mock = use_mock
        
        # Initialize the actual model if we're not using the mock
        if not self.use_mock:
            if model_path is None:
                raise ValueError("Model path must be provided when use_mock is False")
                
            self.model = SiameseUNet(encoder_name='resnet18', pretrained=True)
            
            if os.path.exists(model_path):
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Loaded model from {model_path}")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Continuing with initialized model weights")
            else:
                print(f"Model path {model_path} does not exist. Using initialized weights.")
                
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def predict(self, img1, img2):
        """
        Predict changes between two images.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            
        Returns:
            numpy.ndarray: Change probability map
        """
        # Store original image size for resizing back
        original_height, original_width = img1.shape[:2]
        
        # For demonstration purposes, generate a simple change map
        if self.use_mock:
            # Create a simple difference-based change map
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
            
            # Normalize images
            img1_norm = img1_gray.astype(float) / 255.0
            img2_norm = img2_gray.astype(float) / 255.0
            
            # Calculate absolute difference
            diff = np.abs(img1_norm - img2_norm)
            
            # Apply Gaussian blur to smooth the difference map
            diff_smooth = cv2.GaussianBlur(diff, (15, 15), 0)
            
            # Normalize to 0-1 range
            change_prob = (diff_smooth - diff_smooth.min()) / (diff_smooth.max() - diff_smooth.min() + 1e-8)
            
            return change_prob
        else:
            # Real model prediction code
            # Convert to PIL images
            img1_pil = Image.fromarray(img1)
            img2_pil = Image.fromarray(img2)
            
            # Apply transforms
            img1_tensor = self.transform(img1_pil).unsqueeze(0).to(self.device)
            img2_tensor = self.transform(img2_pil).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                self.model.eval()  # Ensure model is in evaluation mode
                pred = self.model(img1_tensor, img2_tensor)
            
            # Convert to numpy
            pred_np = pred.squeeze().cpu().numpy()
            
            # Resize to original size
            pred_np = cv2.resize(pred_np, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            
            return pred_np
            
    def predict_batch(self, images1, images2):
        """
        Predict changes for a batch of image pairs.
        
        Args:
            images1 (torch.Tensor): Batch of first images
            images2 (torch.Tensor): Batch of second images
            
        Returns:
            numpy.ndarray: Batch of change probability maps
        """
        if self.use_mock:
            # Handle mock prediction for batch
            batch_size = images1.shape[0]
            preds = []
            
            for i in range(batch_size):
                # Convert tensor to numpy
                img1 = images1[i].permute(1, 2, 0).cpu().numpy()
                img2 = images2[i].permute(1, 2, 0).cpu().numpy()
                
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img1 = img1 * std + mean
                img2 = img2 * std + mean
                
                # Clip to valid range
                img1 = np.clip(img1, 0, 1)
                img2 = np.clip(img2, 0, 1)
                
                # Convert to uint8
                img1 = (img1 * 255).astype(np.uint8)
                img2 = (img2 * 255).astype(np.uint8)
                
                # Predict
                pred = self.predict(img1, img2)
                preds.append(pred)
            
            return np.stack(preds)
        else:
            # Real model prediction
            with torch.no_grad():
                self.model.eval()  # Ensure model is in evaluation mode
                preds = self.model(images1.to(self.device), images2.to(self.device))
            
            return preds.cpu().numpy()
    
    def predict_and_visualize(self, img1, img2, threshold=0.5):
        """
        Predict changes and visualize the results.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            threshold (float): Threshold for binary change mask
            
        Returns:
            tuple: (change_prob, change_mask, visualization)
        """
        # Predict changes
        change_prob = self.predict(img1, img2)
        
        # Create binary mask
        change_mask = (change_prob > threshold).astype(np.uint8)
        
        # Create visualization
        vis = img2.copy()
        vis[change_mask == 1] = [255, 0, 0]  # Highlight changes in red
        
        return change_prob, change_mask, vis

# Environmental Change Analysis
class EnvironmentalChangeAnalyzer:
    """
    Class for analyzing environmental changes.
    """
    def __init__(self, _detector):
        """
        Initialize the analyzer.
        
        Args:
            _detector (ChangeDetector): Change detection model (with leading underscore for Streamlit caching)
        """
        self.change_detector = _detector
    
    def calculate_ndvi(self, img):
        """
        Calculate Normalized Difference Vegetation Index (NDVI) from RGB image.
        This is a simplified version since we don't have NIR band.
        
        Args:
            img (numpy.ndarray): RGB image
            
        Returns:
            numpy.ndarray: Pseudo-NDVI
        """
        # For RGB images, we use a pseudo-NDVI using red and green bands
        # Real NDVI would use NIR and Red: (NIR - Red) / (NIR + Red)
        red = img[:, :, 0].astype(float)
        green = img[:, :, 1].astype(float)
        
        # Avoid division by zero
        denominator = green + red
        denominator[denominator == 0] = 1e-10
        
        # Calculate pseudo-NDVI
        pseudo_ndvi = (green - red) / denominator
        
        return pseudo_ndvi
    
    def calculate_ndwi(self, img):
        """
        Calculate Normalized Difference Water Index (NDWI) from RGB image.
        This is a simplified version since we don't have NIR band.
        
        Args:
            img (numpy.ndarray): RGB image
            
        Returns:
            numpy.ndarray: Pseudo-NDWI
        """
        # For RGB images, we use a pseudo-NDWI using green and blue bands
        # Real NDWI would use Green and NIR: (Green - NIR) / (Green + NIR)
        green = img[:, :, 1].astype(float)
        blue = img[:, :, 2].astype(float)
        
        # Avoid division by zero
        denominator = green + blue
        denominator[denominator == 0] = 1e-10
        
        # Calculate pseudo-NDWI
        pseudo_ndwi = (green - blue) / denominator
        
        return pseudo_ndwi
    
    def calculate_ndbi(self, img):
        """
        Calculate Normalized Difference Built-up Index (NDBI) from RGB image.
        This is a simplified version since we don't have SWIR band.
        
        Args:
            img (numpy.ndarray): RGB image
            
        Returns:
            numpy.ndarray: Pseudo-NDBI
        """
        # For RGB images, we use a pseudo-NDBI using red and blue bands
        # Real NDBI would use SWIR and NIR: (SWIR - NIR) / (SWIR + NIR)
        red = img[:, :, 0].astype(float)
        blue = img[:, :, 2].astype(float)
        
        # Avoid division by zero
        denominator = red + blue
        denominator[denominator == 0] = 1e-10
        
        # Calculate pseudo-NDBI
        pseudo_ndbi = (red - blue) / denominator
        
        return pseudo_ndbi
    
    def analyze_changes(self, img1, img2, threshold=0.5):
        """
        Analyze environmental changes between two images.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            threshold (float): Threshold for binary change mask
            
        Returns:
            dict: Analysis results
        """
        # Detect changes
        change_prob, change_mask, vis = self.change_detector.predict_and_visualize(img1, img2, threshold)
        
        # Calculate change statistics
        total_pixels = change_mask.size
        changed_pixels = change_mask.sum()
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Analyze change patterns
        num_regions, labels = cv2.connectedComponents(change_mask)
        
        # Calculate spectral indices for both images
        ndvi1 = self.calculate_ndvi(img1)
        ndvi2 = self.calculate_ndvi(img2)
        ndwi1 = self.calculate_ndwi(img1)
        ndwi2 = self.calculate_ndwi(img2)
        ndbi1 = self.calculate_ndbi(img1)
        ndbi2 = self.calculate_ndbi(img2)
        
        # Calculate differences in indices
        ndvi_diff = ndvi2 - ndvi1
        ndwi_diff = ndwi2 - ndwi1
        ndbi_diff = ndbi2 - ndbi1
        
        # Determine change types based on spectral indices
        change_types = []
        
        if num_regions > 1:  # At least one change region (excluding background)
            for i in range(1, min(num_regions, 6)):  # Limit to 5 regions for simplicity
                # Create mask for current region
                region_mask = (labels == i)
                
                # Skip small regions
                if np.sum(region_mask) < 100:  # Arbitrary threshold
                    continue
                
                # Calculate average index differences in the region
                avg_ndvi_diff = np.mean(ndvi_diff[region_mask])
                avg_ndwi_diff = np.mean(ndwi_diff[region_mask])
                avg_ndbi_diff = np.mean(ndbi_diff[region_mask])
                
                # Determine change type based on index differences
                if avg_ndvi_diff < -0.1 and avg_ndbi_diff > 0.05:
                    change_type = "Urban Development"
                elif avg_ndvi_diff < -0.1:
                    change_type = "Deforestation"
                elif avg_ndvi_diff > 0.1:
                    change_type = "Vegetation Regrowth"
                elif avg_ndwi_diff > 0.1:
                    change_type = "Water Body Expansion"
                elif avg_ndwi_diff < -0.1:
                    change_type = "Water Body Reduction"
                elif avg_ndbi_diff > 0.1:
                    change_type = "Infrastructure Development"
                else:
                    change_type = "Land Cover Change"
                
                change_types.append(change_type)
        
        # Return results
        return {
            'change_probability': change_prob,
            'change_mask': change_mask,
            'visualization': vis,
            'change_percentage': change_percentage,
            'num_change_regions': num_regions - 1,  # Subtract background
            'change_types': change_types
        }
    
    def generate_report(self, location, date1, date2, results):
        """
        Generate a report of environmental changes.
        
        Args:
            location (str): Name of the location
            date1 (str): Date of the first image
            date2 (str): Date of the second image
            results (dict): Analysis results
            
        Returns:
            str: Report text
        """
        from datetime import datetime
        
        # Format report
        report = f"Environmental Change Analysis Report\n"
        report += f"===================================\n\n"
        report += f"Location: {location}\n"
        report += f"Period: {date1} to {date2}\n\n"
        report += f"Change Summary:\n"
        report += f"- Changed Area: {results['change_percentage']:.2f}% of the total area\n"
        report += f"- Number of Change Regions: {results['num_change_regions']}\n\n"
        
        if results['num_change_regions'] > 0 and len(results['change_types']) > 0:
            report += f"Detected Change Types:\n"
            for i, change_type in enumerate(results['change_types']):
                report += f"- Region {i+1}: {change_type}\n"
            
            report += f"\nPotential Environmental Impacts:\n"
            
            # Create a dictionary of environmental impacts by change type
            impacts = {
                "Deforestation": [
                    "Loss of habitat for wildlife",
                    "Increased carbon emissions",
                    "Reduced biodiversity",
                    "Potential soil erosion and landslides",
                    "Disruption of water cycles"
                ],
                "Urban Development": [
                    "Increased impervious surface area",
                    "Heat island effect",
                    "Increased stormwater runoff",
                    "Habitat fragmentation",
                    "Air and noise pollution"
                ],
                "Water Body Expansion": [
                    "Potential flooding of adjacent areas",
                    "Changes in local microclimate",
                    "Creation of new aquatic habitats",
                    "Altered groundwater dynamics"
                ],
                "Water Body Reduction": [
                    "Loss of aquatic habitats",
                    "Reduced water availability for ecosystems",
                    "Potential water quality degradation",
                    "Impact on migratory species"
                ],
                "Vegetation Regrowth": [
                    "Increased carbon sequestration",
                    "Habitat restoration",
                    "Improved soil stability",
                    "Enhanced biodiversity",
                    "Reduced erosion"
                ],
                "Infrastructure Development": [
                    "Habitat fragmentation",
                    "Increased human activity in the area",
                    "Potential pollution from construction",
                    "Changes in land use patterns"
                ],
                "Land Cover Change": [
                    "Altered ecosystem services",
                    "Changes in local biodiversity",
                    "Modified surface reflectivity (albedo)",
                    "Potential changes in local climate"
                ]
            }
            
            # Add impacts for each detected change type
            added_impacts = set()  # To avoid duplicates
            for change_type in results['change_types']:
                if change_type in impacts:
                    for impact in impacts[change_type]:
                        if impact not in added_impacts:
                            report += f"- {impact}\n"
                            added_impacts.add(impact)
        
        report += f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"\nNote: This analysis uses spectral indices derived from RGB imagery to estimate\n"
        report += f"environmental changes. For more accurate results, multispectral imagery\n"
        report += f"with NIR and SWIR bands would provide better classification of change types.\n"
        report += f"This is a demonstration using simplified change detection techniques.\n"
        
        return report

# Main function for testing
def main():
    # Test on a sample location
    location = 'dubai'  # Change to any available location
    
    # Load images
    img1, img2 = load_image_pair(location)
    if img1 is None or img2 is None:
        print(f"Failed to load images for {location}")
        return
    
    # Get dates
    date1, date2 = get_dates(location)
    
    # Initialize change detector
    detector = ChangeDetector()
    
    # Initialize analyzer
    analyzer = EnvironmentalChangeAnalyzer(detector)
    
    # Analyze changes
    results = analyzer.analyze_changes(img1, img2)
    
    # Generate report
    report = analyzer.generate_report(location, date1, date2, results)
    print(report)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title(f"Before ({date1})")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title(f"After ({date2})")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(results['change_probability'], cmap='jet')
    plt.title("Change Probability")
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(results['visualization'])
    plt.title("Changes Highlighted")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{location}_changes.png")
    plt.show()

if __name__ == "__main__":
    main()