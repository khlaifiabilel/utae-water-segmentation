import os
import torch
import rasterio
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from models.utae_water_segmentation import create_water_segmentation_model
from utils.vectorize import raster_to_geojson

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def normalize_s2(s2_data, means=None, stds=None):
    """Normalize Sentinel-2 data"""
    if means is None:
        # Default means for S2 bands
        means = [1226.21, 1137.38, 1139.82, 1350.49, 1932.94, 2211.89, 
                2154.36, 2163.57, 2246.07, 2036.50, 1465.38, 986.64, 231.95]
    
    if stds is None:
        # Default stds for S2 bands
        stds = [572.41, 582.87, 675.54, 675.60, 736.38, 878.58, 
                905.21, 943.75, 955.51, 978.05, 825.01, 729.21, 365.72]
    
    for i in range(len(means)):
        if i < s2_data.shape[0]:  # Check if band exists
            s2_data[i] = (s2_data[i] - means[i]) / stds[i]
    
    return s2_data

def predict_water(model, s2_path, output_dir, batch_size=1, img_size=None, 
                 device='cuda', export_geojson=True, export_png=True):
    """
    Generate water/land prediction from Sentinel-2 imagery
    
    Args:
        model: PyTorch model
        s2_path: Path to Sentinel-2 image (GeoTIFF)
        output_dir: Directory to save outputs
        batch_size: Batch size for inference
        img_size: Size to resize image to (if None, use original size)
        device: Device to run inference on ('cuda' or 'cpu')
        export_geojson: Whether to export GeoJSON
        export_png: Whether to export visualization PNG
        
    Returns:
        Dictionary with paths to output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # File name without extension
    base_name = Path(s2_path).stem
    
    # Output paths
    prediction_path = output_dir / f"{base_name}_water_prediction.tif"
    geojson_path = output_dir / f"{base_name}_water.geojson"
    png_path = output_dir / f"{base_name}_visualization.png"
    
    # Open the Sentinel-2 image
    with rasterio.open(s2_path) as src:
        s2_data = src.read().astype(np.float32)  # [channels, height, width]
        profile = src.profile.copy()
        height, width = s2_data.shape[1], s2_data.shape[2]
        
    # Check number of bands
    if s2_data.shape[0] < 13:
        logging.warning(f"Expected 13 Sentinel-2 bands, got {s2_data.shape[0]}. "
                      f"This might not be a standard Sentinel-2 image.")
        
        # Pad with zeros if fewer bands
        if s2_data.shape[0] < 13:
            pad_bands = 13 - s2_data.shape[0]
            s2_data = np.pad(s2_data, ((0, pad_bands), (0, 0), (0, 0)))
    
    # Normalize data
    s2_data = normalize_s2(s2_data)
    
    # Convert to PyTorch tensor
    s2_tensor = torch.from_numpy(s2_data).float()
    
    # Handle image size
    original_size = (height, width)
    if img_size is not None:
        # Resize to target size
        s2_tensor = torch.nn.functional.interpolate(
            s2_tensor.unsqueeze(0),
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    # Set model to evaluation mode
    model.eval()
    
    # Process in tiles if the image is large
    # This example uses a simple approach, but you can implement more sophisticated tiling
    prediction = np.zeros((height, width), dtype=np.uint8)
    
    with torch.no_grad():
        # For this simple example, we process the whole image at once
        # In practice, you'd want to tile large images
        s2_batch = s2_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
        outputs = model(s2_batch)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).cpu().numpy()[0]  # [H, W]
        
        # Resize back to original size if needed
        if img_size is not None:
            pred = torch.nn.functional.interpolate(
                torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='nearest'
            ).squeeze().long().numpy()
        
        prediction = pred
    
    # Save prediction as GeoTIFF
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw',
        nodata=None
    )
    with rasterio.open(prediction_path, 'w', **profile) as dst:
        dst.write(prediction.astype(rasterio.uint8), 1)
    
    # Export GeoJSON
    if export_geojson:
        raster_to_geojson(prediction, s2_path, geojson_path)
    
    # Create visualization
    if export_png:
        # Get RGB bands (assuming bands 4,3,2 are R,G,B)
        with rasterio.open(s2_path) as src:
            rgb = np.zeros((height, width, 3), dtype=np.float32)
            if src.count >= 4:  # Make sure we have enough bands
                r = src.read(4)  # Red band
                g = src.read(3)  # Green band
                b = src.read(2)  # Blue band
                
                # Stack and normalize for visualization
                rgb[:, :, 0] = np.clip(r / 3000, 0, 1)  # Adjust value based on your data
                rgb[:, :, 1] = np.clip(g / 3000, 0, 1)
                rgb[:, :, 2] = np.clip(b / 3000, 0, 1)
        
        # Create water/land colormap (blue for water, green for land)
        water_cmap = plt.cm.colors.ListedColormap(['#33CC33', '#3366FF'])  # Land, Water
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(rgb)
        ax1.set_title('Sentinel-2 Image')
        ax1.axis('off')
        
        ax2.imshow(prediction, cmap=water_cmap, vmin=0, vmax=1)
        ax2.set_title('Water Segmentation')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'prediction': prediction_path,
        'geojson': geojson_path if export_geojson else None,
        'visualization': png_path if export_png else None
    }

def batch_process(model_path, s2_dir, output_dir, pattern='*.tif', img_size=256):
    """
    Process all Sentinel-2 images in a directory
    
    Args:
        model_path: Path to trained model checkpoint
        s2_dir: Directory with Sentinel-2 images
        output_dir: Directory to save outputs
        pattern: File pattern to match
        img_size: Image size to resize to
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    model = create_water_segmentation_model(temporal_length=1, for_training=False)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    logging.info(f"Model loaded from {model_path}")
    
    # Find all S2 images
    s2_dir = Path(s2_dir)
    s2_paths = list(s2_dir.glob(pattern))
    logging.info(f"Found {len(s2_paths)} images to process")
    
    # Process each image
    for s2_path in tqdm(s2_paths, desc="Processing images"):
        try:
            outputs = predict_water(
                model=model,
                s2_path=s2_path,
                output_dir=output_dir,
                img_size=img_size,
                device=device
            )
            logging.info(f"Processed {s2_path.name}: "
                         f"Prediction saved to {outputs['prediction']}, "
                         f"GeoJSON saved to {outputs['geojson']}")
        except Exception as e:
            logging.error(f"Error processing {s2_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Generate water segmentation from Sentinel-2 imagery")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to Sentinel-2 image or directory')
    parser.add_argument('--output', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--batch', action='store_true', help='Process all images in input directory')
    parser.add_argument('--img_size', type=int, default=None, help='Size to resize images to')
    parser.add_argument('--no_geojson', action='store_true', help='Skip GeoJSON export')
    parser.add_argument('--no_png', action='store_true', help='Skip PNG visualization export')
    args = parser.parse_args()
    
    if args.batch:
        # Process all images in directory
        batch_process(
            model_path=args.model,
            s2_dir=args.input,
            output_dir=args.output,
            img_size=args.img_size
        )
    else:
        # Process single image
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_water_segmentation_model(temporal_length=1, for_training=False)
        checkpoint = torch.load(args.model, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(device)
        
        outputs = predict_water(
            model=model,
            s2_path=args.input,
            output_dir=args.output,
            img_size=args.img_size,
            device=device,
            export_geojson=not args.no_geojson,
            export_png=not args.no_png
        )
        
        logging.info(f"Processing complete. Outputs saved to: {args.output}")

if __name__ == "__main__":
    main()