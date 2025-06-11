import os
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape, mapping
import json
import logging

def raster_to_geojson(prediction, source_raster_path, output_path, 
                      simplify_tolerance=1.0, min_area=100):
    """
    Convert binary water prediction to GeoJSON
    
    Args:
        prediction: numpy array with binary prediction (1=water, 0=land)
        source_raster_path: Path to the source raster file (to get georeferencing)
        output_path: Path to save the GeoJSON file
        simplify_tolerance: Tolerance for polygon simplification (in pixels)
        min_area: Minimum area for polygons to keep (in pixels)
    
    Returns:
        Path to saved GeoJSON file
    """
    logging.info(f"Converting water prediction to GeoJSON: {output_path}")
    
    try:
        # Open the source raster to get the transform and CRS
        with rasterio.open(source_raster_path) as src:
            transform = src.transform
            crs = src.crs
            
        # Create mask for water pixels (value=1)
        water_mask = prediction == 1
        
        # Extract shapes from the binary mask
        shapes = features.shapes(
            prediction.astype('uint8'),
            mask=water_mask,
            transform=transform
        )
        
        # Create GeoJSON features
        features_list = []
        for geom, value in shapes:
            if value == 1:  # Only include water features
                # Convert to shapely geometry
                poly = shape(geom)
                
                # Skip tiny polygons
                if poly.area < min_area:
                    continue
                
                # Simplify polygon to reduce size
                if simplify_tolerance > 0:
                    poly = poly.simplify(simplify_tolerance)
                
                features_list.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "class": "water",
                        "area_sqm": poly.area * (transform[0] ** 2)  # Approximate area in sq meters
                    }
                })
        
        # Create the GeoJSON structure
        geojson_dict = {
            "type": "FeatureCollection",
            "features": features_list
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(geojson_dict, f)
        
        # Also create a GeoDataFrame for additional analysis if needed
        gdf = gpd.GeoDataFrame.from_features(geojson_dict, crs=crs)
        
        logging.info(f"Successfully created GeoJSON with {len(features_list)} water features")
        return output_path
        
    except Exception as e:
        logging.error(f"Error creating GeoJSON: {str(e)}")
        raise e