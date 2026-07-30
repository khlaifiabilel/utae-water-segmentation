import json
import logging
from pathlib import Path

import rasterio
from rasterio import features
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)


def raster_to_geojson(
    prediction,
    source_raster_path,
    output_path,
    simplify_tolerance=1.0,
    min_area=100,
):
    """Convert water pixels to georeferenced GeoJSON polygons."""
    logger.info("Converting water prediction to GeoJSON: %s", output_path)
    with rasterio.open(source_raster_path) as source:
        transform = source.transform
        crs = source.crs

    water_mask = prediction == 1
    extracted_shapes = features.shapes(
        prediction.astype("uint8"), mask=water_mask, transform=transform
    )
    feature_list = []
    for geometry, value in extracted_shapes:
        if value != 1:
            continue
        polygon = shape(geometry)
        if polygon.area < min_area:
            continue
        if simplify_tolerance > 0:
            polygon = polygon.simplify(simplify_tolerance)
        feature_list.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "class": "water",
                    "area_crs_units": polygon.area,
                },
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": feature_list,
    }
    if crs is not None:
        geojson["name"] = str(crs)

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(geojson, output_file)
    logger.info("Created GeoJSON with %d water features", len(feature_list))
    return output_path
