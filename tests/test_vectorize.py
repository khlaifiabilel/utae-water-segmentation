import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from utils.vectorize import raster_to_geojson


def test_raster_to_geojson_reports_geometry_area_in_crs_units(tmp_path):
    source_path = tmp_path / "source.tif"
    output_path = tmp_path / "water.geojson"
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 20, 10, 10),
    }
    with rasterio.open(source_path, "w", **profile) as destination:
        destination.write(np.zeros((2, 2), dtype=np.uint8), 1)

    prediction = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    raster_to_geojson(
        prediction,
        source_path,
        output_path,
        simplify_tolerance=0,
        min_area=0,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(result["features"]) == 1
    assert result["features"][0]["properties"]["area_crs_units"] == pytest.approx(100)
