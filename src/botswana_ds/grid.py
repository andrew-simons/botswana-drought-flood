"""The common Botswana analysis grid that EVERYTHING snaps to.

Design choice: a single fixed grid means every variable (rainfall, NDVI, soil
moisture, ...) lines up pixel-for-pixel and over time, so we can stack them into a
clean (T, C, H, W) tensor for PyTorch. We align to CHIRPS' native 0.05 deg cells so
rainfall (our most important variable) needs no resampling.

This module has NO Earth Engine dependency on purpose — you can build and inspect the
grid offline. Run `python -m botswana_ds.grid` to print the grid summary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Botswana bounding box (lon/lat, WGS84). Generous box covering the whole country.
#   West ~20.0E (Namibia border), East ~29.4E (Zimbabwe), South ~26.9S, North ~17.8S
BOTSWANA_BBOX = (20.0, -26.9, 29.4, -17.8)  # (min_lon, min_lat, max_lon, max_lat)

# CHIRPS native resolution = 0.05 degrees (~5.5 km). We adopt it as the grid step.
GRID_RES_DEG = 0.05

# GEE crsTransform that locks every export to this exact pixel grid.
# Format: [xScale, xShear, xTranslation, yShear, yScale, yTranslation]
# xTranslation/yTranslation are the top-left CORNER of the top-left pixel.
GEE_CRS_TRANSFORM = [GRID_RES_DEG, 0, 20.0, 0, -GRID_RES_DEG, -17.8]

# Exact pixel dimensions of the grid — used alongside crsTransform in GEE exports
# to guarantee an integer pixel count without relying on floating-point bbox math.
GRID_COLS = int(round((BOTSWANA_BBOX[2] - BOTSWANA_BBOX[0]) / GRID_RES_DEG))  # 188
GRID_ROWS = int(round((BOTSWANA_BBOX[3] - BOTSWANA_BBOX[1]) / GRID_RES_DEG))  # 182
GEE_DIMENSIONS = f"{GRID_COLS}x{GRID_ROWS}"                                    # "188x182"


@dataclass(frozen=True)
class Grid:
    """A regular lat/lon grid. Pixel centers are at the returned coordinates."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    res: float

    @property
    def lons(self) -> np.ndarray:
        """1-D array of pixel-center longitudes (west -> east)."""
        n = int(round((self.max_lon - self.min_lon) / self.res))
        return self.min_lon + (np.arange(n) + 0.5) * self.res

    @property
    def lats(self) -> np.ndarray:
        """1-D array of pixel-center latitudes (NORTH -> south, image convention)."""
        n = int(round((self.max_lat - self.min_lat) / self.res))
        # top row = northernmost, matching how rasters/images are stored
        return self.max_lat - (np.arange(n) + 0.5) * self.res

    @property
    def shape(self) -> tuple[int, int]:
        """(H, W) — rows (lat) x columns (lon)."""
        return (len(self.lats), len(self.lons))

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lon2d, lat2d), each of shape (H, W)."""
        lon2d, lat2d = np.meshgrid(self.lons, self.lats)
        return lon2d, lat2d

    def transform(self):
        """Affine transform (rasterio-style) for writing GeoTIFFs.

        Imported lazily so the module works without rasterio installed.
        """
        from rasterio.transform import from_origin

        # origin = top-left CORNER (not center)
        return from_origin(self.min_lon, self.max_lat, self.res, self.res)


def make_grid() -> Grid:
    """The canonical project grid."""
    return Grid(*BOTSWANA_BBOX, res=GRID_RES_DEG)


if __name__ == "__main__":
    g = make_grid()
    h, w = g.shape
    print("Botswana project grid")
    print(f"  bbox (lon/lat) : {BOTSWANA_BBOX}")
    print(f"  resolution     : {g.res} deg (~5.5 km, CHIRPS-aligned)")
    print(f"  shape (H, W)   : {h} x {w}  = {h * w:,} pixels")
    print(f"  lon range      : {g.lons[0]:.3f} .. {g.lons[-1]:.3f}")
    print(f"  lat range      : {g.lats[0]:.3f} .. {g.lats[-1]:.3f}")
