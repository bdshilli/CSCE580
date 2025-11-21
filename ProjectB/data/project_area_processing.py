#!/usr/bin/env python3
"""
project_area_processing_to_geojson_with_arcgis_tiles.py

Project area -> streamed GeoJSON points (EPSG:4326) with sample properties:
  dem, filled, slope, aspect, flow, wetland (0/1), soil, precip

Notes:
 - Wetland detection: raster-first (aligned to DEM grid) for canonical 0/1 flag;
   fallback to prepared geometry.intersects() when raster unavailable.
 - precip sampling uses bilinear interpolation when mosaics are available; soil uses nearest (categorical).
 - If mosaics for soil/precip are missing, script falls back to exportImage requests in chunks.
"""
from pathlib import Path
import argparse
import logging
import os
import math
from typing import List, Dict, Optional, Tuple, Any
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio import transform as rio_transform
import fiona
from shapely.geometry import shape as shapely_shape, mapping, Point
from shapely.ops import unary_union, transform as shapely_transform
from shapely.prepared import prep
import numpy as np
import pyarrow.parquet as pq
import richdem as rd
from tqdm import tqdm
from pyproj import Transformer, CRS
import json
from datetime import datetime
import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.io import MemoryFile
import io
import warnings
import subprocess
import sys
import glob

# optional pandas for preview printing
try:
    import pandas as pd
except Exception:
    pd = None

# logging + env
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
try:
    import pyproj
    os.environ.setdefault("PROJ_LIB", pyproj.datadir.get_data_dir())
except Exception:
    pass

_WORKING_CRS = "EPSG:32617"

# Path to the downloader script (adjust if placed elsewhere)
DOWNLOADER_SCRIPT = Path(__file__).parent / "download_arcgis_images_sc.py"
# Default downloader args
DOWNLOADER_DEFAULTS = {
    "arcgis_config": "arcgis_config.yaml",
    "services": ["soil_url", "precip_url"],
    "out_dir": "raw_arcgis",
    "tile_size_m": 50000,
    "res_m": 100
}

# -------------------------
# Utilities (vector/catalog)
# -------------------------
def _make_requests_session(total_retries=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    sess = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor, status_forcelist=list(status_forcelist))
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def read_geom_union_via_fiona(path: str) -> Tuple[Any, Optional[str]]:
    """
    Read vector via fiona, union geometries, and attempt to return a normalized CRS string.
    Handles common GeoJSON CRS variants (e.g. 'urn:ogc:def:crs:OGC:1.3:CRS84').
    Returns (unioned_geometry, crs_string_or_None).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Vector not found: {path}")

    geoms = []
    crs_obj = None
    with fiona.Env():
        with fiona.open(str(p), "r") as src:
            # fiona returns src.crs_wkt or src.crs (dict) depending on driver
            crs_obj = src.crs_wkt or src.crs
            for feat in src:
                geom = feat.get("geometry")
                if geom is None:
                    continue
                try:
                    g = shapely_shape(geom)
                    if not g.is_empty:
                        geoms.append(g)
                except Exception:
                    continue

    if not geoms:
        raise RuntimeError(f"No valid geometries found in {path}")

    union = unary_union(geoms)

    # Normalize CRS into a string that pyproj/Transformer understands
    crs_str = None
    try:
        # If fiona returned a dict (common), try to build a pyproj CRS
        if isinstance(crs_obj, dict) and crs_obj:
            # handle form like {'init': 'epsg:4326'} or {'properties': {'name': 'EPSG:4326'}}
            if "init" in crs_obj:
                try:
                    crs_str = CRS.from_user_input(crs_obj["init"]).to_string()
                except Exception:
                    crs_str = str(crs_obj["init"])
            elif "properties" in crs_obj and isinstance(crs_obj["properties"], dict):
                name = crs_obj["properties"].get("name")
                if name:
                    # some GeoJSONs use 'urn:ogc:def:crs:OGC:1.3:CRS84'
                    if "CRS84" in name or "EPSG:4326" in name or name.strip().endswith("4326"):
                        crs_str = "EPSG:4326"
                    else:
                        try:
                            crs_str = CRS.from_user_input(name).to_string()
                        except Exception:
                            crs_str = str(name)
                else:
                    crs_str = str(crs_obj)
            else:
                # try generic conversion
                try:
                    crs_str = CRS(crs_obj).to_string()
                except Exception:
                    crs_str = str(crs_obj)
        elif isinstance(crs_obj, str) and crs_obj:
            # handle WKT or URN like CRS84
            if "CRS84" in crs_obj or "4326" in crs_obj or "EPSG:4326" in crs_obj:
                crs_str = "EPSG:4326"
            else:
                try:
                    crs = CRS.from_wkt(crs_obj)
                    crs_str = crs.to_string()
                except Exception:
                    # fallback to raw string
                    crs_str = crs_obj
        else:
            crs_str = None
    except Exception:
        try:
            crs_str = str(crs_obj)
        except Exception:
            crs_str = None

    return union, crs_str


def reproject_geom(geom: Any, src_crs: Optional[str], dst_crs: str):
    if src_crs is None or src_crs == dst_crs:
        return geom
    try:
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        return shapely_transform(lambda x, y: transformer.transform(x, y), geom)
    except Exception as e:
        logging.warning(f"Reprojection failed ({src_crs} -> {dst_crs}): {e}. Proceeding with original geometry.")
        return geom


def transform_bbox(tbox: Tuple[float, float, float, float], src_crs: Optional[str], dst_crs: str) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = tbox
    if src_crs is None or src_crs == dst_crs:
        return minx, miny, maxx, maxy
    try:
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        tx = [transformer.transform(x, y)[0] for (x, y) in corners]
        ty = [transformer.transform(x, y)[1] for (x, y) in corners]
        return float(min(tx)), float(min(ty)), float(max(tx)), float(max(ty))
    except Exception as e:
        logging.warning(f"Failed to transform bbox ({tbox}) from {src_crs} to {dst_crs}: {e}")
        return minx, miny, maxx, maxy


def load_catalog_pyarrow(parquet_path: Path) -> List[Dict[str, Any]]:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(str(parquet_path))
    rows: List[Dict[str, Any]] = []
    for batch in table.to_batches():
        rows.extend(batch.to_pylist())
    logging.info(f"Loaded catalog via pyarrow: {len(rows)} rows")
    return rows


def catalog_get_crs(catalog_rows: List[Dict[str, Any]]) -> Optional[str]:
    for r in catalog_rows:
        crs = r.get("crs", None)
        if crs:
            return str(crs)
    return None


def select_tifs_from_catalog_rows(catalog_rows: List[Dict[str, Any]], bounds_working: Tuple[float, float, float, float], catalog_crs: Optional[str], working_crs: str) -> List[str]:
    sel: List[str] = []
    a, b, c, d = bounds_working
    for r in catalog_rows:
        try:
            rminx = r.get("minx", None); rminy = r.get("miny", None)
            rmaxx = r.get("maxx", None); rmaxy = r.get("maxy", None)
            if any(v is None for v in (rminx, rminy, rmaxx, rmaxy)):
                continue
            if catalog_crs and catalog_crs != working_crs:
                tminx, tminy, tmaxx, tmaxy = transform_bbox((float(rminx), float(rminy), float(rmaxx), float(rmaxy)), catalog_crs, working_crs)
            else:
                tminx, tminy, tmaxx, tmaxy = float(rminx), float(rminy), float(rmaxx), float(rmaxy)
            if (a < tmaxx) and (c > tminx) and (b < tmaxy) and (d > tminy):
                p = r.get("path", None) or r.get("filepath", None) or r.get("file", None)
                if p:
                    sel.append(str(p))
        except Exception:
            continue
    return sel


def compute_mosaic(tif_paths: List[str], bounds: Tuple[float, float, float, float], res: float, dst_crs: str):
    srcs = []
    for p in tif_paths:
        try:
            src = rasterio.open(p)
        except Exception as e:
            logging.warning(f"Could not open {p}: {e}")
            continue
        if src.crs is None:
            logging.warning(f"{p} has no CRS; skipping")
            try:
                src.close()
            except Exception:
                pass
            continue
        try:
            if dst_crs and str(src.crs) != dst_crs:
                vrt = WarpedVRT(src, crs=dst_crs, resampling=Resampling.bilinear)
                srcs.append(vrt)
            else:
                srcs.append(src)
        except Exception as e:
            logging.warning(f"Failed to build VRT for {p}: {e}")
            try:
                src.close()
            except Exception:
                pass
            continue

    if not srcs:
        raise RuntimeError("No readable sources provided for mosaic.")

    logging.info(f"Merging {len(srcs)} sources into mosaic with res={res} and bounds={bounds}")
    mosaic, out_trans = rio_merge(srcs, bounds=bounds, res=res)
    ref = srcs[0]
    profile = ref.meta.copy()
    profile.update({
        "crs": dst_crs,
        "transform": out_trans,
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": mosaic.shape[0],
    })
    for s in srcs:
        try:
            s.close()
        except Exception:
            pass
    return mosaic, out_trans, profile


def extract_patch(array2d: np.ndarray, center_row: int, center_col: int, patch_size: int, fill_value=np.nan):
    half = patch_size // 2
    r0 = center_row - half
    r1 = center_row + half + 1
    c0 = center_col - half
    c1 = center_col + half + 1
    h, w = array2d.shape
    patch = np.full((patch_size, patch_size), fill_value, dtype=array2d.dtype)
    rr0 = max(0, r0); rr1 = min(h, r1)
    cc0 = max(0, c0); cc1 = min(w, c1)
    dst_r0 = rr0 - r0; dst_r1 = dst_r0 + (rr1 - rr0)
    dst_c0 = cc0 - c0; dst_c1 = dst_c0 + (cc1 - cc0)
    patch[dst_r0:dst_r1, dst_c0:dst_c1] = array2d[rr0:rr1, cc0:cc1]
    return patch


# Bilinear interpolation sampler for continuous mosaics
def bilinear_sample_array(band: np.ndarray, transform: Any, xs: List[float], ys: List[float]) -> List[Optional[float]]:
    """
    Bilinear sample band (2D numpy) at coordinates xs, ys given rasterio transform.
    Returns float or None.
    """
    if band is None or transform is None:
        return [None] * len(xs)
    inv = ~transform  # Affine inverse
    out = []
    H, W = band.shape
    for x, y in zip(xs, ys):
        # fractional col,row
        col_f, row_f = inv * (x, y)
        if not np.isfinite(col_f) or not np.isfinite(row_f):
            out.append(None); continue
        # neighbors
        c0 = math.floor(col_f); r0 = math.floor(row_f)
        c1 = c0 + 1; r1 = r0 + 1
        # check bounds
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            # if point is on border, try nearest fallback
            rc = int(round(row_f)); cc = int(round(col_f))
            if 0 <= rc < H and 0 <= cc < W:
                v = band[rc, cc]
                out.append(None if (isinstance(v, float) and math.isnan(v)) else float(v))
            else:
                out.append(None)
            continue
        # fractional weights
        dx = col_f - c0
        dy = row_f - r0
        v00 = band[r0, c0]; v10 = band[r0, c1]; v01 = band[r1, c0]; v11 = band[r1, c1]
        vals = [v00, v10, v01, v11]
        # if all nan-ish, return None
        if all((isinstance(v, float) and math.isnan(v)) for v in vals):
            out.append(None); continue
        # treat nan as 0 weight by replacing with nearest neighbor where appropriate
        def safe_val(v):
            return 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)
        s00 = safe_val(v00); s10 = safe_val(v10); s01 = safe_val(v01); s11 = safe_val(v11)
        w00 = (1 - dx) * (1 - dy)
        w10 = dx * (1 - dy)
        w01 = (1 - dx) * dy
        w11 = dx * dy
        denom = w00 + w10 + w01 + w11
        if denom == 0:
            out.append(None)
            continue
        val = (s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11) / denom
        out.append(float(val))
    return out


def sample_raster_at_points(mosaic: np.ndarray, trans: Any, xs: List[float], ys: List[float], method: str = "nearest") -> List[Optional[float]]:
    """
    Sample mosaic at given points. method: "nearest" or "bilinear".
    """
    if mosaic is None or trans is None:
        return [None] * len(xs)
    # mosaic may be 3D: (bands, H, W) or 2D
    if mosaic.ndim == 3:
        band = mosaic[0]
    else:
        band = mosaic
    if method == "nearest":
        rows, cols = rio_transform.rowcol(trans, xs, ys, op=round)
        out = []
        H, W = band.shape
        for r, c in zip(rows, cols):
            if r < 0 or c < 0 or r >= H or c >= W:
                out.append(None)
            else:
                val = band[r, c]
                if isinstance(val, (float, np.floating)) and math.isnan(val):
                    out.append(None)
                else:
                    out.append(float(val))
        return out
    elif method == "bilinear":
        return bilinear_sample_array(band, trans, xs, ys)
    else:
        raise ValueError("Unknown sampling method")


# GeoJSON streaming writer (EPSG:4326)
class GeoJSONStreamer:
    def __init__(self, out_path: Path, src_crs: str = _WORKING_CRS):
        self.out_path = out_path
        self.src_crs = src_crs
        self.fp = None
        self.first = True
        self.transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        self.count = 0
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def open(self):
        if self.fp is not None:
            return
        self.fp = self.out_path.open("w", encoding="utf-8")
        self.fp.write('{"type":"FeatureCollection","features":[\n')
        self.first = True
        self.fp.flush()
        logging.info(f"GeoJSON streamer opened: {self.out_path}")

    def write_feature(self, x, y, props: Dict[str, Any]):
        # x,y expected in src_crs (working CRS)
        if self.fp is None:
            self.open()
        try:
            lon, lat = self.transformer.transform(float(x), float(y))
        except Exception:
            lon, lat = None, None
        feat = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props}
        if not self.first:
            self.fp.write(",\n")
        self.fp.write(json.dumps(feat, ensure_ascii=False))
        self.first = False
        self.count += 1
        if (self.count % 1000) == 0:
            self.fp.flush()
            logging.debug(f"Wrote {self.count} features so far")

    def close(self, generated_iso: Optional[str] = None, count: Optional[int] = None):
        if self.fp is None:
            # write empty collection
            try:
                with self.out_path.open("w", encoding="utf-8") as fh:
                    fh.write('{"type":"FeatureCollection","features":[],')
                    meta = {}
                    if generated_iso:
                        meta["generated"] = generated_iso
                    meta["count"] = int(count) if count is not None else 0
                    fh.write(json.dumps(meta, ensure_ascii=False)[1:-1])
                    fh.write("}\n")
                logging.info(f"Wrote empty GeoJSON with metadata: {self.out_path}")
            except Exception as e:
                logging.warning(f"Failed to write empty GeoJSON: {e}")
            return

        self.fp.write("\n],")
        meta = {}
        if generated_iso:
            meta["generated"] = generated_iso
        meta["count"] = int(count) if count is not None else int(self.count)
        try:
            self.fp.write(json.dumps(meta, ensure_ascii=False)[1:-1])
            self.fp.write("}\n")
            self.fp.flush()
            self.fp.close()
            logging.info(f"Closed GeoJSON writer: {self.out_path} (features={self.count})")
        except Exception as e:
            logging.warning(f"Failed to finalize GeoJSON file cleanly: {e}")
            try:
                self.fp.close()
            except Exception:
                pass
        finally:
            self.fp = None


# -------------------------
# ArcGIS tile / mosaic helpers (downloader integration)
# -------------------------
def call_downloader_if_needed(service_keys: List[str], arcgis_config_path: str, out_dir: str,
                              tile_size_m: int = 50000, res_m: int = 100, downloader_script_path: Path = DOWNLOADER_SCRIPT) -> None:
    missing = []
    for key in service_keys:
        pattern = Path(out_dir) / key / "*.tif"
        found = list(glob.glob(str(pattern)))
        if not found:
            missing.append(key)

    if not missing:
        logging.info("Found existing tiles for services: %s", ", ".join(service_keys))
        return

    if not downloader_script_path.exists():
        logging.warning("Downloader script not found at %s; cannot download missing tiles for: %s", downloader_script_path, missing)
        return

    cmd = [sys.executable, str(downloader_script_path),
           "--arcgis-config", str(arcgis_config_path),
           "--out-dir", str(out_dir),
           "--tile-size-m", str(tile_size_m),
           "--res-m", str(res_m),
           "--workers", "4"]
    for k in missing:
        cmd.extend(["--services", k])
    logging.info("Calling downloader to fetch missing services: %s", missing)
    logging.debug("Downloader command: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logging.info("Downloader finished (exit code 0).")
    except subprocess.CalledProcessError as e:
        logging.warning("Downloader failed (exit code %s): %s", e.returncode, e)


def build_mosaic_from_tiles(tile_dir: Path, dst_crs: str, bounds: Tuple[float, float, float, float], res: float):
    tile_paths = sorted([str(p) for p in tile_dir.glob("*.tif")])
    if not tile_paths:
        return None, None

    srcs = []
    for p in tile_paths:
        try:
            src = rasterio.open(p)
        except Exception as e:
            logging.warning("Could not open tile %s: %s", p, e)
            continue
        try:
            if dst_crs and str(src.crs) != dst_crs:
                # Use nearest for category by default, but WarpedVRT will be used
                vrt = WarpedVRT(src, crs=dst_crs, resampling=Resampling.nearest)
                srcs.append(vrt)
            else:
                srcs.append(src)
        except Exception as e:
            logging.warning("Failed to build VRT for %s: %s", p, e)
            try:
                src.close()
            except Exception:
                pass
            continue

    if not srcs:
        return None, None

    try:
        mosaic, out_trans = rio_merge(srcs, bounds=bounds, res=res)
    except Exception as e:
        logging.warning("rio_merge failed for tiles in %s: %s", tile_dir, e)
        mosaic = None
        out_trans = None
    finally:
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass
    return mosaic, out_trans


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Project area -> GeoJSON samples (EPSG:4326) using downloaded ArcGIS tiles.")
    parser.add_argument("--index", required=True, help="Parquet catalog (pyarrow) produced by tif_catalog")
    parser.add_argument("--project-geojson", required=True, help="Project area geojson/shapefile (used to clip/select points)")
    parser.add_argument("--wetland-geojson", default=None, help="Optional wetland geojson/shapefile")
    parser.add_argument("--buffer", type=float, default=0.0, help="Buffer (meters) around project area")
    parser.add_argument("--res", type=float, default=1.0, help="Working resolution in meters")
    parser.add_argument("--patch-size", type=int, default=7, help="Patch size (odd integer) used to compute wetland_frac")
    parser.add_argument("--samples-per-chunk", type=int, default=20000, help="Samples per memory chunk to process/write")
    parser.add_argument("--out-dir", default="processed_project", help="Output directory (geojson will be written here)")
    parser.add_argument("--src-crs-if-missing", default=None, help="If input vector lacks CRS, assume this (e.g. EPSG:4326). If unset, assume working CRS (EPSG:32617).")
    parser.add_argument("--catalog-crs", default=None, help="Override catalog CRS (if parquet lacks 'crs' or it's incorrect).")
    parser.add_argument("--arcgis-config", default=DOWNLOADER_DEFAULTS["arcgis_config"], help="Path to YAML/JSON configuration file with ArcGIS REST settings.")
    parser.add_argument("--arcgis-tile-out", default=DOWNLOADER_DEFAULTS["out_dir"], help="Directory where downloader places tiles (raw_arcgis by default).")
    parser.add_argument("--arcgis-tile-size", type=int, default=DOWNLOADER_DEFAULTS["tile_size_m"], help="Tile size passed to downloader (meters).")
    parser.add_argument("--arcgis-res-m", type=int, default=DOWNLOADER_DEFAULTS["res_m"], help="Resolution passed to downloader (meters).")
    parser.add_argument("--chunk-size-arcgis", type=int, default=1000, help="Points per ArcGIS exportImage request if falling back.")
    parser.add_argument("--edge-buffer", type=float, default=5000.0,
                        help="Meters of extra buffer around working bounds to include when building mosaics and computing derived rasters (avoid edge effects).")
    args = parser.parse_args()

    if args.patch_size < 1 or args.patch_size % 2 == 0:
        raise SystemExit("--patch-size must be an odd positive integer (e.g. 7)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = out_dir / "rename.geojson"

    # load arcgis config path for downloader invocation
    arcgis_cfg_path = args.arcgis_config
    arc_tile_root = Path(args.arcgis_tile_out)

    # Load config for ArcGIS service url discovery
    arcgis_cfg: Dict[str, Any] = {}
    if arcgis_cfg_path:
        cfg_path = Path(arcgis_cfg_path)
        if cfg_path.exists():
            with open(cfg_path, "r") as fh:
                if cfg_path.suffix.lower() in (".yaml", ".yml"):
                    arcgis_cfg = yaml.safe_load(fh) or {}
                else:
                    arcgis_cfg = json.load(fh) or {}
            logging.info(f"Loaded ArcGIS config from {cfg_path}")
    arcgis_section = arcgis_cfg.get("arcgis", {}) if isinstance(arcgis_cfg, dict) else {}
    soil_url = arcgis_section.get("soil_url")
    precip_url = arcgis_section.get("precip_url")
    service_keys = []
    if soil_url:
        service_keys.append("soil_url")
    if precip_url:
        service_keys.append("precip_url")

    # Ensure tiles exist (call downloader if missing)
    if service_keys:
        call_downloader_if_needed(service_keys, arcgis_cfg_path, str(arc_tile_root),
                                  tile_size_m=args.arcgis_tile_size, res_m=args.arcgis_res_m, downloader_script_path=DOWNLOADER_SCRIPT)
    else:
        logging.info("No soil/precip URLs found in arcgis_config; skipping downloader check.")

    # load catalog, project geom, compute mosaic of DEM etc
    catalog_rows = load_catalog_pyarrow(Path(args.index))
    catalog_crs_detected = catalog_get_crs(catalog_rows)
    if args.catalog_crs:
        catalog_crs = args.catalog_crs
        logging.info(f"Using user-specified catalog CRS override: {catalog_crs}")
    elif catalog_crs_detected:
        catalog_crs = catalog_crs_detected
        logging.info(f"Detected catalog CRS in parquet: {catalog_crs}")
    else:
        catalog_crs = _WORKING_CRS
        logging.info(f"No catalog CRS found: assuming working CRS {_WORKING_CRS}")

    project_geom, project_src_crs = read_geom_union_via_fiona(args.project_geojson)
    if project_src_crs is None:
        if args.src_crs_if_missing:
            project_src_crs = args.src_crs_if_missing
            logging.info(f"Project vector lacked CRS; assuming {project_src_crs} per --src-crs-if-missing")
        else:
            project_src_crs = _WORKING_CRS
            logging.info(f"Project vector lacked CRS; assuming it is already in working CRS {_WORKING_CRS}")
    project_geom = reproject_geom(project_geom, project_src_crs, _WORKING_CRS)
    logging.info(f"Project geometry ensured in {_WORKING_CRS}")

    # Working geometry (project area with optional user buffer) - this is the sampling area
    area_geom = project_geom.buffer(args.buffer)
    bounds_working = area_geom.bounds
    logging.info(f"Working bounds (EPSG:32617): {bounds_working}")

    # Expand processing bounds for mosaic creation (avoid edge effects for slope/aspect/flow)
    edge_buf = float(args.edge_buffer) if hasattr(args, "edge_buffer") else 0.0
    if edge_buf and edge_buf > 0:
        minx, miny, maxx, maxy = bounds_working
        processing_bounds = (minx - edge_buf, miny - edge_buf, maxx + edge_buf, maxy + edge_buf)
        logging.info(f"Processing bounds expanded by edge-buffer={edge_buf} m -> {processing_bounds}")
    else:
        processing_bounds = bounds_working
        logging.info("No edge-buffer requested; using working bounds for processing.")

    # Select tifs (for DEM mosaic) using expanded processing bounds so derived rasters have context
    tif_list = select_tifs_from_catalog_rows(catalog_rows, processing_bounds, catalog_crs, _WORKING_CRS)
    if not tif_list:
        raise SystemExit("No TIFFs selected from catalog after transforming catalog bboxes into working CRS. Try increasing --buffer or verify CRS.")
    logging.info(f"Selected {len(tif_list)} TIFFs for mosaic (using processing bounds)")

    mosaic, transform, profile = compute_mosaic(tif_list, bounds=processing_bounds, res=args.res, dst_crs=_WORKING_CRS)
    dem = mosaic[0].astype('float32')
    dem = np.where(np.isfinite(dem), dem, np.nan)
    H, W = dem.shape
    logging.info(f"Mosaic shape: {W} x {H}, resolution {args.res} m")

    rd_dem = rd.rdarray(dem.copy(), no_data=float(np.nan))
    t = transform
    rd_dem.geotransform = (t.c, t.a, t.b, t.f, t.d, t.e)

    logging.info("Computing filled DEM ...")
    filled = rd.FillDepressions(rd_dem, in_place=False)
    filled_arr = np.asarray(filled)

    channels_map = {"dem": dem, "filled": filled_arr}
    logging.info("Computing slope and aspect and flow ...")
    channels_map["slope"] = np.asarray(rd.TerrainAttribute(rd_dem, attrib="slope_degrees"))
    channels_map["aspect"] = np.asarray(rd.TerrainAttribute(rd_dem, attrib="aspect"))
    channels_map["flow"] = np.asarray(rd.FlowAccumulation(filled, method="D8"))

    for ch in ("dem", "filled", "slope", "aspect", "flow"):
        channels_map.setdefault(ch, np.full_like(dem, np.nan))

    # -------------------------
    # Wetland geometry: raster-first setup (reproject, rasterize to DEM grid, prepare)
    # -------------------------
    wet_geom = None
    wet_prep = None
    wet_raster = None
    if args.wetland_geojson:
        wet_geom_read, wet_src_crs = read_geom_union_via_fiona(args.wetland_geojson)
        # if fiona didn't provide a CRS, honor CLI override or assume EPSG:4326 for GeoJSON-like inputs
        if wet_src_crs is None:
            if args.src_crs_if_missing:
                wet_src_crs = args.src_crs_if_missing
            else:
                # many GeoJSONs use lon/lat; default to EPSG:4326 when unsure
                wet_src_crs = "EPSG:4326"

        # Reproject wetland geometry into working CRS (EPSG:32617)
        wet_geom = reproject_geom(wet_geom_read, wet_src_crs, _WORKING_CRS)

        # Rasterize onto the DEM grid (use same transform and shape) -> boolean mask
        try:
            raw_wet = rasterize([(wet_geom, 1)], out_shape=(H, W), transform=transform,
                                fill=0, dtype='uint8', all_touched=False)
            wet_raster = (raw_wet == 1)
            logging.info("wet_raster created: shape=%s, total_wet_pixels=%d", wet_raster.shape, int(np.count_nonzero(wet_raster)))
        except Exception as e:
            wet_raster = None
            logging.warning("wetland rasterize failed: %s", e)

        # Prepare prepared-geometry for fast intersects() tests (fallback)
        try:
            wet_prep = prep(wet_geom)
            logging.info("Prepared wetland geometry for prepared.intersects() tests.")
        except Exception as e:
            wet_prep = None
            logging.warning("Failed to prepare wetland geometry: %s", e)

        # Diagnostics: log bounds to help detect CRS/extent mismatches
        try:
            logging.info("wet_geom bounds (working CRS): %s", (wet_geom.bounds if wet_geom is not None else None))
            logging.info("project_geom bounds (working CRS): %s", (project_geom.bounds if project_geom is not None else None))
        except Exception:
            pass
    else:
        wet_geom = None
        wet_prep = None
        wet_raster = None

    # Prepare project geometry for point-in-polygon tests (boundary-inclusive)
    proj_prep = None
    try:
        proj_prep = prep(project_geom)
        logging.info("Prepared project geometry for fast point-in-polygon tests.")
    except Exception as e:
        logging.warning(f"Failed to prepare project geometry: {e}")
        proj_prep = None

    # Rasterize project polygon to boolean mask (True = inside)
    try:
        proj_mask_r = rasterize(
            [(project_geom, 1)],
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype='uint8',
            all_touched=False
        )
        proj_mask = proj_mask_r.astype(bool)
        logging.info("Project polygon rasterized to explicit boolean mask (True=inside).")
    except Exception as e:
        logging.warning(f"Project polygon rasterize failed, falling back to geometry_mask: {e}")
        proj_mask = geometry_mask([project_geom], transform=transform, invert=True, out_shape=(H, W))
        proj_mask_r = proj_mask.astype('uint8')

    # If wet_raster exists and proj_mask_r exists, log intersection
    try:
        if wet_raster is not None and 'proj_mask_r' in locals():
            try:
                overlap = int(np.count_nonzero(wet_raster & proj_mask_r))
                logging.info("wet_raster ∩ project_mask nonzero pixels = %d", overlap)
            except Exception:
                pass
    except Exception:
        pass

    # Pixel centers
    cols = np.arange(W)
    rows_arr = np.arange(H)
    col_grid, row_grid = np.meshgrid(cols, rows_arr)
    xs, ys = rio_transform.xy(transform, row_grid, col_grid, offset='center')
    xs = np.asarray(xs); ys = np.asarray(ys)

    xs_f = xs.ravel(); ys_f = ys.ravel()
    dem_f = channels_map["dem"].ravel()
    filled_f = channels_map["filled"].ravel()
    slope_f = channels_map["slope"].ravel()
    aspect_f = channels_map["aspect"].ravel()
    flow_f = channels_map["flow"].ravel()
    proj_mask_f = proj_mask.ravel()

    # Choose valid pixels: fast raster prefilter, then exact geometry-based filter (boundary-inclusive)
    valid_dem = np.isfinite(dem_f)
    fast_keep = valid_dem & proj_mask_f   # fast prefilter using rasterized project mask
    candidate_indices = np.nonzero(fast_keep)[0]
    logging.info(f"Fast-prefilter candidates inside rasterized project: {len(candidate_indices)}")

    # Now apply exact geometry test (prepared.intersects) for every candidate to ensure boundary-inclusive behavior
    indices_list = []
    if proj_prep is not None:
        for flat_idx in candidate_indices:
            x = float(xs_f[flat_idx]); y = float(ys_f[flat_idx])
            try:
                if proj_prep.intersects(Point(x, y)):
                    indices_list.append(flat_idx)
            except Exception:
                try:
                    if project_geom.intersects(Point(x, y)):
                        indices_list.append(flat_idx)
                except Exception:
                    continue
    else:
        for flat_idx in candidate_indices:
            x = float(xs_f[flat_idx]); y = float(ys_f[flat_idx])
            try:
                if project_geom.intersects(Point(x, y)):
                    indices_list.append(flat_idx)
            except Exception:
                continue

    indices = np.asarray(indices_list, dtype=np.int64)
    total_samples = len(indices)
    logging.info(f"Exact geometry-filtered samples (project intersects): {total_samples}")
    if total_samples == 0:
        logging.info("No samples to produce after exact project geometry test; exiting.")
        return

    # -------------------------
    # Wetland two-stage intersection (raster prefilter -> exact geometry check)
    # Precompute a flattened boolean array wet_flags_f (length = number of raster pixels)
    # -------------------------
    n_pixels = dem_f.size
    wet_flags_f = np.zeros((n_pixels,), dtype=np.uint8)

    if wet_raster is not None:
        try:
            wet_mask_f = wet_raster.ravel().astype(bool)
        except Exception:
            wet_mask_f = np.zeros((n_pixels,), dtype=bool)

        valid_dem_mask = np.isfinite(dem_f)
        fast_wet_keep = valid_dem_mask & wet_mask_f
        candidate_wet_indices = np.nonzero(fast_wet_keep)[0]
        logging.info(f"Fast-prefilter wetland candidates from raster: {len(candidate_wet_indices)}")

        if len(candidate_wet_indices) > 0 and (wet_prep is not None or wet_geom is not None):
            if wet_prep is not None:
                for flat_idx in candidate_wet_indices:
                    x = float(xs_f[flat_idx]); y = float(ys_f[flat_idx])
                    try:
                        if wet_prep.intersects(Point(x, y)):
                            wet_flags_f[flat_idx] = 1
                    except Exception:
                        try:
                            if wet_geom is not None and wet_geom.intersects(Point(x, y)):
                                wet_flags_f[flat_idx] = 1
                        except Exception:
                            pass
            else:
                for flat_idx in candidate_wet_indices:
                    x = float(xs_f[flat_idx]); y = float(ys_f[flat_idx])
                    try:
                        if wet_geom.intersects(Point(x, y)):
                            wet_flags_f[flat_idx] = 1
                    except Exception:
                        pass
        else:
            # no geometry provided or no candidates: leave wet_flags_f as zeros
            pass

    else:
        # No wetland raster available; fall back to geometry-only precomputation across all valid pixels
        logging.info("No wetland raster available; performing geometry-only precomputation for wet flags (this may be slower).")
        valid_dem_mask = np.isfinite(dem_f)
        candidate_indices_geom = np.nonzero(valid_dem_mask)[0]
        if len(candidate_indices_geom) > 0 and (wet_prep is not None or wet_geom is not None):
            if wet_prep is not None:
                for flat_idx in candidate_indices_geom:
                    x = float(xs_f[flat_idx]); y = float(ys_f[flat_idx])
                    try:
                        if wet_prep.intersects(Point(x, y)):
                            wet_flags_f[flat_idx] = 1
                    except Exception:
                        try:
                            if wet_geom is not None and wet_geom.intersects(Point(x, y)):
                                wet_flags_f[flat_idx] = 1
                        except Exception:
                            pass
            else:
                for flat_idx in candidate_indices_geom:
                    x = float(xs_f[flat_idx]); y = float(ys_f[flat_idx])
                    try:
                        if wet_geom.intersects(Point(x, y)):
                            wet_flags_f[flat_idx] = 1
                    except Exception:
                        pass
        else:
            # nothing to mark
            pass

    # Build mosaics for soil/precip from downloaded tiles (if available)
    soil_mosaic = None; soil_trans = None
    precip_mosaic = None; precip_trans = None
    soil_tile_dir = arc_tile_root / "soil_url"
    if soil_tile_dir.exists() and any(soil_tile_dir.glob("*.tif")):
        logging.info("Building soil mosaic from tiles in %s", soil_tile_dir)
        soil_mosaic, soil_trans = build_mosaic_from_tiles(soil_tile_dir, _WORKING_CRS, bounds_working, res=args.res)
        if soil_mosaic is not None:
            logging.info("Soil mosaic built (shape=%s).", str(soil_mosaic.shape))
    else:
        logging.info("No soil tiles found at %s; will fall back to API sampling later.", soil_tile_dir)

    precip_tile_dir = arc_tile_root / "precip_url"
    if precip_tile_dir.exists() and any(precip_tile_dir.glob("*.tif")):
        logging.info("Building precip mosaic from tiles in %s", precip_tile_dir)
        precip_mosaic, precip_trans = build_mosaic_from_tiles(precip_tile_dir, _WORKING_CRS, bounds_working, res=args.res)
        if precip_mosaic is not None:
            logging.info("Precip mosaic built (shape=%s).", str(precip_mosaic.shape))
    else:
        logging.info("No precip tiles found at %s; will fall back to API sampling later.", precip_tile_dir)

    # Prepare fallback ArcGIS session/token if mosaics not available
    arc_cfg = arcgis_section if 'arcgis_section' in locals() else (yaml.safe_load(open(arcgis_cfg_path))['arcgis'] if arcgis_cfg_path and Path(arcgis_cfg_path).exists() else {})
    session = _make_requests_session()
    access_token = None
    client_id = arc_cfg.get("client_id") if isinstance(arc_cfg, dict) else None
    client_secret = arc_cfg.get("client_secret") if isinstance(arc_cfg, dict) else None
    if client_id and client_secret:
        logging.info("Requesting OAuth2 token using client_id/client_secret (client_credentials)...")
        access_token = fetch_access_token_from_client_credentials(session, client_id, client_secret, portal=arc_cfg.get("portal", "https://www.arcgis.com"), timeout=int(arc_cfg.get("timeout", 60)))
        if access_token:
            logging.info("Successfully obtained access token via client_credentials.")
    if not access_token and arc_cfg.get("token"):
        access_token = arc_cfg.get("token")
        logging.info("Using token provided in config (arcgis.token).")
    if access_token:
        token_header_type = arc_cfg.get("token_header_type", "Bearer")
        if token_header_type:
            session.headers.update({"Authorization": f"{token_header_type} {access_token}"})
        else:
            session.headers.update({"Authorization": f"Bearer {access_token}"})

    # Create GeoJSON streamer
    geojson_streamer = GeoJSONStreamer(geojson_path, src_crs=_WORKING_CRS)

    spc = int(args.samples_per_chunk)
    n_chunks = math.ceil(total_samples / spc)
    logging.info(f"Processing in {n_chunks} chunk(s) of up to {spc} samples each")

    patch_half = args.patch_size // 2

    sample_counter = 0
    try:
        for chunk_idx in range(n_chunks):
            start = chunk_idx * spc
            end = min(total_samples, (chunk_idx + 1) * spc)
            sel_idx = indices[start:end]
            N = len(sel_idx)
            if N == 0:
                continue

            easts = np.empty((N,), dtype='float64')
            norths = np.empty((N,), dtype='float64')
            dem_vals = np.empty((N,), dtype='float32')
            filled_vals = np.empty((N,), dtype='float32')
            slope_vals = np.empty((N,), dtype='float32')
            aspect_vals = np.empty((N,), dtype='float32')
            flow_vals = np.empty((N,), dtype='float32')
            wet_vals = np.zeros((N,), dtype='int8')  # 0 or 1
            wet_frac_vals = np.zeros((N,), dtype='float32')  # fraction 0..1

            for i_local, flat_idx in enumerate(sel_idx):
                easts[i_local] = float(xs_f[flat_idx])
                norths[i_local] = float(ys_f[flat_idx])
                dem_vals[i_local] = float(dem_f[flat_idx]) if np.isfinite(dem_f[flat_idx]) else float("nan")
                filled_vals[i_local] = float(filled_f[flat_idx]) if np.isfinite(filled_f[flat_idx]) else float("nan")
                slope_vals[i_local] = float(slope_f[flat_idx]) if np.isfinite(slope_f[flat_idx]) else float("nan")
                aspect_vals[i_local] = float(aspect_f[flat_idx]) if np.isfinite(aspect_f[flat_idx]) else float("nan")
                flow_vals[i_local] = float(flow_f[flat_idx]) if np.isfinite(flow_f[flat_idx]) else float("nan")

                # --- Wetland assignment using precomputed wet_flags_f ---
                try:
                    wet_vals[i_local] = int(bool(wet_flags_f[int(flat_idx)]))
                except Exception:
                    wet_vals[i_local] = 0

            # Sample soil & precip from mosaics if present, else fallback to exportImage per-subchunk
            soil_vals = [None] * N
            precip_vals = [None] * N

            if soil_mosaic is not None and soil_trans is not None:
                soil_vals = sample_raster_at_points(soil_mosaic, soil_trans, easts.tolist(), norths.tolist(), method="nearest")
            if precip_mosaic is not None and precip_trans is not None:
                precip_vals = sample_raster_at_points(precip_mosaic, precip_trans, easts.tolist(), norths.tolist(), method="bilinear")

            # Identify which still need fallback
            need_soil_idx = [i for i, v in enumerate(soil_vals) if v is None]
            need_precip_idx = [i for i, v in enumerate(precip_vals) if v is None]

            # local exportImage sampling helper (unchanged)
            def sample_imageserver_export_local(session_local, base_url, pts_x, pts_y, token_local, service_srid, pts_crs=_WORKING_CRS, timeout=60):
                if len(pts_x) == 0:
                    return []
                xs_local = [float(x) for x in pts_x]; ys_local = [float(y) for y in pts_y]
                minx = float(min(xs_local)); miny = float(min(ys_local)); maxx = float(max(xs_local)); maxy = float(max(ys_local))
                pad_x = (maxx - minx) * 0.01 if (maxx - minx) > 0 else 0.5
                pad_y = (maxy - miny) * 0.01 if (maxy - miny) > 0 else 0.5
                minx -= pad_x; maxx += pad_x; miny -= pad_y; maxy += pad_y

                bbox_for_service = (minx, miny, maxx, maxy)
                bbox_sr_id = None
                if service_srid is not None:
                    try:
                        transformer = Transformer.from_crs(pts_crs, f"EPSG:{service_srid}", always_xy=True)
                        c1 = transformer.transform(minx, miny)
                        c2 = transformer.transform(maxx, maxy)
                        bx_min = min(c1[0], c2[0]); bx_max = max(c1[0], c2[0])
                        by_min = min(c1[1], c2[1]); by_max = max(c1[1], c2[1])
                        bbox_for_service = (bx_min, by_min, bx_max, by_max)
                        bbox_sr_id = service_srid
                    except Exception:
                        bbox_for_service = (minx, miny, maxx, maxy)
                        bbox_sr_id = None
                else:
                    if pts_crs.upper().startswith("EPSG:"):
                        try:
                            bbox_sr_id = int(pts_crs.split(":")[-1])
                        except Exception:
                            bbox_sr_id = None

                dx = bbox_for_service[2] - bbox_for_service[0]
                dy = bbox_for_service[3] - bbox_for_service[1]
                est_px = int(max(32, min(2048, math.ceil(max(abs(dx), abs(dy)) / 1.0 if max(abs(dx), abs(dy)) > 0 else 64))))
                width = min(2048, max(32, est_px))
                height = min(2048, max(32, est_px))

                params = {
                    "f": "image",
                    "bbox": f"{bbox_for_service[0]},{bbox_for_service[1]},{bbox_for_service[2]},{bbox_for_service[3]}",
                    "bboxSR": str(bbox_sr_id) if bbox_sr_id is not None else pts_crs.split(":")[-1],
                    "imageSR": str(bbox_sr_id) if bbox_sr_id is not None else pts_crs.split(":")[-1],
                    "size": f"{width},{height}",
                    "format": "tiff",
                    "pixelType": "F32"
                }
                if token_local:
                    params["token"] = token_local

                export_url = base_url.rstrip("/") + "/exportImage"
                try:
                    r = session_local.get(export_url, params=params, timeout=timeout, stream=True)
                    r.raise_for_status()
                except Exception as e:
                    logging.warning("exportImage failed for %s: %s", base_url, e)
                    return [None] * len(xs_local)

                try:
                    mem = io.BytesIO(r.content)
                    with MemoryFile(mem) as mf:
                        with mf.open() as src:
                            with WarpedVRT(src, crs=pts_crs, resampling=Resampling.nearest) as vrt:
                                coords = list(zip(xs_local, ys_local))
                                out_local = []
                                for s in vrt.sample(coords):
                                    if s is None:
                                        out_local.append(None)
                                    else:
                                        v = s[0] if isinstance(s, (list, tuple, np.ndarray)) else s
                                        if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)):
                                            out_local.append(None)
                                        else:
                                            out_local.append(float(v))
                                return out_local
                except Exception as e:
                    logging.warning("Failed to read exported image: %s", e)
                    return [None] * len(xs_local)

            # fallback sampling for soil
            if need_soil_idx and soil_url:
                a_chunk = int(args.chunk_size_arcgis)
                for a0 in range(0, len(need_soil_idx), a_chunk):
                    sub_idx_slice = need_soil_idx[a0:a0 + a_chunk]
                    sub_x = [easts[i] for i in sub_idx_slice]
                    sub_y = [norths[i] for i in sub_idx_slice]
                    try:
                        vals = sample_imageserver_export_local(session, soil_url, sub_x, sub_y, access_token, None, pts_crs=_WORKING_CRS, timeout=int(arc_cfg.get("timeout", 60)))
                        for k, v in enumerate(vals):
                            soil_vals[sub_idx_slice[k]] = v
                    except Exception as e:
                        logging.warning("Soil fallback sampling failed for indices %s: %s", sub_idx_slice, e)

            # fallback sampling for precip
            if need_precip_idx and precip_url:
                a_chunk = int(args.chunk_size_arcgis)
                for a0 in range(0, len(need_precip_idx), a_chunk):
                    sub_idx_slice = need_precip_idx[a0:a0 + a_chunk]
                    sub_x = [easts[i] for i in sub_idx_slice]
                    sub_y = [norths[i] for i in sub_idx_slice]
                    try:
                        vals = sample_imageserver_export_local(session, precip_url, sub_x, sub_y, access_token, None, pts_crs=_WORKING_CRS, timeout=int(arc_cfg.get("timeout", 60)))
                        for k, v in enumerate(vals):
                            precip_vals[sub_idx_slice[k]] = v
                    except Exception as e:
                        logging.warning("Precip fallback sampling failed for indices %s: %s", sub_idx_slice, e)

            # write features to geojson (transformer inside streamer converts to EPSG:4326)
            for i_local in range(N):
                sval = soil_vals[i_local]
                pval = precip_vals[i_local]
                soil_code = None
                if sval is not None:
                    try:
                        soil_code = int(round(float(sval)))
                    except Exception:
                        soil_code = None
                precip_val = None
                if pval is not None:
                    try:
                        precip_val = float(pval)
                    except Exception:
                        precip_val = None

                props = {
                    "elv": (None if math.isnan(dem_vals[i_local]) else float(dem_vals[i_local])),
                    "fill": (None if math.isnan(filled_vals[i_local]) else float(filled_vals[i_local])),
                    "slope": (None if math.isnan(slope_vals[i_local]) else float(slope_vals[i_local])),
                    "asp": (None if math.isnan(aspect_vals[i_local]) else float(aspect_vals[i_local])),
                    "flow": (None if math.isnan(flow_vals[i_local]) else float(flow_vals[i_local])),
                    "soil": soil_code if soil_code is not None else None,
                    "precip": precip_val if precip_val is not None else None,
                    "wet": int(wet_vals[i_local])
                }
                geojson_streamer.write_feature(easts[i_local], norths[i_local], props)

            sample_counter += N
            logging.info(f"Chunk {chunk_idx+1}/{n_chunks} written: {N} samples; total_written={sample_counter}")

    finally:
        try:
            geojson_streamer.close(generated_iso=datetime.utcnow().isoformat() + "Z", count=sample_counter)
        except Exception:
            logging.warning("Failed to close geojson streamer cleanly")

    logging.info(f"Wrote GeoJSON: {geojson_path}")

    try:
        if pd is not None:
            with open(geojson_path, "r", encoding="utf-8") as fh:
                gj = json.load(fh)
            feats = [f["properties"] for f in gj.get("features", [])[:5]]
            df = pd.DataFrame(feats)
            print("\nSample preview (first 5 features):")
            print(df.head(5).to_string(index=False))
        else:
            with open(geojson_path, "r", encoding="utf-8") as fh:
                for _ in range(20):
                    line = fh.readline()
                    if not line:
                        break
                    print(line.strip())
    except Exception as e:
        logging.debug(f"Preview failed: {e}")

    logging.info("All done.")

def fetch_access_token_from_client_credentials(session: requests.Session, client_id: str, client_secret: str, portal: str = "https://www.arcgis.com", timeout: int = 30) -> Optional[str]:
    token_url = portal.rstrip("/") + "/sharing/rest/oauth2/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "f": "json"
    }
    try:
        r = session.post(token_url, data=data, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        access_token = j.get("access_token") or j.get("token")
        if not access_token:
            logging.warning(f"No access_token found in token response: {j}")
            return None
        return access_token
    except Exception as e:
        logging.warning(f"Failed to fetch access token from {token_url}: {e}")
        return None


if __name__ == "__main__":
    main()