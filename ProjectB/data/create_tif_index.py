#!/usr/bin/env python3

from pathlib import Path
import argparse
import logging
import math
from typing import List, Dict, Optional
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
from datetime import datetime
from pyproj import Transformer, CRS

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

# try to set PROJ_LIB to pyproj datadir for consistent reprojection where available
try:
    os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
except Exception:
    pass

_NODATA_FALLBACK = None
_WORKING_CRS = "EPSG:32617"   # ALWAYS write this into parquet 'crs' field
_WORKING_CRS_NORMALIZED = None
try:
    _WORKING_CRS_NORMALIZED = CRS.from_user_input(_WORKING_CRS).to_string()
except Exception:
    _WORKING_CRS_NORMALIZED = _WORKING_CRS


def inspect_tif(path: str, dst_crs: Optional[str], resampling: str, include_wkt: bool) -> Dict:
    """
    Inspect a single TIFF and return metadata dict.
    If dst_crs is provided, compute bounds/resolution in that CRS (via WarpedVRT).
    The returned dict will include 'crs' field set to _WORKING_CRS_NORMALIZED (always).
    """
    p = Path(path)
    out: Dict = {"path": str(p.resolve()), "name": p.name}
    try:
        src = rasterio.open(p)
    except Exception as e:
        out["error"] = f"open_failed: {e}"
        return out

    vrt = None
    try:
        if dst_crs:
            try:
                vrt = WarpedVRT(src, crs=dst_crs, resampling=Resampling[resampling])
                r = vrt
            except Exception as e:
                logging.warning(f"Failed to build VRT for {p}: {e}; falling back to native dataset for inspection.")
                r = src
        else:
            r = src

        b = r.bounds
        out["minx"], out["miny"], out["maxx"], out["maxy"] = float(b.left), float(b.bottom), float(b.right), float(b.top)

        # Always write the working CRS string to parquet metadata to ensure consistency downstream
        out["crs"] = _WORKING_CRS_NORMALIZED

        out["width"] = int(r.width)
        out["height"] = int(r.height)

        try:
            res_x, res_y = r.res
            out["pixel_size_x"] = float(res_x)
            out["pixel_size_y"] = float(res_y)
            out["pixel_area_m2"] = float(abs(res_x * res_y))
        except Exception:
            out["pixel_size_x"] = None
            out["pixel_size_y"] = None
            out["pixel_area_m2"] = None

        out["driver"] = src.driver
        try:
            out["dtype"] = str(src.dtypes[0]) if src.count >= 1 else None
        except Exception:
            out["dtype"] = None

        try:
            nd = r.nodatavals[0]
            out["nodatavals"] = nd if nd is not None else _NODATA_FALLBACK
        except Exception:
            out["nodatavals"] = _NODATA_FALLBACK

        if out.get("pixel_area_m2") is not None and out.get("width") is not None and out.get("height") is not None:
            out["approx_area_m2"] = float(out["width"] * out["height"] * out["pixel_area_m2"])
        else:
            out["approx_area_m2"] = None

        if include_wkt:
            minx, miny, maxx, maxy = out["minx"], out["miny"], out["maxx"], out["maxy"]
            # bbox written in the same coordinates as recorded (i.e., in dst_crs if used)
            wkt = f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"
            out["bbox_wkt"] = wkt

        if vrt is not None:
            try:
                vrt.close()
            except Exception:
                pass
    except Exception as e:
        out["error"] = f"inspect_failed: {e}"
    finally:
        try:
            src.close()
        except Exception:
            pass

    return out


def scan_folder(tif_dir: Path, dst_crs: Optional[str], pattern: str, workers: int, resampling: str, include_wkt: bool) -> List[Dict]:
    tif_files = sorted(tif_dir.glob(pattern))
    paths = [str(p) for p in tif_files]
    rows: List[Dict] = []

    if workers <= 1:
        for p in tqdm(paths, desc="Scanning TIFFs", unit="file"):
            rows.append(inspect_tif(p, dst_crs, resampling, include_wkt))
        return rows

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(inspect_tif, p, dst_crs, resampling, include_wkt): p for p in paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning TIFFs", unit="file"):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"path": str(futures[fut]), "error": f"worker_failed: {e}"})
    return rows


def coerce_for_parquet(rows: List[Dict]) -> pa.Table:
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    columns = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k, None)
            if isinstance(v, (bytes, bytearray)):
                try:
                    v = v.decode("utf-8")
                except Exception:
                    v = str(v)
            columns[k].append(v)
    return pa.table(columns)


def write_parquet(table: pa.Table, out_path: Path, compression: str = "zstd"):
    pq.write_table(table, out_path.as_posix(), compression=compression)


def build_geojson_epsg4326(rows: List[Dict], out_geojson: Path, catalog_crs: Optional[str]) -> None:
    """
    Build GeoJSON in EPSG:4326. Assumes rows hold bbox coords in the coordinate system used to compute them
    (if --dst-crs was used this will be that CRS). We will treat recorded coords as EPSG:32617 (the working CRS),
    and reproject to EPSG:4326 for GeoJSON output.
    """
    features = []
    # we will reproject from working CRS (_WORKING_CRS_NORMALIZED) -> 4326
    try:
        transformer = Transformer.from_crs(_WORKING_CRS_NORMALIZED, "EPSG:4326", always_xy=True)
    except Exception:
        transformer = None

    for r in rows:
        if "error" in r:
            continue
        try:
            minx, miny, maxx, maxy = float(r["minx"]), float(r["miny"]), float(r["maxx"]), float(r["maxy"])
        except Exception:
            continue

        ring = [
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
            (minx, miny),
        ]

        if transformer is not None:
            try:
                reproj_ring = [tuple(transformer.transform(x, y)) for (x, y) in ring]
                coords = [[float(lon), float(lat)] for (lon, lat) in reproj_ring]
            except Exception:
                coords = [[float(x), float(y)] for (x, y) in ring]
        else:
            coords = [[float(x), float(y)] for (x, y) in ring]

        props = dict(r)
        # ensure we explicitly record that rows are interpreted as the working CRS
        props["orig_written_crs"] = _WORKING_CRS_NORMALIZED
        props["geojson_crs"] = "EPSG:4326"

        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": props,
        }
        features.append(feature)

    fc = {
        "type": "FeatureCollection",
        "features": features,
        "generated": datetime.utcnow().isoformat() + "Z",
        "count": len(features),
    }

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    with out_geojson.open("w", encoding="utf-8") as fh:
        json.dump(fc, fh, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description="Create a Parquet and/or GeoJSON catalog of GeoTIFFs. The parquet 'crs' field will always be EPSG:32617.")
    p.add_argument("--tif-dir", type=str, default="raw_tif", help="Directory containing .tif files")
    p.add_argument("--out", type=str, default="tif_index.parquet", help="Output parquet file (optional; use --no-parquet to skip)")
    p.add_argument("--out-geojson", type=str, default=None, help="Output GeoJSON file (optional). GeoJSON will be written in EPSG:4326.")
    p.add_argument("--no-parquet", action="store_true", help="Do not write parquet output")
    p.add_argument("--dst-crs", type=str, default=_WORKING_CRS, help="Reproject bounds/res to this CRS (default enforced to EPSG:32617)")
    p.add_argument("--pattern", type=str, default="*.tif", help="Glob pattern for TIFFs")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1)")
    p.add_argument("--resampling", type=str, default="nearest", choices=["nearest", "bilinear", "cubic"], help="Resampling for bounds reprojection")
    p.add_argument("--wkt", action="store_true", help="Include bbox WKT polygon in the catalog (parquet only)")
    args = p.parse_args()

    tif_dir = Path(args.tif_dir)
    if not tif_dir.exists():
        raise SystemExit(f"tif-dir not found: {tif_dir}")

    workers = args.workers if args.workers > 0 else 1
    if workers > 1:
        workers = min(workers, multiprocessing.cpu_count())

    # enforce dst_crs default to working CRS to avoid confusion
    dst_crs = args.dst_crs or _WORKING_CRS
    logging.info(f"Scanning {tif_dir} pattern={args.pattern} dst_crs={dst_crs} workers={workers}")
    rows = scan_folder(tif_dir, dst_crs, args.pattern, workers, args.resampling, args.wkt)

    # Ensure every row contains the canonical working CRS string
    for r in rows:
        # set/overwrite the crs field to the working CRS normalized string
        r["crs"] = _WORKING_CRS_NORMALIZED

    if not args.no_parquet:
        table = coerce_for_parquet(rows)
        out_path = Path(args.out)
        write_parquet(table, out_path)
        logging.info(f"Wrote Parquet catalog with {len(rows)} entries to {out_path} (crs field set to {_WORKING_CRS_NORMALIZED})")
    else:
        logging.info("Skipping Parquet output (--no-parquet).")

    if args.out_geojson:
        out_geojson = Path(args.out_geojson)
        # build geojson by reprojecting recorded bboxes (assumed EPSG:32617) -> EPSG:4326
        build_geojson_epsg4326(rows, out_geojson, _WORKING_CRS_NORMALIZED)
        logging.info(f"Wrote GeoJSON catalog with {len([r for r in rows if 'error' not in r])} features to {out_geojson}")

if __name__ == "__main__":
    main()
