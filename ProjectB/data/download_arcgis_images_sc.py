#!/usr/bin/env python3
"""
download_arcgis_images_sc.py

Download whole-image tiles from one or more ArcGIS ImageServer endpoints
covering the (approximate) extent of South Carolina.

Improvements:
 - default resolution set to 10 m (--res-m 10)
 - default tile size reduced to 20000 m (--tile-size-m 20000) so at 10 m/pixel tiles are ~2000 px (likely within server limits)
 - default max pixels increased to 4096 (when service doesn't advertise a limit)
 - automatic scaling of effective resolution to respect server max pixels
 - robust detection of JSON error responses and avoidance of saving non-image/HTML payloads
"""
from pathlib import Path
import argparse
import logging
import math
import json
import io
import time
from typing import Optional, Tuple, List, Dict, Any
import concurrent.futures

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import yaml
except Exception:
    yaml = None

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------------- helpers ----------------
def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise SystemExit("PyYAML required to read YAML config. Install: pip install pyyaml")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        return json.loads(path.read_text(encoding="utf-8"))


def make_session(retries: int = 3, backoff: float = 0.3, status_forcelist=(429, 500, 502, 503, 504)) -> requests.Session:
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=list(status_forcelist), raise_on_status=False)
    a = HTTPAdapter(max_retries=r)
    s.mount("https://", a)
    s.mount("http://", a)
    return s


def fetch_token_client_credentials(session: requests.Session, client_id: str, client_secret: str, portal: str, timeout: int = 30) -> Optional[str]:
    token_url = portal.rstrip("/") + "/sharing/rest/oauth2/token"
    data = {"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials", "f": "json"}
    try:
        r = session.post(token_url, data=data, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        return j.get("access_token") or j.get("token")
    except Exception as e:
        logging.warning("Failed to fetch token via client_credentials: %s", e)
        return None


def get_service_info(session: requests.Session, base_url: str, token: Optional[str], timeout: int = 30) -> Optional[dict]:
    params = {"f": "json"}
    if token:
        params["token"] = token
    try:
        r = session.get(base_url.rstrip("/") + "?f=json", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning("Failed to fetch service info for %s: %s", base_url, e)
        return None


def get_service_srid(info: Optional[dict]) -> Optional[int]:
    if not info:
        return None
    sr = info.get("spatialReference") or info.get("extent", {}).get("spatialReference") or info.get("tileInfo", {}).get("spatialReference")
    if isinstance(sr, dict):
        wk = sr.get("wkid") or sr.get("latestWkid")
        try:
            return int(wk) if wk is not None else None
        except Exception:
            return None
    return None


def detect_service_max_pixels(info: Optional[dict]) -> Optional[int]:
    """
    Try several keys in service info that might indicate a max pixel dimension.
    Fallback is None (caller can use a conservative default).
    """
    if not info:
        return None
    candidates = [
        "maxImageHeight", "maxImageWidth", "maxImageSize",
        "maxExportSize", "maxImagePixels", "maxPixelSize", "maxSize"
    ]
    for k in candidates:
        v = info.get(k)
        if v:
            try:
                return int(v)
            except Exception:
                continue
    ti = info.get("tileInfo") or {}
    for k in ("rows", "cols", "width", "height", "tileHeight", "tileWidth"):
        v = ti.get(k)
        if v:
            try:
                return int(v)
            except Exception:
                pass
    return None


def tile_grid_for_bbox_projected(minx: float, miny: float, maxx: float, maxy: float, tile_size_m: float, overlap_m: float = 0.0) -> List[Tuple[float, float, float, float]]:
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be > 0")
    x0 = minx
    y0 = miny
    tiles = []
    stride = tile_size_m - overlap_m
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile size")
    nx = int(math.ceil((maxx - minx) / stride))
    ny = int(math.ceil((maxy - miny) / stride))
    for iy in range(ny):
        for ix in range(nx):
            tx0 = x0 + ix * stride
            ty0 = y0 + iy * stride
            tx1 = min(tx0 + tile_size_m, maxx)
            ty1 = min(ty0 + tile_size_m, maxy)
            if tx1 <= tx0 or ty1 <= ty0:
                continue
            tiles.append((tx0, ty0, tx1, ty1))
    return tiles


def bbox_transform(bbox: Tuple[float, float, float, float], src_crs: str, dst_crs: str) -> Tuple[float, float, float, float]:
    if src_crs == dst_crs:
        return bbox
    if Transformer is None:
        raise SystemExit("pyproj required for reprojection (pip install pyproj)")
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    minx, miny, maxx, maxy = bbox
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    xs = []; ys = []
    for x, y in corners:
        tx, ty = transformer.transform(x, y)
        xs.append(tx); ys.append(ty)
    return (min(xs), min(ys), max(xs), max(ys))


def compute_exportimage_size(bbox_proj: Tuple[float, float, float, float], target_res_m: float, max_pixels: Optional[int] = None) -> Tuple[int, int, float]:
    """
    Compute width,height for exportImage given bbox (projected units) and target_res_m (meters/pixel).
    If max_pixels is provided and either width or height exceeds max_pixels, increase the effective
    pixel size (m/pixel) so both width & height <= max_pixels. Returns (width, height, effective_res_m).
    """
    minx, miny, maxx, maxy = bbox_proj
    dx = maxx - minx
    dy = maxy - miny
    if target_res_m <= 0:
        raise ValueError("target_res_m must be > 0")
    w = max(1, int(math.ceil(dx / target_res_m)))
    h = max(1, int(math.ceil(dy / target_res_m)))

    if max_pixels is None:
        return w, h, target_res_m

    max_pix = int(max_pixels)
    if w <= max_pix and h <= max_pix:
        return w, h, target_res_m

    sf_w = math.ceil(w / max_pix)
    sf_h = math.ceil(h / max_pix)
    sf = max(sf_w, sf_h, 1)
    eff_res = target_res_m * sf
    w2 = max(1, int(math.ceil(dx / eff_res)))
    h2 = max(1, int(math.ceil(dy / eff_res)))
    logging.info("Scaling resolution for bbox %.3f×%.3f m -> effective_res=%s m/px to satisfy max_pixels=%d (w,h %d×%d -> %d×%d)",
                 dx, dy, eff_res, max_pix, w, h, w2, h2)
    return w2, h2, eff_res


def download_tile(session: requests.Session, base_url: str, bbox_for_service: Tuple[float, float, float, float],
                  bbox_srid: Optional[int], width: int, height: int, out_path: Path,
                  token: Optional[str], timeout: int = 120, fmt: str = "tiff") -> Tuple[bool, str]:
    """
    Download tile via exportImage. Inspect content-type and JSON error responses to avoid writing bad files.
    Returns (success_bool, message)
    """
    params = {
        "f": "image",
        "bbox": f"{bbox_for_service[0]},{bbox_for_service[1]},{bbox_for_service[2]},{bbox_for_service[3]}",
        "bboxSR": str(bbox_srid) if bbox_srid is not None else "3857",
        "imageSR": str(bbox_srid) if bbox_srid is not None else "3857",
        "size": f"{width},{height}",
        "format": fmt
    }
    if token:
        params["token"] = token

    url = base_url.rstrip("/") + "/exportImage"
    try:
        resp = session.get(url, params=params, timeout=timeout, stream=True)
    except Exception as e:
        return False, f"request-failed:{e}"

    try:
        resp.raise_for_status()
    except requests.HTTPError:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "json" in ctype:
            try:
                j = resp.json()
                if isinstance(j, dict) and "error" in j:
                    return False, f"server-error:{json.dumps(j['error'])}"
                return False, f"server-json-error:{json.dumps(j)}"
            except Exception:
                return False, f"http-error-status:{resp.status_code}"
        else:
            return False, f"http-error-status:{resp.status_code}"

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "json" in content_type:
        try:
            j = resp.json()
            if isinstance(j, dict) and "error" in j:
                return False, f"server-error:{json.dumps(j['error'])}"
            return False, f"unexpected-json-response:{json.dumps(j)[:200]}"
        except Exception as e:
            return False, f"json-parse-failed:{e}"

    if not any(x in content_type for x in ("tiff", "image/tiff", "image/vnd", "application/octet-stream")):
        try:
            snippet = resp.content[:200].decode("utf-8", errors="replace")
        except Exception:
            snippet = str(resp.content[:64])
        return False, f"unexpected-content-type:{content_type} snippet:{snippet[:200]}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with tmp.open("wb") as fh:
            total = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
        if total < 256:
            try:
                txt = tmp.read_bytes()[:512].decode("utf-8", errors="replace")
            except Exception:
                txt = "<binary>"
            tmp.unlink(missing_ok=True)
            return False, f"payload-too-small:{total} snippet:{txt[:200]}"
        tmp.replace(out_path)
        return True, "OK"
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, f"write-failed:{e}"


# ---------------- main CLI ----------------
def main():
    p = argparse.ArgumentParser(description="Download ImageServer tiles for South Carolina (or custom bbox).")
    p.add_argument("--arcgis-config", default="arcgis_config.yaml", help="YAML/JSON config with arcgis: (...)")
    p.add_argument("--services", nargs="+", default=["soil_url", "precip_url"], help="Keys under arcgis in config (default: soil_url precip_url)")
    p.add_argument("--out-dir", default="raw_arcgis", help="Output directory to write tiles and manifest (default: raw_arcgis)")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("MINX","MINY","MAXX","MAXY"),
                   help="Bounding box in EPSG:4326 (lon min, lat min, lon max, lat max). If omitted defaults to SC bbox.")
    p.add_argument("--proj-crs", default="EPSG:3857", help="Projected CRS used to tile (default EPSG:3857).")
    p.add_argument("--tile-size-m", type=float, default=20000.0, help="Tile size in meters (default 20000).")
    p.add_argument("--overlap-m", type=float, default=100.0, help="Tile overlap in meters.")
    p.add_argument("--res-m", type=float, default=10.0, help="Target pixel size in meters for downloads (default 10).")
    p.add_argument("--workers", type=int, default=4, help="Parallel download workers.")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds per exportImage request.")
    p.add_argument("--portal", default=None, help="Optional portal override (otherwise read from config arcgis.portal)")
    p.add_argument("--max-retries", type=int, default=3, help="Retries per tile on failure.")
    p.add_argument("--default-max-pixels", type=int, default=4096, help="Default max pixels (width/height) to request when service doesn't advertise a limit.")
    args = p.parse_args()

    cfg = load_config(Path(args.arcgis_config))
    arc = cfg.get("arcgis", {}) if isinstance(cfg, dict) else {}
    portal = args.portal or arc.get("portal", "https://www.arcgis.com")
    client_id = arc.get("client_id")
    client_secret = arc.get("client_secret")
    token_cfg = arc.get("token")
    token_header_type = arc.get("token_header_type", "Bearer")
    arc_timeout = int(arc.get("timeout", args.timeout))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_sc_bbox = (-83.75, 32.00, -78.00, 35.35)
    bbox4326 = tuple(args.bbox) if args.bbox else default_sc_bbox
    logging.info("Using bbox (EPSG:4326): %s", bbox4326)

    proj_crs = args.proj_crs
    if Transformer is None:
        logging.warning("pyproj not installed; bbox reprojection won't work. Please install pyproj.")
    try:
        bbox_proj = bbox_transform(bbox4326, "EPSG:4326", proj_crs)
    except Exception as e:
        logging.error("Failed to transform bbox: %s", e)
        return
    logging.info("Projected bbox (%s): %s", proj_crs, bbox_proj)

    tiles = tile_grid_for_bbox_projected(bbox_proj[0], bbox_proj[1], bbox_proj[2], bbox_proj[3], args.tile_size_m, args.overlap_m)
    logging.info("Generated %d tile(s) at tile_size_m=%s overlap_m=%s", len(tiles), args.tile_size_m, args.overlap_m)
    if len(tiles) == 0:
        logging.error("No tiles generated; exiting.")
        return

    # Warning about resource use
    logging.warning("Requested default resolution is %s m/pixel. At this resolution downloads and storage "
                    "for the entire state can be large. Consider reducing tile-size or workers if you hit rate/space limits.",
                    args.res_m)

    session = make_session(retries=3, backoff=0.5)
    access_token = None
    if client_id and client_secret:
        logging.info("Requesting OAuth token via client_credentials...")
        access_token = fetch_token_client_credentials(session, client_id, client_secret, portal=portal, timeout=arc_timeout)
        if access_token:
            session.headers.update({"Authorization": f"{token_header_type} {access_token}"})
            logging.info("Got token via client_credentials.")
    if not access_token and token_cfg:
        access_token = token_cfg
        session.headers.update({"Authorization": f"{token_header_type} {access_token}"})
        logging.info("Using static token from config.")

    arc_services = {}
    for key in args.services:
        url = arc.get(key)
        if not url:
            logging.error("Service key '%s' not found in config 'arcgis' section.", key)
            continue
        arc_services[key] = url
    if not arc_services:
        logging.error("No valid services to download; check --services and config file.")
        return

    service_infos: Dict[str, Dict[str, Any]] = {}
    service_srids: Dict[str, Optional[int]] = {}
    service_max_pixels: Dict[str, int] = {}
    for key, url in arc_services.items():
        info = get_service_info(session, url, access_token, timeout=arc_timeout)
        service_infos[key] = info or {}
        srid = get_service_srid(info)
        service_srids[key] = srid
        detected_max = detect_service_max_pixels(info)
        service_max_pixels[key] = int(detected_max) if detected_max is not None else int(args.default_max_pixels)
        logging.info("Service %s SRID=%s max_pixels=%s", key, srid, service_max_pixels[key])

    tasks = []
    for key, url in arc_services.items():
        srid = service_srids.get(key)
        max_pixels = service_max_pixels.get(key, args.default_max_pixels)
        for ti, tb in enumerate(tiles):
            if srid is not None and Transformer is not None:
                try:
                    bbox_for_service = bbox_transform(tb, proj_crs, f"EPSG:{srid}")
                    bbox_srid = srid
                except Exception:
                    bbox_for_service = tb
                    bbox_srid = None
            else:
                bbox_for_service = tb
                bbox_srid = None

            width, height, eff_res = compute_exportimage_size(bbox_for_service, args.res_m, max_pixels=max_pixels)
            out_fname = f"{key}_tile_{ti:04d}_x{int(tb[0])}_y{int(tb[1])}_res{int(eff_res)}m_{width}x{height}.tif"
            out_path = out_dir / key / out_fname

            tasks.append({
                "service_key": key,
                "service_url": url,
                "tile_index": ti,
                "bbox_proj": tb,
                "bbox_for_service": bbox_for_service,
                "bbox_srid": bbox_srid,
                "width": width,
                "height": height,
                "out_path": out_path,
                "max_pixels": max_pixels,
                "effective_res_m": eff_res
            })

    logging.info("Prepared %d download task(s) across %d service(s).", len(tasks), len(arc_services))

    def worker(task: dict) -> dict:
        s = make_session(retries=2, backoff=0.2)
        s.headers.update(session.headers)
        attempts = 0
        success = False
        last_err = None
        while attempts < args.max_retries and not success:
            attempts += 1
            ok, msg = download_tile(s, task["service_url"], task["bbox_for_service"], task["bbox_srid"],
                                    task["width"], task["height"], task["out_path"],
                                    token=access_token, timeout=args.timeout, fmt="tiff")
            if ok:
                success = True
                return {
                    "service_key": task["service_key"],
                    "tile_index": task["tile_index"],
                    "out_path": str(task["out_path"].resolve()),
                    "width": task["width"],
                    "height": task["height"],
                    "bbox_proj": task["bbox_proj"],
                    "bbox_srid": task["bbox_srid"],
                    "effective_res_m": task.get("effective_res_m"),
                    "status": "ok",
                    "message": msg
                }
            else:
                last_err = msg
                logging.warning("Attempt %d/%d failed for %s tile %s: %s", attempts, args.max_retries, task["service_key"], task["tile_index"], msg)
                if "exceeds the size limit" in str(msg) or "requested image exceeds" in str(msg) or "size" in str(msg).lower():
                    new_res = task.get("effective_res_m", args.res_m) * 2
                    w2, h2, eff2 = compute_exportimage_size(task["bbox_for_service"], new_res, max_pixels=task.get("max_pixels"))
                    task["width"], task["height"], task["effective_res_m"] = w2, h2, eff2
                    logging.info("Retrying with coarser resolution %s m/px (size %dx%d)", eff2, w2, h2)
                time.sleep(1.0 * attempts)
        return {
            "service_key": task["service_key"],
            "tile_index": task["tile_index"],
            "out_path": str(task["out_path"].resolve()),
            "width": task["width"],
            "height": task["height"],
            "bbox_proj": task["bbox_proj"],
            "bbox_srid": task["bbox_srid"],
            "effective_res_m": task.get("effective_res_m"),
            "status": "failed",
            "message": last_err
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                logging.exception("Worker crashed: %s", e)
                continue
            results.append(res)
            if res["status"] == "ok":
                logging.info("Downloaded %s tile %s -> %s (size %dx%d res=%s)", res["service_key"], res["tile_index"], res["out_path"], res["width"], res["height"], res.get("effective_res_m"))
            else:
                logging.error("Failed %s tile %s: %s (tried size %dx%d res=%s)", res["service_key"], res["tile_index"], res["message"], res["width"], res["height"], res.get("effective_res_m"))

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok_count = sum(1 for r in results if r["status"] == "ok")
    fail_count = sum(1 for r in results if r["status"] != "ok")
    logging.info("Download complete: %d ok, %d failed (manifest at %s)", ok_count, fail_count, manifest_path)


if __name__ == "__main__":
    main()
