#!/usr/bin/env python3

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from time import sleep

import requests
from requests.adapters import HTTPAdapter, Retry
from rasterio.merge import merge
import rasterio
from tqdm import tqdm

DEFAULT_BASE_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/OPR/Projects/"
    "SC_SavannahPeeDee_2019_B19"
)

def create_session(retries=5, backoff=0.5, status_forcelist=(500,502,503,504)):
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=status_forcelist)
    s.mount('https://', HTTPAdapter(max_retries=r))
    s.mount('http://', HTTPAdapter(max_retries=r))
    return s

def read_links_from_txt(session, links_url):
    r = session.get(links_url, timeout=60)
    r.raise_for_status()
    # text file may be one long line or many lines
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    # filter only .tif links (defensive)
    tif_links = [ln for ln in lines if ln.lower().endswith('.tif')]
    return tif_links

def download_url(session, url, out_path, chunk_size=1024*1024):
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        tmp_path = out_path.with_suffix(out_path.suffix + '.part')
        with open(tmp_path, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=out_path.name, leave=False
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp_path.replace(out_path)

def merge_tifs(tif_paths, out_path, compress='deflate', bigtiff='YES'):
    src_files_to_mosaic = []
    for p in tif_paths:
        src = rasterio.open(str(p))
        src_files_to_mosaic.append(src)

    try:
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src_files_to_mosaic[0].meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": compress,
            "BIGTIFF": bigtiff
        })

        with rasterio.open(str(out_path), "w", **out_meta) as dest:
            dest.write(mosaic)
    finally:
        for src in src_files_to_mosaic:
            try:
                src.close()
            except Exception:
                pass

def main(base_url, out_filename, workers=1, start_idx=1, end_idx=9):
    session = create_session()
    tmpdir = (Path(__file__).resolve().parent / "raw_tif")
    tmpdir.mkdir(parents=True, exist_ok=True)
    print(f"Using local temp folder: {tmpdir}")

    all_downloaded = []

    try:
        for i in range(start_idx, end_idx + 1):
            folder_name = f"SC_SavannahPeeDee_{i}_2019"
            links_url = f"{base_url}/{folder_name}/0_file_download_links.txt"
            print(f"\nProcessing folder {i}: {links_url}")

            try:
                tif_links = read_links_from_txt(session, links_url)
            except requests.HTTPError as e:
                print(f"  WARNING: could not read {links_url}: {e}")
                continue

            if not tif_links:
                print("  No .tif links found in this txt.")
                continue

            # download each tif
            for link in tif_links:
                fname = Path(link).name
                out_path = tmpdir / fname
                if out_path.exists():
                    print(f"  already downloaded: {fname}")
                    all_downloaded.append(out_path)
                    continue
                try:
                    print(f"  downloading {fname} ...")
                    download_url(session, link, out_path)
                    all_downloaded.append(out_path)
                except Exception as e:
                    print(f"    ERROR downloading {link}: {e}")
                    # do not crash; try next file
                    continue

        if not all_downloaded:
            raise RuntimeError("No files downloaded; aborting merge.")

        print(f"\nMerging {len(all_downloaded)} files into {out_filename} ...")
        merge_tifs(all_downloaded, Path(out_filename))
        print("Merge complete:", out_filename)

    except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download and merge Savannah PeeDee 2019 tiles")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="Base project URL (parent of SC_SavannahPeeDee_{i}_2019 folders)")
    ap.add_argument("--out", default="merged_savannah.tif", help="Output merged GeoTIFF")
    ap.add_argument("--start", type=int, default=1, help="Start index (inclusive)")
    ap.add_argument("--end", type=int, default=9, help="End index (inclusive)")
    args = ap.parse_args()

    main(args.base_url, args.out, start_idx=args.start, end_idx=args.end)
