"""FastSpecFit extraction pipeline for DESI spectra using Producer-Consumer architecture."""

from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import fitsio
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

SURVEY = 'sv3'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DUST_DIR = DATA_DIR / "dust" / "maps"
FTEMPLATES_DIR = DATA_DIR / "templates"


@dataclass
class FastSpecTask:
    temp_dir_obj: tempfile.TemporaryDirectory
    runs: list[tuple[Path, list[int]]] # list of (redrock_path, targetids)


def download_file(url: str, dest_path: Path) -> None:
    """Download a file if it doesn't already exist."""
    if dest_path.exists():
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def setup_global_dependencies() -> None:
    """Download dust maps and templates if missing."""
    DUST_DIR.mkdir(parents=True, exist_ok=True)
    FTEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    dust_base = "https://portal.nersc.gov/project/cosmo/data/dust/v0_1/maps"
    for map_name in ['SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits']:
        download_file(f"{dust_base}/{map_name}", DUST_DIR / map_name)

    template_version = "2.2.0"
    template_name = f"ftemplates-chabrier-{template_version}.fits"
    url = f"https://data.desi.lbl.gov/public/external/templates/fastspecfit/{template_version}/{template_name}"
    dest = FTEMPLATES_DIR / template_version / template_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, dest)


def download_tractor_catalog(tractor_path_str: str) -> None:
    """Parse the missing tractor path and download it from NERSC."""
    tractor_path = Path(tractor_path_str)
    parts = tractor_path.parts
    if 'tractor' in parts:
        idx = parts.index('tractor')
        rel_parts = parts[idx-1:]
        rel_path = "/".join(rel_parts)
    else:
        raise ValueError(f"Could not parse tractor path: {tractor_path_str}")
        
    url = f"https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{rel_path}"
    tractor_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, tractor_path)


def producer_worker(task_queue: queue.Queue, healpix_groups: list[tuple[int, set[int]]], pbar: tqdm, error_log_path: Path) -> None:
    """Worker that prepares tasks for a HEALPix and puts them in the bounded queue."""
    for healpix_29, targetids_set in healpix_groups:
        healpix = healpix_29 // (4**23)
        group = healpix // 100
        
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir_obj.name)
        
        programs = ['dark', 'bright', 'backup']
        runs = []
        
        mock_dir = temp_path / "spectro" / "redux" / "fuji" / "healpix" / SURVEY / "{prog}" / str(group) / str(healpix)
        
        for prog in programs:
            prog_mock_dir = Path(str(mock_dir).format(prog=prog))
            prog_mock_dir.mkdir(parents=True, exist_ok=True)
            
            rr_file = f"redrock-{SURVEY}-{prog}-{healpix}.fits"
            base_url = f"https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/{SURVEY}/{prog}/{group}/{healpix}"
            rr_url = f"{base_url}/{rr_file}"
            rr_path = prog_mock_dir / rr_file
            
            try:
                download_file(rr_url, rr_path)
                fits = fitsio.FITS(str(rr_path))
                targets = fits['REDSHIFTS'].read(columns=['TARGETID'])
                rr_targetids = set(targets['TARGETID'])
                
                matched = targetids_set.intersection(rr_targetids)
                if matched:
                    # Download coadd
                    coadd_file = f"coadd-{SURVEY}-{prog}-{healpix}.fits"
                    coadd_path = prog_mock_dir / coadd_file
                    
                    try:
                        download_file(f"{base_url}/{coadd_file}", coadd_path)
                        runs.append((rr_path, list(matched)))
                    except requests.exceptions.HTTPError as e:
                        logger.warning(f"Failed to download COADD for {prog} healpix {healpix}: {e}")
                        with open(error_log_path, 'a') as f:
                            f.write(f"{healpix},{prog},COADD_HTTPError,{e}\n")
                        
            except requests.exceptions.HTTPError:
                pass # Try next program
            except requests.exceptions.HTTPError as e:
                if e.response.status_code != 404:
                    with open(error_log_path, 'a') as f:
                        f.write(f"{healpix},{prog},REDROCK_HTTPError,{e}\n")
            except Exception as e:
                logger.warning(f"Error checking {prog} for healpix {healpix}: {e}")
                with open(error_log_path, 'a') as f:
                    f.write(f"{healpix},{prog},REDROCK_Error,{e}\n")
                
        if runs:
            # Block if the queue is full
            task_queue.put(FastSpecTask(temp_dir_obj=temp_dir_obj, runs=runs))
        else:
            # Cleanup if no targets matched in any program
            temp_dir_obj.cleanup()
            with open(error_log_path, 'a') as f:
                f.write(f"{healpix},ALL,NoTargetsFound,Could not extract targets for this HEALPix\n")
            
        pbar.update(1)


def run_extraction(config: dict, limit: int | None = None, max_workers: int = 4) -> None:
    """Run extraction using a Producer-Consumer architecture."""
    logger.info("Setting up global dependencies...")
    setup_global_dependencies()
    
    parquet_path = PROJECT_ROOT / "data" / "crossmatch_cache" / "crossmatch_merged_1.0arcsec.parquet"
    if not parquet_path.exists():
        logger.error(f"Dataset not found at {parquet_path}")
        return
        
    logger.info("Loading dataset...")
    df = pd.read_parquet(parquet_path)
    desi_df = df[df['survey'] == 'desi'].drop_duplicates(subset=['object_id'])
    
    # Sort by HEALPix and Object ID before applying the limit
    desi_df = desi_df.sort_values(by=['_healpix_29', 'object_id'])
    
    if limit:
        desi_df = desi_df.head(limit)
        
    # Group by healpix_29
    grouped = desi_df.groupby('_healpix_29')['object_id'].apply(lambda x: set(int(tid) for tid in x)).reset_index()
    healpix_groups = list(grouped.itertuples(index=False, name=None))
    
    logger.info(f"Processing {len(desi_df)} targets across {len(healpix_groups)} HEALPix regions...")
    
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    run_dir = DATA_DIR / f"fastspec_extraction_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = run_dir / "extracted_emission_lines.csv"
    raw_csv = run_dir / "raw_metrics.csv"
    error_log_path = run_dir / "failed_downloads.csv"
    
    with open(error_log_path, 'w') as f:
        f.write("healpix,program,error_type,message\n")
    
    # Bounded queue to limit disk usage
    task_queue = queue.Queue(maxsize=3)
    
    # Split HEALPix jobs among producer threads
    chunks = np.array_split(healpix_groups, max_workers)
    chunks = [c.tolist() for c in chunks if len(c) > 0]
    
    producer_pbar = tqdm(total=len(healpix_groups), desc="[1/2] Downloading HEALPix Data")
    consumer_pbar = tqdm(total=len(desi_df), desc="[2/2] Extracting Targets via fastspec")
    
    # Start producers
    threads = []
    for chunk in chunks:
        t = threading.Thread(target=producer_worker, args=(task_queue, chunk, producer_pbar))
        t = threading.Thread(target=producer_worker, args=(task_queue, chunk, producer_pbar, error_log_path))
        t.start()
        threads.append(t)
        
    # Monitor producers in a separate thread to send a sentinel when done
    def monitor_producers():
        for t in threads:
            t.join()
        task_queue.put(None) # Sentinel value to stop consumer
        
    monitor_thread = threading.Thread(target=monitor_producers)
    monitor_thread.start()
    
    fastspec_bin = __import__('shutil').which('fastspec') or 'fastspec'
    
    # Consumer loop (Main thread)
    while True:
        task = task_queue.get()
        if task is None:
            break
            
        temp_path = Path(task.temp_dir_obj.name)
        
        batch_results = []
        batch_raw = []
        
        for redrock_path, targetids in task.runs:
            targetids_str = ",".join(map(str, targetids))
            output_fits = temp_path / f"fastspec_{targetids[0]}.fits"
            
            env = os.environ.copy()
            env['DESI_ROOT'] = str(temp_path.absolute())
            env['DESI_SPECTRO_REDUX'] = str((temp_path / 'spectro' / 'redux').absolute())
            env['SPECPROD'] = 'fuji'
            env['DUST_DIR'] = str(DUST_DIR.parent.absolute())
            env['FPHOTO_DIR'] = str(temp_path.absolute())
            env['FTEMPLATES_DIR'] = str(FTEMPLATES_DIR.absolute())
            
            env['OMP_NUM_THREADS'] = '1'
            env['NUMEXPR_NUM_THREADS'] = '1'
            env['OPENBLAS_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            
            cmd = [
                fastspec_bin,
                str(redrock_path),
                "--targetids", targetids_str,
                "--mp", str(max(1, os.cpu_count() or 1)),
                "--ignore-photometry",
                "--outfile", str(output_fits)
            ]
            
            while True:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if result.returncode == 0:
                    break
                else:
                    match = re.search(r"Unable to find Tractor catalog\s+(\S+)", result.stderr)
                    if not match:
                        match = re.search(r"Unable to find Tractor catalog\s+(\S+)", result.stdout)
                    
                    if match:
                        missing_tractor = match.group(1)
                        download_tractor_catalog(missing_tractor)
                    else:
                        logger.error(f"FastSpecFit failed for batch {targetids_str}:\n{result.stderr}")
                        break
                        
            # Parse FITS if generated
            if output_fits.exists():
                fits = fitsio.FITS(str(output_fits))
                if 'FASTSPEC' in fits:
                    data = fits['FASTSPEC'].read()
                    out_df = pd.DataFrame(data)
                    
                    if not out_df.empty:
                        for _, row in out_df.iterrows():
                            tid = row['TARGETID']
                            
                            flux_cols = [col for col in out_df.columns if col.endswith('_FLUX') and 'BOX' not in col]
                            for flux_col in flux_cols:
                                line_name = flux_col.replace('_FLUX', '')
                                ivar_col = f"{line_name}_FLUX_IVAR"
                                
                                if ivar_col in out_df.columns:
                                    flux = row[flux_col]
                                    ivar = row[ivar_col]
                                    
                                    if pd.notna(flux) and pd.notna(ivar) and ivar > 0:
                                        snr = flux * np.sqrt(ivar)
                                        if snr >= 3.0:
                                            batch_results.append({
                                                'TARGETID': tid,
                                                'LINE_NAME': line_name,
                                                'FLUX': flux,
                                                'SNR': snr
                                            })
                                            
                            # Save raw metrics
                            raw_dict = {}
                            for col in out_df.columns:
                                val = row[col]
                                if isinstance(val, bytes):
                                    val = val.decode('utf-8')
                                raw_dict[col] = val
                            batch_raw.append(raw_dict)
                            
            consumer_pbar.update(len(targetids))
            
        # Write incremental batch to CSVs
        if batch_results:
            out_df = pd.DataFrame(batch_results)
            out_df.to_csv(output_csv, mode='a', header=not output_csv.exists(), index=False)
            
        if batch_raw:
            raw_df = pd.DataFrame(batch_raw)
            raw_df.to_csv(raw_csv, mode='a', header=not raw_csv.exists(), index=False)
            
        # Clean up disk space
        task.temp_dir_obj.cleanup()
        task_queue.task_done()
        
    producer_pbar.close()
    consumer_pbar.close()
    
    logger.info("Extraction complete!")
    logger.info(f"Final results saved to {output_csv}")
    logger.info(f"Raw metrics saved to {raw_csv}")

