"""FastSpecFit extraction pipeline for DESI spectra"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time
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
FASTSPEC_BIN = __import__('shutil').which('fastspec') or 'fastspec'


def download_file(url: str, dest_path: Path) -> None:
    """Download a file if it doesn't already exist."""
    if dest_path.exists():
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def setup_global_dependencies() -> None:
    """Download dust maps and templates if missing."""
    dust_base = "https://portal.nersc.gov/project/cosmo/data/dust/v0_1/maps"
    for map_name in ['SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits']:
        download_file(f"{dust_base}/{map_name}", DUST_DIR / map_name)

    template_version = "2.2.0"
    template_name = f"ftemplates-chabrier-{template_version}.fits"
    url = f"https://data.desi.lbl.gov/public/external/templates/fastspecfit/{template_version}/{template_name}"
    download_file(url, FTEMPLATES_DIR / template_version / template_name)


def download_tractor_catalog(tractor_path_str: str) -> None:
    """Parse the missing tractor path and download it from NERSC to the persistent DATA_DIR."""
    tractor_path = Path(tractor_path_str)
    parts = tractor_path.parts
    if 'tractor' in parts:
        idx = parts.index('tractor')
        rel_parts = parts[idx-1:] # e.g. ('north', 'tractor', '218', 'tractor-2185p330.fits')
        rel_path = "/".join(rel_parts)
    else:
        raise ValueError(f"Could not parse tractor path: {tractor_path_str}")
        
    url = f"https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{rel_path}"
    dest = DATA_DIR / "desi_raw_fits" / rel_path
    download_file(url, dest)


def load_and_filter_dataset(parquet_path: Path, limit: int | None = None, retry_failed: Path | None = None) -> tuple[list[tuple[int, set[int]]], dict[int, str]]:
    """Load dataset, compute HEALPIX_64, filter, and group by HEALPix."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found at {parquet_path}")
        
    logger.info("Loading dataset...")
    df = pd.read_parquet(parquet_path)
    desi_df = df[df['survey'] == 'desi'].drop_duplicates(subset=['object_id']).copy()
    
    desi_df['healpix_64'] = desi_df['_healpix_29'] // (4**23)
    
    if retry_failed:
        logger.info(f"Filtering dataset to retry failed HEALPix values from {retry_failed}")
        failed_df = pd.read_csv(retry_failed)
        desi_df = desi_df[desi_df['healpix_64'].isin(failed_df['healpix'].unique())]
        
    desi_df = desi_df.sort_values(by=['healpix_64', 'object_id'])
    
    if limit:
        desi_df = desi_df.head(limit)
        
    grouped = desi_df.groupby('healpix_64')['object_id'].apply(lambda x: set(int(tid) for tid in x)).reset_index()
    target_to_wiki = dict(zip(desi_df['object_id'].astype(int), desi_df['wiki_entity_id'].astype(str)))
    return list(grouped.itertuples(index=False, name=None)), target_to_wiki


def extract_and_save_results(output_fits: Path, output_csv: Path, raw_csv: Path, target_to_wiki: dict[int, str]) -> None:
    """Parse FastSpecFit FITS output, compute SNR, and safely save to CSVs."""
    if not output_fits.exists():
        return
        
    out_data = fitsio.FITS(str(output_fits))['FASTSPEC'].read()
    out_df = pd.DataFrame(out_data)
    
    if out_df.empty:
        return
        
    batch_results, batch_raw = [], []
    for _, row in out_df.iterrows():
        tid = int(row['TARGETID'])
        wiki_id = target_to_wiki.get(tid)
        
        flux_cols = [col for col in out_df.columns if col.endswith('_FLUX') and 'BOX' not in col]
        for flux_col in flux_cols:
            ivar_col = f"{flux_col}_IVAR"
            if ivar_col in out_df.columns:
                flux, ivar = row[flux_col], row[ivar_col]
                if pd.notna(flux) and pd.notna(ivar) and ivar > 0:
                    snr = flux * np.sqrt(ivar)
                    if snr >= 3.0:
                        batch_results.append({
                            'wiki_entity_id': wiki_id,
                            'TARGETID': tid,
                            'LINE_NAME': flux_col.replace('_FLUX', ''),
                            'FLUX': flux,
                            'SNR': snr
                        })
                        
        raw_dict = {'wiki_entity_id': wiki_id}
        for col in out_df.columns:
            val = row[col]
            raw_dict[col] = val.decode('utf-8') if isinstance(val, bytes) else val
        batch_raw.append(raw_dict)
        
    if batch_results:
        pd.DataFrame(batch_results).to_csv(output_csv, mode='a', header=not output_csv.exists(), index=False)
    if batch_raw:
        pd.DataFrame(batch_raw).to_csv(raw_csv, mode='a', header=not raw_csv.exists(), index=False)


def run_fastspec_command(rr_path: Path, targetids: set[int], output_fits: Path, temp_path: Path, error_log_path: Path, healpix: int, prog: str) -> bool:
    """Configure environment and execute the fastspec command."""
    targetids_str = ",".join(map(str, targetids))
    
    env = os.environ.copy()
    env['DESI_ROOT'] = str(temp_path.absolute())
    env['DESI_SPECTRO_REDUX'] = str((temp_path / 'spectro' / 'redux').absolute())
    env['SPECPROD'] = 'fuji'
    env['DUST_DIR'] = str(DUST_DIR.parent.absolute())
    fphoto_dir = DATA_DIR / "desi_raw_fits"
    fphoto_dir.mkdir(parents=True, exist_ok=True)
    env['FPHOTO_DIR'] = str(fphoto_dir.absolute())
    env['FTEMPLATES_DIR'] = str(FTEMPLATES_DIR.absolute())
    
    env['OMP_NUM_THREADS'] = '1'
    env['NUMEXPR_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    
    cmd = [
        FASTSPEC_BIN, str(rr_path),
        "--targetids", targetids_str,
        "--mp", str(max(1, os.cpu_count() or 1)),
        "--ignore-photometry", "--outfile", str(output_fits)
    ]
    
    while True:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode == 0:
            return True
            
        match = re.search(r"Unable to find Tractor catalog\s+(\S+)", result.stderr) or \
                re.search(r"Unable to find Tractor catalog\s+(\S+)", result.stdout)
        
        if match:
            try:
                download_tractor_catalog(match.group(1))
            except Exception as e:
                logger.error(f"Failed to fetch Tractor catalog: {e}")
                log_error(error_log_path, healpix, prog, "TractorError", str(e))
                return False
        else:
            logger.error(f"FastSpecFit crashed on {healpix} ({prog}):\n{result.stderr}")
            log_error(error_log_path, healpix, prog, "FastSpecCrash", "See terminal logs")
            return False


def log_error(error_log_path: Path, healpix: int, prog: str, error_type: str, message: str) -> None:
    """Helper to log errors to the CSV."""
    with open(error_log_path, 'a') as f:
        msg = message.replace('\n', ' ')
        f.write(f"{healpix},{prog},{error_type},{msg}\n")


def process_healpix_region(healpix: int, all_targetids: set[int], run_dir: Path, pbar: tqdm, target_to_wiki: dict[int, str]) -> None:
    """Process all targets within a specific HEALPix region."""
    group = healpix // 100
    pending_targets = set(all_targetids)
    
    output_csv = run_dir / "extracted_emission_lines.csv"
    raw_csv = run_dir / "raw_metrics.csv"
    error_log_path = run_dir / "failed_downloads.csv"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        mock_base = temp_path / "spectro" / "redux" / "fuji" / "healpix" / SURVEY
        
        for prog in ['dark', 'bright', 'backup']:
            if not pending_targets:
                break
                
            prog_dir = mock_base / prog / str(group) / str(healpix)
            prog_dir.mkdir(parents=True, exist_ok=True)
            
            rr_file = f"redrock-{SURVEY}-{prog}-{healpix}.fits"
            coadd_file = f"coadd-{SURVEY}-{prog}-{healpix}.fits"
            base_url = f"https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/{SURVEY}/{prog}/{group}/{healpix}"
            
            rr_path = prog_dir / rr_file
            coadd_path = prog_dir / coadd_file
            
            try:
                pbar.set_postfix_str(f"Downloading redrock ({prog})")
                download_file(f"{base_url}/{rr_file}", rr_path)
                fits = fitsio.FITS(str(rr_path))
                rr_targets = set(fits['REDSHIFTS'].read(columns=['TARGETID'])['TARGETID'])
                
                matched_targets = pending_targets.intersection(rr_targets)
                if not matched_targets:
                    continue
                    
                pbar.set_postfix_str(f"Downloading coadd ({prog})")
                download_file(f"{base_url}/{coadd_file}", coadd_path)
                
                output_fits = temp_path / f"fastspec_{prog}.fits"
                
                pbar.set_postfix_str(f"Running fastspec ({prog})")
                success = run_fastspec_command(rr_path, matched_targets, output_fits, temp_path, error_log_path, healpix, prog)
                
                if success:
                    pbar.set_postfix_str(f"Extracting lines ({prog})")
                    extract_and_save_results(output_fits, output_csv, raw_csv, target_to_wiki)
                    
                pending_targets -= matched_targets
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code != 404:
                    log_error(error_log_path, healpix, prog, "HTTPError", str(e))
            except Exception as e:
                logger.error(f"Unexpected error processing HEALPix {healpix} in {prog}: {e}")
                log_error(error_log_path, healpix, prog, "UnknownError", str(e))
        
        if pending_targets:
            log_error(error_log_path, healpix, "ALL", "NotFound", f"Could not locate target(s) {pending_targets} in any program")


def run_extraction(config: dict, limit: int | None = None, retry_failed: Path | None = None) -> None:
    """Main extraction pipeline entry point."""
    logger.info("Setting up global dependencies...")
    setup_global_dependencies()
    
    parquet_path = PROJECT_ROOT / "data" / "crossmatch_cache" / "crossmatch_merged_1.0arcsec.parquet"
    healpix_groups, target_to_wiki = load_and_filter_dataset(parquet_path, limit, retry_failed)
    
    logger.info(f"Processing {sum(len(t) for _, t in healpix_groups)} targets across {len(healpix_groups)} HEALPix regions...")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = DATA_DIR / f"fastspec_extraction_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    error_log_path = run_dir / "failed_downloads.csv"
    with open(error_log_path, 'w') as f:
        f.write("healpix,program,error_type,message\n")
        
    pbar = tqdm(healpix_groups, desc="Extracting Targets")
    for healpix, all_targetids in pbar:
        process_healpix_region(healpix, all_targetids, run_dir, pbar, target_to_wiki)
        
    pbar.set_postfix_str("Done")

    logger.info("Extraction complete!")
    logger.info(f"Final results saved to {run_dir}")
