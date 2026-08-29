import os
import requests
import subprocess
import fitsio
import pandas as pd
import re
from pathlib import Path

# Hardcoded target values identified from the dataset
SURVEY = 'sv3'
HEALPIX_29 = 643592388338055503
TARGETID = 39632951360620188
HEALPIX = HEALPIX_29 // (4**23)
GROUP = HEALPIX // 100

# Directories
DATA_DIR = Path("data/desi_raw_fits")
DUST_DIR = Path("data/dust/maps")
FTEMPLATES_DIR = Path("data/templates")
OUTPUT_DIR = Path("experiments/fastspec_healpix/output")

DUST_DIR.mkdir(parents=True, exist_ok=True)
FTEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"File already exists, skipping download: {dest_path}")
        return
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Saved to {dest_path}")

def setup_dust_maps():
    dust_base = "https://portal.nersc.gov/project/cosmo/data/dust/v0_1/maps"
    for map_name in ['SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits']:
        url = f"{dust_base}/{map_name}"
        dest = DUST_DIR / map_name
        download_file(url, dest)

def setup_templates():
    template_version = "2.2.0"
    template_name = f"ftemplates-chabrier-{template_version}.fits"
    url = f"https://data.desi.lbl.gov/public/external/templates/fastspecfit/{template_version}/{template_name}"
    dest = FTEMPLATES_DIR / "2.2.0" / template_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, dest)

def find_target_program():
    programs = ['dark', 'bright', 'backup']
    for prog in programs:
        print(f"Checking program: {prog}")
        redrock_file = f"redrock-{SURVEY}-{prog}-{HEALPIX}.fits"
        base_url = f"https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/{SURVEY}/{prog}/{GROUP}/{HEALPIX}"
        redrock_url = f"{base_url}/{redrock_file}"
        
        mock_dir = DATA_DIR / "spectro" / "redux" / "fuji" / "healpix" / SURVEY / prog / str(GROUP) / str(HEALPIX)
        mock_dir.mkdir(parents=True, exist_ok=True)
        redrock_path = mock_dir / redrock_file
        
        try:
            download_file(redrock_url, redrock_path)
            # Check if TARGETID is in this file
            fits = fitsio.FITS(str(redrock_path))
            targets = fits['REDSHIFTS'].read(columns=['TARGETID'])
            if TARGETID in targets['TARGETID']:
                print(f"Target {TARGETID} found in {prog} program!")
                return prog, base_url, mock_dir, redrock_path
            else:
                print(f"Target not found in {prog}.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"No redrock file for {prog}.")
            else:
                print(f"Error downloading {prog}: {e}")
        except Exception as e:
            print(f"Error checking {prog}: {e}")
            
    raise ValueError(f"Target {TARGETID} not found in any program for HEALPIX {HEALPIX}")

def download_tractor_catalog(tractor_path_str):
    tractor_path = Path(tractor_path_str)
    
    # Extract the relative path like 'north/tractor/218/tractor-2185p330.fits'
    parts = tractor_path.parts
    if 'tractor' in parts:
        idx = parts.index('tractor')
        rel_parts = parts[idx-1:] # e.g. ('north', 'tractor', '218', 'tractor-2185p330.fits')
        rel_path = "/".join(rel_parts)
    else:
        raise ValueError(f"Could not parse tractor path: {tractor_path_str}")
        
    url = f"https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{rel_path}"
    tractor_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, tractor_path)

def run_fastspec(redrock_path, output_fits):
    env = os.environ.copy()
    env['DESI_ROOT'] = str(DATA_DIR.absolute())
    env['DESI_SPECTRO_REDUX'] = str((DATA_DIR / 'spectro' / 'redux').absolute())
    env['SPECPROD'] = 'fuji'
    env['DUST_DIR'] = str(DUST_DIR.parent.absolute())
    env['FPHOTO_DIR'] = str(DATA_DIR.absolute())
    env['FTEMPLATES_DIR'] = str(FTEMPLATES_DIR.absolute())

    cmd = [
        "fastspec",
        str(redrock_path),
        "--targetids", str(TARGETID),
        "--ignore-photometry",
        "--outfile", str(output_fits)
    ]
    
    while True:
        print(f"Running Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print("FastSpecFit completed successfully.")
            break
        else:
            # Parse error for tractor catalog
            match = re.search(r"Unable to find Tractor catalog\s+(\S+)", result.stderr)
            if not match:
                match = re.search(r"Unable to find Tractor catalog\s+(\S+)", result.stdout)
            
            if match:
                missing_tractor = match.group(1)
                print(f"Missing Tractor catalog detected: {missing_tractor}")
                download_tractor_catalog(missing_tractor)
            else:
                print(result.stderr)
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

def extract_lines(output_fits, output_csv):
    print(f"Extracting lines from {output_fits}...")
    if not output_fits.exists():
        raise FileNotFoundError(f"Output FITS not found: {output_fits}")
    
    fits = fitsio.FITS(str(output_fits))
    data = fits['FASTSPEC'].read()
    
    df = pd.DataFrame(data)
    
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
    df.to_csv(output_csv, index=False)
    print(f"Saved extracted measurements to {output_csv}")

if __name__ == "__main__":
    # 1. Setup global dependencies
    setup_dust_maps()
    setup_templates()
    
    # 2. Find which program this target belongs to
    prog, base_url, mock_dir, redrock_path = find_target_program()
    
    # 3. Download the heavy coadd file for that program
    coadd_file = f"coadd-{SURVEY}-{prog}-{HEALPIX}.fits"
    coadd_url = f"{base_url}/{coadd_file}"
    coadd_path = mock_dir / coadd_file
    download_file(coadd_url, coadd_path)
    
    # 4. Run FastSpec (will auto-download missing tractor catalogs if needed)
    output_fits = OUTPUT_DIR / f"fastspec-{SURVEY}-{prog}-{HEALPIX}.fits"
    run_fastspec(redrock_path, output_fits)
    
    # 5. Extract to CSV
    output_csv = OUTPUT_DIR / f"extracted_lines_{TARGETID}.csv"
    extract_lines(output_fits, output_csv)
    print("Experiment completed successfully!")
