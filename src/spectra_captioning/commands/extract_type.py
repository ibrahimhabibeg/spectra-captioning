"""Command to extract object types (class/subclass) from SDSS and DESI catalogs."""

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd
from astropy.table import vstack
from astroquery.sdss import SDSS
from dl import queryClient as qc


def to_int(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    m = re.match(r"^b['\"](.*)['\"]$", s)
    if m:
        s = m.group(1)
    return int(s.strip())

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_sdss(df_sdss: pd.DataFrame) -> pd.DataFrame:
    df_sdss = df_sdss.copy()
    df_sdss['object_id'] = df_sdss['object_id'].apply(to_int)
    
    batch_size = 64
    results = []
    all_ids = df_sdss['object_id'].tolist()
    total_batches = (len(all_ids) + batch_size - 1) // batch_size
    
    logging.info(f"Fetching {total_batches} batches for SDSS.")
    for i in range(0, len(all_ids), batch_size):
        batch_num = i // batch_size + 1
        logging.info(f"Fetching SDSS batch {batch_num}/{total_batches}...")
        batch_ids = all_ids[i:i + batch_size]
        ids_str = ','.join(str(x) for x in batch_ids if pd.notna(x))
        if not ids_str:
            continue
            
        query = f"""\
            SELECT specobjid, class, subclass
            FROM SpecObjAll
            WHERE specobjid IN ({ids_str})
        """
        try:
            res = SDSS.query_sql(query)
            if res is not None:
                results.append(res)
        except Exception as e:
            logging.error(f"Error at SDSS batch {batch_num}: {e}")
        
        time.sleep(1) # Respect API rate limits
        
    if results:
        full_res = vstack(results).to_pandas()
    else:
        full_res = pd.DataFrame(columns=['specobjid', 'class', 'subclass'])
        
    # Merge back to get wiki_entity_id
    merged = df_sdss.merge(full_res, left_on='object_id', right_on='specobjid', how='inner')
    merged = merged[['wiki_entity_id', 'class', 'subclass']]
    return merged

def process_desi(df_desi: pd.DataFrame) -> pd.DataFrame:
    df_desi = df_desi.copy()
    df_desi['object_id'] = df_desi['object_id'].astype('int64')
    targetids = df_desi['object_id'].tolist()
    
    results = []
    batch_size = 64
    total_batches = (len(targetids) + batch_size - 1) // batch_size
    
    logging.info(f"Fetching {total_batches} batches for DESI.")
    for i, batch in enumerate(chunks(targetids, batch_size)):
        logging.info(f"Fetching DESI batch {i + 1}/{total_batches}...")
        id_str = ','.join(str(t) for t in batch if pd.notna(t))
        if not id_str:
            continue
        q = f"SELECT targetid, spectype, subtype FROM desi_dr1.zpix WHERE targetid IN ({id_str})"
        try:
            res = qc.query(sql=q, fmt='pandas')
            results.append(res)
        except Exception as e:
            logging.error(f"Error at DESI batch {i + 1}: {e}")
            
    if results:
        zcat = pd.concat(results, ignore_index=True)
    else:
        zcat = pd.DataFrame(columns=['targetid', 'spectype', 'subtype'])
        
    merged = df_desi.merge(zcat, left_on='object_id', right_on='targetid', how='inner')
    merged.rename(columns={'spectype': 'class'}, inplace=True)
    merged['subclass'] = None
    merged = merged[['wiki_entity_id', 'class', 'subclass']]
    return merged

def run_extract_type(args_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract object types from surveys (SDSS and DESI) and store them in a CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/crossmatch_cache/crossmatch_merged_1.0arcsec.parquet"),
        help="Input parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/extracted_types.csv"),
        help="Output CSV file path.",
    )
    args = parser.parse_args(args_list)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    if not args.input.exists():
        logging.error(f"Input file not found at {args.input}")
        return

    logging.info(f"Loading data from {args.input}")
    df_full = pd.read_parquet(args.input)
    df_full = df_full.drop_duplicates(subset=['wiki_entity_id'])

    df_sdss = df_full[df_full['survey'] == 'sdss']
    df_desi = df_full[df_full['survey'] == 'desi']

    logging.info(f"Found {len(df_sdss)} SDSS records and {len(df_desi)} DESI records.")

    result_dfs = []
    
    if not df_sdss.empty:
        sdss_res = process_sdss(df_sdss)
        result_dfs.append(sdss_res)
        
    if not df_desi.empty:
        desi_res = process_desi(df_desi)
        result_dfs.append(desi_res)
        
    if result_dfs:
        final_df = pd.concat(result_dfs, ignore_index=True)
        final_df.to_csv(args.output, index=False)
        logging.info(f"Saved {len(final_df)} records to {args.output}")
    else:
        logging.info("No records processed.")

if __name__ == "__main__":
    run_extract_type()

