import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def identify_lines(csv_path, snr_threshold=3.0):
    df = pd.read_csv(csv_path)
    
    for index, row in df.iterrows():
        targetid = row['TARGETID']
        
        print(f"\nTarget ID: {targetid}")
        print(f"{'Line Name':<15} | {'Flux':<10} | {'SNR':<10}")
        print("-" * 42)
        
        detected = []
        
        flux_cols = [col for col in df.columns if col.endswith('_FLUX') and 'BOX' not in col]
        
        for flux_col in flux_cols:
            line_name = flux_col.replace('_FLUX', '')
            ivar_col = f"{line_name}_FLUX_IVAR"
            
            if ivar_col in df.columns:
                flux = row[flux_col]
                ivar = row[ivar_col]
                
                if pd.notna(flux) and pd.notna(ivar) and ivar > 0:
                    snr = flux * np.sqrt(ivar)
                    
                    if snr >= snr_threshold:
                        detected.append({
                            'line': line_name,
                            'flux': flux,
                            'snr': snr
                        })
        
        detected = sorted(detected, key=lambda x: x['snr'], reverse=True)
        
        if not detected:
            print(f"No emission lines detected with SNR >= {snr_threshold}")
        else:
            for item in detected:
                print(f"{item['line']:<15} | {item['flux']:<10.3f} | {item['snr']:<10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify apparent emission lines from FastSpecFit CSV.")
    parser.add_argument("csv_path", type=str, help="Path to the extracted CSV file.")
    parser.add_argument("--snr", type=float, default=3.0, help="Signal-to-Noise Ratio threshold (default: 3.0)")
    
    args = parser.parse_args()
    
    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        print(f"Error: File '{args.csv_path}' not found.")
    else:
        identify_lines(args.csv_path, args.snr)
