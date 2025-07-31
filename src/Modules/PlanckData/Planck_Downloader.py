"""
Plank_Downloader.py

Author: Andrew Dodge
Date: June 2025

Description:
Download Planck data for use for UMH Validation.

Parameters:
- OUTPUT_FOLDER

Inputs:
- None

Output:
- Produces lcparam_full_long.csv
"""

import numpy as np
import os
import sys
import json
import csv
import requests
import sys
import time
from pathlib import Path
from tqdm import tqdm


def get_default_config(config_overrides=None):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.

        "PLANCK_CMB_TXT_DATA_URL": "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-TT-full_R3.01.txt", #COM_PowerSpect_CMB-tt-full_R3.01.txt
        "PLANCK_CMB_TXT_DATA_FILENAME": "COM_PowerSpect_CMB-tt-full_R3.01.txt",

        "PLANCK_HANFORD_LIGO_DATA_URL": "https://losc.ligo.org/s/events/GW150914/H-H1_LOSC_4_V1-1126259446-32.hdf5", #H-H1_LOSC_4_V1-1126259446-32.hdf5
        "PLANCK_HANFORD_LIGO_DATA_FILENAME": "H-H1_LOSC_4_V1-1126259446-32.hdf5",

        "PLANCK_LIVINGSTON_LIGO_DATA_URL": "https://losc.ligo.org/s/events/GW150914/L-L1_LOSC_4_V1-1126259446-32.hdf5", #L-L1_LOSC_4_V1-1126259446-32.hdf5
        "PLANCK_LIVINGSTON_LIGO_DATA_FILENAME": "L-L1_LOSC_4_V1-1126259446-32.hdf5",

        "PLANCK_CMB_FITS_DATA_URL": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits", #COM_CMB_IQU-smica_2048_R3.00_full.fits
        "PLANCK_CMB_FITS_DATA_FILENAME": "COM_CMB_IQU-smica_2048_R3.00_full.fits",

        "DPI":300, #PNG Resolution.
        "DTYPE":np.float64, #Precision.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }



def download_file(title,url, file_out, chunk_size=1024*1024):
    dwnerr=0
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        mode = 'wb'
        with open(file_out, mode) as file:
            with tqdm(
                desc=f"✅ {title}: Downloading {file_out}",
                total=total_size if total_size > 0 else None,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        bar.update(len(chunk))

        print(f"✅ {title}: Download complete, saved as: {file_out}")

    except requests.exceptions.RequestException as err:
        print(f"❌ {title}: Error occurred during download: {err}.")
        dwnerr+=1

    return dwnerr


def run_pantheon_data_download(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)
    
    PLANCK_CMB_TXT_DATA_URL=config["PLANCK_CMB_TXT_DATA_URL"]
    PLANCK_CMB_TXT_DATA_FILENAME=config["PLANCK_CMB_TXT_DATA_FILENAME"]

    PLANCK_HANFORD_LIGO_DATA_URL=config["PLANCK_HANFORD_LIGO_DATA_URL"]
    PLANCK_HANFORD_LIGO_DATA_FILENAME=config["PLANCK_HANFORD_LIGO_DATA_FILENAME"]

    PLANCK_LIVINGSTON_LIGO_DATA_URL=config["PLANCK_LIVINGSTON_LIGO_DATA_URL"]
    PLANCK_LIVINGSTON_LIGO_DATA_FILENAME=config["PLANCK_LIVINGSTON_LIGO_DATA_FILENAME"]

    PLANCK_CMB_FITS_DATA_URL=config["PLANCK_CMB_FITS_DATA_URL"]
    PLANCK_CMB_FITS_DATA_FILENAME=config["PLANCK_CMB_FITS_DATA_FILENAME"]

    dtype=config["DTYPE"]
    dpi=config["DPI"]
    outdir = config["OUTPUT_FOLDER"]

    title="Planck Data"
    file_hdr="PlanckData"
  
    print(f"✅ Starting Download: {title}.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")
    
    sys.stdout.reconfigure(line_buffering=True)

    dwnerr=0
    
    try:
        file_out=os.path.join(outdir, PLANCK_CMB_TXT_DATA_FILENAME)
        file_out_chk = Path(file_out)
        if not file_out_chk.exists() or input(f"{file_out} exists. Overwrite? (y/n): ").strip().lower() == 'y':
            dwnerr+=download_file(title,PLANCK_CMB_TXT_DATA_URL,file_out)
        else:
            print(f"❌ {title}: Skipping download for: {file_out}.")


    except requests.exceptions.RequestException as e:
        print(f"❌ {title}: Error occurred during download: {e}.")
        dwnerr+=1

    try:
        file_out=os.path.join(outdir, PLANCK_HANFORD_LIGO_DATA_FILENAME)
        file_out_chk = Path(file_out)
        if not file_out_chk.exists() or input(f"{file_out} exists. Overwrite? (y/n): ").strip().lower() == 'y':
            dwnerr+=download_file(title,PLANCK_HANFORD_LIGO_DATA_URL,file_out)
        else:
            print(f"❌ {title}: Skipping download for: {file_out}.")


    except requests.exceptions.RequestException as e:
        print(f"❌ {title}: Error occurred during download: {e}.")
        dwnerr+=1

    try:
        file_out=os.path.join(outdir, PLANCK_LIVINGSTON_LIGO_DATA_FILENAME)
        file_out_chk = Path(file_out)
        if not file_out_chk.exists() or input(f"{file_out} exists. Overwrite? (y/n): ").strip().lower() == 'y':
            dwnerr+=download_file(title,PLANCK_LIVINGSTON_LIGO_DATA_URL,file_out)
        else:
            print(f"❌ {title}: Skipping download for: {file_out}.")


    except requests.exceptions.RequestException as e:
        print(f"❌ {title}: Error occurred during download: {e}.")
        dwnerr+=1

    try:
        file_out=os.path.join(outdir, PLANCK_CMB_FITS_DATA_FILENAME)
        file_out_chk = Path(file_out)
        if not file_out_chk.exists() or input(f"{file_out} exists. Overwrite? (y/n): ").strip().lower() == 'y':
            dwnerr+=download_file(title,PLANCK_CMB_FITS_DATA_URL,file_out)
        else:
            print(f"❌ {title}: Skipping download for: {file_out}.")

    except requests.exceptions.RequestException as e:
        print(f"❌ {title}: Error occurred during download: {e}.")
        dwnerr+=1


    if(dwnerr>0):
        print(f"❌ {title}: Finished, but failed to retrieve the: {title}.")
    else:
        print(f"✅ {title}: Finished, All files necessary downloaded for: {title}.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_pantheon_data_download(config)