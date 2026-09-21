"""
DES_SN5YR_Downloader.py

Author: Andrew Dodge
Date: June 2025

Description:
Download DES_SN5YR data for use for UMH Validation.

Parameters:
- OUTPUT_FOLDER

Inputs:
- None

Output:
- Produces lcparam_full_long.csv
"""

import numpy as np
import os
import io
import sys
import json
import csv
import requests
import pandas as pd
from io import StringIO

def get_default_config(config_overrides=None):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        #2022 + corrections 
        "DES_SN5YR_DATA_URL": "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/DES-Dovekie_HD.csv",
        "DES_SN5YR_DATA_FILENAME": "DES-Dovekie_HD.csv",

        "DES_SN5YR_DATA_BIAS_URL": "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/STAT%2BSYS.npz",
        "DES_SN5YR_DATA_BIAS_FILENAME": "DES_SN5YR_STAT_SYS.npz",

        "DES_SN5YR_DATA_BIAS_SO_URL": "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/STATONLY.npz",
        "DES_SN5YR_DATA_BIAS_SO_FILENAME": "DES_SN5YR_STAT_ONLY.npz",
        #https://raw.githubusercontent.com/des-science/DES_SN5YR/refs/heads/main/4_DISTANCES_COVMAT/DES-Dovekie-SN_Likelihood.py

        "DPI":300, #PNG Resolution.
        "DTYPE":np.float64, #Precision.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }

def download_file(url, outpath, binary=True):
    print(f"Downloading: {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    #os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if binary:
        with open(outpath, "wb") as f: f.write(response.content)
    else:
        with open(outpath, "w", encoding="utf-8") as f: f.write(response.text)
    print(f"Saved: {outpath}")

def read_des_dovekie_hd(path):
    columns = None; rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            if line.startswith("VARNAMES:"): columns = line.split()[1:]; continue
            if line.startswith("SN:"): parts = line.split()[1:]; rows.append(parts)

    if columns is None: raise RuntimeError("Could not find VARNAMES line.")
    if not rows: raise RuntimeError("Could not find any SN: rows.")

    df = pd.DataFrame(rows, columns=columns)

    # Convert numeric columns
    for col in df.columns:
        if col != "CID": df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def run_des_sn5yr_data_download(config_overrides=None):
    config = get_default_config()
    if config_overrides: config.update(config_overrides)

    DES_SN5YR_DATA_URL=config["DES_SN5YR_DATA_URL"]
    DES_SN5YR_DATA_FILENAME=config["DES_SN5YR_DATA_FILENAME"]

    DES_SN5YR_DATA_BIAS_URL=config["DES_SN5YR_DATA_BIAS_URL"]
    DES_SN5YR_DATA_BIAS_FILENAME=config["DES_SN5YR_DATA_BIAS_FILENAME"]

    DES_SN5YR_DATA_BIAS_SO_URL=config["DES_SN5YR_DATA_BIAS_SO_URL"]
    DES_SN5YR_DATA_BIAS_SO_FILENAME=config["DES_SN5YR_DATA_BIAS_SO_FILENAME"]

    dtype=config["DTYPE"]
    dpi=config["DPI"]
    outdir = config["OUTPUT_FOLDER"]

    title="DES_SN5YR Data"
    file_hdr="PantheonData"
  
    print(f"✅ Starting Download: {title}.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)
    file_out=os.path.join(outdir, DES_SN5YR_DATA_FILENAME)
    file_bias_out=os.path.join(outdir, DES_SN5YR_DATA_BIAS_FILENAME)
    file_bias_so_out=os.path.join(outdir, DES_SN5YR_DATA_BIAS_SO_FILENAME)

    print(f"{title}: Files Will be Saved to {outdir}.")

    try:
        print(f"✅ {title}: Downloading data from {DES_SN5YR_DATA_URL}...")
        #response = requests.get(DES_SN5YR_DATA_URL)
        #response.raise_for_status()  # Raises an error for bad status codes
        download_file(DES_SN5YR_DATA_URL, file_out)
        # Decode content and wrap in StringIO
        #data = io.StringIO(response.content.decode('utf-8'))

        # Read with pandas
        #df = pd.read_csv(file_out, sep='\s+')  # Change delimiter if needed, delim_whitespace=True
        df = read_des_dovekie_hd(file_out)

        print(f"\n{file_out} columns:")
        print(list(df.columns))
        print(f"N rows: {len(df)}")

        required_cols = {"CID", "IDSURVEY", "zHD", "zHEL", "MU", "MUERR"}
        missing = required_cols - set(df.columns)
        if missing: raise RuntimeError(f"Missing expected columns in HD file: {missing}")

        # Save as comma-delimited CSV
        df.to_csv(file_out, index=False)

        print(f"✅ {title}: File successfully downloaded and saved as {file_out}")

        #print(f"✅ {title}: Downloading data from {DES_SN5YR_DATA_BIAS_URL}...")
        #response = requests.get(DES_SN5YR_DATA_BIAS_URL)
        #response.raise_for_status()  # Raises an error for bad status codes

        # Decode content and wrap in StringIO
        #data = io.StringIO(response.content.decode('utf-8'))

        for npz_path, npz_URL in [(file_bias_out, DES_SN5YR_DATA_BIAS_URL), (file_bias_so_out, DES_SN5YR_DATA_BIAS_SO_URL)]:
            print(f"✅ {title}: Downloading data from {npz_URL}...")
            download_file(npz_URL, npz_path)

            with open(npz_path, "rb") as f: magic = f.read(4)

            print(f"\n{npz_path} first bytes: {magic}")

            if magic.startswith(b"\x1f\x8b"): print(f"WARNING: {npz_path} appears gzip-compressed. This would be unusual for .npz; inspect before using.")
            elif magic.startswith(b"PK"): print(f"{npz_path} is a normal ZIP/NPZ archive.")
            else: print(f"WARNING: {npz_path} does not look like a standard NPZ archive.")

            data = np.load(npz_path)
            print(f"{npz_path} keys: {list(data.keys())}")

        # Read with pandas
        #df = pd.read_csv(data,sep='\s+')  # Change delimiter if needed, delim_whitespace=True

        # Save as comma-delimited CSV
        #df.to_csv(file_bias_out, index=False)

        print(f"✅ {title}: File successfully downloaded and saved as {file_out}")

        print(f"✅ Finished Dwonloading: {title}.")

    except requests.exceptions.RequestException as e:
        print(f"❌ {title}: Error occurred during download: {e}.")

        print(f"❌ {title}: Finished, but failed to retrieve the: {title}.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_des_sn5yr_data_download(config)