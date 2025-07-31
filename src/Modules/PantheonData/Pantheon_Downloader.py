"""
Pantheon_Downloader.py

Author: Andrew Dodge
Date: June 2025

Description:
Download Pantheon data for use for UMH Validation.

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

def get_default_config(config_overrides=None):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "PANTHEON_DATA_URL": "https://github.com/dscolnic/Pantheon/raw/refs/heads/master/lcparam_full_long.txt", #"https://pantheonplusshoes.github.io/Data/zcmb/lcparam_full_long.csv",
        "PANTHEON_DATA_FILENAME": "lcparam_full_long.csv",

        "DPI":300, #PNG Resolution.
        "DTYPE":np.float64, #Precision.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


def run_pantheon_data_download(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    PANTHEON_DATA_URL=config["PANTHEON_DATA_URL"]
    PANTHEON_DATA_FILENAME=config["PANTHEON_DATA_FILENAME"]

    dtype=config["DTYPE"]
    dpi=config["DPI"]
    outdir = config["OUTPUT_FOLDER"]

    title="Pantheon Data"
    file_hdr="PantheonData"
  
    print(f"✅ Starting Download: {title}.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)
    file_out=os.path.join(outdir, PANTHEON_DATA_FILENAME)

    print(f"{title}: Files Will be Saved to {outdir}.")

    try:
        print(f"✅ {title}: Downloading data from {PANTHEON_DATA_URL}...")
        response = requests.get(PANTHEON_DATA_URL)
        response.raise_for_status()  # Raises an error for bad status codes

        # Decode content and wrap in StringIO
        data = io.StringIO(response.content.decode('utf-8'))

        # Read with pandas
        df = pd.read_csv(data,sep='\s+')  # Change delimiter if needed, delim_whitespace=True

        # Save as comma-delimited CSV
        df.to_csv(file_out, index=False)

        #with open(file_out, "wb") as file:
        #    file.write(response.content)

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
    run_pantheon_data_download(config)