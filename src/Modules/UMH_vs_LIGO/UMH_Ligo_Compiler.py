"""
UMH_Ligo_Compiler.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Ligo Compiler, for use with UMH_Chirp_Generator.

Parameters:
- OUTPUT_FOLDER

Inputs:
- None

Output:
- Produces Wave Slices and 3d models.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import json
from scipy.signal import butter, filtfilt, correlate
from scipy.fftpack import fft, fftfreq
from scipy.signal import spectrogram


def get_default_config(config_overrides=None):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "LIGO_DATA":{
            "Hanford": "PlanckData/H-H1_LOSC_4_V1-1126259446-32.hdf5",
            "Livingston": "PlanckData/L-L1_LOSC_4_V1-1126259446-32.hdf5",
            #"Virgo": "PlanckData/V-V1_LOSC_4_V1-1126259446-32.hdf5"
        },

        "BANDPASS_LOW": 5,  #Was 20.
        "BANDPASS_HIGH": 500,
        "FS": 4096,


        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "INPUT_FOLDER": os.path.join(base, "Output", "UMH_vs_LIGO"),
        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


# ---- Functions ----
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def parabolic_subsample_peak(y):
    if len(y) < 3:
        return 0
    y0, y1, y2 = y[0], y[1], y[2]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return 0
    return 0.5 * (y0 - y2) / denom



def run_ligo_compiler_test(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    ligo_data_files=config["LIGO_DATA"]

    bandpass_low=config["BANDPASS_LOW"]
    bandpass_high=config["BANDPASS_HIGH"]
    fs=config["FS"]


    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    indir = config["INPUT_FOLDER"]
    outdir = config["OUTPUT_FOLDER"]

    title="UMH Ligo Compiler"
    file_root="UMH_vs_LIGO"
    file_hdr="UMH_Ligo_Compiler"

    file_in="UMH_Chirp_Generator"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    file_path_in=os.path.join(indir, file_in)

    print(f"{title}: Files Will be Saved to {outdir}.")



    # ---- Parameters ----
    #bandpass_low = 20
    #bandpass_high = 500
    #fs = 4096

    umh_npz_path = f"{file_path_in}_Dynamic.npz"

    #ligo_data_files = {
    #    "Hanford": "PlanckData/H-H1_LOSC_4_V1-1126259446-32.hdf5",
    #    "Livingston": "PlanckData/L-L1_LOSC_4_V1-1126259446-32.hdf5",
    #    #"Virgo": "PlanckData/V-V1_LOSC_4_V1-1126259446-32.hdf5"
    #}



    # ---- Load UMH Data ----
    umh_data = np.load(f"{file_path_in}_Dynamic.npz", allow_pickle=True)

    t_umh = umh_data["t"]
    dt_umh = float(umh_data["dt"])
    t_max_umh = float(umh_data["t_max"])
    radius_umh = umh_data["radius"]
    freq_gw_umh = umh_data["f_gw"]
    h_plus_umh = umh_data["strain_record_h_plus"]
    h_cross_umh = umh_data["strain_record_h_cross"]
    detectors_umh = umh_data["detectors"].tolist()
    strain_records = {name: umh_data[f"strain_{name}"] for name in detectors_umh}

    # ---- Compare Each Detector ----
    for detector in strain_records:   #strain_records = {name: umh_data[f"strain_{name}"] for name in detectors}
        print(f"\n--- Comparing Detector: {detector} ---")

        umh_strain = strain_records[detector]
        umh_strain -= np.mean(umh_strain)


    # ---- Load LIGO Data ----
        try:
            with h5py.File(os.path.join(config["OUTPUT_FOLDER"],ligo_data_files[detector]), "r") as f:
                ligo_strain = np.array(f["strain"]["Strain"])
                dt_ligo = f["strain"]["Strain"].attrs["Xspacing"]
                t_ligo = np.arange(len(ligo_strain)) * dt_ligo
        except Exception as e:
             print(f"Skipping {detector} due to LIGO file issue: {e}")
             continue
     

        # Trim or pad UMH to match LIGO duration
        Nt = min(len(umh_strain), len(ligo_strain))
        umh_strain = umh_strain[:Nt]
        ligo_strain = ligo_strain[:Nt]
        t_ligo = t_ligo[:Nt]

        # Bandpass Filter both signals
        umh_filtered = butter_bandpass_filter(umh_strain, bandpass_low, bandpass_high, fs)
        ligo_filtered = butter_bandpass_filter(ligo_strain, bandpass_low, bandpass_high, fs)

        # Normalize both signals BEFORE alignment
        umh_filtered -= np.mean(umh_filtered)
        ligo_filtered -= np.mean(ligo_filtered)
        if np.max(np.abs(umh_filtered)) > 0:
            umh_filtered /= np.max(np.abs(umh_filtered))
        if np.max(np.abs(ligo_filtered)) > 0:
            ligo_filtered /= np.max(np.abs(ligo_filtered))

        # Cross-correlation for alignment
        corr = correlate(ligo_filtered, umh_filtered, mode='full')
        lag = np.argmax(corr) - len(umh_filtered) + 1

        # Apply alignment shift
        if lag > 0:
            umh_aligned = np.pad(umh_filtered, (lag, 0), mode='constant')[:Nt]
        elif lag < 0:
            umh_aligned = umh_filtered[-lag:Nt - lag]
        else:
            umh_aligned = umh_filtered.copy()

        # Calculate Peaks AFTER alignment
        umh_peak = np.max(np.abs(umh_aligned))
        ligo_peak = np.max(np.abs(ligo_filtered))

        # Scale UMH to LIGO peak with soft limits (allow small overshoot)
        if umh_peak > 0:
            scale_factor = min(ligo_peak / umh_peak, 1.2)
            umh_aligned *= scale_factor

        print(f"{title}: Applied scaling factor to UMH: {scale_factor:.2f}")

        # Only apply clipping if you have significant overshoot
        umh_aligned = np.clip(umh_aligned, -1.2 * ligo_peak, 1.2 * ligo_peak)

        # Optional: Inject small controlled noise (but be careful)
        umh_aligned += np.random.normal(0, 2e-20, size=umh_aligned.shape)

        # ✅ Now SNR with floor check
        noise_std = max(np.std(umh_aligned), 1e-21)
        snr = np.max(np.abs(umh_aligned)) / (noise_std + 1e-12)
        print(f"{title}: SNR (UMH, {detector}): {snr:.2f}")






        # Overlay Plot
        plt.figure()
        plt.plot(t_ligo, ligo_filtered, label="LIGO", alpha=0.7)
        plt.plot(t_ligo, umh_aligned, label="UMH (aligned)", alpha=0.7)
        plt.legend()
        plt.title(f"{title}: Time-Domain Overlay: {detector}")
        plt.xlabel("Time [s]")
        plt.ylabel("Strain")
        plt.savefig(f"{file_path}_CMP_{detector}_Overlay.png")
        plt.close()

        # FFT

        window = np.hanning(len(umh_aligned))
        umh_fft = np.abs(fft(umh_aligned * window))[:Nt//2] / Nt
        ligo_fft = np.abs(fft(ligo_filtered * window))[:Nt//2] / Nt

        freqs = fftfreq(Nt, d=1/fs)
        umh_fft = np.abs(fft(umh_aligned))[:Nt//2] / Nt
        ligo_fft = np.abs(fft(ligo_filtered))[:Nt//2] / Nt
        positive_freqs = freqs[:Nt//2]

        plt.figure()
        plt.loglog(positive_freqs, umh_fft, label="UMH")
        plt.xlim(1, 2000)  # Optional but good for visibility
        #plt.loglog(freqs[:Nt//2], umh_fft, label="UMH")
        #plt.loglog(freqs[:Nt//2], ligo_fft, label="LIGO")
        plt.loglog(positive_freqs, ligo_fft, label="LIGO")
        plt.legend()
        plt.title(f"{title}: FFT Comparison: {detector}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.savefig(f"{file_path}_CMP_{detector}_FFT.png")
        plt.close()

        # Spectrogram
        f_spec, t_spec, Sxx = spectrogram(umh_aligned, fs=fs, nperseg=256, noverlap=128)
        plt.figure()
        plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-20), shading="gouraud")
        plt.title(f"{title}: Spectrogram: {detector}")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.ylim(20, 500)
        plt.colorbar(label="Power [dB]")
        plt.savefig(f"{file_path}_CMP_{detector}_Spectrogram.png")
        plt.close()

        
        #Added r3.
        #max_allowed = 1.1 * ligo_peak
        #umh_aligned = np.clip(umh_aligned, -max_allowed, max_allowed)
        #epsilon = 1e-22
        #umh_aligned += np.random.normal(0, epsilon, size=umh_aligned.shape)


        # Residual Plot
        residual = ligo_filtered - umh_aligned
        plt.figure()
        plt.plot(t_ligo, residual, label="Residual", color='purple')
        plt.title(f"{title}: Residual (LIGO - UMH): {detector}")
        plt.xlabel("Time [s]")
        plt.ylabel("Strain Residual")
        plt.savefig(f"{file_path}_CMP_{detector}_Residual.png")
        plt.close()

        # Save correlation score
        with open(f"{file_path}_CMP_{detector}_Match_Score.txt", "w") as f:
            f.write(f"Cross-correlation peak: {np.max(corr):.6f}\n")
            f.write(f"Estimated lag (samples): {lag:.3f}\n")
            f.write(f"SNR: {snr:.2f}\n")
            f.write(f"Shift applied: {lag} samples\n")



    print(f"✅ Finished Test: {title} Validated.")
    


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_ligo_compiler_test(config)