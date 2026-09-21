"""
UMH_Chirp_Generator.py

Author: Andrew Dodge
Date: June 2025

Ultronic Medium Hypothesis (UMH) Chirp Generator

This script generates UMH-consistent model gravitational-wave strains for use
with UMH_Ligo_Compiler and data/peer-review comparisons (e.g. GW150914).

High-level design
-----------------
1. Optional UMH soliton source model
   - When SOURCE_ENGINE=soliton_umh or hybrid_umh, evolve a 3D ultronic medium field phi(x,y,z,t) with two orbiting "solitons" as a UMH-native source model.
   - A differential probe of phi defines a soliton-based amplitude trace A_raw_Sol.
   - These soliton fields are UMH diagnostics only; they are never sampled directly as detector strain.

2. Soliton-informed or analytic amplitude envelope
   - Default GW150914-style runs set SOURCE_ENGINE=analytic_umh and use a smooth analytic envelope driven by PN scaling.
   - When enabled, the soliton probe defines a modulation A_raw_Sol → A_hist that shapes the envelope multiplicatively without per-event tuning.

3. Analytic phase and composite track
   - f_GW(t) and phi_GW(t) are generated using the leading-order 0PN-equivalent UMH chirp evolution used in the published analysis,
     followed by the UMH post-merger relaxation/ringdown construction.
   - A single intrinsic track f_GW(t), phi_GW(t) is used for all detectors.

4. Physics normalization (single global scale)
   - If PHYSICS_NORM_ENABLE=True, one scalar gain G_amp is fixed by the UMH quadrupole amplitude evaluated at f_ref_src = (1+z_UMH) f_ref_obs,
     using the source-frame chirp mass and the full UMH effective luminosity distance D_L^UMH from the Pantheon+-calibrated source--observer mapping.
     No per-detector tuning is applied.

#4. Physics normalization (single global scale)
#   - If PHYSICS_NORM_ENABLE=True, one scalar gain G_amp is fixed at f_ref_obs by the UMH quadrupole amplitude newtonian_h_at_f_umh[_fitted] using the
#     UMH tension–redshift law (UMH_z_tension). No per-detector tuning.

5. Detector strains
   - Detector-frame strains are built as: h_det(t) = G_amp * A_composite(t - tau_det) * [F_plus * cos(phi_GW) + F_cross * sin(phi_GW)],
     using standard antenna patterns and geometric delays.

Key clarity points
------------------
- Soliton fields and A_raw_Sol are diagnostics of the ultronic medium.
- The only arrays intended for quantitative comparison to LIGO/Virgo are the stored strain_{det}(t) built from the analytic PN+ringdown track
  with a single UMH-based normalization.
- Visualization (noise, whitening, spectrograms) operates on copies only.
"""

# - The 3D UMH soliton field (phi, A_raw) is currently only an internal UMH diagnostic, this is in the process of being built out to provide full simulation.
# - The physically relevant waveforms for comparison to LIGO/Virgo are: strain_records[det_name]  (stored as strain_{det} in output)
#   built from the analytic PN+ringdown track plus a single global UMH/GR- consistent normalization. 
#   No detector-specific amplitude rescaling is applied.
# - Visualization helpers (noise injection, whitening, spectrograms, custom colormaps)
#   NEVER modify the stored physics strain arrays.


import numpy as np
import os, time
os.environ["MPLBACKEND"] = "Agg"  # must be set before importing matplotlib when run cmd line.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys, json, math
from numba import njit, prange
from scipy.signal import butter, filtfilt, hilbert, spectrogram, savgol_filter
from scipy.special import expit
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse


def get_default_config(config_overrides=None):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # NOTE:
    # The "replica_gw150914" profile is a convenience preset using published
    # GW150914-like source parameters. It does NOT change any UMH physics;
    # it only populates config defaults so the generator is reproducible.
    replica = {
        # Source (GW150914-like; medians/typical)
        "profile":          "replica_gw150914",

        "event_utc":        (2015, 9, 14, 9, 50, 45.391),   # 2015-09-14 09:50:45.391

        "Sites_Used":       {"Hanford", "Livingston"},

        "M1_solar_src":      37.03,
        "M2_solar_src":      35.40,

        "distance_Mpc":     336.25,
        
        "ra_deg":            67.49,
        "dec_deg":          -72.34,

        "pol_psi_deg":      -64.75,
        
        # Binary inclination in degrees (0° = face-on, 90° = edge-on, 180° = face-off)
        "BINARY_IOTA_DEG": -120.25,


        # If RINGDOWN_OVERRIDE is provided:
        # - Merge it into any existing ringdown config.
        # - These override-derived values become the *authoritative* f_rd_obs_Hz,
        #   tau_rd_obs, f_merge_Hz used for both waveform construction AND metadata.
        #   RINGDOWN_OVERRIDE its f_rd_obs_Hz / tau_rd_obs / f_merge_Hz replace the Q-based defaults.
        #"RINGDOWN_OVERRIDE": {"f_rd_obs_Hz": 250.0, "tau_rd_obs": 0.004, "f_merge_Hz": 150.0},

        "f_min_obs":         20.0000,                    # Minimum Frequency hz.
        
        "USE_UMH_DFDT_PROFILE":        False,  # UMH Produces good results at 0pn.
        # Publication configuration: use the leading-order 0PN-equivalent
        # UMH inspiral evolution reported in the paper.
        # Higher-order GR-limit PN corrections are retained only for diagnostic and extension studies and are not enabled in the
        # published GW150914/GW170814 runs.
        "UMH_DFDT_PN_PROFILE":          0.0,
        "USE_UMH_AMP_PN_PROFILE":      False,   # UMH Produces good results at 0pn, no Amplitude PN Necessary, for Diagnostic only.
        
        "UMH_RedShift_Calibration_File": os.path.join(base, "Output", "UMH_RedShift", "UMH_RedShift_Calibration_Fit.json"),
        
        # --- UMH frequency-redshift (detector time dilation) ---
        "APPLY_UMH_FREQ_REDSHIFT":      True,   # default ON → Use Src Obs effect to apply Pantheon+ Calibration Freq RedShift and Time Dilation.
        "APPLY_UMH_AMPLITUDE_SCALING":  True,   # Use UMH Tension Pantheon+ Calibration for Amplitude scaling.

        #All headline comparisons to GW150914 use the analytic UMH/PN track (SOURCE_ENGINE=analytic_umh). 
        #Using soliton_umh, or hybrid_umh SOURCE_ENGINE introduces small, parameter-free fluctuations sourced by the soliton field; 
        #these are presented separately as candidate UMH microstructure, not used to fit the main chirp.
        #This is used to perform true medium wave simulations for upcoming research.
        "SOURCE_ENGINE":      "analytic_umh",   # analytic_umh, soliton_umh, hybrid_umh
    }

    replica_gw170814 = {
        # Source (GW150914-like; medians/typical)
        "profile":          "replica_gw170814",

        "event_utc":        (2017, 8, 14, 10, 30, 43.530),   # 2017-08-14 10:30:43

        "Sites_Used":       {"Hanford", "Livingston", "Virgo"},

        "M1_solar_src":       51.548,
        "M2_solar_src":       22.052,

        "distance_Mpc":       506.125,

        "ra_deg":            359.10,
        "dec_deg":           -80.10,

        "pol_psi_deg":       168.6,
        
        # Binary inclination in degrees (0° = face-on, 90° = edge-on, 180° = face-off)
        "BINARY_IOTA_DEG":    67.0,

        # If RINGDOWN_OVERRIDE is provided:
        # - Merge it into any existing ringdown config.
        # - These override-derived values become the *authoritative* f_rd_obs_Hz,
        #   tau_rd_obs, f_merge_Hz used for both waveform construction AND metadata.
        #   RINGDOWN_OVERRIDE its f_rd_obs_Hz / tau_rd_obs / f_merge_Hz replace the Q-based defaults.
        #"RINGDOWN_OVERRIDE": {"f_rd_obs_Hz": 250.0, "tau_rd_obs": 0.004, "f_merge_Hz": 150.0},

        "f_min_obs":         20.0000,                    # Minimum Frequency hz.
        
        "USE_UMH_DFDT_PROFILE":        False,  # UMH Produces good results at 0pn, can be left False.
        # Publication configuration: use the leading-order 0PN-equivalent
        # UMH inspiral evolution reported in the paper.
        # Higher-order GR-limit PN corrections are retained only for diagnostic and extension studies and are not enabled in the published
        # GW150914/GW170814 runs.
        "UMH_DFDT_PN_PROFILE":          0.0,
        "USE_UMH_AMP_PN_PROFILE":      False,   # UMH Produces good results at 0pn, no Amplitude PN Necessary, for Diagnostic only.
        
        "UMH_RedShift_Calibration_File": os.path.join(base, "Output", "UMH_RedShift", "UMH_RedShift_Calibration_Fit.json"),

        # --- UMH frequency-redshift (detector time dilation) ---
        "APPLY_UMH_FREQ_REDSHIFT":      True,   # default ON → Use Src Obs effect to apply Pantheon+ Calibration Freq RedShift and Time Dilation.       
        "APPLY_UMH_AMPLITUDE_SCALING":  True,   # Use UMH Tension Pantheon+ Calibration for Amplitude scaling.

        #All headline comparisons to GW170814 use the analytic UMH/PN track (SOURCE_ENGINE=analytic_umh). 
        #Using soliton_umh, or hybrid_umh SOURCE_ENGINE introduces small, parameter-free fluctuations sourced by the soliton field; 
        #these are presented separately as candidate UMH microstructure, not used to fit the main chirp.
        #This is used to perform true medium wave simulations for upcoming research.
        "SOURCE_ENGINE":      "analytic_umh",   # analytic_umh, soliton_umh, hybrid_umh
    }

    config = {
        # All Settings.
        # Constants.
        "G_phys":        6.67430e-11,                    # Gravitational constant (m^3 kg^-1 s^-2)
        "c_phys":       2.99792458e8,                    # Physical speed of light (m/s)
        "lambda_u":     1.616255e-35,                    # m (Planck-like)
        "M_sun":          1.98847e30,                    # Solar mass (kg)
        "MPC_TO_M":   3.085677581e22,                    # meters in one megaparsec (CODATA 2018)

        "c":                  1.0000,                    # Medium wave speed (sim units)
        "grid_spacing":    1000.0000,                    # 1 grid unit = 1000 meters (tunable)
        "SIZE":                  256,                    # Grid Size, Nx, Ny, Nz = 128, 128, 128
        "REF_SPACING":             2,
        "r0":                     12,                    # Initial separation in grid units
        "scale_factor":      8.25e04,                    # Meters per grid unit


        # Simulation Settings parameters
        "profile": "default",
        "M1_solar_src":      29.0150,                    # Size of M1_solar_src dimensionless, should always be larger or equal to M2_solar_src.
        "M2_solar_src":      23.1730,                    # Size of M2_solar_src dimensionless, should always be smaller or equal to M1_solar_src.
        "distance_Mpc":     410.0000,                    # Distance from Source of event to Detectors.
        "soliton_radius":    40.0000,                    # Initial Radius between M1_solar_src and M2_solar_src.
        "BINARY_IOTA_DEG":     164.0,                    # Binary inclination in degrees (0° = face-on, 90° = edge-on, 180° = face-off).

        # Spin support:
        #   "none"       : nonspinning baseline
        #   "aligned"    : z-spin affects remnant/ringdown and optional 1.5PN df/dt
        #   "precessing" : approximate simple-precession dominant-mode wrapper; not full IMR precession and no higher modes
        "SPIN_MODE": "none",                             # "none", "aligned", "precessing", or "auto", terms enter the PN df/dt coefficients.
        "spin1x": 0.0, "spin1y": 0.0, "spin1z": 0.0,     # Full Spin Support
        "spin2x": 0.0, "spin2y": 0.0, "spin2z": 0.0,
        # Backward-compatible aliases.
        "chi1z": 0.0, "chi2z": 0.0,                      # Aligned Spin
        "REMNANT_APPROXIMANT":      "SEOBNRv4",          # Final-state fit for Mrem/a_rem used by Kerr diagnostic and UMH remnant seed.

        "PRECESSION_STRENGTH":                       1.0,
        "PRECESSION_MAX_BETA_DEG":                  45.0,
        "PRECESSION_MAX_HZ":                        64.0,
        "PRECESSION_FREEZE_AFTER_PEAK":             True,
        "PRECESSION_INCLUDE_THOMAS_PHASE":         False,
        "PRECESSION_INCLUDE_POLARIZATION_ROTATION": True,

        "dt_obs":       1.0 / (4096 * 4),                # Slower time step, improves resolution and chirp scale

        "Sites_Used": {"Hanford", "Livingston", "Virgo"},


        # Simulation Settings parameters
        "AMPLITUDE_SEED":     1.0000,                    #2500.0,

        "ONSET_TAPER_SEC":    0.0015,                    # or similar; ~15 ms fade-in

        #No SMOOTH_FGW necessary, looks clean without.
        "SMOOTH_FGW":          False,                    # Turn on/off
        "TAU_SMOOTH_SEC":     0.0000,                    # 2 ms smoothing time-constant (tune 0.002–0.005) 0.0020
        "MAX_DFDT_HZ_S":        None,                    # Clamp |df/dt|, #1.5e4 or e.g. 2.5e4 to c


        # RingDown Settings parameters
        "RINGDOWN_ENABLE":                             True,      # Used to disable any ringdown.  For diagnostic purposes.
        "USE_UMH_MERGE_RELAXATION_THRESHOLD":          True,
        "RINGDOWN_SEED_MODE":            "umh_remnant_mode",      # "qnm_diagnostic",

        # Active UMH remnant frequency model.
        # Uses the UMH rotating confinement eigenmode rule for F_chi.
        # Does not use empirical Kerr/QNM Omega_220 or Q_220 fit constants.
        "UMH_REMNANT_FREQ_MODEL":          "umh_radial_fit", #"umh_eigen", "umh_radial_operator", "umh_radial_fit", "umh_omega_fit",

        # Active UMH damping model.
        #   "umh_cycles"      = tau = N_cycles / f_rd_src
        "UMH_REMNANT_DAMPING_MODEL":           "umh_cycles", #"umh_radial_operator"

        # Optional nonlinear confinement boundary thickness in units of GM/c^2.
        # Finite-width correction to the UMH rotating confinement surface.
        # Publication configuration: zero boundary softening is used.
        # The unsmoothed confinement/eigenmode already provides the reported detector overlap, so no additional finite-width smoothing is required
        # for the published GW150914/GW170814 comparisons.
        "UMH_EIGEN_BOUNDARY_SOFTENING":                 0.0,

        # Optional radial relaxation / backscatter phase-delay correction.
        # UMH interpretation:
        #   The sharp rotating confinement eigenmode gives the azimuthal m=2 strain-guide frequency. 
        #   A radial strain-gradient/backscatter delay can add a small extra response time before the remnant emits as a clean outgoing transverse mode.
        # This is NOT boundary softening and does not change r_conf. It changes the eigenmode clock:
        #   t_eff = t_phi + t_rad
        #   omega_hat_eff = m / t_eff
        #
        # Use "none" for the pure sharp-confinement UMH eigenmode.
        # Use "fractional_time" to test a global radial-delay fraction:
        #   t_rad = UMH_RADIAL_PHASE_DELAY_FRAC * t_phi

        # Active UMH remnant frequency model.
        #   "umh_eigen"      = direct rotating confinement eigenmode
        #   "umh_omega_fit"  = fitted UMH dimensionless frequency law:
        #                      Omega_UMH(chi) = A + B*(1-chi)^C
        # UMH fitted frequency-law coefficients.
        "UMH_OMEGA_FIT_A":               1.5381969714219732,
        "UMH_OMEGA_FIT_B":              -1.1556090096475626,
        "UMH_OMEGA_FIT_C":              0.14022090728188466,

        # Keep coefficient fitting diagnostic off during normal waveform generation.
        # Turn on only when intentionally regenerating the coefficient table.
        "UMH_REMNANT_CALC_COEFF_DIAGNOSTIC":          False,

        # Radial phase delay remains separate. Default should stay off.
        "UMH_RADIAL_PHASE_MODEL":                    "none",
        "UMH_RADIAL_PHASE_DELAY_FRAC":                  0.0,
        "UMH_RADIAL_PHASE_DELAY_TIME_GM_C3":            0.0,

        # First-principles UMH radial-operator ringdown branch.
        # This solves a radial variable-coefficient outgoing-boundary problem instead of fitting a Kerr/QNM coefficient table.
        # Use the decay rate from the complex radial eigenfrequency.
        # Radial operator mode.
        "UMH_RADIAL_OP_ELL":                              2,
        "UMH_RADIAL_OP_M":                                2,
        # Radial domain in x = r / (GM/c^2). x_inner is automatically tied to the UMH confinement radius unless explicitly set.
        "UMH_RADIAL_OP_X_INNER_OFFSET":               0.050,
        "UMH_RADIAL_OP_X_OUTER":                       30.0,
        "UMH_RADIAL_OP_X_MATCH":                       None,
        # Simple analytic profile controls. These are placeholders for first-principles UMH remnant profiles.
        "UMH_RADIAL_OP_TENSION_AMP":                   1.25,
        "UMH_RADIAL_OP_DENSITY_AMP":                   0.00,
        "UMH_RADIAL_OP_BARRIER_AMP":                   0.15,
        "UMH_RADIAL_OP_BARRIER_WIDTH":                 0.55,
        "UMH_RADIAL_OP_BARRIER_OFFSET":                0.20,
        "UMH_RADIAL_OP_ROTATION_SCALE":                1.00,
        "UMH_RADIAL_OP_ROTATION_POWER":                1.2415,
        # Spin-dependent rotation-power correction. chi_ref is the reference remnant-spin point used by the radial-operator surrogate.
        "UMH_RADIAL_OP_ROTATION_POWER_CHI_ENABLE":     True,
        "UMH_RADIAL_OP_ROTATION_POWER_CHI_REF": 0.6100942439792432,
        "UMH_RADIAL_OP_ROTATION_POWER_CHI_SLOPE":    -1.965,
        # Safety clamp.
        "UMH_RADIAL_OP_ROTATION_POWER_MIN":            0.85,
        "UMH_RADIAL_OP_ROTATION_POWER_MAX":            1.50,

        "UMH_RADIAL_OP_OMEGA_MIN_FACTOR":              0.88,
        "UMH_RADIAL_OP_OMEGA_MAX_FACTOR":              1.02,
        "UMH_RADIAL_OP_GAMMA_MIN_FACTOR":              0.10,
        "UMH_RADIAL_OP_GAMMA_MAX_FACTOR":              3.00,
        # Complex root initial guess. None = use the sharp UMH eigenmode estimate.
        "UMH_RADIAL_OP_OMEGA_GUESS":                   None,
        "UMH_RADIAL_OP_GAMMA_GUESS":                   0.04,
        # Root / integration settings.
        "UMH_RADIAL_OP_RTOL":                        1.0e-8,
        "UMH_RADIAL_OP_ATOL":                       1.0e-10,
        "UMH_RADIAL_OP_RESID_TOL":                   1.0e-5,
        "UMH_RADIAL_OP_MAXFEV":                         300,
        "UMH_RADIAL_OP_TARGET_RATIO":          0.9544249206,
        "UMH_RADIAL_OP_DEBUG":                         True,

        # Fast surrogate for the UMH radial outgoing-boundary operator.
        # Coefficients reproduce operator solutions; they are not fitted to detector strain.
        "UMH_RADIAL_FIT_CHI_REF":         0.61009424397924320,
        "UMH_RADIAL_FIT_DELAY_REF":       0.04780931608484118,
        "UMH_RADIAL_FIT_DELAY_CHI_SLOPE": 0.03586007393884942,

        "UMH_RADIAL_FIT_DELAY_MIN":                    0.00,
        "UMH_RADIAL_FIT_DELAY_MAX":                    0.20,

        # Publication damping model.
        # The post-merger amplitude decays with one UMH coherence cycle per e-fold: tau_rd_src = UMH_RINGDOWN_DAMPING_CYCLES / f_rd_src.
        # The ringdown frequency itself is supplied by the active UMH remnant frequency model above.
        "UMH_RINGDOWN_DAMPING_CYCLES":                 1.00,
        "USE_ISCO_ANCHOR_FOR_FMERGE":                  True,
        "MERGE_FRAC_ISCO":                             1.00,

         #QNM Overtone Diagnostics, current UMH Medium relaxation works better without this, for diagnostic.
        "COMPUTE_QNM_DIAGNOSTIC":     True,
        "USE_QNM_OVERTONE_ATTACH":   False,
        "QNM_OVERTONE_LIST":     [0, 1, 2],
        "QNM_OVERTONE_DECAY_RATIO":    0.3,
        "QNM_OVERTONE_AMPS":          None,

        #No RING_MERGE_C1_BLEND_SEC necessary, uses medium relaxation instead.
        #"RING_MERGE_C1_BLEND_SEC":    0.0015,                 # ms smoothing time-constant (tune 0.002–0.005) 0.0035

        "f_min_obs":               20.0000,                    # Minimum Frequency hz.


        #Using soliton_umh, or hybrid_umh SOURCE_ENGINE introduces small, parameter-free fluctuations sourced by the soliton field; 
        #these are presented separately as candidate UMH microstructure, not used to fit the main chirp.
        #This is used to perform true medium wave simulations for upcoming research.
        "SOURCE_ENGINE":    "analytic_umh",                    # analytic_umh, soliton_umh, hybrid_umh
        # analytic_umh - UMH Reduced GR Limit
        #    UMH reduced/macroscopic branch.
        #    GR-like chirp behavior appears as the emergent limit of UMH.
        #    Fast, stable, good for LIGO event comparisons.
        #    Best for showing UMH reproduces GR-limit behavior while adding UMH propagation and medium relaxation.
        # soliton_umh
        #    UMH wave-field branch.
        #    Nonlinear medium/soliton source dynamics are explicitly evolved.
        #    Slower, more detailed.
        #    Best for studying beyond-GR-limit wave structure, noise-like microstructure, source asymmetry, nonlinear overlap, and medium effects.
        # hybrid_umh
        #    Analytic UMH track supplies the stable macroscopic carrier.
        #    Soliton branch supplies physically generated modulation, mode corrections, spin/vorticity corrections, or remnant relaxation detail.

        "USE_SOLITON_FOR_ENVELOPE":             False,         # Soliton field modulates the analytic UMH amplitude envelope.
        "USE_SOLITON_FOR_SOURCE_MODES":         False,         # Soliton field generates or corrects h_lm mode content.
        "USE_SOLITON_FOR_SPIN_EVOLUTION":       False,         # Soliton vorticity/torque evolves S1, S2, and L.
        "USE_SOLITON_FOR_REMPROFILE":           False,         # Post-merger soliton/medium relaxation determines remnant frequency and damping.
        "USE_SOLITON_FOR_NOISE_MICROSTRUCTURE": False,         # Extract medium-level fine structure that may appear as non-GR waveform texture or noise-like detail.

        "damping_factor":           0.9999,                    #0.9999,
        "freq_damping":             0.0050,                    # Tweak between 0.0001 and 0.01 to suppress high-frequency noise, #0.0050.


        #Base location, created by UMH Redshift test for Calibration of Frequency, Phase, Time Dilation, and Amplitude.
        "UMH_RedShift_Calibration_File": os.path.join(base, "Output", "UMH_RedShift", "UMH_RedShift_Calibration_Fit.json"),
        
        # --- UMH frequency-redshift (detector time dilation) ---
        # APPLY_UMH_FREQ_REDSHIFT:
        #   - If True, we treat UMH_z_tension as a cosmological / medium redshift.
        #   - All *intrinsic* PN quantities (M_chirp_src, f_min_src, f_merge_src, f_rd_src)
        #     are defined in the source frame; we compute *_obs by dividing by z_factor
        #     once at setup. We then evolve in the observer frame.
        "APPLY_UMH_FREQ_REDSHIFT":      True,   # default ON → Use Src Obs effect to apply Pantheon+ Calibration Freq RedShift and Time Dilation.


        # Normalization Settings parameters
        "PHYSICS_NORM_ENABLE":          True,
        
        # APPLY_UMH_AMPLITUDE_SCALING:
        #   - Optional *amplitude* use of UMH_z_tension.
        #   - When True, we fold the same z_factor = (1 + UMH_z_tension) that would normally stretch the time/frequency track into the overall
        #     strain normalization.
        #   - This lets us test the impact of UMH tension using just amplitude rescaling, without changing the PN/UMH inspiral timing.
        #     In other words, it is an approximate, phenomenological way to encode UMH_z_tension when APPLY_UMH_FREQ_REDSHIFT is disabled.
        #   - If both APPLY_UMH_FREQ_REDSHIFT and APPLY_UMH_AMPLITUDE_SCALING are True, the redshift acts in *both* the timing (frequency track)
        #     and the amplitude normalization.
        "APPLY_UMH_AMPLITUDE_SCALING":  True,   # Use UMH Tension Pantheon+ Calibration for Amplitude scaling.

       
        "USE_PSD_FREF":                False,   #Not used, as this is a diagnostic concept of importing PSD from Compiler.  Do not use.


        # Visual Only parameters
        "ADD_VISUAL_NOISE":    False,
        "VISUAL_SCALE_MULT":  1.0000,


        # --- Internal ---
        "DPI":                   300,                     #PNG Resolution.

        "DTYPE":          np.float64,                     #Precision of Types.

        "PSD_MAP_INPUT_FILENAME": "UMH_Ligo_Compiler_CMP_PSD_Map.npz",
        "PSD_MAP_INPUT_FOLDER": os.path.join(base, "Output", "UMH_vs_LIGO"),
        "OUTPUT_FOLDER":        os.path.join(base, "Output")
    }

    overrides = dict(config_overrides or {})
    requested_profile = overrides.get("profile", config["profile"])

    # If replica requested, layer it on top of base defaults first
    if requested_profile == "replica_gw150914": config.update(replica)
    if requested_profile == "replica_gw170814": config.update(replica_gw170814)

    # Apply all user overrides next (so user can refine replica/default)
    # If RINGDOWN_OVERRIDE is set, its f_rd_obs_Hz / tau_rd_obs / f_merge_Hz replace the Q-based defaults. Effective values are written to metadata.
    #    Deep-merge RINGDOWN_OVERRIDE if both sides have it
    if "RINGDOWN_OVERRIDE" in overrides and isinstance(overrides["RINGDOWN_OVERRIDE"], dict):
        base_rd = dict(config.get("RINGDOWN_OVERRIDE", {}))
        base_rd.update(overrides["RINGDOWN_OVERRIDE"])
        config["RINGDOWN_OVERRIDE"] = base_rd
        # remove so it isn't applied again by config.update below
        overrides = {k: v for k, v in overrides.items() if k != "RINGDOWN_OVERRIDE"}

    config.update(overrides)

    # Backward-compatible aliases: If the user explicitly supplied chi1z/chi2z but did not explicitly supply spin1z/spin2z, copy the aliases into the full spin-vector fields.
    if "chi1z" in overrides and "spin1z" not in overrides: config["spin1z"] = float(config["chi1z"])
    if "chi2z" in overrides and "spin2z" not in overrides: config["spin2z"] = float(config["chi2z"])

    # Stamp the final profile field for provenance
    config["profile"] = requested_profile
    print(f"profile:{requested_profile}")

    return config, base


SITES = {
    "Hanford": {
        "lat_deg":   46.455146622500000, # 46°27′18.528″ N
        "lon_deg": -119.407657133600000, # 119°24′27.566″ W
        "h_m":      142.555000000000000, # WGS-84 ellipsoidal height (vertex)
        "az_x_deg": 324.000700000000000, # N35.9993W (true/geodetic north)
        "az_y_deg": 234.000700000000000, # S54.0007W (true/geodetic north)
    },
    "Livingston": {
        "lat_deg":   30.562894314200000, # 30°33′46.420″ N
        "lon_deg":  -90.774240359400000, # 90°46′27.265″ W
        "h_m":       -6.574000000000000, # WGS-84 ellipsoidal height (vertex)
        "az_x_deg": 252.283600000000000, # S72.2836W (true/geodetic north)
        "az_y_deg": 162.283600000000000, # S17.7164E (true/geodetic north)
    },
    "Virgo": {
        "lat_deg":   43.631418361111110, # 43°37′53.1061″ N
        "lon_deg":   10.504502638888889, # 10°30′16.2095″ E
        "h_m":       53.089000000000000, # Ellipsoidal height at BS (ETRF)
        "az_x_deg":  19.432979590000000, # N19.43298E, BS→NE arm (true/geographic north)
        "az_y_deg": 289.432936660000000, # N70.56706W, BS→WE arm (true/geographic north)
    }
}

#| Site                   | Latitude             | Longitude             | Elevation / Height Approx.  |
#| ---------------------- | -------------------- | --------------------- | --------------------------- |
#| Hanford (LHO)          | **46°27′18.528″  N** | **119°24′27.566″  W** | ~ 142.555 m above sea level |
#| Livingston (LLO)       | **30°33′46.420″  N** | ** 90°46′27.265″  W** | ~  -6.574 m below sea level |
#| Virgo (EGO, near Pisa) | **43°37′53.1061″ N** | ** 10°30′16.2095″ E** | ~  53.089 m above sea level |


EPS_FLOOR      = 1e-40
EPS_SAFE_FLOOR = 1e-24
EPS_SPIN       = 1e-14


# Function to ensure proper RA, DEC values are provided.
# Canonicalize sky coordinates: wrap RA to [0°,360°) and fold DEC into [-90°,90°],
# applying a 180° RA shift when crossing a pole to preserve the same physical sky direction.
def canonicalize_radec(ra_deg, dec_deg):
    # Wrap RA into [0, 360)
    ra = ra_deg % 360.0; dec = dec_deg
    # Fold dec into [-90, +90] with RA shift if needed
    if dec > 90.0: dec = 180.0 - dec; ra = (ra + 180.0) % 360.0
    elif dec < -90.0: dec = -180.0 - dec; ra = (ra + 180.0) % 360.0
    return ra, dec

# --------------------------------------------------------------------
# UMH-native higher-order df/dt correction coefficients (GR-limit form)
# --------------------------------------------------------------------
# All coefficients are dimensionless and can be replaced or constrained by UMH theory once medium-based corrections are derived.
# When these match the GR 3.5 PN series, df/dt reproduces the standard
# TaylorT1/T4 inspiral rate for non-spinning binaries.
def get_gr_dfdt_coeffs(M1_kg_src, M2_kg_src, f_ref_src, eta=None, spin1x=0.0, spin1y=0.0, spin1z=0.0, spin2x=0.0, spin2y=0.0, spin2z=0.0, 
                       SPIN_MODE="None", G_phys=6.67430e-11, c_phys=2.99792458e8, UMH_DFDT_PN_PROFILE=3.5):
    """
    GR-like post-Newtonian correction coefficients {C2..C7}
    for the UMH df/dt law, evaluated at a reference frequency f_ref_src.

    These are dimensionless numbers used in:

        df/dt = K_N * f^(11/3) * [1
                                  + C2 * u**2   # 1PN
                                  + C3 * u**3   # 1.5PN
                                  + C4 * u**4   # 2PN
                                  + C5 * u**5   # 2.5PN
                                  + C6 * u**6   # 3PN
                                  + C7 * u**7]  # 3.5PN

    where u = (f / f_ref_src)**(1/3) is proportional to v / v_ref and
    K_N is the leading-order (0PN) Newtonian/UMH coefficient from energy balance.
    """
    C2,C3,C4,C5,C6,C7=0.0,0.0,0.0,0.0,0.0,0.0

    # --- binary parameters ---
    M_tot_kg_src = M1_kg_src + M2_kg_src
    if(eta is None): eta = (M1_kg_src * M2_kg_src) / (M_tot_kg_src * M_tot_kg_src)
    v_ref = (math.pi * G_phys * M_tot_kg_src * f_ref_src / c_phys**3)**(1/3)

    X1 = M1_kg_src / M_tot_kg_src; X2 = M2_kg_src / M_tot_kg_src
    beta_SO = 0.0
    
    SPIN_MODE = str(SPIN_MODE).lower()
    if SPIN_MODE in ("aligned", "precessing"):
        spin1x=0.0; spin1y=0.0; spin2x=0.0; spin2y=0.0
        # Leading aligned-spin spin-orbit parameter. Enters the 1.5PN df/dt correction as 4*pi - beta_SO.
        beta_SO = ((1.0 / 12.0) * ((113.0 * X1 * X1 + 75.0 * eta) * float(spin1z) + (113.0 * X2 * X2 + 75.0 * eta) * float(spin2z)))
    else: spin1x=0.0; spin1y=0.0; spin1z=0.0; spin2x=0.0; spin2y=0.0; spin2z=0.0
    
    # --- 1 PN–2.5 PN terms (standard) ---
    if(UMH_DFDT_PN_PROFILE>=1.0): C2 = v_ref**2 * (-743.0/336.0 - 11.0/4.0 * eta)                                 # 1 PN
    #if(UMH_DFDT_PN_PROFILE>=1.5): C3 = v_ref**3 * (4.0 * math.pi)                                                # 1.5 PN
    if(UMH_DFDT_PN_PROFILE>=1.5): C3 = v_ref**3 * (4.0 * math.pi - beta_SO)                                       # 1.5 PN
    if(UMH_DFDT_PN_PROFILE>=2.0): C4 = v_ref**4 * (34103.0/18144.0 + 13661.0/2016.0 * eta + 59.0/18.0 * eta**2)   # 2 PN
    if(UMH_DFDT_PN_PROFILE>=2.5): C5 = v_ref**5 * (-4159.0/672.0 - 189.0/8.0 * eta) * math.pi                     # 2.5 PN

    # --- 3 PN term (includes Euler gamma + log terms, absorbed at f_ref_src) ---
    if(UMH_DFDT_PN_PROFILE>=3.0):
        gamma_E = 0.5772156649015328606
        C6 = v_ref**6 * (16447322263.0/139708800.0 - 1712.0/105.0 * (gamma_E + math.log(4.0 * v_ref))
            + (-56198689.0/217728.0 + 451.0/48.0 * math.pi**2) * eta + 541.0/896.0 * eta**2 - 5605.0/2592.0 * eta**3)

    # --- 3.5 PN term ---
    if(UMH_DFDT_PN_PROFILE>=3.5):
        C7 = v_ref**7 * math.pi * (77096675.0/254016.0 + 378515.0/1512.0 * eta - 74045.0/756.0 * eta**2)

    return dict(C2=C2, C3=C3, C4=C4, C5=C5, C6=C6, C7=C7), dict(beta_SO=beta_SO, spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, 
                spin2x=spin2x, spin2y=spin2y, spin2z=spin2z, SPIN_MODE=SPIN_MODE, 
                scalar_dfdt_spin_note=("Scalar df/dt uses only spin components projected onto the initial orbital angular momentum. "
                                       "Transverse spin enters through the precession polarization wrapper." if SPIN_MODE == "precessing" else ""))


# UMH_DFDT_COEFFS:
# Dictionary of dimensionless correction coefficients that modify the
# Newtonian (0PN) frequency-evolution law df/dt = K_N * f**(11/3).
#
#   C2 → 1PN (v^2 term)
#       Encodes relativistic corrections to orbital binding energy and gravitational-wave flux at order (v/c)^2.
#       In GR limit: C2 = v_ref^2 * ( -743/336 - 11/4 * η )
#
#   C3 → 1.5PN (v^3 term)
#       Represents the leading "tail" term, i.e., back-scattering of waves off the curved spacetime geometry.
#       In GR limit: C3 = v_ref^3 * ( 4π )
#
#   C4 → 2PN (v^4 term)
#       Higher-order relativistic corrections to energy flux and conservative dynamics; depends quadratically on η.
#       In GR limit: C4 = v_ref^4 * ( 34103/18144 + 13661/2016*η + 59/18*η**2 )
#
#   C5 → 2.5PN (v^5 term)
#       Next-order tail correction; purely dissipative term entering the phase evolution.  In GR limit:
#       C5 = v_ref^5 * ( -4159/672 - 189/8 * η ) * π
#
# In full UMH theory, each coefficient can incorporate medium-dependent corrections (e.g., tension-redshift, dispersion, or field anisotropy),
# making them physically interpretable rather than phenomenological.


def build_polarizations_nonprecessing(A_obs, phase_obs, iota_rad):
    cosi = np.cos(iota_rad)
    h_plus = A_obs * 0.5 * (1.0 + cosi*cosi) * np.cos(phase_obs)
    h_cross = A_obs * cosi * np.sin(phase_obs)
    return h_plus, h_cross


def _umh_unit_vector(v, fallback=(0.0, 0.0, 1.0)):
    # Safe unit-vector helper.
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if (not np.isfinite(n)) or n < EPS_SAFE_FLOOR: return np.asarray(fallback, dtype=float)
    return v / n


def _umh_cumtrapz(y, x):
    """
    Small local cumulative trapezoid integrator to avoid adding scipy.integrate dependency.
    Returns out[0]=0 and out[i]=integral x[0]..x[i] y dx.
    """
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if y.size < 2: return out
    dx = np.diff(x)
    good = np.isfinite(dx)
    dx = np.where(good, dx, 0.0)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def _umh_basis_from_axis(axis, preferred=None):
    # Build an orthonormal basis e1,e2 perpendicular to axis.
    k = _umh_unit_vector(axis)
    if preferred is None:
        preferred = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(preferred, k))) > 0.95: preferred = np.array([1.0, 0.0, 0.0], dtype=float)
    else: preferred = np.asarray(preferred, dtype=float)
    e1 = preferred - float(np.dot(preferred, k)) * k
    e1 = _umh_unit_vector(e1, fallback=(1.0, 0.0, 0.0))
    # If fallback accidentally aligned with k, use y.
    if abs(float(np.dot(e1, k))) > 0.95:
        e1 = np.array([0.0, 1.0, 0.0], dtype=float)
        e1 = e1 - float(np.dot(e1, k)) * k
        e1 = _umh_unit_vector(e1, fallback=(1.0, 0.0, 0.0))
    e2 = np.cross(k, e1)
    e2 = _umh_unit_vector(e2, fallback=(0.0, 1.0, 0.0))
    return e1, e2


def evolve_precession_angles(t_obs, dt_obs, M1_solar_src, M2_solar_src, z_factor, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                             iota_rad, phase_obs=None, f_obs=None, f_min_obs=20.0, t_peak_target=None,
                             PRECESSION_STRENGTH=1.0, PRECESSION_MAX_BETA_DEG=45.0, PRECESSION_MAX_HZ=64.0, PRECESSION_FREEZE_AFTER_PEAK=True,
                             G_phys=6.67430e-11, c_phys=2.99792458e8, M_sun=1.98847e30):
    """
    Approximate simple-precession evolution for the UMH analytic waveform.
    -------
    This is a conservative first-pass precession wrapper:
      - It does not replace the intrinsic UMH/PN phase track.
      - It evolves a simple precessing orbital angular-momentum direction L_hat(t).
      - It computes time-dependent inclination iota(t) and polarization rotation psi(t).
      - It is suitable for diagnostics and controlled spin sweeps.
    Important limitation:
        This is not a full IMRPhenomPv2/SEOBNR precessing model and does not include  higher modes. 
        It is a "twisting-up" style dominant-mode approximation.
    Coordinate convention
    Initial source frame:
        z-axis = initial orbital angular momentum L0.
        x/y spin components are transverse to L0.
        observer line of sight N lies in the x-z plane with angle iota_rad from L0.
    """
    t = np.asarray(t_obs, dtype=float)
    n = t.size
    if n < 1: raise ValueError("evolve_precession_angles: t_obs is empty.")
    zfac = float(z_factor)
    if (not math.isfinite(zfac)) or zfac <= 0.0: zfac = 1.0
    m1 = float(M1_solar_src); m2 = float(M2_solar_src)
    if m1 <= 0.0 or m2 <= 0.0: raise ValueError("evolve_precession_angles: source masses must be positive.")
    Mtot_solar = m1 + m2
    Mtot_kg = Mtot_solar * M_sun
    X1 = m1 / Mtot_solar; X2 = m2 / Mtot_solar
    eta = (m1 * m2) / (Mtot_solar * Mtot_solar)
    # Frequency track in observer frame.
    if f_obs is None:
        if phase_obs is not None: f_obs = ensure_f_hist_from_phase(phase_obs, dt_obs, dtype=float)
        else: f_obs = np.full(n, float(f_min_obs), dtype=float) # Emergency fallback only. Prefer phase-derived f_obs.
    else: f_obs = np.asarray(f_obs, dtype=float)
    if f_obs.size != n:
        if f_obs.size < n:
            pad_val = float(f_obs[-1]) if f_obs.size else float(f_min_obs)
            f_obs = np.pad(f_obs, (0, n - f_obs.size), constant_values=pad_val)
        else: f_obs = f_obs[:n]
    f_obs = np.where(np.isfinite(f_obs), f_obs, float(f_min_obs))
    f_obs = np.maximum(f_obs, max(float(f_min_obs), 1.0e-6))
    # Convert to source-frame frequency for PN velocity estimate.
    f_src = f_obs * zfac
    v = (np.pi * G_phys * Mtot_kg * f_src / (c_phys**3)) ** (1.0 / 3.0)
    v = np.where(np.isfinite(v), v, 1.0e-4); v = np.clip(v, 1.0e-4, 0.95)
    # Dimensionless angular momenta scaled by M^2:
    #   L/M^2 ~ eta / v
    #   S_i/M^2 ~ X_i^2 chi_i
    Lmag = eta / np.maximum(v, 1.0e-6)
    S1_vec = (X1 * X1) * np.array([spin1x, spin1y, spin1z], dtype=float)
    S2_vec = (X2 * X2) * np.array([spin2x, spin2y, spin2z], dtype=float)
    S_vec = S1_vec + S2_vec
    L0_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    # Transverse spin drives precession.
    S_perp_vec = S_vec - float(np.dot(S_vec, L0_hat)) * L0_hat
    S_perp = float(np.linalg.norm(S_perp_vec))
    S_parallel = float(np.dot(S_vec, L0_hat))
    spin_has_precession = S_perp > EPS_SPIN
    # Observer line of sight. Initial inclination is angle between N and L0.
    sini = math.sin(float(iota_rad))
    cosi = math.cos(float(iota_rad))
    N_hat = _umh_unit_vector(np.array([sini, 0.0, cosi], dtype=float), fallback=(0.0, 0.0, 1.0))
    # Sky basis. p_hat is the projection of initial L0 into the sky plane, so nonprecessing beta=0 gives psi=0.
    p_hat = L0_hat - float(np.dot(L0_hat, N_hat)) * N_hat
    p_hat = _umh_unit_vector(p_hat, fallback=(1.0, 0.0, 0.0))
    q_hat = np.cross(N_hat, p_hat)
    q_hat = _umh_unit_vector(q_hat, fallback=(0.0, 1.0, 0.0))
    if not spin_has_precession:
        Lhat = np.tile(L0_hat, (n, 1))
        alpha = np.zeros(n, dtype=float)
        beta = np.zeros(n, dtype=float)
        gamma = np.zeros(n, dtype=float)
        psi = np.zeros(n, dtype=float)
        iota_eff = np.full(n, float(iota_rad), dtype=float)
        omega_p_obs = np.zeros(n, dtype=float)
        return {"alpha": alpha, "beta": beta, "gamma": gamma, "psi": psi, "iota": iota_eff, "Lhat": Lhat, "Nhat": N_hat,
            "Omega_p_obs": omega_p_obs, "f_obs": f_obs, "simple_precession_note": "No transverse spin; precession angles are zero."}
    # Approximate fixed total-angular-momentum direction, initialized from early inspiral.
    Lref = float(Lmag[0]) if Lmag.size else eta / max((np.pi * G_phys * Mtot_kg * max(f_min_obs, 1.0) * zfac / c_phys**3) ** (1.0 / 3.0), 1.0e-6)
    J0_vec = Lref * L0_hat + S_vec
    J_hat = _umh_unit_vector(J0_vec, fallback=L0_hat)
    # Basis around J. Choose e1 so alpha=0 puts L as close as possible to initial L0.
    e1_pref = L0_hat - float(np.dot(L0_hat, J_hat)) * J_hat
    e1, e2 = _umh_basis_from_axis(J_hat, preferred=e1_pref)
    # Cone opening angle grows as L shrinks.
    # Use denominator L + S_parallel to avoid over-tilting when spin is mostly aligned.
    denom = np.maximum(np.abs(Lmag + S_parallel), 1.0e-6)
    beta = np.arctan2(S_perp, denom)
    beta_max = math.radians(float(PRECESSION_MAX_BETA_DEG))
    if not math.isfinite(beta_max) or beta_max <= 0.0: beta_max = math.radians(45.0)
    beta = np.clip(beta, 0.0, beta_max)
    # Leading-order-inspired simple precession frequency.
    # This is intentionally conservative and diagnostic.
    c1 = 2.0 + 1.5 * (m2 / m1); c2 = 2.0 + 1.5 * (m1 / m2)
    S_eff_vec = c1 * S1_vec + c2 * S2_vec
    S_eff_perp_vec = S_eff_vec - float(np.dot(S_eff_vec, L0_hat)) * L0_hat
    S_eff_perp = float(np.linalg.norm(S_eff_perp_vec))
    GM_over_c3 = G_phys * Mtot_kg / (c_phys**3)
    inv_M_sec = 1.0 / max(GM_over_c3, 1.0e-30)
    omega_p_src = float(PRECESSION_STRENGTH) * inv_M_sec * (v**5) * (S_eff_perp / np.maximum(Lmag, 1.0e-6))
    omega_p_obs = omega_p_src / zfac
    omega_p_obs = np.where(np.isfinite(omega_p_obs), omega_p_obs, 0.0)
    omega_max = 2.0 * np.pi * float(PRECESSION_MAX_HZ)
    if math.isfinite(omega_max) and omega_max > 0.0: omega_p_obs = np.clip(omega_p_obs, 0.0, omega_max)
    # Freeze after merger/ringdown attachment to avoid unphysical post-merger plane whipping.
    if PRECESSION_FREEZE_AFTER_PEAK and t_peak_target is not None and math.isfinite(float(t_peak_target)):
        idx_peak = int(np.searchsorted(t, float(t_peak_target))); idx_peak = max(0, min(idx_peak, n - 1))
        omega_p_obs[idx_peak:] = 0.0
        beta[idx_peak:] = beta[idx_peak]
    alpha = _umh_cumtrapz(omega_p_obs, t)
    # Minimal-rotation / Thomas-precession-like angle.
    # Keep available, but the waveform builder can choose whether to apply it.
    gamma_dot = -np.cos(beta) * omega_p_obs
    gamma = _umh_cumtrapz(gamma_dot, t)
    # L_hat(t) precessing around J_hat.
    ca = np.cos(alpha); sa = np.sin(alpha); cb = np.cos(beta); sb = np.sin(beta)
    Lhat = (cb[:, None] * J_hat[None, :] + sb[:, None] * (ca[:, None] * e1[None, :] + sa[:, None] * e2[None, :]))
    # Normalize row-wise for numerical safety.
    Lnorm = np.linalg.norm(Lhat, axis=1)
    bad = (~np.isfinite(Lnorm)) | (Lnorm < EPS_SAFE_FLOOR)
    Lnorm = np.where(bad, 1.0, Lnorm)
    Lhat = Lhat / Lnorm[:, None]
    if np.any(bad): Lhat[bad, :] = L0_hat
    # Effective inclination relative to the fixed line of sight.
    cos_i_eff = np.einsum("ij,j->i", Lhat, N_hat)
    cos_i_eff = np.clip(cos_i_eff, -1.0, 1.0)
    iota_eff = np.arccos(cos_i_eff)
    # Time-dependent polarization angle from projection of L onto the sky.
    Lproj = Lhat - cos_i_eff[:, None] * N_hat[None, :]
    px = np.einsum("ij,j->i", Lproj, p_hat)
    qx = np.einsum("ij,j->i", Lproj, q_hat)
    psi = np.arctan2(qx, px); psi = np.unwrap(psi); psi = np.where(np.isfinite(psi), psi, 0.0)
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "psi": psi, "iota": iota_eff, "Lhat": Lhat, "Jhat": J_hat, "Nhat": N_hat,
            "Omega_p_obs": omega_p_obs, "f_obs": f_obs, "S_vec": S_vec, "S_perp": S_perp,
            "simple_precession_note": ("Approximate dominant-mode simple-precession wrapper. "
                                       "Not a full IMR precessing waveform and does not include higher modes.")}


def rotate_modes_to_inertial(A_obs, phase_obs, precession, INCLUDE_THOMAS_PHASE=False, INCLUDE_POLARIZATION_ROTATION=True):
    """
    Rotate the dominant co-precessing quadrupole-like waveform into the inertial observer basis.
    Parameters
        A_obs, phase_obs: Existing UMH observer-frame amplitude and phase arrays.
        precession: Dict returned by evolve_precession_angles().
        INCLUDE_THOMAS_PHASE:
            If True, applies phase_obs -> phase_obs + 2*gamma.
            Default False preserves the current UMH phase track more strictly.
        INCLUDE_POLARIZATION_ROTATION: If True, rotates plus/cross by the time-dependent precession polarization angle psi(t).
    Returns: h_plus_obs, h_cross_obs
    """
    A = np.asarray(A_obs, dtype=float)
    phase = np.unwrap(np.asarray(phase_obs, dtype=float))
    if A.size != phase.size: raise ValueError(f"rotate_modes_to_inertial: A_obs and phase_obs size mismatch: {A.size} vs {phase.size}")
    n = A.size
    iota_eff = np.asarray(precession.get("iota"), dtype=float)
    psi = np.asarray(precession.get("psi", np.zeros(n)), dtype=float)
    gamma = np.asarray(precession.get("gamma", np.zeros(n)), dtype=float)
    for name, arr in (("iota", iota_eff), ("psi", psi), ("gamma", gamma)):
        if arr.size != n: raise ValueError(f"rotate_modes_to_inertial: precession['{name}'] length {arr.size} != waveform length {n}")
    phase_eff = phase.copy()
    if INCLUDE_THOMAS_PHASE: phase_eff = phase_eff + 2.0 * gamma
    # Dominant-mode co-precessing polarizations, but with time-dependent inclination.
    cosi = np.cos(iota_eff)
    h_plus_cp = A * 0.5 * (1.0 + cosi * cosi) * np.cos(phase_eff)
    h_cross_cp = A * cosi * np.sin(phase_eff)
    if not INCLUDE_POLARIZATION_ROTATION: return h_plus_cp, h_cross_cp
    # Spin-2 polarization-basis rotation.
    c2p = np.cos(2.0 * psi); s2p = np.sin(2.0 * psi)
    h_plus = c2p * h_plus_cp - s2p * h_cross_cp
    h_cross = s2p * h_plus_cp + c2p * h_cross_cp
    h_plus = np.where(np.isfinite(h_plus), h_plus, 0.0)
    h_cross = np.where(np.isfinite(h_cross), h_cross, 0.0)
    return h_plus, h_cross


def build_polarizations_precessing(t_obs, dt_obs, A_obs=None, phase_obs=None, M1_solar_src=30.0, M2_solar_src=30.0, z_factor=1.0,
                                   spin1x=0.0, spin1y=0.0, spin1z=0.0, spin2x=0.0, spin2y=0.0, spin2z=0.0, iota_rad=0.0,
                                   D_L_umh_Mpc=None, f_min_obs=20.0, t_peak_target=None,
                                   PRECESSION_STRENGTH=1.0, PRECESSION_MAX_BETA_DEG=45.0, PRECESSION_MAX_HZ=64.0,
                                   PRECESSION_FREEZE_AFTER_PEAK=True, PRECESSION_INCLUDE_THOMAS_PHASE=False,
                                   PRECESSION_INCLUDE_POLARIZATION_ROTATION=True, return_meta=False,
                                   G_phys=6.67430e-11, c_phys=2.99792458e8, M_sun=1.98847e30):
    """
    Build observer-frame plus/cross polarizations for an approximate precessing-spin UMH waveform.
    This function intentionally reuses the already-generated UMH amplitude and phase track.
    It does not regenerate the chirp. It only twists/rotates the dominant-mode waveform using a simple-precession approximation.
    Required: A_obs, phase_obs
    D_L_umh_Mpc is accepted for signature compatibility/provenance but is not used here, because amplitude normalization 
    has already been applied before this function is called.
    """
    if A_obs is None or phase_obs is None:
        raise ValueError("build_polarizations_precessing requires A_obs and phase_obs. "
                         "Pass the already-built A_obs and phase_obs from run_chirp_generator_test.")
    A = np.asarray(A_obs, dtype=float)
    phase = np.unwrap(np.asarray(phase_obs, dtype=float))
    t = np.asarray(t_obs, dtype=float)
    if A.size != phase.size: raise ValueError(f"build_polarizations_precessing: A_obs and phase_obs size mismatch: {A.size} vs {phase.size}")
    if t.size != A.size: raise ValueError(f"build_polarizations_precessing: t_obs length {t.size} != waveform length {A.size}")
    # Derive f(t) from the same phase used to build the waveform.
    f_obs = ensure_f_hist_from_phase(phase, dt_obs, dtype=float)
    f_obs = np.maximum(f_obs, max(float(f_min_obs), 1.0e-6))
    precession = evolve_precession_angles(t_obs=t, dt_obs=dt_obs, M1_solar_src=M1_solar_src, M2_solar_src=M2_solar_src, z_factor=z_factor,
                                          spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                                          iota_rad=iota_rad, phase_obs=phase, f_obs=f_obs, f_min_obs=f_min_obs, t_peak_target=t_peak_target,
                                          PRECESSION_STRENGTH=PRECESSION_STRENGTH, PRECESSION_MAX_BETA_DEG=PRECESSION_MAX_BETA_DEG,
                                          PRECESSION_MAX_HZ=PRECESSION_MAX_HZ, PRECESSION_FREEZE_AFTER_PEAK=PRECESSION_FREEZE_AFTER_PEAK,
                                          G_phys=G_phys, c_phys=c_phys, M_sun=M_sun)
    h_plus_obs, h_cross_obs = rotate_modes_to_inertial(A_obs=A, phase_obs=phase, precession=precession,
                                                       INCLUDE_THOMAS_PHASE=PRECESSION_INCLUDE_THOMAS_PHASE, 
                                                       INCLUDE_POLARIZATION_ROTATION=PRECESSION_INCLUDE_POLARIZATION_ROTATION)
    if return_meta:
        meta = {"precession_model": "UMH_simple_precession_dominant_mode", "precession_note": precession.get("simple_precession_note", ""),
                "PRECESSION_STRENGTH": float(PRECESSION_STRENGTH), "PRECESSION_MAX_BETA_DEG": float(PRECESSION_MAX_BETA_DEG),
                "PRECESSION_MAX_HZ": float(PRECESSION_MAX_HZ), "PRECESSION_FREEZE_AFTER_PEAK": bool(PRECESSION_FREEZE_AFTER_PEAK),
                "PRECESSION_INCLUDE_THOMAS_PHASE": bool(PRECESSION_INCLUDE_THOMAS_PHASE),
                "PRECESSION_INCLUDE_POLARIZATION_ROTATION": bool(PRECESSION_INCLUDE_POLARIZATION_ROTATION),
                "precession_beta_max_rad": float(np.max(precession["beta"])) if precession["beta"].size else 0.0,
                "precession_beta_max_deg": float(np.degrees(np.max(precession["beta"]))) if precession["beta"].size else 0.0,
                "precession_alpha_final_rad": float(precession["alpha"][-1]) if precession["alpha"].size else 0.0,
                "precession_omega_p_max_rad_s": float(np.max(precession["Omega_p_obs"])) if precession["Omega_p_obs"].size else 0.0,
                "precession_S_perp": float(precession.get("S_perp", 0.0))}
        return h_plus_obs, h_cross_obs, meta
    return h_plus_obs, h_cross_obs


# UMH soliton constructor used for the illustrative ultronic medium field.
# This feeds A_raw (diagnostic amplitude) only; final detector strains do NOT sample phi(x,y,z,t) directly.
@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def gaussian_soliton(phi, center, radius, amplitude):
    cx, cy, cz = center
    Nx, Ny, Nz = phi.shape
    r_int = int(radius)
    r3 = 3 * r_int          # 3 for effective Gaussian support

    x_min = max(cx - r3, 0)
    x_max = min(cx + r3, Nx)
    y_min = max(cy - r3, 0)
    y_max = min(cy + r3, Ny)
    z_min = max(cz - r3, 0)
    z_max = min(cz + r3, Nz)

    for x in prange(x_min, x_max):
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                r2 = dx * dx + dy * dy + dz * dz
                phi[x, y, z] += amplitude * np.exp(-r2 / (2.0 * radius * radius))
    

@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def update_field(Nx,Ny,Nz,phi, phi_prev, phi_next, c, dt_src, damping_factor, freq_damping):
    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                lap = (
                    phi[i+1, j, k] + phi[i-1, j, k] +
                    phi[i, j+1, k] + phi[i, j-1, k] +
                    phi[i, j, k+1] + phi[i, j, k-1] -
                    6.0 * phi[i, j, k]
                )
                phi_next[i, j, k] = damping_factor * (
                    2 * phi[i, j, k] - phi_prev[i, j, k] + (c**2 * dt_src**2) * lap
                    - freq_damping * (phi[i, j, k] - phi_prev[i, j, k]))

def apply_pml(phi_next, PML_N):
    if PML_N > 0:
        # x-faces
        phi_next[:PML_N, :, :]      = 0.0
        phi_next[-PML_N:, :, :]     = 0.0
        # y-faces
        phi_next[:, :PML_N, :]      = 0.0
        phi_next[:, -PML_N:, :]     = 0.0
        # z-faces
        phi_next[:, :, :PML_N]      = 0.0
        phi_next[:, :, -PML_N:]     = 0.0


# Conceptual UMH "Michelson" probe in the soliton field.
# Used to build A_raw and, after smoothing, A_hist (envelope). This probe is NOT
# the actual detector response used in comparisons.
@njit(cache=True, parallel=False, fastmath=False, nogil=True)
def measure_strain(phi, center, orientation, spacing=1):
    # orientation is a pair like ("x","y"), ("x","z"), etc.
    ax, ay = orientation
    axis = {"x": 0, "y": 1, "z": 2}
    Nx, Ny, Nz = phi.shape

    base = np.array(center, dtype=np.int64)

    # (+/−) along first arm
    p1 = base.copy(); p1[axis[ax]] = min(max(base[axis[ax]] + spacing, 1), [Nx,Ny,Nz][axis[ax]] - 2)
    m1 = base.copy(); m1[axis[ax]] = min(max(base[axis[ax]] - spacing, 1), [Nx,Ny,Nz][axis[ax]] - 2)
    f1p = phi[p1[0], p1[1], p1[2]]
    f1m = phi[m1[0], m1[1], m1[2]]

    # (+/−) along second arm (start from the original base)
    p2 = base.copy(); p2[axis[ay]] = min(max(base[axis[ay]] + spacing, 1), [Nx,Ny,Nz][axis[ay]] - 2)
    m2 = base.copy(); m2[axis[ay]] = min(max(base[axis[ay]] - spacing, 1), [Nx,Ny,Nz][axis[ay]] - 2)
    f2p = phi[p2[0], p2[1], p2[2]]
    f2m = phi[m2[0], m2[1], m2[2]]

    df1 = (f1p - f1m) / (2.0 * spacing)
    df2 = (f2p - f2m) / (2.0 * spacing)
    return 0.5 * (df1 + df2)


# UMH-consistent leading-order quadrupole amplitude:
# h_ref ∝ M_chirp_src^(5/3) f_ref_src^(2/3) / D_L^UMH
# D_L^UMH is the full Pantheon+-calibrated UMH effective luminosity distance: D_L^UMH = D_geom(z) * (1+z)^((1+delta)/2) * T(z)^(-1/2)
# This is used ONLY for the single global normalization. Redshift, time-dilation, and transmission enter once through D_L^UMH, not through a separate D_eff factor.
def newtonian_h_at_f_umh(f_ref_src_hz, M_chirp_src_kg, D_L_umh_m, G_phys=6.67430e-11, c_phys=2.99792458e8):
    """
    UMH-consistent quadrupole amplitude normalization.
    ----------
    f_ref_src_hz:
        Source-frame reference GW frequency corresponding to the observer-frame normalization point, f_ref_src = (1+z_UMH) * f_ref_obs.
    M_chirp_src_kg: Source-frame chirp mass.
    D_L_umh_m:
        Full UMH effective luminosity distance in meters, including the Pantheon+-calibrated transmission and time-dilation factors.
    -----
    This implements the convention: h_ref^UMH = 4 G^(5/3) M_c,src^(5/3) (pi f_ref,src)^(2/3) / (c^4 D_L^UMH)
    The envelope sample is still selected using f_ref_obs, because the stored waveform is indexed in detector/observer time. But the amplitude formula
    itself is evaluated in the source-frame convention.
    """
    f = float(f_ref_src_hz); M = float(M_chirp_src_kg); D = float(D_L_umh_m)
    if not math.isfinite(f) or f <= 0.0: raise ValueError(f"Invalid f_ref_src_hz={f_ref_src_hz}")
    if not math.isfinite(M) or M <= 0.0: raise ValueError(f"Invalid M_chirp_src_kg={M_chirp_src_kg}")
    if not math.isfinite(D) or D <= 0.0: raise ValueError(f"Invalid D_L_umh_m={D_L_umh_m}")

    num = 4.0 * (G_phys ** (5.0 / 3.0)) * (M ** (5.0 / 3.0)) * (math.pi * f) ** (2.0 / 3.0)
    return num / (c_phys ** 4 * D)


def create_UMH_z_tension_from_redshift_calibration(calibration_path, H0_km_s_Mpc=70.0, delta=1.0):
    """
    Create the UMH_z_tension model from a calibration JSON and configuration H0.
    Load UMH redshift/tension calibration constants
    (α, β1, β2, etc.) from UMH_RedShift_Calibration_Fit.json.

    Parameters
    ----------
    calibration_path :  Path to UMH_RedShift_Calibration_Fit.json
    H0_km_s_Mpc :       Reference Hubble-scale parameter for comparison (default=70)
    delta :             Power on (1+z) in UMH luminosity-distance law (default=1.0)
    """
    # --- Core UMH functions ---
    def L(z): return math.log1p(z)                          # natural log(1+z)
    def tau(z, beta1, beta2): Lz = L(z); return beta1 * Lz + beta2 * Lz * Lz
    def T(z, beta1, beta2): return math.exp(-tau(z, beta1, beta2))
    def z_from_d(alpha, d_Mpc): return math.expm1(alpha * d_Mpc)   # redshift from geometric distance
    def d_from_z(z, alpha): return math.log1p(z) / alpha           # geometric distance from redshift
    def DL_from_z(z, alpha, beta1, beta2, delta):
        # luminosity distance in Mpc using UMH attenuation and redshift law
        d = d_from_z(z, alpha)
        Tz = T(z, beta1, beta2)
        return d * (1.0 + z)**((1.0 + delta) / 2.0) / math.sqrt(Tz)
    def mu_from_z(z, alpha, beta1, beta2, delta): DL = DL_from_z(z, alpha, beta1, beta2, delta); return 5.0 * math.log10(DL) + 25.0     # distance modulus
    def z_from_DL(DL_Mpc, alpha, beta1, beta2, delta, z_max=5.0):
        # numeric inversion for GWs
        lo, hi = 0.0, z_max
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if DL_from_z(mid, alpha, beta1, beta2, delta) > DL_Mpc: hi = mid
            else: lo = mid
        return 0.5 * (lo + hi)

    # UMH_DEFAULT_CALIBRATION - Only used if Calibration file is unavailable.
    UMH_DEFAULT_CALIBRATION = {
        "model": "UMH non-expansion",
        "alpha_1_per_Mpc": 0.000248180462738863,
        "alpha_1_per_m": 8.042981038169037e-27,
        "alpha": 0.000248180462738863,
        "intercept": 1.3374410762914681,
        "M_best": 5.784551628220646,
        "beta1": 0.4527504384345805,
        "beta2": -0.2703128096327221,
        # Metadata and diagnostics (optional but helpful for provenance)
        "source": "Fallback UMH_RedShift_Calibration_Fit.json constants",
        "H0": 70.0,   # default if config doesn’t override
        "delta": 1.0,          # UMH luminosity distance power term
        "z_from_d": z_from_d,
        "d_from_z": d_from_z,
        "tau": tau,
        "T": T,
        "DL_from_z": DL_from_z,
        "mu_from_z": mu_from_z,
        "z_from_DL": z_from_DL,
    }

    # --- Load calibration JSON ---
    try:
        if(calibration_path != "" and os.path.exists(calibration_path)):
            with open(calibration_path, "r") as f: calib = json.load(f)

            # --- Core UMH calibration parameters ---
            alpha = calib.get("alpha_1_per_Mpc")
            beta1 = calib.get("beta1", 0.0)
            beta2 = calib.get("beta2", 0.0)
            M_best = calib.get("M_best", None)

            # --- Package everything ---
            UMH_z_tension = {
                "alpha": alpha,
                "beta1": beta1,
                "beta2": beta2,
                "M_best": M_best,
                "H0": H0_km_s_Mpc,
                "delta": delta,
                "source": "Loaded directly from UMH_RedShift_Calibration_Fit.json constants",
                "z_from_d": z_from_d,
                "d_from_z": d_from_z,
                "tau": tau,
                "T": T,
                "DL_from_z": DL_from_z,
                "mu_from_z": mu_from_z,
                "z_from_DL": z_from_DL,
            }
            return UMH_z_tension

        else: return UMH_DEFAULT_CALIBRATION

    except Exception as e:
        print(f"[WARN] Could not load UMH calibration: {e}")
        return UMH_DEFAULT_CALIBRATION

def beta_eff_from_pantheon(z_tension: float, beta1: float, beta2: float) -> float:
    """
    Convert Pantheon transport coefficients (beta1,beta2) into a single dimensionless local response strength:
        beta_eff(z) = d tau / d ln(1+z) = beta1 + 2*beta2*ln(1+z)
    This is used as the ringdown frequency relaxation strength (dimensionless),
    i.e. tau_relax = beta_eff * tau_rd.
    """
    z = float(z_tension or 0.0)
    if not math.isfinite(z) or z < 0.0: z = 0.0
    L = math.log1p(z)  # ln(1+z)
    return float(beta1 + 2.0 * beta2 * L)


# This is NOT used in the actual waveform; it's a consistency illustration only.
def umh_macro_frequency_band(lambda_u, M_tot_solar, k_min=6.0, k_max=20.0, G_phys=6.67430e-11, c_phys=2.99792458e8, M_sun=1.98847e30):
    """
    Heuristic UMH diagnostic:
    Connect fundamental UMH frequency f_u to BBH GW band via geometric scaling.

    Treats the observable GW frequency as: f_GW ~ f_u * (lambda_u / R_orb), where R_orb is in [k_min * R_g, k_max * R_g].

    This is NOT used in the actual waveform; it's a consistency illustration only.
    """
    # Fundamental UMH frequency
    f_u = c_phys / (2.0 * math.pi * lambda_u)

    # Gravitational radius for total mass
    M_tot_kg_src = M_tot_solar * M_sun
    R_g = G_phys * M_tot_kg_src / (c_phys**2)

    # Orbital radii range near merger
    R_min = k_min * R_g
    R_max = k_max * R_g

    # Dimensionless scaling factors
    eps_min = lambda_u / R_min
    eps_max = lambda_u / R_max

    # UMH-induced macro frequencies
    f_min = f_u * eps_max  # larger R -> smaller eps -> smaller f
    f_max = f_u * eps_min  # smaller R -> larger eps -> larger f

    return f_min, f_max


def load_psd_map_for_generator(file_path):
    """
    Load PSD arrays saved by the compiler and build callable psd(f) maps.
    Returns a dict: psd_map[detector](freq_array) -> PSD(freq_array).
    """
    data = np.load(file_path)
    psd_map = {}
    #for name in ("Hanford", "Livingston", "Virgo"):
    for name, g in SITES.items():
        f_key = f"{name}_freqs"
        p_key = f"{name}_psd"
        if f_key in data and p_key in data:
            freqs = data[f_key].astype(float)
            Pxx   = data[p_key].astype(float)
            # Build an interpolator using closure
            def make_psd_interp(freqs_local, Pxx_local):
                def psd_func(f):
                    f = np.asarray(f, dtype=float)
                    return np.interp(f, freqs_local, Pxx_local, left=Pxx_local[0], right=Pxx_local[-1])
                return psd_func

            psd_map[name] = make_psd_interp(freqs, Pxx)
    return psd_map

def compute_snr_weighted_fref(psd_map, f_min, f_merge):
    """
    Compute f_ref_src using SNR-weighted criterion ∝ f^(-7/3)/S_n(f)
    for all available detectors in psd_map.
    """
    if not psd_map: raise RuntimeError("compute_snr_weighted_fref: psd_map is empty.")

    freqs = np.linspace(f_min, f_merge, 2000)
    f_refs = []

    for det_name, psd_fn in psd_map.items():
        psd_vals = psd_fn(freqs)
        psd_vals = np.maximum(psd_vals, 1e-48)
        weight   = freqs**(-7.0/3.0) / psd_vals
        idx      = int(np.argmax(weight))
        f_refs.append(freqs[idx])

    return float(np.mean(f_refs))


# PN-0 (Newtonian) chirp time based on df/dt ∝ f^(11/3).
# This realizes f(t) ∝ (t_c - t)^(-3/8), as derived in the UMH framework and GR.
def pn_duration_sec(M_chirp_src_kg, f_min_src, f_merge_use_pn, G_phys=6.67430e-11, c_phys=2.99792458e8):
    """
    Compute post-Newtonian (leading-order) inspiral duration between f_min_src and f_merge_use_pn.
    """
    # Chirp mass (physical units)
    #Mc = (M1_kg_src * M2_kg_src)**(3.0/5.0) / (M1_kg_src + M2_kg_src)**(1.0/5.0)

    # Factor (G * Mc / c^3)
    G_over_c3 = G_phys / (c_phys**3)
    term = G_over_c3 * M_chirp_src_kg  # seconds

    # Coefficient for PN leading-order time-to-frequency relation
    C = (5.0 / (256.0 * np.pi**(8.0/3.0))) * (term**(-5.0/3.0))

    # Duration (seconds) from f_min_src→f_merge_use_pn
    return C * (f_min_src**(-8.0/3.0) - f_merge_use_pn**(-8.0/3.0))

#Numerically estimate time to sweep f_min_src -> f_merge_src using dfdt_highorder.
def estimate_merge_time_from_dfdt(K_N, f_min_src, f_merge_src, f_ref_src, coeffs, n_steps=4000):
    """
    Numerically estimate time to sweep f_min_src -> f_merge_src using dfdt_highorder.
    Used to calibrate K_N so that the high-order profile matches target t_merge.
    """
    if f_min_src <= 0.0 or f_merge_src <= f_min_src: return None

    fs = np.linspace(f_min_src, f_merge_src, n_steps)
    dt_sum = 0.0
    for i in range(n_steps - 1):
        f_mid = 0.5 * (fs[i] + fs[i+1])
        dfdt_mid = dfdt_highorder(K_N, f_ref_src, f_mid, coeffs=coeffs)
        if not np.isfinite(dfdt_mid) or dfdt_mid <= 0.0: continue
        df = fs[i+1] - fs[i]
        dt_sum += df / dfdt_mid

    return dt_sum if dt_sum > 0.0 else None

# UMH-native higher-order df/dt law with optional GR-like PN coefficients.
def dfdt_highorder(K_N, f_ref_src, f_cur, coeffs=None):
    """
    UMH-native higher-order df/dt law with optional GR-like PN coefficients.

    Parameters
    ----------
    K_N : float
        Leading-order Newtonian/UMH coefficient from energy balance.
    f_ref_src : float
        Reference frequency for nondimensionalization (e.g., f_min or 100 Hz).
    f_cur : float
        Current GW frequency.
    coeffs : dict, optional
           Specific content (non-spinning, quasi-circular, GR limit):

              - C2 (1PN, v^2):
                    Encodes relativistic corrections to orbital binding energy
                    and GW flux at O(v^2/c^2).

              - C3 (1.5PN, v^3):
                    Leading tail term (back-scattering off curvature).

              - C4 (2PN, v^4):
                    Higher-order conservative + flux corrections, includes η^2.

              - C5 (2.5PN, v^5):
                    Next-order dissipative tail term.

              - C6 (3PN, v^6):
                    Includes logarithmic and Euler–gamma pieces; in this
                    implementation they are evaluated at f_ref_src so the logs are
                    absorbed into a finite coefficient.

              - C7 (3.5PN, v^7):
                    3.5PN dissipative term.
    Notes
    -----
    df/dt = K_N * f_cur^(11/3) * [1 + Σ Cn * u^n],
    where u = (f_cur / f_ref_src)^(1/3) tracks the relative orbital velocity.
    """
    # Safety guard
    if f_ref_src <= 0.0: f_ref_src = max(f_cur, 1e-9)

    # Dimensionless velocity-like ratio (u = v/v_ref)
    u = (f_cur / f_ref_src)**(1.0/3.0)

    # Retrieve coefficients safely
    if coeffs is None: coeffs = {}
    C2 = coeffs.get("C2", 0.0)
    C3 = coeffs.get("C3", 0.0)
    C4 = coeffs.get("C4", 0.0)
    C5 = coeffs.get("C5", 0.0)
    C6 = coeffs.get("C6", 0.0)
    C7 = coeffs.get("C7", 0.0)

    # Correction series (UMH or GR-like)
    correction = (1.0
        + C2 * u**2    # 1PN
        + C3 * u**3    # 1.5PN
        + C4 * u**4    # 2PN
        + C5 * u**5    # 2.5PN
        + C6 * u**6    # 3PN
        + C7 * u**7)   # 3.5PN

    # Optional safety: prevent negative df/dt
    if correction < 0.0: correction = 0.0

    return K_N * (f_cur ** (11.0 / 3.0)) * correction


def amplitude_pn_factor(f_gw, M_tot_kg_src, eta, G_phys=6.67430e-11, c_phys=2.99792458e8, max_order=3.0):
    """
    PN amplitude correction for the dominant (2,2) mode, non-spinning (GR limit of UMH).

    h_22 ∝ v^2 * [ 1
                   + a2 v^2          (1PN)
                   + a3 v^3          (1.5PN)
                   + a4 v^4          (2PN)
                   + a5 v^5          (2.5PN, non-spinning)
                   + a6 v^6          (3.0PN, non-spinning)
                 ],
    with v = (pi * G * M_tot * f_gw / c^3)^(1/3).

    All constants carried at full double precision.
    """
    f = float(f_gw)
    if f <= 0.0: return 1.0

    # PN velocity parameter
    v = (math.pi * G_phys * M_tot_kg_src * f / c_phys**3) ** (1.0 / 3.0)

    pn = 1.0

    # 1PN
    if max_order >= 1.0:
        a2       = -107.0 / 42.0 + (55.0 / 42.0) * eta
        pn      += a2 * v**2

    # 1.5PN
    if max_order >= 1.5:
        a3       = 2.0 * math.pi  # 6.2831853071795864769
        pn       += a3 * v**3

    # 2PN
    if max_order >= 2.0:
        a4 = (-2173.0 / 1512.0
              - (1069.0 / 216.0) * eta
              + (2047.0 / 1512.0) * (eta**2))
        pn       += a4 * v**4

    # 2.5PN (non-spinning)
    if max_order >= 2.5:
        # −107π/21 + 34π/21 η
        a5        = (-107.0 * math.pi / 21.0) + (34.0 * math.pi / 21.0) * eta
        pn       += a5 * v**5

    # 3.0PN (non-spinning)
    if max_order >= 3.0:
        # Euler–Mascheroni constant to 30 digits
        gamma_E   = 0.5772156649015328606065120900824

        # 3PN constant term coefficients, evaluated to full precision
        a6_const  = (27027409.0 / 646800.0
                    - (856.0 / 105.0) * gamma_E
                    + (2.0 / 3.0) * (math.pi**2)
                    - (1712.0 / 105.0) * math.log(2.0))
        a6_eta    = -278185.0 / 33264.0 + (41.0 * (math.pi**2)) / 96.0
        a6_eta2   = -20261.0 / 2772.0
        a6_eta3   = 114635.0 / 99792.0

        # log term uses ln(4v) convention → ln(v) piece
        log_piece = -(1712.0 / 105.0) * math.log(v)

        a6        = a6_const + log_piece + a6_eta * eta + a6_eta2 * (eta**2) + a6_eta3 * (eta**3)
        pn       += a6 * v**6

    # Numerical protection near merger
    if pn < 0.1: pn = 0.1

    return pn


# Reconstruct instantaneous GW frequency from the unwrapped phase.
# Used to QA and stabilize normalization so that scaling is tied to phase_hist(t), not to any stale or noisy frequency record.
def ensure_f_hist_from_phase(phase_hist, dt, dtype=np.float64):
    ph = np.unwrap(np.asarray(phase_hist, dtype=dtype))
    f  = np.diff(ph) / (2.0*np.pi*dt)
    if f.size == 0: return np.zeros_like(ph)
    return np.concatenate([f, [f[-1]]])


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


# --- Create Detectors GPS ---
def llh_to_ecef(lat, lon, h, a = 6378137.0, e2 = 6.69437999014e-3):
    s = np.sin(lat); c = np.cos(lat)
    N = a / np.sqrt(1 - e2*s*s)
    x = (N + h) * c * np.cos(lon)
    y = (N + h) * c * np.sin(lon)
    z = (N*(1 - e2) + h) * s
    return np.array([x, y, z])

def gmst_rad_from_utc(year, month, day, hour, minute, second):
    # Simple, accurate GMST (UTC-based) good to <~0.1s for our purpose.
    # Convert UTC -> Julian Date
    y, m = year, month
    if m <= 2: y -= 1; m += 12
    A = np.floor(y/100.0)
    B = 2 - A + np.floor(A/4.0)
    day_frac = (hour + (minute + second/60.0)/60.0)/24.0
    JD = (np.floor(365.25*(y+4716)) + np.floor(30.6001*(m+1)) + day + day_frac + B - 1524.5)
    T = (JD - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629*(JD-2451545.0) + 0.000387933*T*T - (T**3)/38710000.0) * np.pi/180.0
    return (gmst % (2*np.pi))

def r3(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, s, 0], [-s, c, 0], [ 0, 0, 1]])

def enu_basis(lat, lon):
    # East, North, Up unit vectors in ECEF at site (lat, lon)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    E = np.array([-so,  co, 0.0])
    N = np.array([-sl*co, -sl*so, cl])
    U = np.array([ cl*co,  cl*so, sl])
    return E, N, U

def arm_from_azimuth(az_deg, E, N):
    # azimuth measured east of north in the local tangent plane
    az = np.deg2rad(az_deg)
    return np.cos(az)*N + np.sin(az)*E

def build_polarization_basis(k):
    # Given propagation direction k (from source to Earth), build any orthonormal transverse basis (e_theta, e_phi); then rotate by psi for polarization.
    # Use a stable choice: take ẑ unless nearly parallel.
    z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(k, z)) > 0.99: z = np.array([1.0, 0.0, 0.0])
    e_theta = np.cross(k, z); e_theta /= np.linalg.norm(e_theta)
    e_phi   = np.cross(k, e_theta)
    return e_theta, e_phi

def antenna_F(u, v, k, psi):
    # Detector tensor
    d = 0.5*(np.outer(u, u) - np.outer(v, v))
    # Polarization basis
    e_t, e_p = build_polarization_basis(k)
    # Rotate by psi
    c, s = np.cos(psi), np.sin(psi)
    p_plus  =  np.outer(c*e_t - s*e_p, c*e_t - s*e_p) - np.outer(s*e_t + c*e_p, s*e_t + c*e_p)
    p_cross =  (np.outer(c*e_t - s*e_p, s*e_t + c*e_p) + np.outer(s*e_t + c*e_p, c*e_t - s*e_p))
    Fp = np.sum(d * p_plus)
    Fx = np.sum(d * p_cross)
    return Fp, Fx


# geom_delay_sec sign convention:
#   geom_delay_sec > 0  ⇒ this site receives the signal LATER than a site with smaller delay.
# Delays are used only as relative time shifts; no amplitude scaling here.
def compute_detector_geo(Sites_Used, ra_deg, dec_deg, utc_tuple, pol_psi_deg=0.0, c_phys=2.99792458e8):
    """
    ra/dec in degrees (ICRS). utc_tuple = (Y,M,D,h,m,s). pol_psi_deg is polarization angle.
    Returns dict: per-detector { geom_delay_sec, Fplus, Fcross, site_ecef, u, v }.
    """
    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    psi = np.deg2rad(pol_psi_deg)

    # Source unit vector (ICRS/ECI)
    s_eci = np.array([np.cos(dec)*np.cos(ra),
                      np.cos(dec)*np.sin(ra),
                      np.sin(dec)])
    gmst  = gmst_rad_from_utc(*utc_tuple)
    s_ecef = r3(gmst) @ s_eci

    # Propagation direction (toward Earth)
    k = -s_ecef / np.linalg.norm(s_ecef)

    site_geo = {}
    for name, g in SITES.items():
        if name in Sites_Used:
            lat = np.deg2rad(g["lat_deg"])
            lon = np.deg2rad(g["lon_deg"])
            r_ecef = llh_to_ecef(lat, lon, g["h_m"])
            E, N, U = enu_basis(lat, lon)
            u = arm_from_azimuth(g["az_x_deg"], E, N); u /= np.linalg.norm(u)
            v = arm_from_azimuth(g["az_y_deg"], E, N); v /= np.linalg.norm(v)
            Fp, Fx = antenna_F(u, v, k, psi)
            site_geo[name] = dict(site_ecef=r_ecef, u=u, v=v, Fplus=Fp, Fcross=Fx)

    # Time delay (seconds) for each detector from source propagation.
    # Convention: h(t, x) = h0(t - k·x/c)
    # Using +np.dot(r_site, k)/c ensures the correct arrival order (Livingston ~7 ms before Hanford for GW150914 geometry).
    taus = {name: float(np.dot(info["site_ecef"], k) / c_phys) for name, info in site_geo.items()}
    t0 = min(taus.values())
    for name in site_geo: site_geo[name]["geom_delay_sec"] = taus[name] - t0

    return site_geo, k, gmst
# --- End Create Detectors GPS ---

# --- Calculate Ringdown ---
def _kerr_isco_loss(chi):
    """
    Specific binding-energy loss at the Kerr ISCO: E_loss = 1 - E_ISCO(a)
        chi > 0 = aligned/prograde
        chi < 0 = anti-aligned/retrograde
    This gives the correct test-particle trend:
      chi=0    -> 0.05719
      chi>0    -> larger radiated fraction
      chi<0    -> smaller radiated fraction
    """
    a = max(-0.999, min(float(chi), 0.999))
    z1 = 1.0 + (1.0 - a*a)**(1.0/3.0) * ((1.0 + a)**(1.0/3.0) + (1.0 - a)**(1.0/3.0))
    z2 = math.sqrt(3.0*a*a + z1*z1)
    # Prograde for a > 0, retrograde for a < 0
    r_isco = 3.0 + z2 - math.copysign(math.sqrt(max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0*z2))), a)
    e_isco = math.sqrt(max(0.0, 1.0 - 2.0 / (3.0 * r_isco)))
    return max(0.0, 1.0 - e_isco)


def estimate_remnant_mass_spin(M1_solar_src, M2_solar_src, spin1x=0.0, spin1y=0.0, spin1z=0.0, spin2x=0.0, spin2y=0.0, spin2z=0.0, 
                               approximant="SEOBNRv4", SPIN_MODE="none", G_phys=6.67430e-11, c_phys=2.99792458e8, M_sun=1.98847e30, f_ref=-1):
    """
    Estimate the final remnant mass, final spin, radiated-energy fraction, and symmetric mass ratio for a quasi-circular aligned-spin BBH merger.
    This function intentionally does NOT use a local hand-fit. It delegates the remnant calculation to standard PyCBC/LALSuite remnant fits.
    Recommended approximants:
      - "SEOBNRv4"    : aligned-spin, robust for this function's inputs.
      - "SEOBNRv4PHM" : precessing-capable LAL path, but here x/y spins are zero.
      - "NRSur7dq4"   : NRSur7dq4Remnant fit, valid only inside its calibration domain and requires LALSuite NRSur data support.
    Returns
    -------
    Mrem_solar_src : float Final remnant mass in source-frame solar masses.
    a_rem_src : float Dimensionless final remnant spin. For aligned-spin inputs this is signed relative to the orbital angular momentum.
    E_rad_frac : float Fraction of source-frame total mass radiated away: E_rad_frac = 1 - Mrem / (M1 + M2).
    eta : float Symmetric mass ratio.
    """
    def _finite_float(x, name):
        x = float(x)
        if not math.isfinite(x): raise ValueError(f"{name} must be finite; got {x!r}")
        return x

    M1_solar_src = _finite_float(M1_solar_src, "M1_solar_src"); M2_solar_src = _finite_float(M2_solar_src, "M2_solar_src")
    if M1_solar_src <= 0.0 or M2_solar_src <= 0.0: raise ValueError(f"Component masses must be positive; got M1={M1_solar_src}, M2={M2_solar_src}")

    SPIN_MODE = str(SPIN_MODE).lower()
    if SPIN_MODE == "aligned": 
        spin1x=0.0; spin1y=0.0; spin2x=0.0; spin2y=0.0
        spin1z = _finite_float(spin1z, "spin1z"); spin2z = _finite_float(spin2z, "spin2z")
        if not (-0.999 <= spin1z <= 0.999): raise ValueError(f"spin1z must be in [-0.999, 0.999]; got {spin1z}")
        if not (-0.999 <= spin2z <= 0.999): raise ValueError(f"spin2z must be in [-0.999, 0.999]; got {spin2z}")
    elif SPIN_MODE == "precessing":
        spin1x = _finite_float(spin1x, "spin1x"); spin2x = _finite_float(spin2x, "spin2x")
        spin1y = _finite_float(spin1y, "spin1y"); spin2y = _finite_float(spin2y, "spin2y")
        spin1z = _finite_float(spin1z, "spin1z"); spin2z = _finite_float(spin2z, "spin2z")
        if not (-0.999 <= spin1z <= 0.999): raise ValueError(f"spin1z must be in [-0.999, 0.999]; got {spin1z}")
        if not (-0.999 <= spin2z <= 0.999): raise ValueError(f"spin2z must be in [-0.999, 0.999]; got {spin2z}")
        mag1 = math.sqrt(spin1x*spin1x + spin1y*spin1y + spin1z*spin1z)
        mag2 = math.sqrt(spin2x*spin2x + spin2y*spin2y + spin2z*spin2z)
        if mag1 >= 0.999: raise ValueError(f"|spin1| must be < 0.999 for precessing remnant fit; got {mag1}")
        if mag2 >= 0.999: raise ValueError(f"|spin2| must be < 0.999 for precessing remnant fit; got {mag2}")
    else: spin1x=0.0; spin1y=0.0; spin1z=0.0; spin2x=0.0; spin2y=0.0; spin2z=0.0

    # Ensure labeling: M1 >= M2, carrying spins with their masses.
    if M2_solar_src > M1_solar_src: 
        M1_solar_src, M2_solar_src = M2_solar_src, M1_solar_src; 
        spin1x, spin2x = spin2x, spin1x;  spin1y, spin2y = spin2y, spin1y; spin1z, spin2z = spin2z, spin1z
    M_tot_src = M1_solar_src + M2_solar_src
    eta = (M1_solar_src * M2_solar_src) / (M_tot_src * M_tot_src)
    approximant = str(approximant)

    # Preferred path: PyCBC wrapper around LALSuite remnant fits.
    try:
        from pycbc.conversions import get_final_from_initial

        Mrem_solar_src, a_rem_src = get_final_from_initial(mass1=M1_solar_src, mass2=M2_solar_src, spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, 
                                                           spin2x=spin2x, spin2y=spin2y, spin2z=spin2z, approximant=approximant, f_ref=f_ref)
        Mrem_solar_src = float(Mrem_solar_src); a_rem_src = float(a_rem_src)

    except ImportError:
        # Direct LALSimulation fallback. This is still a standard LALSuite remnant fit, not a local fit.
        try:
            import lalsimulation as lalsim
        except ImportError as exc:
            raise ImportError("Defensible remnant mass/spin requires PyCBC or LALSuite. Install one of them, e.g. `pip install lalsuite pycbc`,"
                              "or do not claim an NR/EOB-calibrated remnant estimate.") from exc

        spin1 = [spin1x, spin1y, spin1z]; spin2 = [spin2x, spin2y, spin2z]
        if approximant == "NRSur7dq4":
            try:
                import lal
                from lalsimulation import nrfits
            except ImportError as exc: raise ImportError("approximant='NRSur7dq4' requires LALSuite NR fit support.") from exc

            res = nrfits.eval_nrfit(M1_solar_src * lal.MSUN_SI, M2_solar_src * lal.MSUN_SI, spin1, spin2, "NRSur7dq4Remnant", ["FinalMass", "FinalSpin"], f_ref=f_ref)
            Mrem_solar_src = float(res["FinalMass"][0] / lal.MSUN_SI)
            sf = res["FinalSpin"]
            a_rem_src = float(math.sqrt(float(sf[0])**2 + float(sf[1])**2 + float(sf[2])**2))
            if float(sf[2]) < 0.0: a_rem_src *= -1.0
        elif approximant == "SEOBNRv4":
            _, mf_frac, a_rem_src = lalsim.SimIMREOBFinalMassSpin(M1_solar_src, M2_solar_src, spin1, spin2, getattr(lalsim, approximant))
            Mrem_solar_src = float(mf_frac) * M_tot_src; a_rem_src = float(a_rem_src)
        else:
            _, mf_frac, a_rem_src = lalsim.SimIMREOBFinalMassSpinPrec(M1_solar_src, M2_solar_src, spin1, spin2, getattr(lalsim, approximant))
            Mrem_solar_src = float(mf_frac) * M_tot_src; a_rem_src = float(a_rem_src)

    if not math.isfinite(Mrem_solar_src) or Mrem_solar_src <= 0.0: raise RuntimeError(f"Invalid remnant mass returned: {Mrem_solar_src}")
    if not math.isfinite(a_rem_src) or abs(a_rem_src) >= 1.0: raise RuntimeError(f"Invalid remnant spin returned: {a_rem_src}")

    E_rad_frac = 1.0 - (Mrem_solar_src / M_tot_src)
    if not math.isfinite(E_rad_frac) or E_rad_frac < 0.0 or E_rad_frac > 0.25:
        raise RuntimeError(f"Invalid radiated-energy fraction: {E_rad_frac}; Mtot={M_tot_src}, Mrem={Mrem_solar_src}")

    print(f"[REMNANT_FINAL] M1={M1_solar_src:.6f} M2={M2_solar_src:.6f} eta={eta:.9f} Mrem_src={Mrem_solar_src:.6f} a_rem={a_rem_src:.6f} E_rad_frac={E_rad_frac:.9f}")

    return Mrem_solar_src, a_rem_src, E_rad_frac, eta


def umh_gr_limit_220_shape(chi):
    """
    GR-limit dimensionless Kerr/220 shape functions expressed as the macroscopic limit of the UMH remnant eigenmode.
    These are not used as an external QNM seed. They define the GR-limit boundary condition for the UMH spin/tension-winding factor F(chi) and damping eigenvalue gamma(chi).
    """
    a = max(-0.9999, min(float(chi), 0.9999))
    one_minus_a = max(1.0 - a, 1.0e-12)

    # Berti/Cardoso/Will-style 220 real-frequency and quality-factor fits.
    Omega_220 = 1.5251 - 1.1568 * (one_minus_a ** 0.1292)
    Q_220     = 0.7000 + 1.4187 * (one_minus_a ** -0.4990)

    if not math.isfinite(Omega_220) or Omega_220 <= 0.0: raise ValueError(f"Invalid Omega_220={Omega_220} for chi={chi}")
    if not math.isfinite(Q_220) or Q_220 <= 0.0: raise ValueError(f"Invalid Q_220={Q_220} for chi={chi}")

    # UMH spin/tension-winding factor required for the GR/Kerr 220 limit:
    #     omega_R_UMH = pi*c^3/(G*M*F)
    #     omega_R_GR  = Omega_220*c^3/(G*M)
    # so: F_GR_220 = pi / Omega_220
    F_GR_220 = math.pi / Omega_220

    return {"chi": float(a), "Omega_220": float(Omega_220), "Q_220": float(Q_220), "F_GR_220": float(F_GR_220)}


def fit_umh_ringdown_coefficients(chi_values, f_src_Hz_values, tau_src_s_values, Mrem_kg_values, G_phys=6.67430e-11, c_phys=2.99792458e8):
    """
    Fit Kerr-style coefficient forms to UMH remnant eigenmode data.
    Inputs must be source-frame:
        chi_values       remnant spin values
        f_src_Hz_values  UMH source-frame ringdown frequencies
        tau_src_s_values UMH source-frame damping times
        Mrem_kg_values   UMH source-frame remnant masses
    Outputs:
        omega_coeffs = (A, B, C) for Ω(χ) = A + B(1-χ)^C
        Q_coeffs     = (D, E, F) for Q(χ) = D + E(1-χ)^F
    """
    from scipy.optimize import curve_fit

    def omega_fit_form(chi, A, B, C): chi = np.asarray(chi, dtype=float); return A + B * np.power(1.0 - chi, C)
    def Q_fit_form(chi, D, E, F): chi = np.asarray(chi, dtype=float); return D + E * np.power(1.0 - chi, F)

    chi = np.asarray(chi_values, dtype=float); f = np.asarray(f_src_Hz_values, dtype=float); tau = np.asarray(tau_src_s_values, dtype=float)
    M = np.asarray(Mrem_kg_values, dtype=float)
    if not (chi.size == f.size == tau.size == M.size): raise ValueError("chi, f, tau, and M arrays must have the same length.")
    if chi.size < 6: raise ValueError("Use at least 6 spin samples; 20+ is better for stable coefficients.")
    if np.any(~np.isfinite(chi)) or np.any(chi <= -1.0) or np.any(chi >= 1.0): raise ValueError("All chi values must be finite and inside (-1, 1).")
    if np.any(~np.isfinite(f)) or np.any(f <= 0.0): raise ValueError("All source-frame frequencies must be positive and finite.")
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0.0): raise ValueError("All source-frame tau values must be positive and finite.")
    if np.any(~np.isfinite(M)) or np.any(M <= 0.0): raise ValueError("All remnant masses must be positive and finite.")
    # Dimensionless real frequency:
    # Ω = ω_R * G*M/c^3 = 2π f_src * G*M/c^3
    Omega = 2.0 * math.pi * f * G_phys * M / (c_phys ** 3)
    # Ringdown quality factor:
    # Q = π f τ
    Q = math.pi * f * tau
    # Fit Ω(χ) = A + B(1-χ)^C
    omega_guess = (1.5, -1.1, 0.13)
    omega_bounds = ([0.0, -10.0, 0.001], [5.0,  10.0, 5.0])
    omega_coeffs, omega_cov = curve_fit(omega_fit_form, chi, Omega, p0=omega_guess, bounds=omega_bounds, maxfev=100000)

    # Fit Q(χ) = D + E(1-χ)^F
    Q_guess = (0.7, 1.4, -0.5)
    Q_bounds = ([0.0, 0.0, -5.0], [20.0, 50.0, 5.0])
    Q_coeffs, Q_cov = curve_fit(Q_fit_form, chi, Q, p0=Q_guess, bounds=Q_bounds, maxfev=100000)

    Omega_pred = omega_fit_form(chi, *omega_coeffs)
    Q_pred = Q_fit_form(chi, *Q_coeffs)

    return {"omega_coeffs": {"A": float(omega_coeffs[0]), "B": float(omega_coeffs[1]), "C": float(omega_coeffs[2]), "form": "Omega_UMH(chi) = A + B*(1-chi)^C"},
            "Q_coeffs": {"D": float(Q_coeffs[0]), "E": float(Q_coeffs[1]), "F": float(Q_coeffs[2]), "form": "Q_UMH(chi) = D + E*(1-chi)^F"},
            "diagnostics": {"n_samples": int(chi.size), "Omega_rms_residual": float(np.sqrt(np.mean((Omega - Omega_pred) ** 2))),
                            "Q_rms_residual": float(np.sqrt(np.mean((Q - Q_pred) ** 2))), "Omega_values": Omega.tolist(), "Q_values": Q.tolist(),
                            "Omega_fit_values": Omega_pred.tolist(), "Q_fit_values": Q_pred.tolist()}}



def umh_radial_phase_delay_from_model(t_phi_GM_c3, chi, model="none", delay_frac=0.0, delay_time_GM_c3=0.0):
    """
    UMH radial relaxation / backscatter phase-delay model.
    The sharp rotating confinement eigenmode uses: omega_hat_base = m / t_phi
    where t_phi is the dimensionless rotating confinement clock in units GM/c^3.
    A radial strain-gradient/backscatter delay adds an additional response time:
        t_eff = t_phi + t_rad
        omega_hat_eff = m / t_eff
    This lowers the real remnant frequency without changing the confinement radius. That is intentionally different from boundary softening.
    Supported models:
      - "none": t_rad = 0
      - "fractional_time": t_rad = delay_frac * t_phi
    The delay is dimensionless in units GM/c^3.
    """
    t_phi = float(t_phi_GM_c3)
    if not math.isfinite(t_phi) or t_phi <= 0.0: raise ValueError(f"Invalid t_phi_GM_c3={t_phi_GM_c3}")
    a = float(chi)
    if not math.isfinite(a): raise ValueError(f"Invalid chi={chi}")
    model = str(model).lower().strip()
    if model in ("none", "off", "false", "0", ""): t_rad = 0.0; rule = "none"
    elif model == "fractional_time":
        eps = float(delay_frac)
        if not math.isfinite(eps) or eps < 0.0: raise ValueError(f"Invalid UMH_RADIAL_PHASE_DELAY_FRAC={delay_frac}")
        t_rad = eps * t_phi
        rule = "fractional_time"
    elif model == "absolute_time":
        t_rad = float(delay_time_GM_c3)
        if not math.isfinite(t_rad) or t_rad < 0.0: raise ValueError(f"Invalid radial delay_time_GM_c3={delay_time_GM_c3}")
        rule = "absolute_time"
    else: raise ValueError(f"Unsupported UMH_RADIAL_PHASE_MODEL={model!r}; use 'none', 'fractional_time', or 'absolute_time'.")

    t_eff = t_phi + t_rad
    if not math.isfinite(t_eff) or t_eff <= 0.0: raise ValueError(f"Invalid radial delayed t_eff={t_eff}")
    radial_factor = t_phi / t_eff

    return {"model": model, "rule": rule, "chi": float(a), "t_phi_GM_c3": float(t_phi), "t_rad_GM_c3": float(t_rad), "t_eff_GM_c3": float(t_eff),
            "delay_frac_input": float(delay_frac), "radial_frequency_factor": float(radial_factor),
            "note": ("Radial phase delay lowers omega_hat by adding response time. It is not a boundary-softening radius correction.")}


def umh_tau_from_damping_cycles(f_src_Hz, damping_cycles=1.0):
    """
    UMH amplitude-decay timescale from remnant coherence cycles.
    The active UMH interpretation is: tau_src = N_cycles / f_src
    where N_cycles is the number of source-frame oscillation cycles over which the post-merger remnant amplitude decays by one e-fold.
    This gives:
        gamma_src = 1/tau_src
        Q_UMH     = omega_R*tau_src/2 = pi*N_cycles

    This is not the Kerr/QNM Q-factor decay. It is the UMH remnant coherence damping rule.
    """
    f = float(f_src_Hz); N = float(damping_cycles)
    if not math.isfinite(f) or f <= 0.0: raise ValueError(f"Invalid f_src_Hz={f_src_Hz}")
    if not math.isfinite(N) or N <= 0.0: raise ValueError(f"Invalid damping_cycles={damping_cycles}")
    omega_R_src = 2.0 * math.pi * f
    tau_src = N / f; gamma_src = 1.0 / tau_src
    Q_umh = 0.5 * omega_R_src * tau_src
    return {"tau_src_s": float(tau_src), "gamma_src_1_s": float(gamma_src), "Q_umh": float(Q_umh), "damping_cycles": float(N), "damping_rule": "umh_coherence_cycles"}


def umh_eigen_F_chi(chi, m_az=2, branch="corotating", boundary_softening=0.0, radial_phase_model="none", radial_delay_frac=0.0, radial_delay_time_GM_c3=0.0):
    """
    UMH remnant eigenmode spin/tension factor F(chi).
    This is the active UMH-native replacement for using the empirical Kerr/QNM Omega_220 fit constants.

    UMH basis:
      - The remnant is treated as a rotating nonlinear confinement state of the ultronic medium.
      - The dominant post-merger mode is the lowest quadrupolar, phase-locked circulating strain eigenmode, m=2.
      - The wave propagates at the medium wave speed c, while the remnant spin changes the effective confinement path through frame/phase dragging of
        the rotating strain geometry.
      - The real mode frequency is written in UMH form: omega_R = pi*c^3 / (G*Mrem*F_chi)
        so F_chi is the dimensionless spin/tension-winding path factor.

    Interpretation:
      - F_chi is NOT a fitted QNM coefficient.
      - It is the effective dimensionless circumference/phase-lock factor of the dominant trapped m=2 strain mode.
      - The default branch uses the GR-limit rotating confinement geometry because UMH recovers GR macroscopically, but it does not use the
        empirical Kerr/QNM Omega_220 or Q_220 fit formulas.

    Parameters
    ----------
    chi: Final remnant dimensionless spin. Positive values are treated as co-rotating/corotating confinement. The magnitude is clamped below 1.
    m_az: Azimuthal phase-lock number. For the dominant gravitational quadrupole, use m_az=2.
    branch: "corotating" or "counterrotating". For current nonprecessing remnant work, "corotating" is the active/default branch.
    boundary_softening:
        Optional finite-width nonlinear-confinement correction. Keep 0.0 without a UMH remnant simulation/eigenmode result that fixes it.
        Positive values increase F_chi and lower the mode frequency.

    Returns: F_chi, meta (F_chi is dimensionless. meta records the derivation path.)
    """
    a = float(chi)
    if not math.isfinite(a): raise ValueError(f"Invalid chi={chi}")

    # Current nonprecessing remnant path: use spin magnitude for the rotating confinement geometry. Preserve sign handling for future anti-aligned tests.
    a = max(-0.999999, min(a, 0.999999))

    if int(m_az) <= 0: raise ValueError(f"m_az must be positive; got {m_az}")
    m_az = int(m_az)

    branch = str(branch).lower().strip()
    if branch not in ("corotating", "counterrotating"): raise ValueError(f"Unsupported UMH eigen branch={branch!r}; use 'corotating' or 'counterrotating'.")

    # Effective rotating confinement radius in units of GM/c^2.
    # This is the GR-limit rotating strain-guide geometry, used here as the macroscopic limit of the UMH confinement surface. It is NOT the empirical
    # QNM 220 frequency fit. It is the null/strain confinement orbit of the rotating remnant medium.
    # corotating:      r = 2[1 + cos((2/3) arccos(-a))]
    # counterrotating: r = 2[1 + cos((2/3) arccos(+a))]
    if branch == "corotating": r_conf = 2.0 * (1.0 + math.cos((2.0 / 3.0) * math.acos(-a))); spin_term = a
    else: r_conf = 2.0 * (1.0 + math.cos((2.0 / 3.0) * math.acos(a))); spin_term = -a

    # Optional finite-boundary correction from nonlinear confinement thickness.
    # This is left explicit because the current UMH Framework establishes nonlinear confinement and phase-locked modes, but does not yet provide a
    # first-principles closed value for this correction.
    soft = float(boundary_softening)
    if not math.isfinite(soft) or soft < 0.0: raise ValueError(f"Invalid boundary_softening={boundary_softening}")

    r_eff = r_conf + soft

    # Sharp rotating confinement clock.
    # This is the azimuthal m=2 strain-guide time in units GM/c^3.
    t_phi = r_eff ** 1.5 + spin_term
    if not math.isfinite(t_phi) or t_phi <= 0.0: raise ValueError(f"Invalid UMH eigen t_phi={t_phi}; chi={chi}, r_eff={r_eff}, spin_term={spin_term}")

    # Dimensionless angular frequency: omega_hat = omega_R * G*M/c^3
    # for an m_az phase-locked wave circulating around the rotating confinement path. The denominator is the rotating path/time factor.
    omega_hat_base = float(m_az) / t_phi
    if not math.isfinite(omega_hat_base) or omega_hat_base <= 0.0: raise ValueError(f"Invalid UMH base eigen omega_hat={omega_hat_base}; chi={chi}, r_eff={r_eff}, spin_term={spin_term}")

    radial_meta = umh_radial_phase_delay_from_model(t_phi_GM_c3=t_phi, chi=a, model=radial_phase_model, delay_frac=radial_delay_frac, 
                                                    delay_time_GM_c3=radial_delay_time_GM_c3)
    # Delayed effective eigenclock.
    t_eff = float(radial_meta["t_eff_GM_c3"])
    omega_hat = float(m_az) / t_eff
    if not math.isfinite(omega_hat) or omega_hat <= 0.0: raise ValueError(f"Invalid UMH delayed eigen omega_hat={omega_hat}; chi={chi}, t_eff={t_eff}")

    # From omega_R = pi*c^3/(G*M*F_chi):
    #     omega_hat = pi / F_chi
    #     F_chi     = pi / omega_hat
    F_chi = math.pi / omega_hat

    meta = {"F_chi": float(F_chi), "chi": float(a), "m_az": int(m_az), "branch": branch, "r_conf_GM_c2": float(r_conf), "r_eff_GM_c2": float(r_eff),
            "boundary_softening_GM_c2": float(soft), "t_phi_GM_c3": float(t_phi), "t_eff_GM_c3": float(t_eff),
            "omega_hat_base_UMH": float(omega_hat_base), "omega_hat_UMH": float(omega_hat), "radial_frequency_factor": float(omega_hat / omega_hat_base),
            "radial_phase_delay": radial_meta, "rule": "UMH_phase_locked_rotating_confinement_eigenmode_with_optional_radial_delay"}
    return float(F_chi), meta


def umh_omega_fit_F_chi(chi, A=1.5381969714219732, B=-1.1556090096475626, C=0.14022090728188466):
    """
    UMH fitted dimensionless remnant-frequency law. This is not a radial-delay model. It directly defines:
        Omega_UMH(chi) = omega_R * G*M/c^3
                       = A + B*(1 - chi)^C
    The UMH remnant form uses: omega_R = pi*c^3 / (G*M*F_chi), therefore: F_chi = pi / Omega_UMH
    """
    a = float(chi)
    if not math.isfinite(a): raise ValueError(f"Invalid chi={chi}")
    a = max(-0.999999, min(a, 0.999999))
    A = float(A); B = float(B); C = float(C)
    if not math.isfinite(A) or not math.isfinite(B) or not math.isfinite(C): raise ValueError(f"Invalid UMH omega-fit coefficients: A={A}, B={B}, C={C}")
    one_minus_a = max(1.0 - a, 1.0e-12)
    omega_hat = A + B * (one_minus_a ** C)
    if not math.isfinite(omega_hat) or omega_hat <= 0.0: raise ValueError(f"Invalid UMH omega-fit omega_hat={omega_hat}; chi={a}, A={A}, B={B}, C={C}")
    F_chi = math.pi / omega_hat
    meta = {"F_chi": float(F_chi), "chi": float(a), "omega_hat_UMH": float(omega_hat), "omega_hat_base_UMH": float(omega_hat),
            "radial_frequency_factor": 1.0, "omega_fit_coeffs": {"A": float(A), "B": float(B), "C": float(C), "form": "Omega_UMH(chi) = A + B*(1-chi)^C",},
            "radial_phase_delay": {"model": "not_used", "rule": "frequency_law_direct", "t_rad_GM_c3": 0.0,
                                   "note": "UMH omega-fit is a direct frequency law, not a radial-delay model."},
            "rule": "UMH_fitted_dimensionless_remnant_frequency_law"}
    return float(F_chi), meta


def umh_conf_radius_GM_c2(chi):
    """
    UMH rotating confinement / photon-guide radius in x = r/(GM/c^2).
    Same radius rule used by the sharp rotating-confinement eigenmode: r_conf = 2[1 + cos(2/3 arccos(-chi))]
    """
    a = float(chi)
    if not math.isfinite(a): raise ValueError(f"Invalid chi={chi}")
    a = max(-0.999999, min(a, 0.999999))
    return 2.0 * (1.0 + math.cos((2.0 / 3.0) * math.acos(-a)))


def umh_radial_operator_default_params():
    return {"ell": 2, "m": 2, "x_inner_offset": 0.050, "x_outer": 80.0, "x_match": None,
            "tension_amp": 0.00, "density_amp": 0.00, "barrier_amp": 0.12, "barrier_width": 0.90, "barrier_offset": 0.75, "rotation_scale": 1.00,
            "omega_guess": None, "gamma_guess": None, "rtol": 1.0e-8, "atol": 1.0e-10, "maxfev": 80, "debug": False}


def umh_radial_rotation_power_from_chi(chi, params):
    """
    Effective UMH radial rotation-profile exponent.
    Base value is calibrated at chi_ref.  The slope lets the radial confinement/outgoing-boundary profile change with remnant spin.
    """
    base = float(params.get("rotation_power", 1.5))
    if not bool(params.get("rotation_power_chi_enable", False)): return base
    c = float(chi)
    chi_ref = float(params.get("rotation_power_chi_ref", 0.6100942439792432))
    slope = float(params.get("rotation_power_chi_slope", 0.0))
    p_eff = base + slope * (c - chi_ref)
    p_min = float(params.get("rotation_power_min", 0.85))
    p_max = float(params.get("rotation_power_max", 1.50))
    if not math.isfinite(p_eff): p_eff = base
    return float(min(max(p_eff, p_min), p_max))


def umh_radial_profiles_x(x, chi, params):
    """
    Dimensionless UMH remnant radial profiles. x = r / (GM/c^2)
    Lambda(x) = T_eff / T_u
    N(x)      = rho_eff / rho_u
    U(x)      = confinement / curvature barrier term
    Omega(x)  = local rotating-medium angular velocity in GM/c^3 units
    This is the first analytic test profile. Later replace this with profiles extracted from the nonlinear UMH remnant simulation:
        T_eff(r,chi), rho_eff(r,chi), U_eff(r,chi), Omega_U(r,chi).
    """
    x = float(x); a = float(chi); eps = 1.0e-12
    tension_amp = float(params.get("tension_amp", 0.0)); density_amp = float(params.get("density_amp", 0.0))
    barrier_amp = float(params.get("barrier_amp", 0.12)); barrier_width = max(float(params.get("barrier_width", 0.90)), 1.0e-6)
    barrier_offset = float(params.get("barrier_offset", 0.75)); rotation_scale = float(params.get("rotation_scale", 1.0))
    x_conf = umh_conf_radius_GM_c2(a); x_bar = x_conf + barrier_offset; inv_x2 = 1.0 / max(x * x, eps)
    # Effective tension and inertia from inverse-square strain loading.
    Lambda = 1.0 + tension_amp * inv_x2; N = 1.0 + density_amp * inv_x2
    # dLambda/dx for the radial operator.
    dLambda_dx = tension_amp * (-2.0) / max(x**3, eps)
    # Local confinement / strain-curvature barrier.
    z = (x - x_bar) / barrier_width
    U = barrier_amp * math.exp(-0.5 * z * z)
    # Rotating UMH medium clock. This uses the same angular guide rule as the sharp eigenmode, but extended as a radial rotation profile.
    rotation_power = umh_radial_rotation_power_from_chi(a, params)
    denom = (x ** rotation_power) + a
    if denom <= eps: denom = eps
    Omega_U = rotation_scale / denom
    return {"Lambda": float(max(Lambda, eps)), "N": float(max(N, eps)), "dLambda_dx": float(dLambda_dx), "U": float(max(U, 0.0)),
            "Omega_U": float(Omega_U), "x_conf": float(x_conf), "x_barrier": float(x_bar), "rotation_power": float(rotation_power)}


def umh_radial_riccati_rhs(x, y_arr, omega_hat, chi, ell, m, params):
    """
    Riccati/log-derivative form of the radial equation. y = R'/R
    Original radial operator:
        1/x^2 d/dx [ x^2 Lambda dR/dx ]
        + [ N(omega - m Omega_U)^2
            - Lambda ell(ell+1)/x^2
            - U ] R = 0
    Dividing by Lambda: R'' + P R' + Q R = 0
    with:
        P = 2/x + Lambda'/Lambda
        Q = (N/Lambda)(omega - m Omega)^2
            - ell(ell+1)/x^2
            - U/Lambda
    Riccati: y' = -y^2 - P y - Q
    """
    y = y_arr[0]
    prof = umh_radial_profiles_x(float(x), chi, params)
    Lambda = prof["Lambda"]; N = prof["N"]; dLambda_dx = prof["dLambda_dx"]; U = prof["U"]; Omega_U = prof["Omega_U"]
    P = (2.0 / max(float(x), 1.0e-12)) + dLambda_dx / Lambda
    wloc = omega_hat - float(m) * Omega_U
    Q = (N / Lambda) * (wloc * wloc) - float(ell * (ell + 1)) / max(float(x) ** 2, 1.0e-12) - U / Lambda
    dydx = -(y * y) - P * y - Q
    return np.array([dydx], dtype=np.complex128)


def umh_radial_boundary_logderivative(x, omega_hat, chi, ell, m, params, side):
    """
    Boundary log-derivative y = R'/R.
    Convention: deltaPsi ~ R(x) exp(-i omega t)
    Outer outgoing:
        R ~ exp(+i omega r_*)
        y_out = +i omega dr_*/dx
    Inner absorbed/ingoing into rotating remnant:
        R ~ exp(-i(omega - m Omega_core) r_*)
        y_in = -i(omega - m Omega_core) dr_*/dx
    """
    prof = umh_radial_profiles_x(float(x), chi, params)
    drstar_dx = math.sqrt(prof["N"] / prof["Lambda"])
    if side == "outer": return 1j * omega_hat * drstar_dx
    if side == "inner": omega_core = prof["Omega_U"]; return -1j * (omega_hat - float(m) * omega_core) * drstar_dx
    raise ValueError(f"Unknown boundary side={side!r}")


def umh_radial_shooting_residual(vars_real, chi, params):
    """
    Root residual for complex omega_hat.
    vars_real = [omega_R_hat, gamma_hat]
    omega_hat = omega_R_hat - i gamma_hat
    Match log-derivatives at x_match: y_inner(x_match) - y_outer(x_match) = 0
    """
    from scipy.integrate import solve_ivp

    omega_R_hat = float(vars_real[0]); gamma_hat = float(vars_real[1])
    # Keep solver away from unphysical negative frequencies/negative damping.
    if omega_R_hat <= 0.0 or gamma_hat <= 0.0: return np.array([1.0e6 + abs(omega_R_hat), 1.0e6 + abs(gamma_hat)], dtype=float)
    omega_hat = complex(omega_R_hat, -gamma_hat)
    ell = int(params.get("ell", 2)); m = int(params.get("m", 2))
    x_conf = umh_conf_radius_GM_c2(chi)
    x_inner = x_conf + float(params.get("x_inner_offset", 0.050))
    x_outer = float(params.get("x_outer", 80.0))
    x_match_cfg = params.get("x_match", None)
    if x_match_cfg is None: x_match = max(x_inner + 2.0, x_conf + float(params.get("barrier_offset", 0.75)) + 1.0)
    else: x_match = float(x_match_cfg)
    if not (x_inner < x_match < x_outer): raise ValueError(f"Bad radial domain: x_inner={x_inner}, x_match={x_match}, x_outer={x_outer}")
    rtol = float(params.get("rtol", 1.0e-8)); atol = float(params.get("atol", 1.0e-10))
    y0_inner = np.array([umh_radial_boundary_logderivative(x_inner, omega_hat, chi, ell, m, params, side="inner")], dtype=np.complex128)
    sol_in = solve_ivp(fun=lambda x, y: umh_radial_riccati_rhs(x, y, omega_hat, chi, ell, m, params), t_span=(x_inner, x_match), y0=y0_inner,
                       method="DOP853", rtol=rtol, atol=atol)
    if not sol_in.success or sol_in.y.shape[1] < 1: return np.array([1.0e5, 1.0e5], dtype=float)
    y0_outer = np.array([umh_radial_boundary_logderivative(x_outer, omega_hat, chi, ell, m, params, side="outer")], dtype=np.complex128)
    sol_out = solve_ivp(fun=lambda x, y: umh_radial_riccati_rhs(x, y, omega_hat, chi, ell, m, params), t_span=(x_outer, x_match),
                        y0=y0_outer, method="DOP853", rtol=rtol, atol=atol)
    if not sol_out.success or sol_out.y.shape[1] < 1: return np.array([1.0e5, 1.0e5], dtype=float)
    y_inner_match = sol_in.y[0, -1]; y_outer_match = sol_out.y[0, -1]
    residual = y_inner_match - y_outer_match
    return np.array([float(residual.real), float(residual.imag)], dtype=float)


def solve_umh_radial_operator_qnm(chi, params=None):
    """
    Solve the UMH radial outgoing-boundary eigenproblem.
    Returns dimensionless:
        omega_hat_R = Re(omega) GM/c^3
        gamma_hat   = decay rate GM/c^3
        Q           = omega_hat_R / (2 gamma_hat)
        F_chi       = pi / omega_hat_R
    """
    from scipy.optimize import root, least_squares
    p = umh_radial_operator_default_params()
    if params: p.update(params)
    a = float(chi)
    if not math.isfinite(a): raise ValueError(f"Invalid chi={chi}")
    a = max(-0.999999, min(a, 0.999999))
    ell = int(p.get("ell", 2)); m = int(p.get("m", 2))
    x_conf = umh_conf_radius_GM_c2(a); t_phi = (x_conf ** 1.5) + a; omega_eigen_guess = float(m) / t_phi
    omega_guess = p.get("omega_guess", None)
    if omega_guess is None: omega_guess = omega_eigen_guess
    omega_guess = float(omega_guess)
    gamma_guess = p.get("gamma_guess", None)
    if gamma_guess is None: gamma_guess = omega_guess / (2.0 * math.pi) # Current coherence-cycle damping gives Q≈pi, so gamma≈omega/(2*pi).
    gamma_guess = float(gamma_guess)
    # More forgiving than root(...): try nearby frequencies and damping rates.
    start_guesses = []
    for om_fac in (1.00, 0.98, 0.95, 0.92, 0.90, 1.03, 1.06, 1.10):
        for gam_fac in (1.00, 0.75, 0.50, 0.25, 0.10, 1.50, 2.00): 
            start_guesses.append([max(1.0e-8, omega_guess * om_fac), max(1.0e-10, gamma_guess * gam_fac)])
    # Also try the Kerr-like target neighborhood because the UMH radial correction is expected to lower omega by roughly a few percent.
    for om_fac in (0.955, 0.950, 0.945, 0.940):
        for gam_fac in (0.50, 0.75, 1.00, 1.25): start_guesses.append([max(1.0e-8, omega_guess * om_fac), max(1.0e-10, gamma_guess * gam_fac)])
    best_any = None; candidates, valid_candidates = [], []; valid_keys = set()
    omega_min = omega_eigen_guess * float(p.get("omega_min_factor", 0.88))
    omega_max = omega_eigen_guess * float(p.get("omega_max_factor", 1.05))
    gamma_min = gamma_guess * float(p.get("gamma_min_factor", 0.10))
    gamma_max = gamma_guess * float(p.get("gamma_max_factor", 3.00))
    gamma_min = max(gamma_min, 1.0e-10)
    gamma_max = max(gamma_max, gamma_min * 1.01)
    max_nfev = int(p.get("maxfev", 80)) * 20
    resid_tol = float(p.get("resid_tol", 1.0e-5))
    target_ratio = float(p.get("target_ratio", 0.953))
    for x0_try in start_guesses:
        try:
            x0_try = np.array(x0_try, dtype=float)
            if not (omega_min <= x0_try[0] <= omega_max): continue
            if not (gamma_min <= x0_try[1] <= gamma_max): continue
            ls = least_squares(fun=lambda z: umh_radial_shooting_residual(z, a, p), x0=x0_try, bounds=([omega_min, gamma_min], [omega_max, gamma_max]), 
                               xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11, max_nfev=max_nfev)
            if not (omega_min <= ls.x[0] <= omega_max): continue
            if not (gamma_min <= ls.x[1] <= gamma_max): continue
            fun = np.asarray(ls.fun, dtype=float); norm = float(np.linalg.norm(fun))
            omega_tmp = float(ls.x[0]); gamma_tmp = float(ls.x[1])
            ratio_to_eigen = omega_tmp / omega_eigen_guess
            delay_frac = (omega_eigen_guess / omega_tmp) - 1.0
            Q_tmp = omega_tmp / (2.0 * gamma_tmp)
            candidate = {"x": np.asarray(ls.x, dtype=float), "fun": fun, "norm": norm, "success": bool(ls.success), "message": str(ls.message),
                "nfev": int(ls.nfev), "start": np.asarray(x0_try, dtype=float), "ratio_to_eigen": float(ratio_to_eigen),
                "delay_frac": float(delay_frac), "Q": float(Q_tmp), "target_error": float(abs(ratio_to_eigen - target_ratio))}
            candidates.append(candidate)

            if best_any is None or candidate["norm"] < best_any["norm"]: best_any = candidate
            if candidate["success"] and candidate["norm"] < resid_tol:
                # Collapse duplicate roots reached from different starting guesses.
                # Tolerances are intentionally tighter than physics differences we care about.
                root_key = (round(float(omega_tmp), 9), round(float(gamma_tmp), 9))
                is_new_valid_root = root_key not in valid_keys
                if is_new_valid_root:
                    valid_keys.add(root_key); valid_candidates.append(candidate)
                    if bool(p.get("debug", False)):
                        print(f"[UMH_RADIAL_OP_VALID_CANDIDATE] omega={omega_tmp:.9f} gamma={gamma_tmp:.9f} "
                              f"ratio_to_eigen={ratio_to_eigen:.6f} delay_frac={delay_frac:.6f} Q={Q_tmp:.6f} "
                              f"target_error={abs(ratio_to_eigen - target_ratio):.6e} resid_norm={norm:.3e}")
            elif bool(p.get("debug", False)):
                print(f"[UMH_RADIAL_OP] start={candidate['start']} x={candidate['x']} resid={candidate['fun']} norm={candidate['norm']:.6e} "
                      f"success={candidate['success']}")
        except Exception as exc:
            if bool(p.get("debug", False)): print(f"[UMH_RADIAL_OP] start={x0_try} failed: {exc}")

    if best_any is None: raise RuntimeError("UMH radial-operator solve failed before producing any candidate.")
    if not valid_candidates:
        raise RuntimeError(f"UMH radial-operator solve did not converge to a valid outgoing-boundary mode. "
                            f"best_message={best_any['message']}; start omega={omega_guess:.12g}, gamma={gamma_guess:.12g}; "
                            f"best_start={best_any['start']}; best_x={best_any['x']}; residual={best_any['fun']}; "
                            f"residual_norm={best_any['norm']:.6e}; nfev={best_any['nfev']}. "
                            f"This usually means the current analytic placeholder radial profile does not yet support "
                            f"a clean trapped/outgoing QNM-like resonance inside the selected search band.")

    # Select the valid mode closest to the desired fundamental correction. Use residual norm as a secondary tie-breaker.
    best = min(valid_candidates, key=lambda c: (c["target_error"], c["norm"]))
    omega_R_hat = float(best["x"][0]); gamma_hat = float(best["x"][1])
    if not math.isfinite(omega_R_hat) or omega_R_hat <= 0.0: raise RuntimeError(f"Invalid radial operator omega_R_hat={omega_R_hat}")
    if not math.isfinite(gamma_hat) or gamma_hat <= 0.0: raise RuntimeError(f"Invalid radial operator gamma_hat={gamma_hat}")
    print(f"[UMH_RADIAL_OP_SELECTED] omega={omega_R_hat:.9f} gamma={gamma_hat:.9f} ratio_to_eigen={best['ratio_to_eigen']:.6f} "
          f"delay_frac={best['delay_frac']:.6f} Q={best['Q']:.6f} target_ratio={target_ratio:.6f} target_error={best['target_error']:.6e} "
          f"resid_norm={best['norm']:.3e} unique_valid_count={len(valid_candidates)} raw_candidate_count={len(candidates)}")

    Q = omega_R_hat / (2.0 * gamma_hat); F_chi = math.pi / omega_R_hat
    x_inner = x_conf + float(p.get("x_inner_offset", 0.050)); x_outer = float(p.get("x_outer", 80.0)); x_match_cfg = p.get("x_match", None)
    if x_match_cfg is None: x_match = max(x_inner + 2.0, x_conf + float(p.get("barrier_offset", 0.75)) + 1.0)
    else: x_match = float(x_match_cfg)

    rotation_power_eff = umh_radial_rotation_power_from_chi(a, p)

    meta = {"rule": "UMH_radial_variable_coefficient_outgoing_boundary_operator", "chi": float(a), "ell": int(ell), "m": int(m),
            "omega_hat_UMH": float(omega_R_hat), "gamma_hat_UMH": float(gamma_hat), "Q_umh": float(Q), "F_chi": float(F_chi),
            "omega_hat_base_UMH": float(omega_eigen_guess), "omega_hat_sharp_eigen_guess": float(omega_eigen_guess), 
            "ratio_radial_to_sharp_eigen": float(omega_R_hat / omega_eigen_guess),
            "target_ratio": float(target_ratio), "target_error": float(best["target_error"]), "delay_frac": float(best["delay_frac"]),
            "unique_valid_count": int(len(valid_candidates)), "raw_candidate_count": int(len(candidates)),
            "x_conf_GM_c2": float(x_conf), "x_inner": float(x_inner), "x_match": float(x_match), "x_outer": float(x_outer),
            "profiles": {"tension_amp": float(p.get("tension_amp", 0.0)), "density_amp": float(p.get("density_amp", 0.0)),
                         "barrier_amp": float(p.get("barrier_amp", 0.12)), "barrier_width": float(p.get("barrier_width", 0.90)),
                         "barrier_offset": float(p.get("barrier_offset", 0.75)), "rotation_scale": float(p.get("rotation_scale", 1.0)),
                         "rotation_power_base": float(p.get("rotation_power", 1.5)),
                         "rotation_power_effective": float(rotation_power_eff),
                         "rotation_power_chi_enable": bool(p.get("rotation_power_chi_enable", False)),
                         "rotation_power_chi_ref": float(p.get("rotation_power_chi_ref", 0.6100942439792432)),
                         "rotation_power_chi_slope": float(p.get("rotation_power_chi_slope", 0.0)),
                         "rotation_power_min": float(p.get("rotation_power_min", 0.85)),
                         "rotation_power_max": float(p.get("rotation_power_max", 1.50)),
                         },
            "root": {"success": bool(best["success"]), "message": str(best["message"]), "nfev": int(best["nfev"]), "start": best["start"].tolist(),
                     "residual_real": float(best["fun"][0]), "residual_imag": float(best["fun"][1]), "residual_norm": float(best["norm"])}}

    return {"omega_hat_R": float(omega_R_hat), "gamma_hat": float(gamma_hat), "Q_umh": float(Q), "F_chi": float(F_chi), "meta": meta}


def umh_radial_fit_delay_frac(chi, rad_chi_ref=0.6100942439792432, rad_chi_d0=0.04780931608484118, rad_chi_d1=0.03586007393884942,
                              rad_chi_dmin=0.00, rad_chi_dmax=0.20):
    chi = float(chi)
    delay = rad_chi_d0 + rad_chi_d1 * (chi - rad_chi_ref)
    if not math.isfinite(delay): delay = rad_chi_d0
    return float(min(max(delay, rad_chi_dmin), rad_chi_dmax))


def umh_radial_fit_frequency_factor(chi, rad_chi_ref=0.6100942439792432, rad_chi_d0=0.04780931608484118, rad_chi_d1=0.03586007393884942,
                                    rad_chi_dmin=0.00, rad_chi_dmax=0.20): 
    delay = umh_radial_fit_delay_frac(chi, rad_chi_ref, rad_chi_d0, rad_chi_d1, rad_chi_dmin, rad_chi_dmax)
    return 1.0 / (1.0 + delay)


def umh_remnant_mode_from_Mchi(Mrem_kg_src, chi_rem, freq_model="umh_eigen", damping_model="umh_cycles", damping_cycles=1.0, boundary_softening=0.0, 
                               radial_phase_model="none", radial_delay_frac=0.0, radial_delay_time_GM_c3=0.0, 
                               omega_fit_A=1.5381969714219732, omega_fit_B=-1.1556090096475626, omega_fit_C=0.14022090728188466, 
                               radial_operator_params=None, rad_chi_ref=0.6100942439792432, rad_chi_d0=0.04780931608484118, rad_chi_d1=0.03586007393884942,
                               rad_chi_dmin=0.00, rad_chi_dmax=0.20, G_phys=6.67430e-11, c_phys=2.99792458e8):
    """
    UMH-native remnant relaxation-mode seed.
    Active UMH form: omega_R = pi*T_u*L^2 / (Mrem*c*F_chi)
    Using the UMH macroscopic identity: T_u*L^2 = c^4/G
    gives: omega_R = pi*c^3 / (G*Mrem*F_chi)
    In freq_model='umh_eigen', F_chi is computed from the UMH rotating confinement eigenmode rule, not from empirical Kerr/QNM 220 coefficients.
    Damping is intentionally separate. The active/default UMH damping model can remain 'umh_cycles', because the GR/Kerr Q-factor decay is only a compatibility
    diagnostic unless independently derived from UMH.
    """
    M = float(Mrem_kg_src)
    if not math.isfinite(M) or M <= 0.0: raise ValueError(f"Invalid Mrem_kg_src={Mrem_kg_src}")
    freq_model = str(freq_model).lower().strip();  damping_model = str(damping_model).lower().strip()

    M_geom_time = G_phys * M / (c_phys ** 3)
    radial_operator_solution, radial_operator_gamma_hat = None, None
    if freq_model == "umh_eigen":
        F_chi, shape_meta = umh_eigen_F_chi(chi=chi_rem, m_az=2, branch="corotating", boundary_softening=boundary_softening,
            radial_phase_model=radial_phase_model, radial_delay_frac=radial_delay_frac, radial_delay_time_GM_c3=radial_delay_time_GM_c3)
        freq_rule = "umh_eigen_F_chi_phase_locked_confinement"
        omega_hat_R = math.pi / float(F_chi)
    elif freq_model in ("umh_omega_fit", "umh_frequency_law", "umh_coeff_fit"):
        F_chi, shape_meta = umh_omega_fit_F_chi(chi=chi_rem, A=omega_fit_A, B=omega_fit_B, C=omega_fit_C)
        freq_rule = "umh_fitted_dimensionless_frequency_law"
        omega_hat_R = math.pi / float(F_chi)
    elif freq_model in ("umh_radial_operator", "umh_shooting", "umh_radial_qnm"):
        radial_operator_solution = solve_umh_radial_operator_qnm(chi=chi_rem, params=radial_operator_params)
        F_chi = float(radial_operator_solution["F_chi"])
        omega_hat_R = float(radial_operator_solution["omega_hat_R"])
        radial_operator_gamma_hat = float(radial_operator_solution["gamma_hat"])
        shape_meta = radial_operator_solution["meta"]
        freq_rule = "umh_radial_operator_outgoing_boundary_complex_eigenfrequency"
    elif freq_model == "umh_radial_fit":
        F_chi_sharp, sharp_meta = umh_eigen_F_chi(chi=chi_rem, m_az=2, branch="corotating", boundary_softening=boundary_softening,
            radial_phase_model="None", radial_delay_frac=0.0, radial_delay_time_GM_c3=0.0)
        omega_hat_sharp = math.pi / float(F_chi_sharp)
        radial_factor = umh_radial_fit_frequency_factor(chi_rem, rad_chi_ref=rad_chi_ref, rad_chi_d0=rad_chi_d0, rad_chi_d1=rad_chi_d1,
            rad_chi_dmin=rad_chi_dmin, rad_chi_dmax=rad_chi_dmax)
        delay_frac = umh_radial_fit_delay_frac(chi_rem, rad_chi_ref=rad_chi_ref, rad_chi_d0=rad_chi_d0, rad_chi_d1=rad_chi_d1,
            rad_chi_dmin=rad_chi_dmin, rad_chi_dmax=rad_chi_dmax)
        omega_hat_R = omega_hat_sharp * radial_factor
        F_chi = math.pi / omega_hat_R
        freq_rule = "umh_radial_surrogate"
        shape_meta = {"rule": "UMH_radial_operator_surrogate", "chi": float(chi_rem), 
                      "omega_hat_base_UMH": float(omega_hat_sharp), "omega_hat_sharp_eigen_guess": float(omega_hat_sharp), 
                      "omega_hat_UMH": float(omega_hat_R), "ratio_radial_to_sharp_eigen": float(radial_factor), 
                      "radial_frequency_factor": float(radial_factor), "delay_frac": float(delay_frac), "F_chi_sharp": float(F_chi_sharp), 
                      "F_chi": float(F_chi), "chi_ref": float(rad_chi_ref), "delay_ref": float(rad_chi_d0), "delay_chi_slope": float(rad_chi_d1),
                      "delay_min": float(rad_chi_dmin), "delay_max": float(rad_chi_dmax), "sharp_meta": sharp_meta,
                      "note": "Fast surrogate for the solved UMH radial outgoing-boundary operator. QNM is not used as the active seed."}
    else: raise ValueError(f"Unsupported UMH remnant freq_model={freq_model!r}; use 'umh_eigen' or 'umh_omega_fit'.")

    omega_R_src = omega_hat_R / M_geom_time
    f_src = omega_R_src / (2.0 * math.pi)

    omega_hat_base = shape_meta.get("omega_hat_base_UMH", None)
    if omega_hat_base is None: omega_hat_base = shape_meta.get("omega_hat_sharp_eigen_guess", None)
    if omega_hat_base is None and isinstance(shape_meta.get("sharp_meta", None), dict): omega_hat_base = shape_meta["sharp_meta"].get("omega_hat_base_UMH", None)
    if omega_hat_base is None: omega_hat_base = shape_meta.get("omega_hat_UMH", None)
    if omega_hat_base is None: raise ValueError(f"shape_meta missing omega_hat information: {shape_meta}")
    omega_hat_base = float(omega_hat_base)
    omega_R_base_src = omega_hat_base * c_phys**3 / (G_phys * M)
    f_src_no_radial_delay = omega_R_base_src / (2.0 * math.pi)

    tau_src, gamma_src, damping_rule, Q_umh = None, None, None, None
    if damping_model in ("umh_radial_operator", "from_radial_operator", "radial_operator"):
        if radial_operator_gamma_hat is None: raise ValueError("damping_model='umh_radial_operator' requires freq_model='umh_radial_operator'.")
        gamma_src = radial_operator_gamma_hat / M_geom_time
        tau_src = 1.0 / gamma_src
        Q_umh = omega_hat_R / (2.0 * radial_operator_gamma_hat)
        damping_rule = "umh_radial_operator_complex_decay"
    elif damping_model == "umh_cycles":
        decay = umh_tau_from_damping_cycles(f_src_Hz=f_src, damping_cycles=damping_cycles)
        tau_src = decay["tau_src_s"]; gamma_src = decay["gamma_src_1_s"]; Q_umh = decay["Q_umh"]; damping_rule = decay["damping_rule"]
    elif damping_model == "defer": damping_rule = "defer_to_shared_tau_block"
    else: raise ValueError(f"Unsupported damping_model={damping_model!r}")

    return {"f_src_Hz": float(f_src), "omega_R_src_rad_s": float(omega_R_src), "tau_src_s": None if tau_src is None else float(tau_src),
            "gamma_src_1_s": None if gamma_src is None else float(gamma_src), "Q_umh": None if Q_umh is None else float(Q_umh), 
            "F_chi": float(F_chi), "freq_rule": str(freq_rule), "damping_rule": str(damping_rule), "shape_meta": shape_meta,
            "f_src_no_radial_delay_Hz": float(f_src_no_radial_delay), 
            "radial_frequency_factor": float(shape_meta.get("ratio_radial_to_sharp_eigen", shape_meta.get("radial_frequency_factor", 1.0)))}


# ------------------------------------------------------------
# QNM (ℓ,m,n) = (2,2,0) fits (Berti–Cardoso–Will style)
# Reference form:
#   2π (G M_f / c^3) f_rd  =  f1 + f2 (1 - a_f)^{f3}
#   Q = π f_rd τ           =  q1 + q2 (1 - a_f)^{q3}
#
# Notes
# - These are the standard “220” Kerr QNM fits widely used in GW data analysis. They are calibrated to numerical-relativity
#   BH perturbation theory; they do not depend on UMH directly.
# - The only physics inputs are the remnant mass M_f and dimensionless spin a_f = J c / (G M_f^2).
# - Units: M_f in kilograms, we convert to geometric time M_geom = G M_f / c^3 (seconds).
# - Returns detector-frame (f_rd, tau).
# ------------------------------------------------------------
def qnm_22n_modes_from_Ma(Mrem_kg_src, a_rem_src, overtones=(0, 1, 2), G_phys=6.67430e-11, c_phys=2.99792458e8):
    """
    Return dict: n -> (f_22n_Hz, tau_22n_sec) for (l,m)=(2,2), n in {0,1,2}
    using Berti+ fits for (M * omega_R) and Q(a).
    """
    # Geometric mass time-scale: M_geom_time = G M / c^3   [seconds]
    M_geom_time = G_phys * Mrem_kg_src / (c_phys ** 3)

    # Berti et al. (l=m=2) fits; coefficients carried as floats
    # M * omega_R = f1 + f2 * (1 - a) ** f3
    # Q = q1 + q2 * (1 - a) ** q3
    coeffs = {
        0: dict(
            f1=1.5251000000000000, f2=-1.1568000000000000, f3=0.1292000000000000,
            q1=0.7000000000000000, q2= 1.4187000000000000, q3=-0.4990000000000000
        ),
        1: dict(
            f1=1.7067000000000000, f2=-0.9783000000000000, f3=0.1526000000000000,
            q1=0.5743000000000000, q2= 1.6845000000000000, q3=-0.4955000000000000
        ),
        2: dict(
            f1=1.9227000000000000, f2=-0.8833000000000000, f3=0.1630000000000000,
            q1=0.4900000000000000, q2= 1.8360000000000000, q3=-0.4920000000000000
        ),
    }

    modes = {}
    one_minus_a = 1.0 - float(a_rem_src)

    for n in overtones:
        c = coeffs[n]
        M_omega_R = c["f1"] + c["f2"] * (one_minus_a ** c["f3"])  # dimensionless
        Q = c["q1"] + c["q2"] * (one_minus_a ** c["q3"])          # dimensionless

        # Physical angular frequency: omega_R_phys = (M_omega_R / M_geom_time)
        omega_R_phys = M_omega_R / M_geom_time  # [rad/s]
        f_Hz = omega_R_phys / (2.0 * math.pi)   # [Hz]

        # Damping time: tau = 2 Q / omega_R_phys (consistent with Q = omega_R * tau / 2)
        tau_sec = (2.0 * Q) / omega_R_phys

        modes[n] = (f_Hz, tau_sec)

    return modes


# non-negative overtone weights
def compute_overtone_weights(A_attach, modes_dict, use_continuity_fit=True, decay_ratio=0.7):
    ns = sorted(modes_dict.keys())
    N  = len(ns)
    # geometric baseline (strictly positive)
    base = np.array([decay_ratio**i for i in range(N)], float)
    base /= base.sum()
    A0 = A_attach * base
    if not use_continuity_fit or N < 2: return A0
    # light LS toward baseline, then project to nonnegative and renormalize
    M = np.ones((1, N))
    b = np.array([A_attach], float)
    ridge = 1e-4 * np.diag(1.0 / (1 + np.arange(N)))
    A_fit = np.linalg.lstsq(np.vstack([M, ridge]), np.concatenate([b, ridge @ (A0)]), rcond=None)[0]
    A_fit = np.maximum(A_fit, 0.0)
    s = A_fit.sum()
    return (A_fit / s) * A_attach if s > 0 else A0

# complex QNM sum + analytic instantaneous frequency
def synthesize_qnm_sum_and_freq(t, t_attach, A_attach, phi_attach, modes_dict, A_n):
    t = np.asarray(t, float)
    A_rd   = np.zeros_like(t, float)
    phi_rd = np.zeros_like(t, float)
    f_inst = np.zeros_like(t, float)

    mask = (t >= t_attach)
    if not np.any(mask): return A_rd, phi_rd, f_inst

    dt = t[mask] - float(t_attach)
    ns = sorted(modes_dict.keys())
    omegas = np.array([2.0*np.pi*modes_dict[n][0] for n in ns], float)  # rad/s
    taus   = np.array([modes_dict[n][1]         for n in ns], float)
    kappa  = -1.0/taus + 1j*omegas

    E    = np.exp(dt[:, None] * kappa[None, :]) * np.exp(1j * phi_attach)
    x    = (E * A_n[None, :]).sum(axis=1)
    xdot = (E * (A_n * kappa)[None, :]).sum(axis=1)

    A_seg   = np.abs(x)
    phi_seg = np.unwrap(np.angle(x))
    safe = A_seg > (1e-12 * np.max(A_seg))
    omega_inst   = np.zeros_like(A_seg)
    omega_inst[safe] = np.imag(xdot[safe] / x[safe])
    f_seg = omega_inst / (2.0 * np.pi)

    A_rd[mask]   = A_seg
    phi_rd[mask] = phi_seg
    f_inst[mask] = f_seg
    return A_rd, phi_rd, f_inst
# --- End Calculate Ringdown ---


# --- Calculate Phase Accuracy ---
def estimate_hilbert_inst_freq(y, dt, lowcut=20.0, highcut=512.0, taper_edge_sec=0.05, amp_frac_thresh=0.02):
    """
    Robust instantaneous-frequency estimate for diagnostics ONLY.

    - Operates on a COPY of y.
    - Bandpasses, tapers edges, uses Hilbert transform, and clamps outliers.
    - Returns (t, f_inst) with same dt as input.
    """
    y = np.asarray(y, float)
    N = len(y)
    if N < 8: return np.arange(N) * dt, np.zeros(N, float)

    fs = 1.0 / dt
    nyq = 0.5 * fs

    # Light bandpass to isolate the chirp band
    lo = max(lowcut, 0.1)
    hi = min(highcut, nyq * 0.98)
    if lo < hi:
        b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
        sig = filtfilt(b, a, y)
    else: sig = y.copy()

    # Amplitude-based soft mask (avoid crazy phase where signal ~ 0)
    abs_y = np.abs(sig)
    thr = amp_frac_thresh * np.max(abs_y) + 1e-15
    mask = abs_y > thr
    if not np.any(mask): mask[:] = True

    # Edge tapers around significant region to reduce boundary artifacts
    sig = sig.astype(float, copy=True)
    tailN = int(max(3, min(N // 20, taper_edge_sec / dt)))

    i0 = int(np.argmax(mask))
    i1 = int(len(mask) - np.argmax(mask[::-1]) - 1)

    # Fade-in before i0
    if i0 > 0:
        n = min(tailN, i0)
        w = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
        sig[i0 - n:i0] *= w

    # Fade-out after i1
    if i1 < N - 1:
        n = min(tailN, (N - 1 - i1))
        w = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
        sig[i1 + 1:i1 + 1 + n] *= w[::-1]

    # Hilbert analytic signal → phase → instantaneous frequency
    a = hilbert(sig)
    ph = np.unwrap(np.angle(a))
    f = np.diff(ph) / (2.0 * np.pi * dt)
    if len(f) == 0: f = np.zeros_like(sig)
    else: f = np.concatenate([f, [f[-1]]])

    # Clamp insane spikes using median + MAD, only for positive freqs
    pos = f[f > 0.0]
    if len(pos):
        hi_clip = np.percentile(pos, 99.5)   # allows late chirp rise
        hi_clip = min(max(hi_clip, lo + 1.0), hi)
        f = np.clip(f, lo, hi_clip)
    else: f = np.clip(f, lo, hi)

    # Light smoothing for readability (no physics impact; diag only)
    if len(f) >= 13: win = 13; f = savgol_filter(f, win, 3, mode='interp')

    t = np.arange(N) * dt
    return t, f

#Diagnostic Only.
def quantify_phase_accuracy(reference_detector, dt, t_array, strain_records,
                            phase_hist_use, freq_record_use, lowcut, highcut,
                            Fp=None, Fx=None, iota=None, phi0_det=None,
                            amp_threshold=0.05, taper_edge_sec=0.05, bp_order=4):
    """
    Compare analytic vs numeric (and Hilbert) phases, robust to src-vs-observer bookkeeping.
    """
    results = {}

    # --- Inputs ---
    y_ref = np.asarray(strain_records[reference_detector], dtype=float).copy()
    t_array = np.asarray(t_array, dtype=float)
    n = len(t_array)

    # --- Analytic phase from generation history ---
    phi_analytic = np.asarray(phase_hist_use, dtype=float).copy()
    if len(phi_analytic) != n: phi_analytic = np.interp(t_array, np.arange(len(phi_analytic)) * dt, phi_analytic)
    phi_analytic = np.unwrap(phi_analytic)

    # --- Frequency track ---
    f_track = np.asarray(freq_record_use, dtype=float).copy()
    if len(f_track) != n: f_track = np.interp(t_array, np.arange(len(f_track)) * dt, f_track)

    # --- Bandpass + taper BEFORE Hilbert (stabilizes Hilbert phase/frequency) ---
    fs = 1.0 / float(dt)
    nyq = 0.5 * fs
    lo = max(float(lowcut), 0.1)
    hi = min(float(highcut), nyq * 0.98)

    if lo < hi and n > 8:
        b, a = butter(int(bp_order), [lo / nyq, hi / nyq], btype='band')
        y_bp = filtfilt(b, a, y_ref)
    else: y_bp = y_ref.copy()

    # Edge taper to suppress Hilbert edge artifacts
    n_edge = int(max(1, round(float(taper_edge_sec) / float(dt))))
    if 2 * n_edge < n:
        w = np.ones(n, dtype=float)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_edge))
        w[:n_edge] *= ramp
        w[-n_edge:] *= ramp[::-1]
        y_bp = y_bp * w

    # --- Hilbert phase on conditioned reference signal ---
    analytic_signal = hilbert(y_bp)
    phi_hilbert_raw = np.unwrap(np.angle(analytic_signal))

    # High-SNR mask
    amp = np.abs(analytic_signal)
    amp_max = amp.max() if amp.size and amp.max() != 0.0 else 1.0
    amp_norm = amp / amp_max
    hilbert_mask = (amp_norm > float(amp_threshold))
    if np.count_nonzero(hilbert_mask) < 10: hilbert_mask = np.ones_like(amp_norm, dtype=bool)

    # --- Infer how f_track relates to phi_analytic on THIS dt grid ---
    dphi_dt = np.gradient(phi_analytic, dt)
    f_from_phi = dphi_dt / (2.0 * np.pi)

    mask_scale = hilbert_mask & np.isfinite(f_track) & np.isfinite(f_from_phi)
    mask_scale &= (np.abs(f_from_phi) > 1e-6) & (f_track > 0)

    inferred_scale = np.nan
    if np.count_nonzero(mask_scale) > 50:
        ratio = f_track[mask_scale] / f_from_phi[mask_scale]
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size > 0: inferred_scale = float(np.median(ratio))

    if np.isfinite(inferred_scale) and (0.1 < inferred_scale < 10.0): f_track_eff = f_track / inferred_scale
    else: f_track_eff = f_track.copy()

    results["inferred_ftrack_scale_vs_dphi"] = float(inferred_scale) if np.isfinite(inferred_scale) else np.nan

    # --- Numeric phase reconstructed from corrected frequency (TRAPEZOID + align) ---
    if n > 1:
        f_mid = 0.5 * (f_track_eff[:-1] + f_track_eff[1:])
        phi_numeric = np.empty(n, dtype=float)
        phi_numeric[0] = 0.0
        phi_numeric[1:] = 2.0 * np.pi * np.cumsum(f_mid) * dt
    else: phi_numeric = np.zeros(n, dtype=float)

    phi_numeric = np.unwrap(phi_numeric)

    # Align numeric phase to analytic by constant offset over mask
    offset_num = np.mean(phi_numeric[hilbert_mask] - phi_analytic[hilbert_mask])
    phi_numeric_aligned = phi_numeric - offset_num

    # --- Build DETECTOR phase reference for Hilbert comparison ---
    # If you provide phi0_det directly, use it.
    # Else, if (Fp, Fx, iota) are provided, compute a constant phi0.
    # Otherwise, fall back to comparing against phi_analytic (less correct).
    if phi0_det is None and (Fp is not None) and (Fx is not None) and (iota is not None):
        ci = float(np.cos(iota))
        Aplus = 0.5 * (1.0 + ci * ci)
        Across = ci
        phi0_det = float(np.arctan2(float(Fx) * Across, float(Fp) * Aplus))

    if phi0_det is not None:
        phi_ref_for_hilbert = phi_analytic + float(phi0_det)
        # unwrap for consistency
        phi_ref_for_hilbert = np.unwrap(phi_ref_for_hilbert)
    else: phi_ref_for_hilbert = phi_analytic                # fallback (not ideal): intrinsic phase

    # --- Align Hilbert phase to detector reference allowing sign ambiguity ---
    phi_ref_m = phi_ref_for_hilbert[hilbert_mask]
    phi_h_m   = phi_hilbert_raw[hilbert_mask]

    best = None
    for s in (+1.0, -1.0):
        c = np.mean(s * phi_h_m - phi_ref_m)
        phi_h_aligned = s * phi_hilbert_raw - c
        res = np.angle(np.exp(1j * (phi_ref_m - phi_h_aligned[hilbert_mask])))
        rms = float(np.sqrt(np.mean(res ** 2)))
        if (best is None) or (rms < best[0]): best = (rms, s, c, phi_h_aligned)

    phi_hilbert = best[3]
    results["hilbert_align_sign"] = float(best[1])

    # --- Instantaneous freq diagnostic & correlation (use SAME dt and corrected f_track_eff) ---
    t_hilb, f_hilb = estimate_hilbert_inst_freq(y_ref, dt, lowcut=lowcut, highcut=highcut)
    if len(t_hilb) > 1 and len(f_hilb) > 1:
        f_model_on_hilb = np.interp(t_hilb, t_array, f_track_eff)

        if len(hilbert_mask) == len(f_hilb): mask_corr = hilbert_mask & np.isfinite(f_hilb) & np.isfinite(f_model_on_hilb)
        else: mask_corr = np.isfinite(f_hilb) & np.isfinite(f_model_on_hilb)

        if np.count_nonzero(mask_corr) > 10:
            f_m = f_model_on_hilb[mask_corr]
            f_h = f_hilb[mask_corr]
            if np.std(f_m) > 0.0 and np.std(f_h) > 0.0:
                corr = np.corrcoef(f_m, f_h)[0, 1]
                print(f"[UMH Diagnostics][Ref:{reference_detector}] Correlation between f_model and f_Hilbert (high-SNR window): {corr:.4f}")
            else: print(f"[UMH Diagnostics][Ref:{reference_detector}] Correlation not computed (zero variance in high-SNR window).")
        else: print(f"[UMH Diagnostics][Ref:{reference_detector}] Correlation not computed (insufficient high-SNR Hilbert samples).")
    else: print(f"[UMH Diagnostics][Ref:{reference_detector}] Correlation between f_model and f_Hilbert: insufficient Hilbert samples")

    # --- Phase residuals robustly in [-pi, pi] ---
    phi_an_phase  = phi_analytic[hilbert_mask]
    phi_num_phase = phi_numeric_aligned[hilbert_mask]
    phi_hil_phase = phi_hilbert[hilbert_mask]
    phi_ref_phase = phi_ref_for_hilbert[hilbert_mask]

    dphi_an_num = np.angle(np.exp(1j * (phi_an_phase - phi_num_phase)))
    results["max_err_an_num"] = float(np.max(np.abs(dphi_an_num)))
    results["rms_err_an_num"] = float(np.sqrt(np.mean(dphi_an_num ** 2)))

    # Hilbert residual should be against detector reference (if available)
    dphi_ref_hil = np.angle(np.exp(1j * (phi_ref_phase - phi_hil_phase)))
    results["max_err_an_hil"] = float(np.max(np.abs(dphi_ref_hil)))
    results["rms_err_an_hil"] = float(np.sqrt(np.mean(dphi_ref_hil ** 2)))

    print("[UMH Diagnostics] Phase consistency check:")
    for k, v in results.items():
        if isinstance(v, float): print(f"  {k:26s}: {v:.6e}")
        else: print(f"  {k:26s}: {v}")

    return results


# --- Detector-weighted phase QA -------------------------------
def detector_weighted_phase_check(dt, t_array, strain_records, phase_hist_use, geom_delays, amp_threshold=0.05, 
                                                               lowcut=20.0, highcut=512.0, taper_edge_sec=0.05):
    """
    Compute detector-weighted phase residuals vs the analytic phase history.
    Weighting: w_i(t) = |Hilbert(h_det_i(t))|^2 (energy-like), normalized.
    We remove a per-detector constant phase offset before measuring the residuals.
    """
    # Analytic reference phase (source track) on generator grid
    phi_ref_src = np.asarray(phase_hist_use, dtype=float).copy()
    if len(phi_ref_src) != len(t_array): phi_ref_src = np.interp(t_array, np.arange(len(phi_ref_src)) * dt, phi_ref_src)
    phi_ref_src = np.unwrap(phi_ref_src)

    out = {"per_detector": {}, "combined": {"weighted_rms_rad": np.nan, "total_weight": 0.0}}
    #def _integrate(y, x): return float(np.trapz(y, x)) # trapz with float output
    def _integrate(y, x): #return float(np.trapezoid(y, x))  # trapezoid with float output
        if hasattr(np, "trapezoid"): return float(np.trapezoid(y, x))
        else: return float(np.trapz(y, x))
        
    num_accum = 0.0; den_accum = 0.0
    for det_name, y in strain_records.items():
        y = np.asarray(y, dtype=float)

        # --- Shift the reference phase by this detector's delay ---
        tau = float(geom_delays.get(det_name, 0.0))
        # evaluate phi_ref at (t - tau); out-of-range edges are held (nearest)
        t_shift = t_array - tau
        phi_ref = np.interp(t_shift, t_array, phi_ref_src, left=phi_ref_src[0], right=phi_ref_src[-1])

        # Bandpass + edge taper before Hilbert phase extraction (stabilizes phase, esp. when |h| is small)
        fs = 1.0 / float(dt)
        nyq = 0.5 * fs
        lo = max(float(lowcut), 0.1)
        hi = min(float(highcut), nyq * 0.98)
        if lo < hi:
            b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
            y_f = filtfilt(b, a, y)
        else: y_f = y.copy()

        # Soft edge taper to reduce Hilbert edge artifacts
        n_edge = int(max(1, round(float(taper_edge_sec) / float(dt))))
        if 2 * n_edge < len(y_f):
            w = np.ones_like(y_f)
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_edge))
            w[:n_edge] *= ramp
            w[-n_edge:] *= ramp[::-1]
            y_f = y_f * w

        # Hilbert analytic signal for amplitude weighting and recovered phase
        h = hilbert(y_f)
        amp = np.abs(h)
        amp_max = amp.max() if amp.size and amp.max() != 0.0 else 1.0
        mask = (amp / amp_max) > float(amp_threshold)
        if np.count_nonzero(mask) < 10: mask = np.ones_like(amp, dtype=bool)  # robustness fallback

        phase_rec = np.unwrap(np.angle(h))

        # Remove a single constant offset (align detector phase to reference)
        offset = np.mean((phase_rec - phi_ref)[mask])
        phase_resid = (phase_rec - offset) - phi_ref  # residual in radians

        w = amp**2; w_masked = w[mask]; r_masked = phase_resid[mask]; t_masked = t_array[mask]

        W   = _integrate(w_masked, t_masked) if np.any(mask) else 1.0
        num = _integrate(w_masked * (r_masked**2), t_masked)
        w_rms = np.sqrt(num / W) if W > 0.0 else np.nan
        w_max = float(np.max(np.abs(r_masked))) if r_masked.size else np.nan

        out["per_detector"][det_name] = {"weighted_rms_rad": float(w_rms),
            "weighted_max_rad": float(w_max), "total_weight": float(W),
            "amp_threshold": float(amp_threshold), "geom_delay_sec": tau,}

        num_accum += num; den_accum += W

    if den_accum > 0.0:
        out["combined"]["weighted_rms_rad"] = float(np.sqrt(num_accum / den_accum))
        out["combined"]["total_weight"]     = float(den_accum)

    print("[UMH QA] Detector-weighted phase residuals (radians):")
    for k, v in out["per_detector"].items():
        print(f"  {k:10s}  RMS={v['weighted_rms_rad']:.4f}  MAX={v['weighted_max_rad']:.4f}  W={v['total_weight']:.3e}  tau={v['geom_delay_sec']*1e3:.3f} ms")
    print(f"  {'COMBINED':10s}  RMS={out['combined']['weighted_rms_rad']:.4f}  Wtot={out['combined']['total_weight']:.3e}")

    return out
# --- End: detector-weighted phase QA ------------------------------------

# --- End Calculate Phase Accuracy ---


# --- Visual Enhancements Not Applied to Physics ---
def _aligodesign_asd(f: np.ndarray) -> np.ndarray:
    """
    Approximate aLIGO design amplitude spectral density (Hanford-like), in 1/sqrt(Hz).
    Valid-ish over ~10–2000 Hz. Outside that, we just clamp.
    """
    PSD_NORM  = 1e-49       # physical normalization

    f = np.asarray(f, dtype=float)
    # Avoid division by zero
    f = np.maximum(f, 1.0)

    # Analytic fit (Ajith et al style) — gives the 'bucket' near ~100 Hz
    x = f / 215.0
    psd = (PSD_NORM) * (x**(-4.14) - 5.0 * x**(-2.0) + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2))

    # We only care about a smooth positive curve
    psd = np.maximum(psd, PSD_NORM)
    return np.sqrt(psd)  # ASD = sqrt(PSD)


# Optional Utility: generate approximate aLIGO-like colored noise.
# Used ONLY for visualization overlays; never for setting normalization.
def make_ligo_psd_noise(N: int, dt: float, target_rms: float = 1.0, rng: np.random.Generator | None = None,) -> np.ndarray:
    if rng is None: rng = np.random.default_rng(None)

    fs = 1.0 / float(dt)
    freqs = np.fft.rfftfreq(N, d=dt)

    # aLIGO design ASD and PSD
    asd = _aligodesign_asd(freqs).astype(float)
    psd = asd**2

    df = freqs[1] - freqs[0] if len(freqs) > 1 else fs / max(N, 1)

    re = rng.normal(0.0, 1.0, len(freqs))
    im = rng.normal(0.0, 1.0, len(freqs))
    im[0] = 0.0  # DC real

    sigma = np.sqrt(0.5 * psd * df)
    coeffs = (re + 1j * im) * sigma

    noise = np.fft.irfft(coeffs, n=N)

    rms = float(np.sqrt(np.mean(noise**2)))
    if not np.isfinite(rms) or rms < EPS_FLOOR or target_rms <= 0.0: noise[:] = 0.0
    else: noise *= (target_rms / rms)

    return noise.astype(float)


# Optional Utility: simple whitening for plots and sanity checks.
# IMPORTANT: Not used when writing the physics NPZ; does not affect strain_records saved for analysis.
def whiten_for_display(y: np.ndarray, dt: float) -> np.ndarray:
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=dt)
    asd = _aligodesign_asd(freqs)
    Yw = Y / (asd + EPS_SAFE_FLOOR)
    return np.fft.irfft(Yw, n=len(y))


# Provide a better CMAP feel for Spectrogram Visual. Custom colormap for spectrogram plots only; purely aesthetic.
def ligo_cmap():
    base = plt.get_cmap("viridis")
    colors = base(np.linspace(0, 1, 256))

    # --- Gentle adjustments ---
    # Dark end: shift purple → blue-green
    colors[:40, 0] *= 0.6      # reduce red
    colors[:40, 1] *= 0.8      # keep some green for balance
    colors[:40, 2] *= 1.15     # boost blue → cooler tone

    # Midrange: keep viridis shape but slightly brighter
    colors[80:160, :] *= 1.03

    # Highlights: preserve yellow/green but avoid clipping to white
    colors[200:, 2] *= 0.95
    colors = np.clip(colors, 0, 1)

    return mcolors.LinearSegmentedColormap.from_list("ligo_soft", colors)
# --- Visual Enhancements Not Applied to Physics ---


# Main Function to Run Generator.
def run_chirp_generator_test(config_overrides=None):
    config, base   = get_default_config(config_overrides)

    #Read iIn Configuration Settings.

    # ----- Physics Settings -----
    G_phys         = float(config.get("G_phys",   6.67430e-11))        # Gravitational constant (m^3 kg^-1 s^-2)
    c_phys         = float(config.get("c_phys",   2.99792458e8))       # Physical speed of light (m/s)
    lambda_u       = float(config.get("lambda_u", 1.616255e-35))       # m (Planck-like)
    M_sun          = float(config.get("M_sun",    1.98847e30))         # Solar mass (kg)
    MPC_TO_M       = float(config.get("MPC_TO_M", 3.085677581e22))     # meters in one megaparsec (CODATA 2018)


    # ----- Simulation Settings -----
    c              = float(config.get("c", 1.0000))
    grid_spacing   = float(config.get("grid_spacing", 1000.0000))

    soliton_radius = float(config.get("soliton_radius", 40.0000))
    
    M1_solar_src   = float(config.get("M1_solar_src", 30.0))
    M2_solar_src   = float(config.get("M2_solar_src", 30.0))
    if M1_solar_src <= 0.0 or M2_solar_src <= 0.0: raise ValueError("M1_solar_src and M2_solar_src cannot be less than or equal to zero.")

    # Spin full vector.
    spin1x = float(config.get("spin1x", 0.0)); spin1y = float(config.get("spin1y", 0.0)); spin1z = float(config.get("spin1z", 0.0))
    spin2x = float(config.get("spin2x", 0.0)); spin2y = float(config.get("spin2y", 0.0)); spin2z = float(config.get("spin2z", 0.0))
    if M2_solar_src > M1_solar_src: 
        M1_solar_src, M2_solar_src = M2_solar_src, M1_solar_src; 
        spin1x, spin2x = spin2x, spin1x; spin1y, spin2y = spin2y, spin1y; spin1z, spin2z = spin2z, spin1z
    spin1 = np.array([spin1x, spin1y, spin1z], dtype=float); spin2 = np.array([spin2x, spin2y, spin2z], dtype=float)
    mag1 = float(np.linalg.norm(spin1)); mag2 = float(np.linalg.norm(spin2))
    if mag1 >= 1.0 or mag2 >= 1.0: raise ValueError(f"Dimensionless spin magnitudes must be < 1: |spin1|={mag1}, |spin2|={mag2}")
    has_spin = (mag1 > EPS_SPIN) or (mag2 > EPS_SPIN); 
    has_transverse = (abs(spin1x) > EPS_SPIN or abs(spin1y) > EPS_SPIN or abs(spin2x) > EPS_SPIN or abs(spin2y) > EPS_SPIN)
    remnant_approximant = str(config.get("REMNANT_APPROXIMANT", "SEOBNRv4"))
    SPIN_MODE = str(config.get("SPIN_MODE", "none")).lower()
    if SPIN_MODE == "auto":
        if not has_spin: SPIN_MODE = "none"
        elif has_transverse: SPIN_MODE = "precessing"
        else: SPIN_MODE = "aligned"
    if SPIN_MODE not in ("none", "aligned", "precessing"): raise ValueError(f"Invalid SPIN_MODE={SPIN_MODE!r}; use none, aligned, precessing, or auto.")
    if SPIN_MODE == "none": 
        if has_spin: print("[SPIN WARNING] Nonzero spin inputs were provided, but SPIN_MODE='none'. All component spins will be forced to zero for the UMH nonspinning baseline.")
        spin1[:] = 0.0; spin2[:] = 0.0; spin1x = spin1y = spin1z = 0.0; spin2x = spin2y = spin2z = 0.0; mag1 = mag2 = 0.0; 
        has_spin = False; has_transverse = False
    elif SPIN_MODE == "aligned" and has_transverse:
        if has_transverse: raise ValueError("SPIN_MODE='aligned' only allows spin1z/spin2z or chi1z/chi2z. Set SPIN_MODE='precessing' for spin x/y components.")
    elif SPIN_MODE == "precessing":
        if not has_transverse:
            print("[SPIN WARNING] SPIN_MODE='precessing' was requested, but no transverse spin was supplied. The precession wrapper will reduce to the nonprecessing/aligned limit.")
        else: print("[SPIN] Using approximate UMH simple-precession dominant-mode wrapper. This is diagnostic and does not include higher modes or full spin-vector PN evolution.")

    # masses must be SI (kg)
    M1_kg_src = M1_solar_src * M_sun; M2_kg_src = M2_solar_src * M_sun
    M_tot_kg_src = M1_kg_src + M2_kg_src
    eta = (M1_kg_src * M2_kg_src) / (M_tot_kg_src * M_tot_kg_src)
    chi_eff = (M1_solar_src * spin1[2] + M2_solar_src * spin2[2]) / (M1_solar_src + M2_solar_src)

    Mrem_solar_src, a_rem_src, E_rad_frac, eta = estimate_remnant_mass_spin(M1_solar_src, M2_solar_src, spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, 
                                                                            spin2x=spin2x, spin2y=spin2y, spin2z=spin2z, approximant=remnant_approximant, 
                                                                            SPIN_MODE=SPIN_MODE, G_phys=G_phys, c_phys=c_phys)
    Mrem_kg_src = Mrem_solar_src * M_sun

    # The config/table distance is the UMH effective luminosity distance input, not the geometric distance used directly in z_UMH(D_geom; alpha).
    D_L_umh_input_Mpc = float(config.get("distance_Mpc", 410.0))
    distance_Mpc      = D_L_umh_input_Mpc  # legacy alias used elsewhere in logs/config
    D_L_umh_Mpc = D_L_umh_input_Mpc; D_L_umh_m   = D_L_umh_Mpc * MPC_TO_M
    d_geom_Mpc = None; T_umh = None; d_geom_m = float("nan")

    # Binary parameters
    M_chirp_src_kg = ((M1_kg_src * M2_kg_src)**(3/5)) / ((M1_kg_src + M2_kg_src)**(1/5))  # Chirp mass
    # Source-frame chirp mass; detector-frame "effective" chirp is redshifted, handled via UMH_z_tension.
    config["M_chirp_src_kg"] = M_chirp_src_kg

    # Binary inclination:
    BINARY_IOTA_DEG = float(config.get("BINARY_IOTA_DEG", 165.0))  # Binary inclination in degrees (0° = face-on, 90° = edge-on, 180° = face-off)
    BINARY_IOTA     = np.deg2rad(BINARY_IOTA_DEG)                  # BINARY_IOTA_DEG in radians.
   
    Sites_Used = config.get("Sites_Used", {"Hanford", "Livingston", "Virgo"})

    AMPLITUDE_SEED = float(config.get("AMPLITUDE_SEED",  1.0000))
    
    r0             = int(config.get("r0",    12))
    size           = int(config.get("SIZE", 128))

    scale_factor   = float(config.get("scale_factor",    8.25e04))

    damping_factor = float(config.get("damping_factor",  1.0000))
    freq_damping   = float(config.get("freq_damping",    0.0000))
    REF_SPACING    = int(config.get("REF_SPACING",            2))

    SOURCE_ENGINE  = str(config.get("SOURCE_ENGINE", "soliton_umh")).lower()    #SOURCE_ENGINE=analytic_umh
    if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")):
        SOL_ENV_MODE             = str(config.get("SOL_ENV_MODE", "mix")).lower()
        SOL_ENV_ALPHA            = float(config.get("SOL_ENV_ALPHA",            0.35))
        SOL_ENV_MAX_DEV          = float(config.get("SOL_ENV_MAX_DEV",           0.4))
        SOL_ENV_POST_TAPER_MS    = float(config.get("SOL_ENV_POST_TAPER_MS",    12.0))
        SOL_ENV_SMOOTH_CUTOFF_HZ = float(config.get("SOL_ENV_SMOOTH_CUTOFF_HZ", 80.0))

        SOL_FREQ_HILBERT_WINDOW_SEC = float(config.get("SOL_FREQ_HILBERT_WINDOW_SEC", 0.20))  # ~0.2 s window
        SOL_FREQ_EVERY_STEPS        = int(config.get("SOL_FREQ_EVERY_STEPS", 8))              # compute every N steps
        SOL_FREQ_SMOOTH_TAU_SEC     = float(config.get("SOL_FREQ_SMOOTH_TAU_SEC", 0.02))      # smoothing for f_soliton
        SOL_FREQ_MIN_HZ             = float(config.get("SOL_FREQ_MIN_HZ", 5.0))
        SOL_FREQ_MAX_HZ             = float(config.get("SOL_FREQ_MAX_HZ", fs_src * 0.45))

        # Optional blend; keep default 0.0 so generator behavior is unchanged
        SOL_FREQ_BLEND              = float(config.get("SOL_FREQ_BLEND", 0.0))                # 0 = PN only; 1 = soliton only
        SOL_FREQ_MODE               = str(config.get("SOL_FREQ_MODE", "blend")).lower()       # blend, soliton


    #No Need if using High Order.
    SMOOTH_FGW     = config.get("SMOOTH_FGW",     False)  # turn on/off
    TAU_SMOOTH_SEC = config.get("TAU_SMOOTH_SEC", 0.000)  # 2 ms smoothing time-constant (tune 0.02–0.05) 3*dt
    MAX_DFDT_HZ_S  = config.get("MAX_DFDT_HZ_S",   None)  # or e.g. 2.5e4 to clamp |df/dt|

    ONSET_TAPER_SEC = float(config.get("ONSET_TAPER_SEC", 0.0000))


    # Load and Calibrate against UMH Tension for UMH Redshift based on Tension.
    UMH_z_tension = None; UMH_z_tension_info = None; z_source = None; FREQ_RELAX_BETA_CAL = None
    
    H0_km_s_Mpc   = float(config.get("H0_km_s_Mpc", 70.0000))   #0.09537846021692506
    alpha_GR = (H0_km_s_Mpc / (c_phys / 1000.0))                # Only used for Log Comparison, unless UMH_RedShift_Calibration_File is unavailable.
    z_GR = alpha_GR * distance_Mpc

    if "UMH_z_tension" in config: UMH_z_tension = float(config.get("UMH_z_tension", None)) # For 410, 0.09537846021692506.
    else:
        UMH_redshift_json_path = str(config.get("UMH_RedShift_Calibration_File", os.path.join(base, "Output", "UMH_RedShift", "UMH_RedShift_Calibration_Fit.json")))
        UMH_z_tension_ref = create_UMH_z_tension_from_redshift_calibration(UMH_redshift_json_path, H0_km_s_Mpc=70.0000, delta=1.0)
        if(UMH_z_tension_ref is not None):
            alpha    = UMH_z_tension_ref["alpha"]
            beta1    = UMH_z_tension_ref["beta1"]
            beta2    = UMH_z_tension_ref["beta2"]
            M_best   = UMH_z_tension_ref["M_best"]
            H0       = UMH_z_tension_ref["H0"]
            delta    = UMH_z_tension_ref["delta"]
            z_source = UMH_z_tension_ref["source"]

            UMH_z_tension = UMH_z_tension_ref["z_from_DL"](distance_Mpc, alpha, beta1, beta2, delta)
            config["UMH_z_tension"] = UMH_z_tension

            config["FREQ_RELAX_BETA_CAL"] = FREQ_RELAX_BETA_CAL = beta_eff_from_pantheon(UMH_z_tension, beta1, beta2)
                
            UMH_z_tension_info = f"[CALIB] alpha_tgt={alpha}1/m, beta1={beta1}, beta2={beta2},\nUMH_z_tension={UMH_z_tension}, z_GR={z_GR},\nz_source={z_source}"
            UMH_z_tension_info += f", [RELAX] Using Pantheon-derived beta_eff: {FREQ_RELAX_BETA_CAL:.6f}"

    if(UMH_z_tension is None): 
        UMH_z_tension = z_GR
        z_source = f"[CALIB] Fallback to z_GR={z_GR}"
        UMH_z_tension_info = f"[CALIB] Fallback to z_GR={z_GR}"

    # --- Finalize UMH source--observer distances ---------------------------
    # distance_Mpc / D_L_umh_input_Mpc is the tabulated event-level UMH effective luminosity distance. 
    # The geometric distance is recovered from the inverted UMH redshift using the Pantheon+-calibrated relation.
    D_L_umh_Mpc = float(D_L_umh_input_Mpc); D_L_umh_m   = D_L_umh_Mpc * MPC_TO_M
    if ("UMH_z_tension_ref" in locals() and UMH_z_tension_ref is not None and alpha is not None and beta1 is not None and beta2 is not None and delta is not None):
        d_geom_Mpc = float(UMH_z_tension_ref["d_from_z"](UMH_z_tension, alpha))
        D_L_check_Mpc = float(UMH_z_tension_ref["DL_from_z"](UMH_z_tension, alpha, beta1, beta2, delta))
        T_umh = float(UMH_z_tension_ref["T"](UMH_z_tension, beta1, beta2))

        # This should be very close to the input distance because z_UMH was obtained by inverting D_L^UMH(z).
        rel_err = abs(D_L_check_Mpc - D_L_umh_Mpc) / max(abs(D_L_umh_Mpc), EPS_SAFE_FLOOR)
        if rel_err > 1.0e-6: print(f"[WARN] D_L^UMH inversion mismatch: input={D_L_umh_Mpc:.9g} Mpc, check={D_L_check_Mpc:.9g} Mpc, rel_err={rel_err:.3e}")
    else:
        # Fallback only. Primary paper runs should use the Pantheon calibration JSON. If we do not have beta/T(z), approximate D_geom by removing one redshift factor.
        T_umh = None
        if UMH_z_tension is not None and math.isfinite(UMH_z_tension) and UMH_z_tension > -1.0: d_geom_Mpc = D_L_umh_Mpc / max(1.0 + UMH_z_tension, EPS_SAFE_FLOOR)
        else: d_geom_Mpc = D_L_umh_Mpc
    d_geom_m = d_geom_Mpc * MPC_TO_M
    config["D_L_umh_input_Mpc"] = float(D_L_umh_input_Mpc); config["D_L_umh_Mpc"] = float(D_L_umh_Mpc); config["D_L_umh_m"] = float(D_L_umh_m)
    config["d_geom_Mpc"] = float(d_geom_Mpc); config["d_geom_m"] = float(d_geom_m); config["T_umh"] = None if T_umh is None else float(T_umh)
    print(f"[UMH distances] D_L^UMH={D_L_umh_Mpc:.6f} Mpc, D_geom={d_geom_Mpc:.6f} Mpc, z_UMH={UMH_z_tension:.9f}, T(z)={T_umh if T_umh is not None else 'N/A'}")


    # --- UMH redshift (tension) -----------------------------------------
    # This is the only cosmological redshift we use for frequencies.
    # UMH redshift factor:
    #   - All PN / df/dt / f_gw(t) dynamics are built in the *source* frame.
    #   - If APPLY_UMH_FREQ_REDSHIFT is True we use z_factor = 1+z to map
    #     those quantities into the observer frame when we DEFINE the *_obs values.
    #   - After that, we evolve everything in the observer frame; we do NOT
    #     re-apply z_factor later by resampling t.
    APPLY_UMH_FREQ_REDSHIFT = bool(config.get("APPLY_UMH_FREQ_REDSHIFT", True))
    z_factor = 1.0 + UMH_z_tension if APPLY_UMH_FREQ_REDSHIFT else 1.0
    config["UMH_freq_z_factor"] = z_factor
    config["APPLY_UMH_FREQ_REDSHIFT"] = APPLY_UMH_FREQ_REDSHIFT


    dt_obs = config["dt_obs"]
    fs_obs = float(1.0/dt_obs); Fn_obs = 0.5 * fs_obs

    dt_src = dt_obs / z_factor; fs_src = fs_obs * z_factor; Fn_src = Fn_obs * z_factor


    # ----- Ringdown Settings -----
    # These values are now the single source of truth for the ringdown.
    # Metadata JSON and all waveform construction below use this finalized set.
    # --- Inspiral / ringdown characteristic frequencies -----------------
    # --- Frequency band definitions (source vs observer) ---
    # *_src: intrinsic source-frame frequencies.
    # *_obs: what the detector "sees" after UMH redshift scaling.
    # When APPLY_UMH_FREQ_REDSHIFT is True, f_*_obs = f_*_src / z_factor.
    # When False, src and obs are identical.

    # --- ISCO frequency (SOURCE & OBS frames) f_isco_src is an ORBITAL source-frame ISCO frequency. ---
    f_isco_src = (c_phys**3) / (6.0**1.5 * np.pi * G_phys * M_tot_kg_src)
    # --- Map to OBSERVED frame using UMH_z_tension ----------------------
    # Detector-frame frequencies are source-frame / (1+z_umh) when redshift is active.
    f_isco_obs = f_isco_src / z_factor
    f_isco_gw_src = 2.0 * f_isco_src  # GW (2,2) ISCO in source frame
    f_isco_gw_obs = f_isco_gw_src / z_factor
    f_isco_obs  = f_isco_src  / z_factor

    f_min_obs = float(config.get("f_min_obs", 20.0))
    f_min_src = f_min_obs * z_factor

    RINGDOWN_ENABLE         = bool(config.get("RINGDOWN_ENABLE",         True))
    USE_UMH_MERGE_RELAXATION_THRESHOLD = bool(config.get("USE_UMH_MERGE_RELAXATION_THRESHOLD", True))
    USE_QNM_OVERTONE_ATTACH = bool(config.get("USE_QNM_OVERTONE_ATTACH", False))
    ringdown_merge_strategy = None
    frequency_smoothing     = None
    ring_down_modes, f_rd_src_est, f_rd_obs_est, tau_rd_src_est, tau_rd_obs_est = None, None, None, None, None; 
    umh_mode, tau_seed_rule = None, None

    if USE_QNM_OVERTONE_ATTACH:
        print("[QNM_DIAGNOSTIC WARNING] USE_QNM_OVERTONE_ATTACH=True will overwrite the UMH medium-law ringdown after attach. Use only as an explicit "
              "non-UMH diagnostic waveform branch.")

    # ---------------- Ringdown / Merger configuration ----------------
    # Optional overrides (can contain f_merge_Hz, f_merge_src_Hz, f_rd_obs_Hz, tau_rd_obs, etc.)

    rd_cfg   = dict(config.get("RINGDOWN_OVERRIDE", {}))
    tau_rd_obs = None; tau_rd_src = None

    # --- Inspiral end frequency f_merge_src (SOURCE FRAME) --------------
    # Priority:
    #   (1) RINGDOWN_OVERRIDE.f_merge_src_Hz   (explicit source-frame value)
    #   (2) RINGDOWN_OVERRIDE.f_merge_Hz       (obs-frame, converted via UMH_z_tension)
    #   (3) config["f_merge_src_Hz"]           (global source-frame setting)
    #   (4) F_MERGE_TO_FRD_SCALE * f_rd_obs    (model rule in source frame)

    RINGDOWN_SEED_MODE = str(config.get("RINGDOWN_SEED_MODE", "umh_remnant_mode")).lower()
    COMPUTE_QNM_DIAGNOSTIC = bool(config.get("COMPUTE_QNM_DIAGNOSTIC", True))
    qnm_diagnostic = None; umh_qnm_compare = None
    if COMPUTE_QNM_DIAGNOSTIC or RINGDOWN_SEED_MODE == "qnm_diagnostic" or USE_QNM_OVERTONE_ATTACH:
        # --- Optional QNM diagnostic only ------------------------------------
        # QNM is not used to seed the active UMH ringdown unless explicitly requested.
        # --- QNM ringdown frequencies (SOURCE FRAME) ------------------------
        qnm_overtone_list = tuple(int(n) for n in config.get("QNM_OVERTONE_LIST", [0, 1, 2]))
        qnm_decay_ratio   = float(config.get("QNM_OVERTONE_DECAY_RATIO", 0.3)) #0.7
        qnm_amp_vector    = config.get("QNM_OVERTONE_AMPS", None)

        ring_down_modes   = qnm_22n_modes_from_Ma(Mrem_kg_src, a_rem_src, overtones=qnm_overtone_list, G_phys=G_phys, c_phys=c_phys)
        qnm_diagnostic = {"f_220_src_Hz": float(ring_down_modes[0][0]), "f_220_obs_Hz": float(ring_down_modes[0][0] / z_factor),
                          "tau_220_src_s": float(ring_down_modes[0][1]), "tau_220_obs_s": float(ring_down_modes[0][1] * z_factor),
                          "note": "QNM diagnostic only; not used as active UMH ringdown seed."}

    # Choose preliminary merge anchor first.
    if "f_merge_obs_Hz" in rd_cfg: f_merge_obs = float(rd_cfg["f_merge_obs_Hz"]); f_merge_src = f_merge_obs * z_factor; merge_rule = "override_f_merge_obs"
    elif "f_merge_src_Hz" in rd_cfg: f_merge_src = float(rd_cfg["f_merge_src_Hz"]); f_merge_obs = f_merge_src / z_factor; merge_rule = "override_f_merge_src"
    elif config.get("f_merge_obs_Hz", None) is not None: 
        f_merge_obs = float(config["f_merge_obs_Hz"]); f_merge_src = f_merge_obs * z_factor; merge_rule = "config_f_merge_obs_Hz"
    elif config.get("f_merge_src_Hz", None) is not None: 
        f_merge_src = float(config["f_merge_src_Hz"]); f_merge_obs = f_merge_src / z_factor; merge_rule = "config_f_merge_src_Hz"
    else:
        use_isco_anchor = bool(config.get("USE_ISCO_ANCHOR_FOR_FMERGE", True))
        if use_isco_anchor:
            MERGE_FRAC_ISCO = float(config.get("MERGE_FRAC_ISCO", 1.0))
            f_merge_src = MERGE_FRAC_ISCO * f_isco_gw_src; f_merge_obs = f_merge_src / z_factor; merge_rule = "isco_anchor"
        else:
            raise ValueError("USE_ISCO_ANCHOR_FOR_FMERGE=False now requires explicit f_merge_src_Hz or f_merge_obs_Hz. "
                             "The generator no longer allows f_merge_src to be scaled from f_rd_src.")

    # Choose active ringdown target.
    if "f_rd_obs_Hz" in rd_cfg: f_rd_obs = float(rd_cfg["f_rd_obs_Hz"]); f_rd_src = f_rd_obs * z_factor; ringdown_seed_rule = "override_f_rd_obs"
    elif "f_rd_src_Hz" in rd_cfg: f_rd_src = float(rd_cfg["f_rd_src_Hz"]); f_rd_obs = f_rd_src / z_factor; ringdown_seed_rule = "override_f_rd_src"
    elif RINGDOWN_SEED_MODE == "qnm_diagnostic":
        f_rd_src = float(ring_down_modes[0][0]); f_rd_obs = f_rd_src / z_factor
        ringdown_seed_rule = "qnm_explicit_diagnostic_mode"
    else:
        if bool(config.get("UMH_REMNANT_CALC_COEFF_DIAGNOSTIC", True)):
            chi_grid = np.linspace(0.0, 0.95, 40)
            Mrem_kg = 60.0 * 1.98847e30  # fixed mass for coefficient extraction
            f_vals, tau_vals, M_vals = [], [], []
            for chi in chi_grid:
                mode = umh_remnant_mode_from_Mchi(Mrem_kg_src=Mrem_kg, chi_rem=chi, freq_model="umh_eigen", damping_model="umh_cycles",
                                                  damping_cycles=1.0, boundary_softening=0.0, radial_phase_model="none", radial_delay_frac=0.0, 
                                                  radial_delay_time_GM_c3=0.0)
                f_vals.append(mode["f_src_Hz"]); tau_vals.append(mode["tau_src_s"]); M_vals.append(Mrem_kg)
            fit = fit_umh_ringdown_coefficients(chi_values=chi_grid, f_src_Hz_values=f_vals, tau_src_s_values=tau_vals, Mrem_kg_values=M_vals)
            print(fit["omega_coeffs"]); print(fit["Q_coeffs"])
            UMH_RADIAL_OMEGA_FIT_A = fit["omega_coeffs"]["A"]; UMH_RADIAL_OMEGA_FIT_B = fit["omega_coeffs"]["B"]; UMH_RADIAL_OMEGA_FIT_C = fit["omega_coeffs"]["C"]
        else:
            UMH_RADIAL_OMEGA_FIT_A=float(config.get("UMH_OMEGA_FIT_A", 1.5381969714219732))
            UMH_RADIAL_OMEGA_FIT_B=float(config.get("UMH_OMEGA_FIT_B", -1.1556090096475626))
            UMH_RADIAL_OMEGA_FIT_C=float(config.get("UMH_OMEGA_FIT_C", 0.14022090728188466))

        boundary_softening = float(config.get("UMH_EIGEN_BOUNDARY_SOFTENING", 0.0))
        freq_model_active = str(config.get("UMH_REMNANT_FREQ_MODEL", "umh_eigen")).lower().strip()
        try:
            radial_operator_params = {"ell": int(config.get("UMH_RADIAL_OP_ELL", 2)), "m": int(config.get("UMH_RADIAL_OP_M", 2)),
                                      "x_inner_offset": float(config.get("UMH_RADIAL_OP_X_INNER_OFFSET", 0.050)),
                                      "x_outer": float(config.get("UMH_RADIAL_OP_X_OUTER", 80.0)), "x_match": config.get("UMH_RADIAL_OP_X_MATCH", None),
                                      "tension_amp": float(config.get("UMH_RADIAL_OP_TENSION_AMP", 0.0)), 
                                      "density_amp": float(config.get("UMH_RADIAL_OP_DENSITY_AMP", 0.0)),
                                      "barrier_amp": float(config.get("UMH_RADIAL_OP_BARRIER_AMP", 0.12)), 
                                      "barrier_width": float(config.get("UMH_RADIAL_OP_BARRIER_WIDTH", 0.90)),
                                      "barrier_offset": float(config.get("UMH_RADIAL_OP_BARRIER_OFFSET", 0.75)), 
                                      "rotation_scale": float(config.get("UMH_RADIAL_OP_ROTATION_SCALE", 1.0)),

                                      "rotation_power": float(config.get("UMH_RADIAL_OP_ROTATION_POWER", 1.50)),
                                      "rotation_power_chi_enable": bool(config.get("UMH_RADIAL_OP_ROTATION_POWER_CHI_ENABLE", False)),
                                      "rotation_power_chi_ref": float(config.get("UMH_RADIAL_OP_ROTATION_POWER_CHI_REF", 0.6100942439792432)),
                                      "rotation_power_chi_slope": float(config.get("UMH_RADIAL_OP_ROTATION_POWER_CHI_SLOPE", 0.0)),
                                      "rotation_power_min": float(config.get("UMH_RADIAL_OP_ROTATION_POWER_MIN", 0.85)),
                                      "rotation_power_max": float(config.get("UMH_RADIAL_OP_ROTATION_POWER_MAX", 1.50)), 
                                      "omega_min_factor": float(config.get("UMH_RADIAL_OP_OMEGA_MIN_FACTOR", 0.88)),
                                      "omega_max_factor": float(config.get("UMH_RADIAL_OP_OMEGA_MAX_FACTOR", 1.05)),
                                      "gamma_min_factor": float(config.get("UMH_RADIAL_OP_GAMMA_MIN_FACTOR", 0.10)),
                                      "gamma_max_factor": float(config.get("UMH_RADIAL_OP_GAMMA_MAX_FACTOR", 3.00)),
                                      "omega_guess": config.get("UMH_RADIAL_OP_OMEGA_GUESS", None), "gamma_guess": config.get("UMH_RADIAL_OP_GAMMA_GUESS", None),
                                      "rtol": float(config.get("UMH_RADIAL_OP_RTOL", 1.0e-8)), "atol": float(config.get("UMH_RADIAL_OP_ATOL", 1.0e-10)),
                                      "resid_tol": float(config.get("UMH_RADIAL_OP_RESID_TOL", 1.0e-5)), "maxfev": int(config.get("UMH_RADIAL_OP_MAXFEV", 80)), 
                                      "target_ratio": float(config.get("UMH_RADIAL_OP_TARGET_RATIO", 0.953)), "debug": bool(config.get("UMH_RADIAL_OP_DEBUG", False))}
            umh_mode = umh_remnant_mode_from_Mchi(Mrem_kg_src=Mrem_kg_src, chi_rem=a_rem_src, 
                                                  freq_model=freq_model_active,
                                                  damping_model=str(config.get("UMH_REMNANT_DAMPING_MODEL", "umh_cycles")),
                                                  damping_cycles=float(config.get("UMH_RINGDOWN_DAMPING_CYCLES", 1.00)),
                                                  boundary_softening=boundary_softening,
                                                  # Only used by freq_model="umh_eigen".
                                                  radial_phase_model=str(config.get("UMH_RADIAL_PHASE_MODEL", "absolute_time")),
                                                  radial_delay_frac=float(config.get("UMH_RADIAL_PHASE_DELAY_FRAC", 0.0)),
                                                  radial_delay_time_GM_c3=float(config.get("UMH_RADIAL_PHASE_DELAY_TIME_GM_C3", 0.0)),
                                                  # Only used by freq_model="umh_omega_fit".
                                                  omega_fit_A=UMH_RADIAL_OMEGA_FIT_A, omega_fit_B=UMH_RADIAL_OMEGA_FIT_B, omega_fit_C=UMH_RADIAL_OMEGA_FIT_C,
                                                  radial_operator_params=radial_operator_params, 
                                                  rad_chi_ref=float(config.get("UMH_RADIAL_FIT_CHI_REF", 0.6100942439792432)),
                                                  rad_chi_d0=float(config.get("UMH_RADIAL_FIT_DELAY_REF", 0.04780931608484118)),
                                                  rad_chi_d1=float(config.get("UMH_RADIAL_FIT_DELAY_CHI_SLOPE", 0.03586007393884942)),
                                                  rad_chi_dmin=float(config.get("UMH_RADIAL_FIT_DELAY_MIN", 0.00)),
                                                  rad_chi_dmax=float(config.get("UMH_RADIAL_FIT_DELAY_MAX", 0.20)),
                                                  G_phys=G_phys, c_phys=c_phys)
        except RuntimeError as exc:
            if freq_model_active in ("umh_radial_operator", "umh_shooting", "umh_radial_qnm"): 
                print(f"[WARN] UMH radial operator failed; falling back to umh_eigen for this run: {exc}")
                umh_mode = umh_remnant_mode_from_Mchi(Mrem_kg_src=Mrem_kg_src, chi_rem=a_rem_src, freq_model="umh_eigen", damping_model="umh_cycles",
                    damping_cycles=float(config.get("UMH_RINGDOWN_DAMPING_CYCLES", 1.00)), boundary_softening=boundary_softening,
                    radial_phase_model="none", radial_delay_frac=0.0, radial_delay_time_GM_c3=0.0,
                    G_phys=G_phys, c_phys=c_phys)
                umh_mode["shape_meta"]["radial_operator_failure"] = str(exc)
            else: raise

        f_rd_src = float(umh_mode["f_src_Hz"]); f_rd_obs = f_rd_src / z_factor
        tau_rd_src = float(umh_mode["tau_src_s"]); tau_rd_obs = tau_rd_src * z_factor
        if freq_model_active in ("umh_radial_operator", "umh_shooting", "umh_radial_qnm"): ringdown_seed_rule = "umh_remnant_mode_umh_radial_operator"
        elif freq_model_active == "umh_radial_fit": ringdown_seed_rule = "umh_remnant_mode_umh_radial_fit"
        elif freq_model_active in ("umh_omega_fit", "umh_frequency_law", "umh_coeff_fit"): ringdown_seed_rule = "umh_remnant_mode_umh_fitted_frequency_law"
        else: ringdown_seed_rule = "umh_remnant_mode_umh_eigen_F_chi"
        tau_seed_rule = str(umh_mode["damping_rule"])


    # Choose active damping time.
    if "tau_rd_obs" in rd_cfg: tau_rd_obs = float(rd_cfg["tau_rd_obs"]); tau_rd_src = tau_rd_obs / z_factor; tau_seed_rule = "override_tau_obs"
    elif "tau_rd_src" in rd_cfg: tau_rd_src = float(rd_cfg["tau_rd_src"]); tau_rd_obs = tau_rd_src * z_factor; tau_seed_rule = "override_tau_src"
    elif config.get("tau_rd_src", None) is not None: tau_rd_src = float(config["tau_rd_src"]); tau_rd_obs = tau_rd_src * z_factor; tau_seed_rule = "config_tau_src"
    else:
        if(umh_mode is not None and tau_rd_src is not None): pass
        else:
            UMH_RINGDOWN_DAMPING_CYCLES = float(config.get("UMH_RINGDOWN_DAMPING_CYCLES", 1.00))
            tau_rd_src = UMH_RINGDOWN_DAMPING_CYCLES / max(f_rd_src, EPS_SAFE_FLOOR); tau_rd_obs = tau_rd_src * z_factor
            tau_seed_rule = "umh_damping_cycles"


    # --- Optional UMH merge anchor: driver timescale vs medium relaxation ----------------
    # UMH interpretation: inspiral remains quasi-stationary while the medium can respond
    # adiabatically to the driving. Define the merger transition as the moment when the
    # inspiral driving timescale t_drive = f/(df/dt) becomes comparable to the medium
    # relaxation time tau_rd_src. This is source-frame physics and does not rely on any
    # cosmology/expansion inference.
    f_rd_src_eff = f_rd_src; f_rd_obs_eff = f_rd_obs; f_merge_src_eff = f_merge_src; f_merge_obs_eff = f_merge_obs
    if USE_UMH_MERGE_RELAXATION_THRESHOLD:
        term_merge = (G_phys * M_chirp_src_kg) / (c_phys**3)
        K_N_merge = (96.0 / 5.0) * (math.pi**(8.0 / 3.0)) * (term_merge**(5.0 / 3.0))
        tau_relax = max(float(tau_rd_src), 1e-6)
        inv_const = (5.0 / 96.0) * (math.pi**(-8.0 / 3.0)) * (term_merge**(-5.0 / 3.0))
        f_thresh = (inv_const / tau_relax) ** (3.0 / 8.0)
        f_thresh = max(float(f_thresh), f_min_src + 3.0)

        f_merge_src_eff = f_thresh; f_merge_obs_eff = f_merge_src_eff / z_factor
        merge_rule = "umh_driver_timescale_equals_relaxation"

        dfdt_merge = K_N_merge * (max(f_merge_src_eff, f_min_src) ** (11.0 / 3.0))
        driver_timescale_merge_src = float(f_merge_src_eff / max(dfdt_merge, EPS_FLOOR))

    # Ensure ringdown is not below inspiral end (both are already OBSERVED frame)
    if f_rd_obs_eff < f_merge_obs_eff: f_rd_obs_eff = 1.05 * f_merge_obs_eff; f_rd_src_eff = f_rd_obs_eff * z_factor #Adjusted, added f_rd_src_eff = f_rd_obs_eff * z_factor 6/7/2026

    # Diagnostic Block comparing QNM and UMH Mode
    if qnm_diagnostic is not None and umh_mode is not None:
        qnm_f_src = float(qnm_diagnostic["f_220_src_Hz"]); qnm_tau_src = float(qnm_diagnostic["tau_220_src_s"])
        f_umh_src = float(umh_mode["f_src_Hz"]); f_umh_eff_src = float(f_rd_src_eff)
        f_umh_base_src = float(umh_mode.get("f_src_no_radial_delay_Hz", f_umh_src))
        umh_qnm_compare = {"qnm_f_220_src_Hz": qnm_f_src, "qnm_tau_220_src_s": qnm_tau_src,
                            "umh_f_rd_src_Hz": f_umh_src, "umh_f_rd_eff_src_Hz": f_umh_eff_src, "umh_f_no_radial_delay_src_Hz": f_umh_base_src,
                            "delta_umh_minus_qnm_Hz": float(f_umh_src - qnm_f_src), "delta_umh_eff_minus_qnm_Hz": float(f_umh_eff_src - qnm_f_src),
                            "ratio_umh_to_qnm": float(f_umh_src / qnm_f_src), "ratio_umh_eff_to_qnm": float(f_umh_eff_src / qnm_f_src),
                            "qnm_equivalent_radial_delay_frac_from_base": float((f_umh_base_src / qnm_f_src) - 1.0),
                            "note": ("QNM is diagnostic only. qnm_equivalent_radial_delay_frac_from_base is the fractional radial-delay value that "
                                        "would map the sharp UMH eigen frequency to the Kerr/QNM diagnostic frequency.")}


    # --- Physically motivated RING_MERGE_C1_BLEND_SEC (source-frame seconds) ---------
    # If RING_MERGE_C1_BLEND_SEC explicitly set respect it.
    RING_MERGE_C1_BLEND_SEC_cfg = config.get("RING_MERGE_C1_BLEND_SEC", 0.0)     # 0.0 uses no Merge_Ramp, and instead uses a constrained-frequency transition that maps to ringdown.  Set to None to calculate RING_MERGE_C1_BLEND_SEC.
    if RING_MERGE_C1_BLEND_SEC_cfg is not None: RING_MERGE_C1_BLEND_SEC = float(RING_MERGE_C1_BLEND_SEC_cfg)
    else:
        # Time for N_cycles of GW at the merger frequency
        N_cycles_merge = float(config.get("N_CYCLES_MERGE_RAMP", 2.0))
        t_cycles = N_cycles_merge / max(f_merge_src_eff, 1e-6)
        # Fraction of the ringdown damping time (medium relaxation timescale)
        frac_tau = float(config.get("F_TAU_MERGE_RAMP", 0.170))  # consider default 0.5
        t_tau = frac_tau * max(tau_rd_src, 0.0)
        # Hard cap so it can’t get overly long
        t_cap = float(config.get("MERGE_RAMP_CAP_SEC", 0.03))
        # Ensure ramp is not too short, but also not overly long
        t_min = min(t_cycles, t_tau)
        RING_MERGE_C1_BLEND_SEC = min(t_min, t_cap)
        #print(f"RING_MERGE_C1_BLEND_SEC_cfg: N_cycles_merge={N_cycles_merge}, t_cycles={t_cycles}, frac_tau={frac_tau}, t_tau={t_tau}, t_cap={t_cap}, t_min={t_min}, RING_MERGE_C1_BLEND_SEC={t_min}")
        # Persist for metadata / downstream diagnostics
        config["RING_MERGE_C1_BLEND_SEC"] = RING_MERGE_C1_BLEND_SEC
    
    FREQ_RELAX_SEC, FREQ_RELAX_KAPPA, FREQ_RELAX_MIN_SEC, FREQ_RELAX_MAX_SEC, FREQ_RELAX_BETA = None, None, None, None, None
    if(FREQ_RELAX_BETA_CAL is None or "FREQ_RELAX_BETA" in config): FREQ_RELAX_BETA = float(config.get("FREQ_RELAX_BETA", 0.45)) #Fallback or specifically defined.
    else: FREQ_RELAX_BETA = FREQ_RELAX_BETA_CAL
    FREQ_RELAX_MIN_SEC = float(config.get("FREQ_RELAX_MIN_SEC", 2.0 * dt_src))  # Minimum Floor
    FREQ_RELAX_MAX_SEC = float(config.get("FREQ_RELAX_MAX_SEC", 0.0060))        # Maximum Cap
    FREQ_RELAX_SEC = config.get("FREQ_RELAX_SEC", None)  # 0.0006, 0.0008, ~0.6 ms default
    if(FREQ_RELAX_SEC is None):
        FREQ_RELAX_KAPPA_EFF = FREQ_RELAX_KAPPA = config.get("FREQ_RELAX_KAPPA", None)              # ~1.25 -> ~0.8 ms for f_rd~252 Hz
        if(FREQ_RELAX_KAPPA is not None): FREQ_RELAX_SEC_EFF = FREQ_RELAX_KAPPA / (2.0*math.pi*max(f_rd_src_eff, EPS_SAFE_FLOOR))
        else:
            FREQ_RELAX_SEC_EFF = (FREQ_RELAX_BETA * tau_rd_src)
            #The same ultronic medium response that produces the Pantheon β transport coefficients also governs local frequency relaxation near compact binary merger. 
            #In the present work we parameterize the local response via a dimensionless relaxation constant κ, which is fixed by the ringdown timescale and wave coherence length. 
            #A full derivation of cosmological β parameters from local κ is left for future work.

            #The UMH distance–redshift relation (α) fixes the source–observer time dilation. 
            #Therefore the ringdown relaxation timescale used in the chirp generator is not a free per-event fit, 
            #but is determined by the source-frame dynamics mapped consistently to the observer frame via UMH redshift.

            # UMH frequency relaxation model:
            # We assume the medium retunes its instantaneous oscillation frequency over
            # approximately one ringdown cycle. This is implemented as an exponential relaxation with timescale
            #     FREQ_RELAX_SEC = beta * tau_rd, where tau_rd is the source-frame ringdown damping time and
            # Ringdown medium relaxation:
            # beta_eff(z_tension) = beta_1 + 2*beta_2*ln(1 + z_tension)
            # This relaxation strength is inherited from Pantheon+ cosmological calibration,
            # encoding medium tension transport. It is not a GW waveform fitting parameter.
    else: FREQ_RELAX_SEC_EFF = FREQ_RELAX_SEC

    FREQ_RELAX_SEC_EFF = min(FREQ_RELAX_SEC_EFF, FREQ_RELAX_MAX_SEC)  # e.g. 1.5 ms
    FREQ_RELAX_SEC_EFF = max(FREQ_RELAX_SEC_EFF, FREQ_RELAX_MIN_SEC)  # >=2 samples
    FREQ_RELAX_KAPPA_EFF = float(2.0 * math.pi * max(f_rd_src_eff, EPS_SAFE_FLOOR) * FREQ_RELAX_SEC_EFF)
    config["FREQ_RELAX_SEC_EFF"]   = FREQ_RELAX_SEC_EFF
    config["FREQ_RELAX_KAPPA_EFF"] = FREQ_RELAX_KAPPA_EFF

    # --- UMH-derived amplitude ramp parameters ---
    # How long before merger the nonlinear ramp starts: Define in terms of "number of cycles before merger" and
    #    "fraction of the damping time". These are fixed UMH constants, *not* fit per event.
    N_RAMP_CYCLES   = 1.5    # about 1.5 GW cycles before f_merge
    FRAC_TAU_RAMP   = 0.3    # or 0.2–0.5; pick once and freeze

    t_cycles_ramp   = N_RAMP_CYCLES / f_merge_src_eff           # [s]
    t_tau_ramp      = FRAC_TAU_RAMP * tau_rd_src            # [s]
    AMP_RAMP_START_SEC = min(t_cycles_ramp, t_tau_ramp)

    # How sharp the logistic transition is: expit(x) goes 0.1 → 0.9 over Δx ≈ 4.394. We want that to span N_WIDTH_CYCLES at f_merge.
    N_WIDTH_CYCLES  = 1.0    # ramp goes from 10%→90% over ~1 cycle
    delta_t_width   = N_WIDTH_CYCLES / f_merge_src_eff          # [s] - desired 10–90% width

    AMP_RAMP_SLOPE  = 2.0 * math.log(9.0) / delta_t_width   # [1/s]

    # Tail taper length: a fixed multiple of the damping time.
    C_TAIL          = 1.0    # taper over one τ_rd (can choose 0.5–2 and freeze)
    AMP_RAMP_TAIL_SEC = C_TAIL * tau_rd_src

    # Boost factor: keep as a single global UMH constant for now. This is no longer per-event; pick it once (e.g. from a
    #    soliton envelope calibration) and never change it in comparisons.
    if config.get("AMP_RAMP_BOOST", None) is None: AMP_RAMP_BOOST = 0.00 #0.88
    else: AMP_RAMP_BOOST = float(config["AMP_RAMP_BOOST"])


    eta = (M1_kg_src * M2_kg_src) / (M_tot_kg_src ** 2)

    # Persist both SOURCE and OBSERVED values into config for downstream use / metadata
    config["f_merge_src_Hz"]   = f_merge_src
    config["f_merge_src_eff_Hz"]   = f_merge_src_eff
    config["f_rd_src_Hz"]      = f_rd_src
    config["f_rd_src_eff_Hz"]  = f_rd_src_eff
    config["f_isco_src_Hz"]    = f_isco_src
    config["f_isco_gw_src_Hz"] = f_isco_gw_src
    config["tau_rd_obs"]       = tau_rd_obs

    config["f_merge_obs_Hz"]   = f_merge_obs
    config["f_merge_obs_eff_Hz"] = f_merge_obs_eff
    config["f_rd_obs_Hz"]      = f_rd_obs
    config["f_rd_obs_eff_Hz"]  = f_rd_obs_eff
    config["f_isco_obsHz"]     = f_isco_obs
    config["f_isco_gw_obs_Hz"] = f_isco_gw_obs
    config["tau_rd_src"]       = tau_rd_src


    # ----- Normalization Specific Settings -----
    PHYSICS_NORM_ENABLE = bool(config.get("PHYSICS_NORM_ENABLE", True))

    APPLY_UMH_AMPLITUDE_SCALING = bool(config.get("APPLY_UMH_AMPLITUDE_SCALING", True))

    # Physics-driven defaults
    # Allow explicit override
    f_ref_obs_cfg = config.get("NORM_F_REF_OBS_HZ", None); f_ref_obs_desc="NORM_F_REF_OBS_HZ CALCULATED"
    if f_ref_obs_cfg is not None: f_ref_obs = float(f_ref_obs_cfg)
    else:
        # If 100 Hz is inside [f_min_obs, f_merge_obs_eff], use it (GW150914-like case)
        if f_min_obs <= 100.0 <= f_merge_obs_eff: f_ref_obs = 100.0
        else: f_ref_obs  = 0.5 * (f_min_obs + f_merge_obs_eff) # Otherwise, pick the midpoint of inspiral band

        if bool(config.get("USE_PSD_FREF", False)):
            psd_map = {}
            PSD_MAP_INPUT_FOLDER = str(config.get("PSD_MAP_INPUT_FOLDER", os.path.join(base, "Output", "UMH_vs_LIGO")))
            PSD_MAP_FILENAME = str(config.get("PSD_MAP_INPUT_FILENAME", "UMH_vs_LIGO_CMP_PSD_Map.npz"))
            psd_map_path = os.path.join(PSD_MAP_INPUT_FOLDER, PSD_MAP_FILENAME)
            if os.path.exists(psd_map_path):
                psd_map = load_psd_map_for_generator(psd_map_path)
                f_ref_obs = compute_snr_weighted_fref(psd_map, f_min_obs, f_merge_obs_eff)
                f_ref_obs_desc=f"NORM_F_REF_OBS_HZ CREATED FROM PSD MAP ({psd_map_path}) - ({f_ref_obs} Hz)"
                
    # Hard safety clamps from physics / numerics, NOT viz: must be above f_min_obs, must be below merger and Nyquist margin
    f_ref_obs = max(f_ref_obs, 1.05 * f_min_obs)
    f_ref_obs = min(f_ref_obs, 0.95 * f_merge_obs_eff, 0.90 * Fn_obs)
    f_ref_src = f_ref_obs * z_factor

    # f_ref_obs is an OBSERVED-frame reference frequency (detector band).
    config["NORM_F_REF_OBS_HZ"] = f_ref_obs
    config["NORM_F_REF_FRAME"]  = "observed"

    
    # ----- Use UMH, DFDT_PROFILE -----
    # NOTE: get_gr_dfdt_coeffs() returns the GR post-Newtonian df/dt coefficients evaluated for (M1,M2,f_ref_src). In UMH, this corresponds
    # to the GR limit of the medium dynamics; deviations would enter via a modified UMH_DFDT_COEFFS.
    USE_UMH_DFDT_PROFILE  = bool(config.get("USE_UMH_DFDT_PROFILE", False))
    if SPIN_MODE in ("aligned", "precessing") and has_spin and not USE_UMH_DFDT_PROFILE:
        print("[SPIN WARNING] Spin is active, but USE_UMH_DFDT_PROFILE=False, so inspiral phase spin effects are disabled. Spin still affects the remnant/ringdown fit; "
              "for SPIN_MODE='precessing', transverse spin also affects the polarization wrapper.")
    if USE_UMH_DFDT_PROFILE: 
        # Higher-order GR-limit PN coefficients are available for diagnostic and extension studies. The published GW150914/GW170814 runs use
        # the leading-order 0PN-equivalent configuration.
        UMH_DFDT_PN_PROFILE = float(config.get("UMH_DFDT_PN_PROFILE", 0.0))
        UMH_DFDT_COEFFS, UMH_DFDT_SPIN_META = get_gr_dfdt_coeffs(M1_kg_src, M2_kg_src, f_ref_src, eta=eta, spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, 
                                                                 spin2x=spin2x, spin2y=spin2y, spin2z=spin2z, SPIN_MODE=SPIN_MODE, 
                                                                 G_phys=G_phys, c_phys=c_phys, UMH_DFDT_PN_PROFILE=UMH_DFDT_PN_PROFILE)
    else: UMH_DFDT_COEFFS = None; UMH_DFDT_SPIN_META = None; UMH_DFDT_PN_PROFILE = None


    # ----- Use UMH, Amplitude PN Profile -----
    # a2 v^2          (1PN)
    # a3 v^3          (1.5PN)
    # a4 v^4          (2PN)
    # a5 v^5          (2.5PN, non-spinning)
    # a6 v^6          (3.0PN, non-spinning)
    USE_UMH_AMP_PN_PROFILE = bool(config.get("USE_UMH_AMP_PN_PROFILE", False))
    if SPIN_MODE in ("aligned", "precessing") and has_spin and USE_UMH_AMP_PN_PROFILE:
        raise RuntimeError("[SPIN WARNING] USE_UMH_AMP_PN_PROFILE=True uses nonspinning amplitude PN terms; spin amplitude corrections are not included.")
    if(USE_UMH_AMP_PN_PROFILE):
        UMH_AMP_PN_PROFILE  = float(config.get("UMH_AMP_PN_PROFILE", 3.0))
        if(UMH_AMP_PN_PROFILE>3.0): UMH_AMP_PN_PROFILE = 3.0
        elif(UMH_AMP_PN_PROFILE<2.0): UMH_AMP_PN_PROFILE = 2.0

    
    # ----- Visual Only Settings -----
    VISUAL_SCALE_MULT = float(config.get("VISUAL_SCALE_MULT",    1.0))

    # Preview / whitening band (VIZ ONLY; does NOT constrain physics track)
    lowcut_default_obs  = max(10.0, 0.8 * f_min_obs)
    highcap_obs         = min(0.90 * Fn_obs, 1.25 * max(f_merge_obs_eff, f_rd_obs_eff))
    lowcut_obs          = float(config.get("lowcut_obs",  lowcut_default_obs))
    highcut_obs         = float(config.get("highcut_obs", highcap_obs))

    # Enforce ordering + sanity for the *filters only*
    if highcut_obs > highcap_obs: highcut_obs = highcap_obs
    if lowcut_obs < 0.0: lowcut_obs = 0.0
    if highcut_obs <= lowcut_obs:
        highcut_obs = highcap_obs
        if highcut_obs <= lowcut_obs: lowcut_obs = max(10.0, 0.5 * highcut_obs)

    lowcut_src  = lowcut_obs  * z_factor
    highcut_src = highcut_obs * z_factor
    config["lowcut_obs"] = lowcut_obs; config["highcut_obs"] = highcut_obs
    config["lowcut_src"] = lowcut_src; config["highcut_src"] = highcut_src

    # ----- Internal Wiring -----
    dpi             = config["DPI"]
    dtype           = config["DTYPE"]

    outdir          = config["OUTPUT_FOLDER"]

    title           = "UMH Chirp Generator"
    file_root       = "UMH_vs_LIGO"
    file_hdr        = "UMH_Chirp_Generator"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")

    # ----- Start -----

    print(f"[CFG] profile={config['profile']} M1_solar_src={config['M1_solar_src']} M2_solar_src={config['M2_solar_src']}")

    print(f"[REMNANT] Mtot_src={M_tot_kg_src / M_sun:.6f} Msun Mrem_src={Mrem_solar_src:.6f} Msun E_rad_frac={E_rad_frac:.9f} a_rem={a_rem_src:.6f} " 
          f"SPIN_MODE={SPIN_MODE} spin1x={spin1x:+.4f} spin1y={spin1y:+.4f} spin1z={spin1z:+.4f} spin2x={spin2x:+.4f} spin2y={spin2y:+.4f} spin2z={spin2z:+.4f} "
          f"fit={remnant_approximant}")

    print(f"[BAND] f_min_obs={f_min_obs:.1f},  f_merge_obs_eff={f_merge_obs_eff:.1f}, f_rd_obs_eff={f_rd_obs_eff:.1f}, f_isco_obs={f_isco_obs:.1f}, "
          f"lowcut_obs={lowcut_obs:.1f}, highcut_obs={highcut_obs:.1f}, Fn_obs={Fn_obs:.1f}")
    if(UMH_z_tension_info is not None): print(UMH_z_tension_info)

    # Diagnostic call (e.g. for GW150914-like system), for Illustration Only.
    print()
    f_umh_min, f_umh_max = umh_macro_frequency_band(lambda_u, Mrem_solar_src)
    print(f"DIAGNOSTIC ONLY: UMH micro → macro band ≈ {f_umh_min:.1f} - {f_umh_max:.1f} Hz | (GR ISCO ≈ {f_isco_obs})")


    # Simulation parameters
    Nx, Ny, Nz = size, size, size

    center = (Nx//2, Ny//2, Nz//2)  # shared soliton center
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    # Source origin position (can be off-center to simulate directionality)
    source_origin = np.array([cx, cy, cz])

    REF_CENTER    = (cx, cy, cz)         # fixed center point
    REF_ORIENT    = ("x", "y")           # fixed unit direction
    

    # ------------------------------------------------------------
    # Real detector geometry: delays + antenna patterns
    # ------------------------------------------------------------

    # Example: GW150914-like context.  Read from Config.
    profile      = config.get("profile",  None);
    if(profile is not None): profile = str(profile)

    event_utc    = config.get("event_utc",  None) #(2015, 9, 14, 9, 50, 45.391) 2015-09-14 09:50:45.391
    ra_deg       = float(config.get("ra_deg",     111.71))
    dec_deg      = float(config.get("dec_deg",    -72.28))
    # Ensure proper RA / DEC Coordinates.
    ra_deg, dec_deg = canonicalize_radec(ra_deg, dec_deg)
    pol_psi_deg  = float(config.get("pol_psi_deg", 80.00))  # polarization angle ψ

    if(event_utc is not None):
        event_utc = tuple(event_utc)
        sidereal_date_str = (f"{event_utc[0]}-{event_utc[1]:02d}-{event_utc[2]:02d}T"
                        f"{event_utc[3]:02d}:{event_utc[4]:02d}:{event_utc[5]:06.3f}Z")
    else: sidereal_date_str = None
    sky_location_note=f"Orientation (RA~{ra_deg} deg, Dec~{dec_deg} deg)"
    #sky_location_note=f"Orientation (RA≈{ra_deg}°, Dec≈{dec_deg}°)"


    geo, k_dir, gmst = compute_detector_geo(Sites_Used, ra_deg=ra_deg,dec_deg=dec_deg,utc_tuple=event_utc,pol_psi_deg=pol_psi_deg, c_phys=c_phys)


    # Build detectors dict from geo solution
    detectors = {}
    for name, info in geo.items():
        detectors[name] = {
            # physics
            "F_plus":        float(info["Fplus"]),
            "F_cross":       float(info["Fcross"]),
            "geom_delay_sec":float(info["geom_delay_sec"]),
            # geometry (for metadata / debugging)
            "u_arm_ecef":    np.asarray(info["u"], dtype=float),
            "v_arm_ecef":    np.asarray(info["v"], dtype=float),
            "site_ecef":     np.asarray(info["site_ecef"], dtype=float),
        }
    if not detectors: raise RuntimeError("No detectors defined from compute_detector_geo.")

    
    # === Compute largest geometric delay across detectors (seconds) ===
    print("\n=== UMH Chirp Generator: Geometry Diagnostics ===")
    print(f"Source RA={ra_deg:.3f} deg, Dec={dec_deg:.3f} deg, psi={pol_psi_deg:.3f} deg, IOTA={BINARY_IOTA_DEG:.3f} deg")
    print(f"Distance={distance_Mpc}")
    print(f"Event UTC={sidereal_date_str}")
    # List detectors with delays and antenna patterns
    for name, info in geo.items():
        print(f"{name}: geom_delay={info['geom_delay_sec']*1e3:+7.3f} ms, F_plus={info['Fplus']:+.3f}, F_cross={info['Fcross']:+.3f}")
    # Which Detector is first.
    ref_det = min(geo.items(), key=lambda kv: kv[1]['geom_delay_sec'])[0]
    print(f"Reference (earliest arrival by geometry): {ref_det}")

    # For GW150914 sanity: check H1/L1 ordering
    if 'Hanford' in geo and 'Livingston' in geo:
        geo_diff = (geo['Hanford']['geom_delay_sec'] - geo['Livingston']['geom_delay_sec'])
        print(f"H - L delay (geom, this config): {geo_diff*1e3:+7.3f} ms (+ = Hanford later, - = Hanford earlier)")
        # Expected ≈ +6.9 ms for GW150914 (L1 first)

    name_list = list(detectors.keys())

    # --- PN-0 (Newtonian) inspiral frequency sweep setup ---
    # --- PN-0 duration for f_min_src → f_merge_src_eff (use the same target we will splice at) ---
    # Choose the *actual* merge target for PN timing (bounded by the band)
    f_merge_use_pn = float(np.clip(f_merge_src_eff, f_min_src + 3.0, 0.90 * highcut_src))

    # --- PN-0 duration for f_min_src → f_merge_src_eff ---
    # Example target: 35→250 Hz with (36,29) Msun
    T_pn_src = pn_duration_sec(M_chirp_src_kg, f_min_src, f_merge_use_pn, G_phys, c_phys)  # ~0.19 s  # duration of inspiral until f_merge_src_eff
    t_merge_src = T_pn_src                        # when inspiral reaches f_merge_src_eff
    t_merge_obs = t_merge_src * z_factor
    # Largest geometric delay across detectors (already in seconds)
    tau_max = max(det["geom_delay_sec"] for det in detectors.values())

    RD_TAIL_SEC = max(10.0 * tau_rd_src, 0.08)
    # total duration: inspiral (to f_merge_src_eff) + merge window + ringdown tail
    T_total_src = t_merge_src + RD_TAIL_SEC              # source-frame physics
    # total duration: inspiral (to f_merge_obs_eff) + merge window + ringdown tail, expressed in *observer* seconds
    T_total_obs = t_merge_obs + tau_max + RD_TAIL_SEC    # observer-frame window

    Nt_obs = int(np.ceil(T_total_obs / dt_obs))
    t_max = Nt_obs * dt_obs; tc_eff = t_merge_src; tc_dynamic = None
    print(f"[PN] Duration {f_min_src:.1f}→{f_merge_use_pn:.1f} Hz = {T_pn_src:.3f} s | "
          f"t_merge_src={t_merge_src:.3f}s | T_total_src={T_total_src:.3f}s | Nt_obs={Nt_obs} | f_rd_src_eff≈{f_rd_src_eff:.1f} Hz | f_merge_src_eff={f_merge_src_eff:.1f}")

    #Term and Kn Defined.
    term = (G_phys * M_chirp_src_kg) / (c_phys**3)
    K_N  = (96.0 / 5.0) * (math.pi**(8.0/3.0)) * (term**(5.0/3.0))

    # If using UMH / PN-style high-order df/dt, rescale K_N so that the inspiral f_min_src -> f_merge_src_eff duration matches the target t_merge_src.
    if USE_UMH_DFDT_PROFILE and (UMH_DFDT_COEFFS is not None):
        T_dfdt = estimate_merge_time_from_dfdt(K_N, f_min_src, f_merge_src_eff, f_ref_src, UMH_DFDT_COEFFS, n_steps=16000)
        if(config.get("SPIN_PRESERVE_PN0_DURATION", True)):
            if (T_dfdt is not None) and (T_dfdt > 0.0):
                scale = T_dfdt / t_merge_src   # T(K_N) = I / K_N → K_eff = K_N * (T_dfdt / t_merge_src)
                K_N *= scale
                print(f"[UMH dfdt] Base T_dfdt={T_dfdt:.6f}s, target t_merge_src={t_merge_src:.6f}s, rescaling K_N by {scale:.6f}")
        else: 
            t_merge_src = T_dfdt
            print(f"[UMH dfdt] Base T_dfdt={T_dfdt:.6f}s, target t_merge_src={t_merge_src:.6f}s, not rescaling K_N, SPIN_PRESERVE_PN0_DURATION=False")
        dfdt_merge = dfdt_highorder(K_N, f_ref_src, f_merge_src_eff, coeffs=UMH_DFDT_COEFFS)
    else: dfdt_merge = K_N * (f_merge_src_eff ** (11.0 / 3.0))  # Newtonian fallback if profile disabled

    # --- Clamp the merge slope for a stable Hermite ramp ---
    dfdt_merge_eff = dfdt_merge
    delta_f = max(f_rd_src_eff - f_merge_src_eff, 0.0)
    if RING_MERGE_C1_BLEND_SEC > 0.0:
        secant  = delta_f / RING_MERGE_C1_BLEND_SEC
        # Fritsch–Carlson style monotone limit: s0 <= 3 * secant
        max_s0  = 3.0 * secant
        if dfdt_merge_eff > max_s0:
            print(f"[Hermite clamp] dfdt_merge={dfdt_merge_eff:.3f} > 3*Δf/Δt={max_s0:.3f}, clamping.")
            dfdt_merge_eff = max_s0

    if RING_MERGE_C1_BLEND_SEC > 0.0: ringdown_merge_strategy = "Local C¹ Hermite ramp of f_gw plus raised-cosine amplitude over RING_MERGE_C1_BLEND_SEC near merger; elsewhere analytic track unchanged."
    else: ringdown_merge_strategy = "relaxation-only (RELAX_FREQ + EXP_AMP, no explicit merge window; constrained-frequency transition at t_merge_src)"
    if SMOOTH_FGW and TAU_SMOOTH_SEC > 0.0: frequency_smoothing = "Global one-pole exponential smoothing of f_gw(t) with time constant TAU_SMOOTH_SEC (ensures continuous f_gw and phase derivative; does not retune QNM)."
    else: frequency_smoothing = "None"

    # Default Arrays Setup.
    if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")):
        # Setup Soliton Arrays.
        PML_N    = int(config.get("PML_THICKNESS", 8))
        phi      = np.zeros((Nx, Ny, Nz), dtype=dtype)
        phi_prev = np.zeros_like(phi)
        phi_next = np.zeros_like(phi)
        
        A_raw_Sol         = np.zeros(Nt_obs, dtype=dtype)                                        # UMH soliton probe amplitude 
        sol_strain_record = np.zeros(Nt_obs, dtype=dtype)
        freq_record_sol   = np.zeros(Nt_obs, dtype=dtype) # Soliton-derived f_GW(t) (diagnostic, source frame)

    A_raw          = np.zeros(Nt_obs, dtype=dtype)                                        # UMH soliton probe amplitude 
    phase_hist     = np.zeros(Nt_obs, dtype=dtype)
    freq_record    = np.zeros(Nt_obs, dtype=dtype)
    radius_record  = np.zeros(Nt_obs, dtype=dtype)
    strain_records = {name: np.zeros(Nt_obs, dtype=dtype) for name in detectors}

    t_grid  = np.arange(Nt_obs) * dt_obs
    t_array = t_grid.copy()

    phase_gw    = 0.0                              # GW phase; advances by 2π f_gw dt
    phase_accum = 0.0
    
    # --- PN constant so that f_raw( t=0 ) = f_min_src ---
    tau0    = max(t_merge_src - 0.5*dt_src, 1e-3)  # time-to-coalescence at start
    A_PN    = f_min_src * (tau0/5.0)**(3.0/8.0)
    f0_raw  = A_PN * (5.0/tau0)**(3.0/8.0)  # equals f_min_src if calibrated above

    # Safety: if it's off from f_min_src due to config tweaks, snap or warn
    if f_min_src > 0 and abs(f0_raw - f_min_src) / f_min_src < 1e-3: f_start = f_min_src
    else: f_start = max(f_min_src, f0_raw)
    f_insp    = f_start
    f_gw      = f_start
    f_prev    = f_start
    f_smooth  = f_start

    if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")):
        f_sol_smooth = None   # current smoothed soliton f_gw estimate (Hz, source time)
        sol_win_len = max(64, int(round(SOL_FREQ_HILBERT_WINDOW_SEC / dt_src)))
        sol_win_len = min(sol_win_len, Nt_obs)
    else:  sol_win_len = 0

    ringdown_started = False
    A_merge   = 0.0                             # will capture amplitude at transition
    t_rd0     = None
    last_amp  = EPS_SAFE_FLOOR                  # track previous-step amplitude for continuity


    print(f"Total simulation steps (Nt_obs): {Nt_obs}, Total time (t_max): {t_max}")
    # Time evolution
    for t_idx in range(Nt_obs):
        time = t_idx * dt_src

        # Inspiral frequency via ODE (UMH/GR quadrupole)
        if (tc_dynamic is None):
            # --- Higher-order energy-balance ODE ---
            # dfdt_highorder(f) is used for UMH or GR-limit corrections.
            # The main loop presently uses either higher order or the leading-order (Newtonian) energy-balance form:
            #     df/dt = K_N * f^(11/3)
            # dfdt  = K_N * (max(f_insp, f_min_src)**(11.0/3.0))
            # This is set via USE_UMH_DFDT_PROFILE to use UMH-specific flux terms.
            if(UMH_DFDT_COEFFS is not None): dfdt = dfdt_highorder(K_N, f_ref_src, f_insp, coeffs=UMH_DFDT_COEFFS)
            else: dfdt  = K_N * (max(f_insp, f_min_src)**(11.0/3.0))

            f_trial = f_insp + dfdt * dt_src
            if (f_trial >= f_merge_src_eff):
                tc_dynamic = time + dt_src            # first step where we reach merge frequency
                tc_eff     = tc_dynamic
                f_insp     = f_merge_src_eff              # lock at exact f_merge_src_eff at tc_eff
                f_smooth   = f_merge_src_eff

                dfdt_merge = max(dfdt, EPS_FLOOR)
            else: f_insp = f_trial

        if (not RINGDOWN_ENABLE) or (tc_eff is None) or (time < tc_eff) or tc_dynamic is None: f_raw = f_insp  # Inspiral only (no ringdown), or we haven't reached merge yet  
        elif USE_QNM_OVERTONE_ATTACH: 
            if (time >= tc_eff) and (time <= tc_eff + RING_MERGE_C1_BLEND_SEC): f_raw = f_merge_src_eff; f_gw  = f_merge_src_eff
            else: f_raw = f_merge_src_eff
        elif (RING_MERGE_C1_BLEND_SEC > 0.0) and (time <= tc_eff + RING_MERGE_C1_BLEND_SEC):
            # C¹ Hermite ramp: f_merge_src_eff → f_rd_src_eff
            u = (time - tc_eff) / RING_MERGE_C1_BLEND_SEC
            u = max(0.0, min(1.0, u))
            h00 =  2.0*u**3 - 3.0*u**2 + 1.0
            h10 =      u**3 - 2.0*u**2 + u
            h01 = -2.0*u**3 + 3.0*u**2
            h11 =      u**3 -     u**2
            dfdt_merge_end = 0.0  # flat at f_rd_src_eff
            f_raw = (h00 * f_merge_src_eff + h10 * RING_MERGE_C1_BLEND_SEC * dfdt_merge_eff + h01 * f_rd_src_eff + h11 * RING_MERGE_C1_BLEND_SEC * dfdt_merge_end)
        else:
            if RING_MERGE_C1_BLEND_SEC <= 0.0:          # If no RING_MERGE_C1_BLEND_SEC, adjust to ringdown frequency.
                f_target = f_rd_src_eff
                adj = math.exp(-dt_src / float(FREQ_RELAX_SEC_EFF))
                # f_prev is last-step frequency. This makes the transition continuous.
                f_raw = f_target + (f_prev - f_target) * adj
            else: f_raw = f_rd_src_eff                  # RING_MERGE_C1_BLEND_SEC, it will already merge to ringdown.

        # Optional df/dt limiter
        f_inst = f_raw
        if (MAX_DFDT_HZ_S is not None) and (MAX_DFDT_HZ_S > 0.0):
            df_allowed = MAX_DFDT_HZ_S * dt_src
            # only enforce as an upper bound; inspiral is monotone increasing
            if f_inst > f_prev + df_allowed: f_inst = f_prev + df_allowed

        # Optional one-pole smoothing
        f_gw_candidate = f_inst
        if SMOOTH_FGW and (TAU_SMOOTH_SEC > 0.0):
            alpha = 1.0 - math.exp(-dt_src / TAU_SMOOTH_SEC)
            f_smooth += alpha * (f_inst - f_smooth)
            f_gw_candidate = f_smooth

        # Prevent inspiral frequency from decreasing due to numerical noise; allow flat ringdown afterward
        if time <= tc_eff + RING_MERGE_C1_BLEND_SEC: f_pn = max(f_gw_candidate, f_prev)    # Treat ramp as part of monotone rise to f_rd_src_eff
        else: f_pn = f_gw_candidate                             # after ramp, allow flat/decay

        # Optional soliton blend (source frame)
        if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")) and (f_sol_smooth is not None):
            if (SOL_FREQ_BLEND > 0.0 and SOL_FREQ_MODE == "blend"): f_gw = (1.0 - SOL_FREQ_BLEND) * f_pn + SOL_FREQ_BLEND * f_sol_smooth
            elif SOL_FREQ_MODE == "soliton": f_gw = f_sol_smooth # Soliton is authoritative, with a basic safety fallback
            else: f_gw = f_pn
        else: f_gw = f_pn

        f_prev = f_gw

        # Phase evolution
        phase_accum += 2.0 * np.pi * f_gw * dt_src
        phase_gw = phase_accum

        # Amplitude: inspiral vs ringdown envelope
        if not RINGDOWN_ENABLE:
            # No explicit ringdown model. Use inspiral amplitude up to tc_eff, then taper to zero over a short window.
            if time < tc_eff:
                x = AMP_RAMP_SLOPE * (time - (tc_eff - AMP_RAMP_START_SEC))
                sharp_rise = expit(x)
                modulation = 1.0 + AMP_RAMP_BOOST * sharp_rise
                
                # Newtonian f^(2/3) scaling
                amp_newt   = AMPLITUDE_SEED * ((f_gw + 1e-9) / f_min_src)**(2.0/3.0)

                # PN correction factor (no free params)
                if(USE_UMH_AMP_PN_PROFILE): pn_fac   = amplitude_pn_factor(f_gw, M_tot_kg_src, eta, max_order=UMH_AMP_PN_PROFILE)
                else: pn_fac = 1
                amp_phys   = amp_newt * pn_fac * modulation

            elif time <= tc_eff + RING_MERGE_C1_BLEND_SEC:
                # Linear or smooth taper from last_amp at tc_eff → 0
                u = (time - tc_eff) / RING_MERGE_C1_BLEND_SEC    # 0 → 1
                s = 3.0*u*u - 2.0*u*u*u              # smoothstep for continuity
                A_tc = max(last_amp, EPS_SAFE_FLOOR)
                amp_phys = (1.0 - s) * A_tc          # goes to 0 at end of window

            else: amp_phys = 0.0 # Fully off after taper window
                
        elif time < tc_eff:
            # Ringdown enabled & still inspiral: UMH/PN-style amplitude with modulation
            x = AMP_RAMP_SLOPE * (time - (tc_eff - AMP_RAMP_START_SEC))
            sharp_rise = expit(x)
            modulation = 1.0 + AMP_RAMP_BOOST * sharp_rise
            
            # Newtonian f^(2/3) scaling
            amp_newt = AMPLITUDE_SEED * ((f_gw + 1e-9) / f_min_src)**(2.0/3.0)

            # PN correction factor (no free params)
            if(USE_UMH_AMP_PN_PROFILE): pn_fac   = amplitude_pn_factor(f_gw, M_tot_kg_src, eta, max_order=UMH_AMP_PN_PROFILE)
            else: pn_fac = 1
            amp_phys     = amp_newt * pn_fac * modulation

        elif USE_QNM_OVERTONE_ATTACH: 
            # Hold amplitude at the value at t_c so A_attach is exactly A_tc.
            if not ringdown_started:
                ringdown_started = True
                A_tc  = max(last_amp, EPS_SAFE_FLOOR)
                t_rd0 = tc_eff
            amp_phys = A_tc   # no ramp / no exponential; QNMs will overwrite later
        elif (RING_MERGE_C1_BLEND_SEC > 0.0) and time <= tc_eff + RING_MERGE_C1_BLEND_SEC:
            # Smooth blend between A_tc and exponential ringdown
            if not ringdown_started:
                ringdown_started = True
                A_tc = max(last_amp, EPS_SAFE_FLOOR)
                t_rd0 = tc_eff

            u = (time - tc_eff) / RING_MERGE_C1_BLEND_SEC
            s = 3.0*u*u - 2.0*u*u*u
            A_rd = A_tc * math.exp(-(time - tc_eff) / tau_rd_src)
            amp_phys = (1.0 - s) * A_tc + s * A_rd

        else:
            # After ramp: pure exponential ringdown
            if not ringdown_started:
                ringdown_started = True
                A_tc = max(last_amp, EPS_SAFE_FLOOR)
                t_rd0 = tc_eff

            amp_phys = A_tc * math.exp(-(time - t_rd0) / tau_rd_src)

        if ONSET_TAPER_SEC > 0.0000:
            if time <= ONSET_TAPER_SEC:
                # Hann-like smooth ramp 0 -> 1 over [0, ONSET_TAPER_SEC]
                w = 0.5 * (1.0 - math.cos(math.pi * time / ONSET_TAPER_SEC))
                amp_phys *= w

        # Tail taper: smoothly ramp amplitude → 0 over the final AMP_RAMP_TAIL_SEC
        if AMP_RAMP_TAIL_SEC > 0.0:
            t_tail_start = T_total_src - AMP_RAMP_TAIL_SEC   # source-frame seconds
            if time >= t_tail_start:
                u_tail = (time - t_tail_start) / AMP_RAMP_TAIL_SEC
                # clamp to [0, 1]
                if u_tail < 0.0: u_tail = 0.0
                if u_tail > 1.0: u_tail = 1.0
                # Hann-like window from 1 → 0
                w_tail = 0.5 * (1.0 + math.cos(math.pi * u_tail))
                amp_phys *= w_tail


        # Track true envelope
        last_amp = amp_phys

        # Orbital radius from f_gw (using f_orb = f_gw/2)
        f_orb    = f_gw / 2.0
        r_si     = (G_phys * M_tot_kg_src / (2.0 * np.pi * f_orb)**2)**(1.0/3.0)
        r_grid   = np.clip(r_si / scale_factor, 1.0, 1e4)

        if (t_idx % int(0.25 / dt_src)) == 0: print(f"t_idx={time:6.2f}s  f_insp={f_insp:7.2f}  f_gw={f_gw:7.2f}  r_grid={r_grid:6.3f}")

        # Source 1: full 3D orbit with vertical oscillation
        theta   = 2 * np.pi * f_orb * time
        x1 = np.clip(int(round(cx + r_grid * np.cos(theta))), 1, Nx - 2)
        y1 = np.clip(int(round(cy + r_grid * np.sin(theta))), 1, Nx - 2)
        theta_z = theta + np.pi / 2
        cz1 = np.clip(int(round(cz + r_grid * 0.1 * np.sin(theta_z))), 1, Nx - 2)
        center1 = (x1, y1, cz1)

        # Source 2: apply phase offset only here
        theta_offset = theta + np.pi
        x2  = np.clip(int(round(cx + r_grid * np.cos(theta_offset))), 1, Nx - 2)
        y2  = np.clip(int(round(cy + r_grid * np.sin(theta_offset))), 1, Nx - 2)
        theta_z2 = theta_z + np.pi              # Opposite vertical phase
        cz2 = np.clip(int(round(cz + r_grid * 0.1 * np.sin(theta_z2))), 1, Nx - 2)
        center2 = (x2, y2, cz2)
    
        if t_idx % 1024 == 0: print(f"[Step:{t_idx}]: t={time:.3f}s, f_gw={f_gw:.2f}Hz, r_grid={r_grid:.2f}, amp_phys={amp_phys:.3e}")
        
        if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")):
            # Recompute solitons            
            radius = max(10.0, soliton_radius * (r_grid / r0))

            #USE_SOLITON_PROBE
            gaussian_soliton(phi, center1, radius, amp_phys)
            gaussian_soliton(phi, center2, radius, -amp_phys)
    
            # Finite difference update
            phi_next.fill(0.0)
            update_field(Nx,Ny,Nz,phi, phi_prev, phi_next, c, dt_src, damping_factor, freq_damping)
            apply_pml(phi_next,PML_N)

            a_inst = measure_strain(phi, REF_CENTER, REF_ORIENT, spacing = REF_SPACING) #det["orientation"]

            A_raw_Sol[t_idx] = np.abs(a_inst)

            dx = phi[Nx//2 + 1, Ny//2, Nz//2] - phi[Nx//2 - 1, Ny//2, Nz//2]
            dz = phi[Nx//2, Ny//2, Nz//2 + 1] - phi[Nx//2, Ny//2, Nz//2 - 1]
        
            falloff    = 1.0 / (r_grid**2 + 1e-4)              # Avoid runaway
            raw_strain = 0.5 * (dx + dz) / (2.0 * dt_src)

            sol_strain_record[t_idx] = raw_strain * falloff

            phi_prev[:], phi[:]  = phi[:], phi_next[:]
            
            A_raw[t_idx] = max(abs(amp_phys), EPS_FLOOR)

            # --- Soliton Hilbert instantaneous frequency estimate (source frame) ---
            if (SOL_FREQ_BLEND > 0.0) and (sol_win_len > 0) and (t_idx + 1 >= sol_win_len) and ((t_idx % SOL_FREQ_EVERY_STEPS) == 0):
                # Take a sliding window in source time
                start = t_idx + 1 - sol_win_len
                end   = t_idx + 1
                seg   = sol_strain_record[start:end].astype(float)

                # Avoid all-zero window
                if np.any(np.abs(seg) > 0.0):
                    analytic_seg = hilbert(seg)
                    phase_seg    = np.unwrap(np.angle(analytic_seg))

                    # Instantaneous frequency from phase derivative over the tail
                    dphi = np.diff(phase_seg)
                    if len(dphi) > 0:
                        # use last 1/4 of window to avoid startup transients
                        n_tail = max(4, sol_win_len // 4)
                        dphi_tail = dphi[-n_tail:]

                        f_tail = dphi_tail / (2.0 * np.pi * dt_src)  # Hz in source frame
                        f_raw  = float(np.median(f_tail))

                        # Clamp to a sensible band
                        if np.isfinite(f_raw):
                            f_raw = max(SOL_FREQ_MIN_HZ, min(SOL_FREQ_MAX_HZ, f_raw))

                            # One-pole smoothing in time
                            if f_sol_smooth is None: f_sol_smooth = f_raw
                            else:
                                alpha_sol = 1.0 - math.exp(-dt_src / SOL_FREQ_SMOOTH_TAU_SEC) if SOL_FREQ_SMOOTH_TAU_SEC > 0.0 else 1.0
                                f_sol_smooth = f_sol_smooth + alpha_sol * (f_raw - f_sol_smooth)

                            freq_record_sol[t_idx] = f_sol_smooth
            # ------------------------------------------------------------------------

        else: A_raw[t_idx] = max(abs(amp_phys), EPS_FLOOR) # PURE ANALYTIC ENVELOPE
         
        radius_record[t_idx] = r_grid
        freq_record[t_idx]   = f_gw
        phase_hist[t_idx]    = phase_gw

        if t_idx % 4096 == 0: print(f"[Step:{t_idx}]: t={time:.3f}s f_gw={f_gw:.2f}Hz phase={phase_gw:.2e}")
        #End Main Loop.


    # choose source envelope: analytic or soliton-probe
    if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")): 
        A_raw_phys = A_raw.copy()

        peak_sol     = float(np.max(np.abs(A_raw_Sol))) + EPS_FLOOR
        if peak_sol <= EPS_FLOOR: m_sol = np.ones_like(A_raw_Sol)
        else: m_sol  = A_raw_Sol / peak_sol

        # Smooth modulation: ~2 ms boxcar
        win_sec      = float(config.get("SOL_SMOOTH_WIN_SEC", 0.005)) #0.002
        win          = max(1, int(round(win_sec / dt_src)))
        if win > 1:
            ker      = np.ones(win, dtype=float) / float(win)
            m_sol    = np.convolve(m_sol, ker, mode="same")

        # Clamp modulation to avoid crazy boosts
        m_lo         = float(config.get("SOL_MOD_MIN", 0.1)) #0.8
        m_hi         = float(config.get("SOL_MOD_MAX", 1.9)) #1.2
        m_sol        = np.clip(m_sol, m_lo, m_hi)

        def normalize_modulation(m, eps=1e-12): m = np.asarray(m, float); mu = max(np.mean(m), eps); return m / mu
        def clamp_modulation(m, max_dev=0.5): lo, hi = 1.0 - max_dev, 1.0 + max_dev; return np.clip(m, lo, hi)
        def lowpass_modulation(m, fs, f_c_hz):
            """Low-pass a slow modulation (zero-phase IIR-free via FFT)."""
            if f_c_hz is None or f_c_hz <= 0: return m
            M = len(m); M2 = int(1 << (M-1).bit_length())   # power-of-two pad
            F = np.fft.rfft(m, n=M2)
            freqs = np.fft.rfftfreq(M2, d=1.0/fs)
            H = (freqs <= f_c_hz).astype(float)             # brick-wall; fine for envelope
            m_filt = np.fft.irfft(F * H, n=M2)[:M]
            return m_filt
        def post_attach_taper(t, t_attach, taper_ms):
            """Decay from 1 at t_attach to 0 afterward over 'taper_ms' (raised-cosine)."""
            if taper_ms <= 0: return np.ones_like(t)
            w = np.ones_like(t)
            T = taper_ms * 1e-3; mask = t >= t_attach
            x = np.minimum((t[mask] - t_attach) / T, 1.0)
            s = 3*x**2 - 2*x**3 # smoothstep-like: 0→1 then map to 1→0
            w[mask] = 1.0 - s
            return w  # 1 before attach; smoothly → 0 within T after attach

        # Prepare a gentle soliton mask
        m_sol = np.asarray(m_sol, float)
        m_sol = lowpass_modulation(m_sol, fs=fs, f_c_hz=SOL_ENV_SMOOTH_CUTOFF_HZ)
        m_sol = normalize_modulation(m_sol)                    # make mean 1.0
        m_sol = clamp_modulation(m_sol, max_dev=SOL_ENV_MAX_DEV)

        # Optionally taper out the modulation after attach to protect the tail
        w = post_attach_taper(t_grid, t_attach=t_merge_obs, taper_ms=SOL_ENV_POST_TAPER_MS)
        m_post = 1.0 + (m_sol - 1.0) * w

        # Choose blend mode
        if SOL_ENV_MODE   == "multiply": A_src = A_raw_phys * m_post # pure multiplicative modulation (strongest effect)
        elif SOL_ENV_MODE == "mix":      
            A_src = (1.0 - SOL_ENV_ALPHA) * A_raw_phys + SOL_ENV_ALPHA * (A_raw_phys * m_post) # convex blend between physical-only and multiplicative version
        else: A_src = A_raw_phys


        # --- Soliton-based instantaneous frequency (diagnostic only, source frame) ---
        # Use the soliton-measured strain at the reference point
        sol_signal = np.asarray(sol_strain_record, dtype=float)
        if not np.any(np.isfinite(sol_signal)): freq_record_sol[:] = 0.0 # No meaningful soliton data; keep as zeros
        else:
            # Clean up NaNs/Infs for Hilbert stability
            sol_signal = np.nan_to_num(sol_signal, nan=0.0, posinf=0.0, neginf=0.0)

            # Analytic signal via Hilbert transform
            analytic_sol = hilbert(sol_signal)
            phase_sol    = np.unwrap(np.angle(analytic_sol))

            # dφ/dt → f(t) in Hz (source frame, uses dt_src)
            f_sol = np.gradient(phase_sol, dt_src) / (2.0 * np.pi)

            # Optional smoothing to suppress jitter (controlled by SOL_FREQ_SMOOTH_TAU_SEC)
            if SOL_FREQ_SMOOTH_TAU_SEC > 0.0:
                alpha_sol = 1.0 - math.exp(-dt_src / SOL_FREQ_SMOOTH_TAU_SEC)
                f_sm = np.empty_like(f_sol)
                f_prev = f_sol[0]
                for i, val in enumerate(f_sol):
                    f_prev = f_prev + alpha_sol * (val - f_prev)
                    f_sm[i] = f_prev
                f_sol = f_sm

            # Clip to a reasonable band (configurable, source-frame Hz)
            f_sol = np.clip(f_sol, SOL_FREQ_MIN_HZ, SOL_FREQ_MAX_HZ)
            freq_record_sol[:] = f_sol

    else: A_src = A_raw


    if (RINGDOWN_ENABLE and USE_QNM_OVERTONE_ATTACH):
        # === multi-overtone ringdown attach (vectorized, analytic f_inst) ===
        i_attach   = int(np.searchsorted(t_grid, t_merge_obs, side="left"))
        i_attach   = max(0, min(i_attach, len(t_grid) - 1))
        t_attach   = float(t_grid[i_attach])
        # amplitude & phase at attach (already includes any soliton modulation pre-attach)
        A_attach   = float(A_src[i_attach])
        phi_attach = float(phase_hist[i_attach])


        # If user supplied QNM_OVERTONE_AMPS, respect them (nonnegative, normalized)
        A_n = None
        if qnm_amp_vector is not None:
            # truncate/extend to match number of modes
            raw = np.array([float(a) for a in qnm_amp_vector[:len(ring_down_modes)]], float)
            if raw.size < len(ring_down_modes):
                raw = np.pad(raw, (0, len(ring_down_modes)-raw.size), constant_values=0.0)
            raw = np.maximum(raw, 0.0)
            s   = raw.sum()
            if s > 0: A_n = (raw / s) * A_attach

        # Otherwise fall back to principled default
        if A_n is None:
            # Choose overtone weights (non-negative, normalized)
            A_n = compute_overtone_weights(A_attach=A_attach, modes_dict=ring_down_modes,
                use_continuity_fit=True, decay_ratio=qnm_decay_ratio)

        # Synthesize complex QNM sum and its instantaneous frequency (log-derivative)
        A_rd, phi_rd, f_rd_inst = synthesize_qnm_sum_and_freq(t=t_grid, t_attach=t_attach, A_attach=A_attach,
            phi_attach=phi_attach, modes_dict=ring_down_modes, A_n=A_n)

        # Overwrite post-attach region with the QNM sum
        A_src[i_attach:]       = A_rd[i_attach:]
        phase_hist[i_attach:]  = phi_rd[i_attach:]   # already unwrapped inside helper

        freq_record_rd = ensure_f_hist_from_phase(phase_hist, dt_src, dtype=dtype)

        freq_record_rd[i_attach:] = f_rd_inst[i_attach:]
        f_source    = "loop_inspiral + 22n_overtone_ringdown (analytic f_inst)"

        ns_sorted = sorted(ring_down_modes.keys())  # same order used in compute_overtone_weights
        overtone_entries = []
        for idx, n in enumerate(ns_sorted):
            f_n, tau_n = ring_down_modes[n]   # (Hz, sec)
            A = A_n[idx]                      # corresponding amplitude weight
            overtone_entries.append({"n": int(n), "f_22n_Hz": float(f_n), "tau_22n_sec": float(tau_n), "A_n": float(A)})
        qnm_overtone_info = {"USE_QNM_OVERTONE_ATTACH": True, "t_attach": float(t_attach),
            "phi_attach": float(phi_attach), "A_attach": float(A_attach), "overtones": overtone_entries,
            "QNM_OVERTONE_LIST": list(qnm_overtone_list),
            "QNM_OVERTONE_DECAY_RATIO": float(qnm_decay_ratio),
            "QNM_OVERTONE_AMPS_FROM_CONFIG": bool(qnm_amp_vector is not None)}

    elif (RINGDOWN_ENABLE): freq_record_rd = freq_record; f_source = "loop_pn_plus_ringdown"
    else: freq_record_rd = freq_record; f_source = "loop_inspiral, No Ringdown."

    phase_hist = np.unwrap(phase_hist)

    #No Smoothing necessary, looks clean without.
    #Optional: Boxcar Kernel smoothing, does not currently need any.
    if(bool(config.get("USE_BOXCAR", False))):
        win_sec = float(config.get("win_sec", 0.0015))  # ~2 ms boxcar (>> 1/f_min_src, << chirp timescale)
        win     = max(1, int(win_sec/dt_src))        
        ker     = np.ones(win) / win
        idx_tc = int(round(tc_eff / dt_src))

        A_hist_use      = np.empty_like(A_src)
        phase_hist_use  = np.empty_like(phase_hist)
        freq_record_use = np.empty_like(freq_record_rd)

        #Optional: Run a seperate kernel smooth on the ringdown, if desired.
        if(bool(config.get("USE_SEP_BOXCAR", False))):
            # inspiral side (t < tc_eff)
            A_hist_use[:idx_tc]       = np.convolve(A_src[:idx_tc + win//2], ker, mode="same")[:idx_tc]
            # ringdown side (t >= tc_eff)
            A_hist_use[idx_tc:]       =  np.convolve(A_src[idx_tc - win//2:], ker, mode="same")[win//2:]

            # inspiral side (t < tc_eff)
            phase_hist_use[:idx_tc]   = np.convolve(phase_hist[:idx_tc + win//2], ker, mode="same")[:idx_tc]
            # ringdown side (t >= tc_eff)
            phase_hist_use[idx_tc:]   =  np.convolve(phase_hist[idx_tc - win//2:], ker, mode="same")[win//2:]
    
            # inspiral side (t < tc_eff)
            freq_record_use[:idx_tc]  = np.convolve(freq_record_rd[:idx_tc + win//2], ker, mode="same")[:idx_tc]
            # ringdown side (t >= tc_eff)
            freq_record_use[idx_tc:]  =  np.convolve(freq_record_rd[idx_tc - win//2:], ker, mode="same")[win//2:]
        else:
            A_hist_use      = np.convolve(A_src,          ker, mode="same")
            phase_hist_use  = np.convolve(phase_hist,     ker, mode="same")
            freq_record_use = np.convolve(freq_record_rd, ker, mode="same")
    else:
        A_hist_use      = A_src.copy()
        phase_hist_use  = phase_hist.copy()
        freq_record_use = freq_record_rd.copy()
    
    freq_record_src = freq_record_use.copy()
    t_obs = t_array
    # --- Optional UMH frequency-redshift: detector-time stretching ---
    if APPLY_UMH_FREQ_REDSHIFT:
        if z_factor and np.isfinite(z_factor) and z_factor != 1.0:
            freq_record_obs = freq_record_src / z_factor
            print(f"[UMH time-mapping] Observer-frame time scaled by (1+z_UMH)={z_factor:.6f}; "
                  f"frequencies → f/(1+z_UMH), durations → (1+z_UMH)·t.")
        else: 
            freq_record_obs = freq_record_src
            print("[UMH z_factor] No valid redshift provided; skipping frequency-redshift.")
    else: freq_record_obs = freq_record_src

    # --- Physics-based normalization ---
    # If PHYSICS_NORM_ENABLE is True, set the overall amplitude using a
    # simple GR inspiral scaling:  h ~ (G M_chirp_obs / c^2)^(5/3) (π f_obs)^(2/3) / D_L
    # where f_obs is a characteristic observer-frame frequency (e.g. near f_merge_obs_eff) and D_L is the observer luminosity distance.
    # This is independent of APPLY_UMH_AMPLITUDE_SCALING and APPLY_UMH_FREQ_REDSHIFT: those only change how we define f_obs and
    # M_chirp_obs, not the structure of the scaling.
    # Amplitude scaling here is NOT a GR luminosity-distance correction. Distance is geometric (non-expanding).
    # UMH_z_tension affects amplitude via time dilation of energy transport (arrival rate + phase stretching), applied once globally.
    UMH_z_tension_amp_eff=0.0; D_L_norm_m = None
    if PHYSICS_NORM_ENABLE:
        f_hist = ensure_f_hist_from_phase(phase_hist_use, dt_obs, dtype=dtype) 
        #UMH-consistent quadrupole amplitude (no cosmological redshift expansion factor).
        idx    = int(np.nanargmin(np.abs(f_hist - f_ref_obs)))
        if(APPLY_UMH_AMPLITUDE_SCALING): UMH_z_tension_amp_eff = UMH_z_tension; D_L_norm_m = D_L_umh_m; norm_distance_desc = "full D_L^UMH"
        else: UMH_z_tension_amp_eff = 0.0; D_L_norm_m = d_geom_m; norm_distance_desc = "D_geom diagnostic"
        #h_ref  = newtonian_h_at_f_umh(f_ref_obs, M_chirp_src_kg, d_geom_m, z_tension=UMH_z_tension_amp_eff, G_phys=G_phys, c_phys=c_phys)
        h_ref = newtonian_h_at_f_umh(f_ref_src, M_chirp_src_kg, D_L_norm_m, G_phys=G_phys, c_phys=c_phys)
        A_ref  = float(np.abs(A_hist_use[idx]))
        epsA   = EPS_SAFE_FLOOR     #1e-24  # bigger than machine eps; avoids crazy gain
        if not np.isfinite(A_ref) or A_ref < epsA: 
            print(f"[PHYS_NORM] Warning: bad A_ref at f_ref_obs={f_ref_obs:.1f} Hz; skipping physics norm.")
            h_ref = None; G_amp = None; strain_scale_comment = ""
        else:
            G_amp = h_ref / A_ref
            # optional sanity clamp:
            if not (1e-30 <= abs(G_amp) <= 1e6): 
                PHYSICS_NORM_ENABLE  = False
                strain_scale_comment = ""
                print(f"[PHYS_NORM] Warning: G_amp={G_amp:.3e} out of range; skipping.")
            else:
                A_hist_use *= G_amp
                #print(f"[PHYS_NORM] Applied global gain G_amp={G_amp:.3e} at f_ref_obs={f_ref_obs:.1f} Hz.")
                print(f"[PHYS_NORM] Applied global gain G_amp={G_amp:.3e} at f_ref_obs={f_ref_obs:.3f} Hz / f_ref_src={f_ref_src:.3f} Hz using {norm_distance_desc}={D_L_norm_m:.6e} m.")
                strain_scale_comment = "Calibrated to physics scale (not fit). Peak |h| matches model at given distance."
    else: h_ref = None; G_amp = None; strain_scale_comment = ""
    # ---- End Single global physics normalization (scalar gain)


    # === Build minimal padded amplitude/phase grid for interpolation ===
    A_obs     = np.asarray(A_hist_use, dtype=dtype)
    phase_obs = np.unwrap(np.asarray(phase_hist_use, dtype=dtype))

    # Only a small pre-pad (e.g., a few cycles of the lowest band) for whitening safety
    pre_pad_sec  = max(0.02, 3.0 / max(f_min_obs, 1e-6))  # ≈3 cycles at f_min_obs
    post_pad_sec = 0.0                                # no artificial ringdown pad

    Npre  = int(np.ceil(pre_pad_sec / dt_obs))
    Npost = 0

    # Extend time axis slightly before the start, no post-extension
    t_ext = np.concatenate([t_obs[0] - dt_obs * np.arange(Npre, 0, -1), t_obs]).astype(dtype)

    # Pre-pad amplitude smoothly from 0→A_obs[0]
    if Npre > 0:
        r_pre   = np.linspace(0.0, np.pi, Npre, endpoint=False)
        A_pre   = A_obs[0] * 0.5 * (1.0 - np.cos(r_pre))
    else: A_pre = np.empty(0, dtype=A_obs.dtype)

    A_ext = np.concatenate([A_pre, A_obs]).astype(dtype)

    # Phase pre-extension (linear extrapolation of slope)
    slope0     = (phase_obs[1] - phase_obs[0]) / dt_obs if len(phase_obs) > 1 else 0.0
    phase_pre  = phase_obs[0] - slope0 * dt_obs * np.arange(Npre, 0, -1)
    phase_ext  = np.concatenate([phase_pre, phase_obs]).astype(dtype)

    # Monotone interpolators (no extrapolation)
    A_at      = PchipInterpolator(t_ext, A_ext, extrapolate=False)
    phase_at  = PchipInterpolator(t_ext, phase_ext, extrapolate=False)

    # Polarization scalars
    #cosi      = np.cos(BINARY_IOTA)             #BINARY_IOTA_DEG in radians.
    #APLUS     = 0.5 * (1.0 + cosi * cosi)
    #ACROSS    = cosi

    # Used to generate polarization for NPZ transparently.
    A_obs     = np.asarray(A_hist_use, dtype=dtype)
    phase_obs = np.unwrap(np.asarray(phase_hist_use, dtype=dtype))
    # Intrinsic source-frame polarizations on the same grid:
    
    SPIN_MODE = str(SPIN_MODE).lower(); precession_meta = None
    if SPIN_MODE in ("none", "aligned"): h_plus_obs, h_cross_obs = build_polarizations_nonprecessing(A_obs, phase_obs, BINARY_IOTA)
    elif SPIN_MODE == "precessing":
        t_peak_target = float(t_merge_obs)
        PRECESSION_STRENGTH=float(config.get("PRECESSION_STRENGTH", 1.0))
        PRECESSION_MAX_BETA_DEG=float(config.get("PRECESSION_MAX_BETA_DEG", 45.0))
        PRECESSION_MAX_HZ=float(config.get("PRECESSION_MAX_HZ", 64.0))
        PRECESSION_FREEZE_AFTER_PEAK=bool(config.get("PRECESSION_FREEZE_AFTER_PEAK", True))
        PRECESSION_INCLUDE_THOMAS_PHASE=bool(config.get("PRECESSION_INCLUDE_THOMAS_PHASE", False))
        PRECESSION_INCLUDE_POLARIZATION_ROTATION=bool(config.get("PRECESSION_INCLUDE_POLARIZATION_ROTATION", True))
        h_plus_obs, h_cross_obs, precession_meta = build_polarizations_precessing(t_obs=t_obs, dt_obs=dt_obs, A_obs=A_obs, phase_obs=phase_obs,
                                                                                  M1_solar_src=M1_solar_src, M2_solar_src=M2_solar_src, z_factor=z_factor,
                                                                                  spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, spin2x=spin2x, spin2y=spin2y, 
                                                                                  spin2z=spin2z, iota_rad=BINARY_IOTA, D_L_umh_Mpc=D_L_umh_Mpc,
                                                                                  f_min_obs=f_min_obs, t_peak_target=t_peak_target,
                                                                                  PRECESSION_STRENGTH=PRECESSION_STRENGTH,
                                                                                  PRECESSION_MAX_BETA_DEG=PRECESSION_MAX_BETA_DEG,
                                                                                  PRECESSION_MAX_HZ=PRECESSION_MAX_HZ,
                                                                                  PRECESSION_FREEZE_AFTER_PEAK=PRECESSION_FREEZE_AFTER_PEAK,
                                                                                  PRECESSION_INCLUDE_THOMAS_PHASE=PRECESSION_INCLUDE_THOMAS_PHASE,
                                                                                  PRECESSION_INCLUDE_POLARIZATION_ROTATION=PRECESSION_INCLUDE_POLARIZATION_ROTATION,
                                                                                  return_meta=True, G_phys=G_phys, c_phys=c_phys, M_sun=M_sun)
    else: raise RuntimeError(f"Unhandled SPIN_MODE={SPIN_MODE}")

    #h_plus_obs  = A_obs * APLUS  * np.cos(phase_obs)
    #h_cross_obs = A_obs * ACROSS * np.sin(phase_obs)

    # === Per-detector synthesis (no clipping needed; pad covers all delays) ===
    det_meta, strain_records_viz = {}, {}
    geom_delay, det_Fp, det_Fx, det_Sign = [], [], [], []
    for name, det in detectors.items():
        Fp      = det["F_plus"]
        Fx      = det["F_cross"]
        tau     = det["geom_delay_sec"]                    # geometric delay [s]

        # Delay before sampling (pad ensures we’re in-bounds)
        t_del   = t_array - tau

        # interpolate source polarizations at delayed times:
        hplus_d  = np.interp(t_del, t_obs, h_plus_obs,  left=0.0, right=0.0)
        hcross_d = np.interp(t_del, t_obs, h_cross_obs, left=0.0, right=0.0)

        # Detector form strain
        y_det = Fp * hplus_d + Fx * hcross_d

        strain_records[name]     = y_det.astype(dtype)
        strain_records_viz[name] = y_det.astype(dtype)

        yk = int(np.argmax(np.abs(y_det))); sp = float(y_det[yk])
        sign_pred = 1.0 if sp >= 0.0 else -1.0

        print(f"[{name}] strain max={np.max(np.abs(strain_records[name])):.3e}, rms={np.std(strain_records[name]):.3e}")
        print(f"[{name}] Fp={Fp}, Fx={Fx}")
        det_meta[name] = {
            "F_plus":        float(Fp),
            "F_cross":       float(Fx),
            "geom_delay_sec":float(tau),
            "R_eff":         float(np.hypot(Fp, Fx)),
            "sign_pred":     float(sign_pred),
            "site_ecef":     det["site_ecef"].tolist(),
        }
        geom_delay.append(float(tau)); det_Fp.append(float(Fp)); det_Fx.append(float(Fx)); det_Sign.append(float(sign_pred))
    
    if not strain_records: raise RuntimeError("No detector strain records found.")

    # --- Save arrays to NPZ (arrays only; metadata goes to JSON) ---
    detector_names = np.array(list(strain_records.keys()), dtype='U32')  # unicode strings, no pickle
    name_list = list(strain_records.keys())

    #Referenced Detector, first detector that received the signal
    reference_detector = min(geo.keys(), key=lambda k: geo[k]["geom_delay_sec"])


    # --- Quantify Phase Accuracy ---
    # DIAGNOSTIC ONLY: Does not alter raw physics, Quantify Phase Accuracy
    phase_check = quantify_phase_accuracy(reference_detector, dt_obs, t_array, strain_records, phase_hist_use, freq_record_obs, 
                                          lowcut_obs, highcut_obs, Fp=detectors[reference_detector]["F_plus"], 
                                          Fx=detectors[reference_detector]["F_cross"], iota=BINARY_IOTA)
    # --- Detector-weighted phase QA (rigorous, detector-weighted metric) ---
    geom_delays = {name: info["geom_delay_sec"] for name, info in detectors.items()}
    phase_weighted = detector_weighted_phase_check(dt=dt_obs, t_array=t_array, strain_records=strain_records, phase_hist_use=phase_hist_use,
                                    geom_delays=geom_delays, amp_threshold=0.10, lowcut=lowcut_obs, highcut=highcut_obs, taper_edge_sec=0.05)
    # --- End Quantify Phase Accuracy ---


    # Convert to numpy-friendly arrays for archival
    if(UMH_DFDT_COEFFS is not None):
        UMH_keys = np.array(list(UMH_DFDT_COEFFS.keys()))
        UMH_vals = np.array(list(UMH_DFDT_COEFFS.values()), dtype=float)
        # Ensure coefficients are standard Python floats (not numpy types)
        UMH_DFDT_COEFFS_JSON = {k: float(v) for k, v in UMH_DFDT_COEFFS.items()}

        dfdt_orders_inc=["0PN"]
        if(UMH_DFDT_PN_PROFILE>=1.0): dfdt_orders_inc.append("1PN")
        if(UMH_DFDT_PN_PROFILE>=1.5): dfdt_orders_inc.append("1.5PN")
        if(UMH_DFDT_PN_PROFILE>=2.0): dfdt_orders_inc.append("2PN")
        if(UMH_DFDT_PN_PROFILE>=2.5): dfdt_orders_inc.append("2.5PN")
        if(UMH_DFDT_PN_PROFILE>=3.0): dfdt_orders_inc.append("3PN")
        if(UMH_DFDT_PN_PROFILE>=3.5): dfdt_orders_inc.append("3.5PN")
        
        SPIN_MODE = str(SPIN_MODE).lower()
        if SPIN_MODE == "none": spin_effects_note = "not included; SPIN_MODE='none', component spins forced to zero"
        elif SPIN_MODE == "aligned":
            spin_effects_note = ("aligned-spin, nonprecessing support: component z-spins enter the NR/EOB remnant fit and ringdown; leading spin-orbit df/dt term is "
                                 "included only when UMH_DFDT_PN_PROFILE >= 1.5")
        else:
            spin_effects_note = ("approximate precessing support: scalar df/dt uses only the z-projected spin components when UMH_DFDT_PN_PROFILE >= 1.5; "
                                 "transverse spin enters through the simple-precession polarization wrapper, not through scalar df/dt.")
        if SPIN_MODE in ("aligned", "precessing") and UMH_DFDT_PN_PROFILE >= 1.5:
            coeff_note = (f"Coefficients C2..C7 are GR-limit quasi-circular coefficients evaluated at f_ref_src = {f_ref_src} Hz, with the leading aligned-spin spin-orbit "
                          f"term included in C3 through beta_SO.")
        else: coeff_note = (f"Coefficients C2..C7 are standard nonspinning GR-limit quasi-circular coefficients evaluated at f_ref_src = {f_ref_src} Hz.")
        dfdt_model = {"dfdt_model": "UMH_highorder_dfdt",
          "orders_included": dfdt_orders_inc, #["0PN", "1PN", "1.5PN", "2PN", "2.5PN", "3PN", "3.5PN"],
          "spin_effects": spin_effects_note,
          "implementation_note": coeff_note + "\n"+ "In this run, the UMH phase evolution is locked to the GR limit (UMH_DFDT_COEFFS = GR); "
          "The UMH source--observer mapping is applied by generating the chirp with source-frame masses and source-frame frequencies, "
          "then mapping to observer-frame frequency/time through z_factor = 1 + z_UMH. "
          "Amplitude normalization uses the full UMH effective luminosity distance D_L^UMH, including the Pantheon+-calibrated transmission factor."}
          #"In this implementation, UMH_z_tension is applied only through the effective distance D_eff in the amplitude. The df/dt and phase evolution are computed in the "
          #"GR limit using source-frame masses; any UMH-induced modification to observed frequencies is intentionally not included here."}
    else: dfdt_model = {"dfdt_model": "None"}


    # Amplitude PN metadata block
    if USE_UMH_AMP_PN_PROFILE:
        if UMH_AMP_PN_PROFILE >= 3.0:
            amp_ins_model = ("3PN-corrected amplitude: h(f) = h_Newtonian * (1 + a2 v^2 + a3 v^3 + a4 v^4 + a5 v^5 + a6 v^6)")
            #amp_ins_model = ("3PN-corrected amplitude: h(f) = h_Newtonian × (1 + a₂ v² + a₃ v³ + a₄ v⁴ + a₅ v⁵ + a₆ v⁶)")
            amp_higher_order_PN = ("Includes amplitude PN corrections up to 3PN order (dominant 2,2 mode; Newtonian amplitude multiplied by full 2PN–3PN series).")
        elif UMH_AMP_PN_PROFILE >= 2.5:
            amp_ins_model = ("2.5PN-corrected amplitude: h(f) = h_Newtonian * (1 + a2 v^2 + a3 v^3 + a4 v^4 + a5 v^5)")
            #amp_ins_model = ("2.5PN-corrected amplitude: h(f) = h_Newtonian × (1 + a₂ v² + a₃ v³ + a₄ v⁴ + a₅ v⁵)")
            amp_higher_order_PN = ("Includes amplitude PN corrections up to 2.5PN order (dominant 2,2 mode).")
        elif UMH_AMP_PN_PROFILE >= 2.0:
            amp_ins_model = ("2PN-corrected amplitude: h(f) = h_Newtonian * (1 + a2 v^2 + a3 v^3 + a4 v^4)")
            #amp_ins_model = ("2PN-corrected amplitude: h(f) = h_Newtonian × (1 + a₂ v² + a₃ v³ + a₄ v⁴)")
            amp_higher_order_PN = ("Includes amplitude PN corrections up to 2PN order (dominant 2,2 mode; first PN layer affecting amplitude).")
        else:
            # Below 2PN = physically equivalent to Newtonian for amplitude
            amp_ins_model = ("Amplitude PN requested <2PN, but amplitude PN begins at 2PN; result = Newtonian (0PN) amplitude only.")
            amp_higher_order_PN = ("No valid amplitude PN terms below 2PN; using pure Newtonian amplitude.")
    else: amp_ins_model = "Newtonian (0PN) amplitude: h(f) ~ [M_c^(5/3) f^(2/3) / D_UMH]"; amp_higher_order_PN = "Amplitude PN corrections disabled (pure 0PN)."

    
    if config.get("GENERATE_NPZ", True):
        # --- Main NPZ File Containing all of the RAW Physics ---
        # "_Dynamic.npz" contains:
        #   - A_obs_probe: UMH diagnostic composite (not used for detector comparison)
        #   - freq_track_Hz, phase_track_rad: canonical GW track
        #   - A_hist_use: composite amplitude envelope after smoothing
        #
        # Downstream comparison code SHOULD:
        #   - Use detector-specific strain arrays from the main NPZ (below), not
        #     A_obs_probe, for correlations with LIGO/Virgo data.
        np.savez(f"{file_path}_Dynamic.npz",
            profile             = None if profile is None else str(profile),
            event_utc           = None if sidereal_date_str is None else str(sidereal_date_str),

            SOURCE_ENGINE       = str(SOURCE_ENGINE),
            A_obs_probe         = A_obs,
            radius              = None if radius_record is None else radius_record,
            time                = t_array,
            dt_src              = float(dt_src),
            dt_obs              = float(dt_obs),
            Nt_obs              = int(len(t_array)),
            fs_src              = float(fs_src),
            fs_obs              = float(fs_obs),
            Fn_src              = float(Fn_src),
            Fn_obs              = float(Fn_obs),
            t_max               = float(t_max),
            t_merge_src         = float(t_merge_src),
            t_merge_obs         = float(t_merge_obs),
            d_geom_m            = float(d_geom_m),
            M1_solar_src        = float(M1_solar_src),
            M2_solar_src        = float(M2_solar_src),
            M1_kg_src           = float(M1_kg_src),
            M2_kg_src           = float(M2_kg_src),
            distance_Mpc        = float(distance_Mpc),
            M_chirp_src_kg      = float(M_chirp_src_kg),
            soliton_radius      = None if soliton_radius is None else soliton_radius,
            AMPLITUDE_SEED      = AMPLITUDE_SEED,
            freq_track_src_Hz   = freq_record_src,
            freq_track_obs_Hz   = freq_record_obs,
            phase_track_rad     = phase_hist_use,
            f_min_src           = float(f_min_src), f_merge_src=float(f_merge_src), f_merge_src_eff=float(f_merge_src_eff),
            f_min_obs           = float(f_min_obs), f_merge_obs=float(f_merge_obs), f_merge_obs_eff=float(f_merge_obs_eff),
            lowcut_src          = float(lowcut_src), highcut_src=float(highcut_src),
            lowcut_obs          = float(lowcut_obs), highcut_obs=float(highcut_obs),
            highcap_obs         = float(highcap_obs),

            f_rd_src            = float(f_rd_src),
            f_rd_src_eff        = float(f_rd_src_eff),
            f_rd_obs            = float(f_rd_obs),
            f_rd_obs_eff        = float(f_rd_obs_eff),
            tau_rd_src          = float(tau_rd_src),
            tau_rd_obs          = float(tau_rd_obs),
            pre_pad_sec         = float(pre_pad_sec),
            post_pad_sec        = float(post_pad_sec),
            h_plus_obs          = h_plus_obs.astype(dtype),
            h_cross_obs         = h_cross_obs.astype(dtype),

            PHYSICS_NORM_ENABLE = bool(PHYSICS_NORM_ENABLE),
            APPLY_UMH_AMPLITUDE_SCALING = bool(APPLY_UMH_AMPLITUDE_SCALING),
            UMH_z_tension       = None if UMH_z_tension is None else float(UMH_z_tension),
            z_source            = None if z_source is None else z_source,
            z_GR                = None if z_GR is None else float(z_GR),
            UMH_z_beta1         = None if beta1 is None else float(beta1),
            UMH_z_beta2         = None if beta2 is None else float(beta2),
            UMH_z_M_best        = None if M_best is None else float(M_best),
            H0_km_s_Mpc         = None if H0_km_s_Mpc is None else float(H0_km_s_Mpc),
            
            f_ref_src           = None if f_ref_src is None else float(f_ref_src),
            f_ref_obs           = None if f_ref_obs is None else float(f_ref_obs),
            A_PN                = float(A_PN),

            # Main NPZ: detector-frame strains
            # For each detector name in detector_names, the corresponding strain array:
            #   strain_{det} = analytic PN+ringdown track
            #                  × global physics normalization
            #                  × [F_plus, F_cross, inclination]
            #                  with appropriate geometric delay.
            # These are the ONLY arrays intended for quantitative comparison to data.
            detector_names      = detector_names,
            # store per-detector strains as separate arrays (no pickle)
            **{f"strain_{name}": np.asarray(strain_records[name]) for name in name_list},

            UMH_keys            = None if(UMH_DFDT_COEFFS is None) else UMH_keys,
            UMH_vals            = None if(UMH_DFDT_COEFFS is None) else UMH_vals,
            
            ra_deg              = float(ra_deg),
            dec_deg             = float(dec_deg),
            pol_psi_deg         = float(pol_psi_deg),
            BINARY_IOTA_DEG     = float(BINARY_IOTA_DEG),
            geom_delay          = np.array(geom_delay, dtype=float),
            det_Fp              = np.array(det_Fp,     dtype=float),
            det_Fx              = np.array(det_Fx,     dtype=float),
            det_Sign            = np.array(det_Sign,   dtype=float),
            dtype               = dtype,
        )

        # --- Save frequency evolution (handy for QA) ---
        np.savez(f"{file_path}_Freq.npz", t=np.arange(Nt_obs)*dt_obs, freq=freq_record_obs.astype(dtype), f_for_downstream=f_insp, dtype=dtype)

    # --- Rich JSON metadata for peer review (antenna patterns etc.) ---
    SPIN_MODE = str(SPIN_MODE).lower()
    if SPIN_MODE == "none": spin_model_note = "nonspinning; dominant (2,2) nonprecessing polarization"
    elif SPIN_MODE == "aligned": spin_model_note = "aligned-spin nonprecessing; dominant (2,2) polarization"
    else: spin_model_note = ("approximate simple-precession dominant-mode wrapper; time-dependent iota(t) and polarization rotation psi(t) are applied; "
                             "not a full IMR precessing model and does not include higher modes")
    meta = {
        "profile":          None if profile is None else str(profile),
        "event_utc":        None if sidereal_date_str is None else str(sidereal_date_str),

        "distance_Mpc":     float(config.get("distance_Mpc", 410.0)),

        "M1_kg_src":        float(M1_kg_src), "M1_solar_src": float(M1_solar_src),
        "M2_kg_src":        float(M2_kg_src), "M2_solar_src": float(M2_solar_src),
        "M_chirp_src_kg":   float(M_chirp_src_kg),

        "M1_kg_obs":        float(M1_kg_src * z_factor), "M1_solar_obs": float(M1_solar_src * z_factor),
        "M2_kg_obs":        float(M2_kg_src * z_factor), "M2_solar_obs": float(M2_solar_src * z_factor),
        "M_chirp_obs_kg":   float(M_chirp_src_kg * z_factor),
        
        "spin_model":       spin_model_note,
        "spin_mode":        str(SPIN_MODE),
        "spin1x": float(spin1x), "spin1y": float(spin1y), "spin1z": float(spin1z),
        "spin2x": float(spin2x), "spin2y": float(spin2y), "spin2z": float(spin2z),
        "precession_meta":  precession_meta,

        "remnant_fit_approximant": str(remnant_approximant),
        "remnant_mass_Msun_src":   float(Mrem_solar_src),
        "remnant_spin_src":        float(a_rem_src),
        "remnant_mass_Msun_obs":   float(Mrem_solar_src * z_factor),
        "E_rad_frac":       float(E_rad_frac),
        
        "inclination_rad":  float(BINARY_IOTA),
        "inclination_deg":  float(BINARY_IOTA_DEG),

        "ringdown_seed_rule":     ringdown_seed_rule,
        "tau_seed_rule":          tau_seed_rule,
        "qnm_diagnostic":         qnm_diagnostic,
        "qnm_used_for_active_ringdown": bool(RINGDOWN_SEED_MODE == "qnm_diagnostic"),
        "umh_remnant_mode": None if umh_mode is None else umh_mode,
        "umh_qnm_compare":  umh_qnm_compare,

        "f_min_src":        float(f_min_src),
        "f_min_obs":        float(f_min_obs),
        "f_merge_src":      float(f_merge_src),
        "f_merge_src_eff":  float(f_merge_src_eff),
        "f_merge_obs":      float(f_merge_obs),
        "f_merge_obs_eff":  float(f_merge_obs_eff),
        "merge_rule":       merge_rule,
        "driver_timescale_merge_src": driver_timescale_merge_src,
        "lowcut_src":       float(lowcut_src), "highcut_src": float(highcut_src),
        "lowcut_obs":       float(lowcut_obs), "highcut_obs": float(highcut_obs),
        "highcap_obs":      float(highcap_obs),

        "f_rd_src_Hz":      float(f_rd_src),
        "f_rd_src_eff_Hz":  float(f_rd_src_eff),
        "f_rd_obs_Hz":      float(f_rd_obs),
        "f_rd_obs_eff_Hz":  float(f_rd_obs_eff),
        "tau_rd_src":       float(tau_rd_src),
        "tau_rd_obs":       float(tau_rd_obs),

        "qnm_f_220_src_Hz":         (None if qnm_diagnostic is None else float(qnm_diagnostic["f_220_src_Hz"])),
        "qnm_tau_220_src_s":        (None if qnm_diagnostic is None else float(qnm_diagnostic["tau_220_src_s"])),
        "qnm_f_220_obs_Hz":         (None if qnm_diagnostic is None else float(qnm_diagnostic["f_220_obs_Hz"])),
        "qnm_tau_220_obs_s":        (None if qnm_diagnostic is None else float(qnm_diagnostic["tau_220_obs_s"])),

        "f_isco_src_Hz":    float(f_isco_src),
        "f_isco_gw_src_Hz": float(f_isco_gw_src),
        "f_isco_obs_Hz":    float(f_isco_obs),
        "f_isco_gw_obs_Hz": float(f_isco_gw_obs),

        "A_PN":             float(A_PN),
        "dt_src":           float(dt_src),
        "dt_obs":           float(dt_obs),
        "fs_src":           float(fs_src),
        "fs_obs":           float(fs_obs),
        "Nt_obs":           int(Nt_obs),
        "t_max":            float(t_max),
        "t_merge_src":      float(t_merge_src),
        "t_merge_obs":      float(t_merge_obs),
        "ringdown_merge_strategy":      ringdown_merge_strategy,
        "frequency_smoothing":          frequency_smoothing,
        "pre_pad_sec":      float(pre_pad_sec),
        "post_pad_sec":     float(post_pad_sec),
        
        "f_ref_src":        None if f_ref_src is None else float(f_ref_src),
        "f_ref_obs":        None if f_ref_obs is None else float(f_ref_obs),
        "f_ref_obs_desc":   None if f_ref_obs is None else str(f_ref_obs_desc),

        "FREQ_RELAX_SEC":       None if FREQ_RELAX_SEC     is None else float(FREQ_RELAX_SEC),
        "FREQ_RELAX_KAPPA":     None if FREQ_RELAX_KAPPA   is None else float(FREQ_RELAX_KAPPA),
        "FREQ_RELAX_BETA":      None if FREQ_RELAX_BETA    is None else float(FREQ_RELAX_BETA),
        "FREQ_RELAX_BETA_CAL":  None if FREQ_RELAX_BETA_CAL is None else float(FREQ_RELAX_BETA_CAL),
        "FREQ_RELAX_MIN_SEC":   None if FREQ_RELAX_MIN_SEC is None else float(FREQ_RELAX_MIN_SEC),
        "FREQ_RELAX_MAX_SEC":   None if FREQ_RELAX_MAX_SEC is None else float(FREQ_RELAX_MAX_SEC),
        "FREQ_RELAX_SEC_EFF":   None if config["FREQ_RELAX_SEC_EFF"]   is None else float(config["FREQ_RELAX_SEC_EFF"]),
        "FREQ_RELAX_KAPPA_EFF": None if config["FREQ_RELAX_KAPPA_EFF"] is None else float(config["FREQ_RELAX_KAPPA_EFF"]),
        "FREQ_RELAX_SEC_OBS_EFF": float(FREQ_RELAX_SEC_EFF) * z_factor,

        "RING_MERGE_C1_BLEND_SEC":  None if RING_MERGE_C1_BLEND_SEC is None else float(RING_MERGE_C1_BLEND_SEC),

        "detector_names":   name_list,
        "antenna_patterns": det_meta,
        "polarizations":    {"convention": "h_det(t) = F_plus*h_plus_obs(t - tau_det) + F_cross*h_cross_obs(t - tau_det)",
                            "frame": "source-frame (GW propagation direction fixed by RA/Dec/psi)",
                            "stored_in_npz": ["h_plus_obs", "h_cross_obs", "t_array"]},
        "f_smoothing": {
            "SMOOTH_FGW":               bool(SMOOTH_FGW),
            "TAU_SMOOTH_SEC":           float(TAU_SMOOTH_SEC),
            "MAX_DFDT_HZ_S":            float(MAX_DFDT_HZ_S) if MAX_DFDT_HZ_S is not None else None
        },
        "strain_scale_comment":         str(strain_scale_comment),
        "geometry": {
            "ra_deg":                   ra_deg,
            "dec_deg":                  dec_deg,
            "event_utc":                list(event_utc),
            "pol_psi_deg":              pol_psi_deg,
            "gmst_rad":                 gmst,
            "propagation_vector_ecef":  k_dir.tolist(),
            "reference_detector":       reference_detector,
            "sidereal_date_str":        sidereal_date_str, #"2015-09-14T09:50:45Z",
            "sky_location_note":        sky_location_note, 
        },
        "UMH_DFDT_COEFFS":              "None" if(UMH_DFDT_COEFFS is None) else UMH_DFDT_COEFFS_JSON, 
        "phase_accuracy":               phase_check,
        "phase_accuracy_weighted":      phase_weighted,

        
        "Soliton_Settings": {
            "SOURCE_ENGINE":                str(SOURCE_ENGINE),
            "SOL_ENV_MODE":                 str(config.get("SOL_ENV_MODE", "mix")).lower()      if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")) else None,
            "SOL_ENV_ALPHA":                float(config.get("SOL_ENV_ALPHA",            0.35)) if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")) else None,
            "SOL_ENV_MAX_DEV":              float(config.get("SOL_ENV_MAX_DEV",          0.35)) if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")) else None,
            "SOL_ENV_POST_TAPER_MS":        float(config.get("SOL_ENV_POST_TAPER_MS",    0.35)) if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")) else None,
            "SOL_ENV_SMOOTH_CUTOFF_HZ":     float(config.get("SOL_ENV_SMOOTH_CUTOFF_HZ", 0.35)) if(SOURCE_ENGINE in ("soliton_umh", "hybrid_umh")) else None,
        },

        "UMH_z_Tension": {
            "UMH_z_tension":            None if UMH_z_tension is None else float(UMH_z_tension),
            "z_source":                 None if z_source is None else z_source,
            "z_GR_equivalent":          None if z_GR is None else float(z_GR),
            "UMH_z_beta1":              None if beta1 is None else float(beta1),
            "UMH_z_beta2":              None if beta2 is None else float(beta2),
            "UMH_z_M_best":             None if M_best is None else float(M_best),
            "H0_km_s_Mpc":              None if H0_km_s_Mpc is None else float(H0_km_s_Mpc),
        },

        "Physics_Normalization": {
            "PHYSICS_NORM_ENABLE":          bool(PHYSICS_NORM_ENABLE),
            "APPLY_UMH_AMPLITUDE_SCALING":  bool(APPLY_UMH_AMPLITUDE_SCALING),
            "UMH_z_tension_amp_eff":        None if UMH_z_tension_amp_eff is None else float(UMH_z_tension_amp_eff),
            
            "D_L_umh_input_Mpc":    float(D_L_umh_input_Mpc),
            "D_L_umh_Mpc":          float(D_L_umh_Mpc),
            "D_L_umh_m":            float(D_L_umh_m),
            "D_L_norm_m":           None if D_L_norm_m is None else float(D_L_norm_m),

            "d_geom_Mpc":           float(d_geom_Mpc),
            "d_geom_m":             float(d_geom_m),
            "T_umh":                None if T_umh is None else float(T_umh),

            "f_ref_obs_Hz":         None if f_ref_obs is None else float(f_ref_obs),
            "f_ref_src_Hz":         None if f_ref_src is None else float(f_ref_src),
            "normalization_frequency_convention":
                "Envelope sample selected at observer-frame f_ref_obs; quadrupole amplitude evaluated at source-frame f_ref_src using source-frame chirp mass.",

            "G_amp":                        None if G_amp is None else float(G_amp),
        },

        "freq_redshift": {
            "APPLY_UMH_FREQ_REDSHIFT":      bool(APPLY_UMH_FREQ_REDSHIFT),
            "z_freq_value":                 None if APPLY_UMH_FREQ_REDSHIFT is False else float(UMH_z_tension)
        },

        "PN_phase_model": dfdt_model,

        "amplitude_model": {
          "inspiral": amp_ins_model,
          "higher_order_PN": amp_higher_order_PN,
          "UMH_specifics": {
            "UMH_z_tension": None if UMH_z_tension is None else float(UMH_z_tension),
            "z_GR_equivalent": None if z_GR is None else float(z_GR),
            "normalization_law": "Global amplitude scaling fixed by the UMH tension-redshift relation calibrated on Pantheon+; no per-event or per-detector tuning",
            "envelope_model": "SOURCE_ENGINE is set to analytic_umh for this run for performance, but demonstrates microscopic UMH medium effects when soliton_umh or hybrid_umh."
          },
        },

        "qnm_overtone_info": None if USE_QNM_OVERTONE_ATTACH is False else qnm_overtone_info,

        "implementation_note": (
          "In this configuration, df/dt and phase are generated in the GR limit using source-frame masses. If APPLY_UMH_FREQ_REDSHIFT=True, the source "
          "phase/envelope are mapped to detector time via t_det = (1+z_freq) * t_obs with z_freq taken from UMH_z_tension (or z_GR), so that f_det = f_src/(1+z_freq). "
          "Amplitude normalization uses the full UMH effective luminosity distance D_L^UMH when APPLY_UMH_AMPLITUDE_SCALING=True; "
          "no separate D_eff or extra (1+z)^-1 amplitude factor is applied."
        ),
    }
    with open(f"{file_path}_GEN_Metadata.json", "w") as f:  # encoding="utf-8"
        import json; json.dump(meta, f, indent=2)


    # --- Visuals no physics impact, for preview before comparing to LIGO ---
    # NOTE: All operations below (noise injection, visual scaling, tapers, spectrogram upsampling) are for figures ONLY.
    # They operate on strain_viz_use / strain_viz_spect_use and do NOT touch the stored physics strain_records.
    if config.get("GENERATE_VISUAL_PREVIEWS", True):

        if not strain_records_viz: raise RuntimeError("No detector strain records found (check detectors config / delays).")
        for name in strain_records_viz:
            strain_viz_use=strain_records_viz[name].copy()

            if config.get("NORM_VISUAL", False):
                mx  = np.percentile(np.abs(strain_viz_use), 99.0) + EPS_FLOOR
                strain_viz_use = strain_viz_use / mx

            VISUAL_SCALE_MULT = config.get("VISUAL_SCALE_MULT", 1.0)
            if VISUAL_SCALE_MULT != 1.0: strain_viz_use *= VISUAL_SCALE_MULT

            pre_idx = min(int(0.5/dt_obs), len(strain_viz_use))  # first 0.5 s as "noise-only" window (tune if needed)
            noise_rms = max(np.std(strain_viz_use[:pre_idx]), EPS_FLOOR)
            peak_snr  = np.max(np.abs(strain_viz_use)) / noise_rms

            print(f"{title}: {name} peak / pre-trigger RMS ≈ {peak_snr:.2f} (sanity-check only)")

            print(f"{title}: {name}: Nonzero strain values = {np.count_nonzero(strain_viz_use)}")
            max_amp = np.max(np.abs(strain_viz_use))
            print(f"{title}: {name}: max strain before normalization = {max_amp:.2e}")

            print(f"{title}: {name} strain max: {np.max(strain_viz_use)}, min: {np.min(strain_viz_use)}")
            print(f"{title}: {name} strain abs max: {np.max(np.abs(strain_viz_use))}")

            peak_idx = np.argmax(np.abs(strain_viz_use))
            peak_time = t_array[peak_idx]
            print(f"{title}: Detector '{name}' peak strain = {strain_viz_use[peak_idx]:.2e} at t = {peak_time:.3f} s")


            # Optional Filter to CleanUp before Plots.
            if config.get("VISUAL_FILTER", False):
                b, a = butter(4, [lowcut_obs, highcut_obs], btype='band', fs=fs_obs)
                filtered_strain = filtfilt(b, a, strain_viz_use)
                strain_viz_use  = filtered_strain

        
            # --- Gentle start taper ---
            # Opional Add Taper to Start. Default 0.
            TAPER_START = config.get("VISUAL_TAPER_START", 0.0)
            if TAPER_START > 0:
                # --- Fade-in (avoid impulsive start) ---
                fade_in = int(TAPER_START / dt_obs)  # 0.5 s
                if fade_in > 0:
                    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_in)))
                    strain_viz_use[:fade_in] *= ramp


            # --- Gentle end taper ---
            # Opional Add Taper to End. Default 0.
            TAPER_END = config.get("VISUAL_TAPER_END", 0.0)
            if TAPER_END > 0:
                tailN = min(int(TAPER_END/dt_obs), len(strain_viz_use)//20)
                if tailN > 3:
                    w = 0.5*(1 - np.cos(np.linspace(0, np.pi, tailN)))  # 0→1
                    #strain_viz_use[-tailN:] *= (1.0 - w)                 # forces end → 0 smoothly
                    # Only taper the very end if it's safely beyond merger + a buffer
                    t0 = -float(PAD_START_SEC)
                    t_s_tmp = t0 + np.arange(len(strain_viz_use)) * dt_obs
                    safe_after = float(t_merge_obs) + float(config.get("TAPER_SAFE_AFTER_MERGER_SEC", 0.03))
                    end_mask = t_s_tmp >= safe_after
                    idx_end = np.where(end_mask)[0]
                    if idx_end.size > 0:
                        start = idx_end[0]
                        # only taper the last tailN samples inside the "safe" region
                        s0 = max(start, len(strain_viz_use) - tailN)
                        strain_viz_use[s0:] *= (1.0 - w[-(len(strain_viz_use)-s0):])



            # Opional Add Padding to Start for Visual. Default 0.05s
            PAD_START_SEC = config.get("PREPEND_PAD_SEC_VIZ", 0.00)   # try 0.5–2.0 s
            if PAD_START_SEC > 0:
                n_pad = int(round(PAD_START_SEC * fs_obs))
                pad   = np.zeros(n_pad, dtype=strain_viz_use.dtype)
                strain_viz_use = np.concatenate([pad, strain_viz_use])


            # Opional Add Padding to End for Visual. Default 0.10s
            PAD_END_SEC = config.get("APPEND_PAD_SEC_VIZ", 0.00)  # default 1 s
            if PAD_END_SEC > 0:
                n_pad_end = int(round(PAD_END_SEC * float(fs_obs)))
                pad_end   = np.zeros(n_pad_end, dtype=strain_viz_use.dtype)
                strain_viz_use = np.concatenate([strain_viz_use, pad_end])


            # --- Spectrogram (visual only; can upsample) ---
            strain_viz_spect_use=strain_viz_use.copy()
            if config.get("WHITEN_SPECTROGRAM_VISUAL", False):
                # Visual-only whitening; does NOT affect saved physics strain
                strain_viz_spect_use = whiten_for_display(np.asarray(strain_viz_spect_use, float), dt_obs)
                use_whiten = "Whitened "
            else: use_whiten = ""

            # --- Visualization noise (does NOT affect physics) ---
            if config.get("ADD_VISUAL_NOISE", False):
                # always start from clean chirp for this detector
                strain_viz_clean = np.array(strain_viz_spect_use, copy=True)
                # RMS of clean chirp (over the FULL record)
                sig_rms = float(np.sqrt(np.mean(strain_viz_clean**2)))
                if not np.isfinite(sig_rms) or sig_rms < EPS_FLOOR: sig_rms = 0.0

                cfg_frac = float(config.get("NOISE_TO_SIGNAL_RMS", 0.60))
                if not np.isfinite(cfg_frac): cfg_frac = 0.0
                cfg_frac = max(0.0, cfg_frac)
                eff_frac = min(cfg_frac, 0.30)  # safety cap

                if sig_rms > 0.0 and eff_frac > 0.0:
                    N_len = len(strain_viz_clean)
                
                    seed = config.get("NOISE_SEED", None)
                    rng   = np.random.default_rng(seed)

                    target_noise_rms = eff_frac * sig_rms

                    if config.get("USE_LIGO_PSD_NOISE", True):
                        noise = make_ligo_psd_noise(N_len, dt=dt_obs, target_rms=target_noise_rms, rng=rng)
                    else:
                        # simple white fallback
                        white = rng.normal(0.0, 1.0, N_len)
                        w_rms = float(np.sqrt(np.mean(white**2)))
                        if not np.isfinite(w_rms) or w_rms < EPS_FLOOR: noise = np.zeros_like(white)
                        else: noise = white * (target_noise_rms / w_rms)

                    noise_mult = float(config.get("ADD_VISUAL_NOISE_MULT", 3))
                    strain_viz_spect_use = strain_viz_clean + (noise * noise_mult)

                    n = strain_viz_spect_use - strain_viz_clean
                    sig_rms_dbg   = float(np.sqrt(np.mean(strain_viz_clean**2)))
                    noise_rms_dbg = float(np.sqrt(np.mean(n**2)))
                    ratio_dbg     = (noise_rms_dbg / sig_rms_dbg) if sig_rms_dbg > 0.0 else 0.0

                    #NOISE_DBG is for visual SNR tuning; does not represent physical LIGO SNR.
                    print(f"[NOISE_DBG] sig_rms={sig_rms_dbg:.3e}, noise_rms={noise_rms_dbg:.3e}, "
                        f"ratio={ratio_dbg:.3f}, cfg_frac={cfg_frac:.3f}, eff_frac={eff_frac:.3f}")


            # --- Wave Preview Dynamic Plot---
            # Use a pad-aware time axis so "t_merge_obs" remains the correct physical merger time.
            T_total_sec = t_max + PAD_START_SEC + PAD_END_SEC
            t0 = -float(PAD_START_SEC)  # so the padded zeros live at negative time
            t_s = t0 + np.arange(len(strain_viz_use)) * dt_obs

            # Merger marker to be the physical merger time:
            t_merge_plot = float(t_merge_obs)  # defined earlier in the run (observer-frame)

            plt.figure(figsize=(10, 4), constrained_layout=True)

            # --- strain ---
            ax1 = plt.subplot(1, 2, 1)
            ax1.plot(t_s, strain_viz_use)
            ax1.axvline(t_merge_plot, linestyle="--", linewidth=1.0)
            ax1.set_title(f"{title}: Detector Strain")
            ax1.set_xlabel("Time [s]")
            ax1.set_xlim(-PAD_START_SEC, t_max + PAD_END_SEC)

            # --- separation ---
            ax2 = plt.subplot(1, 2, 2)

            # radius_record is NOT padded; keep its native time base aligned to physical time
            t_sep = np.arange(len(radius_record)) * dt_obs
            ax2.plot(t_sep, radius_record)
            ax2.axvline(t_merge_plot, linestyle="--", linewidth=1.0)
            ax2.set_title(f"{title}: Binary Separation (Grid Units)")
            ax2.set_xlabel("Time [s]")
            ax2.set_xlim(0.0, t_max)

            # Add a zoom inset around merger+ringdown (helps make ringdown obvious)
            axins = inset_axes(ax2, width="35%", height="35%", loc="lower left", bbox_to_anchor=(0.0700, +0.0625, 1, 1), bbox_transform=ax2.transAxes)
            axins.plot(t_s, strain_viz_use)
            axins.yaxis.get_offset_text().set_visible(False)
            axins.axvline(t_merge_plot, linestyle="--", linewidth=1.0)
            zoom_left  = t_merge_plot - float(config.get("PREVIEW_ZOOM_LEFT_SEC", 0.04))
            zoom_right = t_merge_plot + float(config.get("PREVIEW_ZOOM_RIGHT_SEC", 0.04))
            axins.set_xlim(zoom_left, zoom_right)

            # autoscale y in the inset, but keep it sane
            seg = (t_s >= zoom_left) & (t_s <= zoom_right)
            if np.any(seg):
                yy = strain_viz_use[seg]
                ypad = 0.10 * (np.max(np.abs(yy)) + EPS_FLOOR)
                axins.set_ylim(np.min(yy) - ypad, np.max(yy) + ypad)

            #plt.tight_layout()
            plt.savefig(f"{file_path}_GEN_Chirp_Dynamic_Preview_{name}.png", dpi=dpi)
            plt.close()


            # --- Spectrogram Preview Plot---
            y = np.asarray(strain_viz_spect_use, dtype=float)

            # Optional: visual-only normalization to avoid tiny numbers
            vis_scale = np.max(np.abs(y))
            if np.isfinite(vis_scale) and vis_scale > 0.0: y = y / vis_scale

            # --- Optional visual upsampling for smoother spectrogram ---
            fs_base = float(fs_obs)
            dt_base = float(dt_obs)
            fs_spec = float(config.get("SPEC_FS", fs_base))

            if fs_spec > fs_base:
                up_factor = fs_spec / fs_base
                N_orig = len(y)
                t_orig = np.arange(N_orig) / fs_base

                N_up = int(round(N_orig * up_factor))
                t_up = np.arange(N_up) / fs_spec

                # Linear interpolation is sufficient for visualization
                y = np.interp(t_up, t_orig, y)
                dt_spec = dt_base / up_factor
                print(f"Resampled as per SPEC_FS:{fs_spec}")
            else: fs_spec = fs_base; dt_spec = dt_base

            # --- Robust onset detection on smoothed envelope (for crop) ---
            CROP_MODE = bool(config.get("SPECTRO_CROP_MODE", True))
            if CROP_MODE:
                CROP_BEFORE = float(config.get("SPECTRO_CROP_BEFORE_SEC", 0.12))
                CROP_AFTER  = float(config.get("SPECTRO_CROP_AFTER_SEC", 0.06))
                
                # Anchor directly on the known merger time; independent of amplitude profile
                T_total = len(y) * dt_spec
                t0 = max(0.0, float(t_merge_obs) - CROP_BEFORE)
                t1 = min(T_total, float(t_merge_obs) + CROP_AFTER)

                i0 = int(t0 * fs_base); i1 = int(t1 * fs_base)
                if i1 <= i0: i0, i1 = 0, len(y)   # extreme fallback
            else: i0, i1 = 0, len(y)

            # --- STFT parameters tuned for this fs_spec ---
            win_sec  = float(config.get("SPEC_WIN_SEC", 0.024))   # 24 ms default
            n_target = int(round(fs_spec * win_sec))
            nperseg  = 1 << int(math.ceil(math.log2(max(512, n_target))))
            nperseg  = int(min(4096, max(512, nperseg)))
            noverlap = int(0.92 * nperseg)
            window   = "hann"

            y_plot = y[i0 : i1]
            t0     = i0 * dt_spec

            # --- Spectrogram at fs_spec ---
            f_spec, t_spec, Sxx = spectrogram(y_plot, fs=fs_spec, window=window, nperseg=nperseg, noverlap=noverlap, detrend=False, scaling="density", mode="psd")

            Sxx  = np.nan_to_num(Sxx, nan=0.0, posinf=0.0, neginf=0.0)
            logp = 10.0 * np.log10(Sxx + EPS_FLOOR)

            if logp.shape[1] > 6: core = logp[:, 3:-3]
            else: core = logp

            # --- Dynamic range tweak so chirp ridge pops ---
            # Use high percentiles and clamp to a reasonable ~80 dB window
            if config.get("ADD_VISUAL_NOISE", False): p_lo, p_hi = 0.3, 99.7
            else: p_lo, p_hi = 95.3, 99.7
            vmin, vmax = np.nanpercentile(core, [p_lo, p_hi])
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
                max_range = 95.0
                if (vmax - vmin) > max_range: vmin = vmax - max_range
                vmin = max(vmin, -140.0); vmax = min(vmax, -20.0)
            else: vmin, vmax = -120.0, -40.0   # robust fallback
            print(f"Spectrogram: vmin={vmin}, vmax={vmax}")

            # --- Plot spectrogram ---
            T = t_spec + t0

            plt.figure(figsize=(8, 4))

            # Compute extents for imshow (left, right, bottom, top)
            extent = [T[0], T[-1], f_spec[0], f_spec[-1]]
            plt.imshow(logp,extent=extent, aspect='auto', origin='lower', interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=ligo_cmap())
            #plt.pcolormesh(T, f_spec, logp, shading="gouraud", vmin=vmin, vmax=vmax, cmap=ligo_cmap())
            
            t_end_vis = T[-1]

            # Geometric delay in seconds: if geom_delay_sec is already a TIME delay, use it directly.
            delay = float(detectors[name]["geom_delay_sec"])

            # --- Overlay: model f_GW(t) track with detector delay ---
            f_model_plot = np.asarray(freq_record_obs, dtype=dtype)

            # We WANT the model at j_merge to sit exactly at plotted t_merge_obs. So choose a global time offset that enforces that:
            ALIGN_EPS = config.get("SPECTRO_ALIGN_EPS", 0.0000 if SMOOTH_FGW is False or (TAU_SMOOTH_SEC == 0.0) else (TAU_SMOOTH_SEC*35))

            if(bool(config.get("SPECTRO_DRAW_JMERGE", True))):
                # Intrinsic model time axis shifted to detector frame
                t_model = np.arange(len(f_model_plot), dtype=float) * dt_spec + delay + ALIGN_EPS

                # Restrict to visible window for neat overlay
                m = (t_model >= t0) & (t_model <= t_end_vis) & np.isfinite(f_model_plot)
                if np.any(m): plt.plot(t_model[m], f_model_plot[m], 'w--', lw=1.0, alpha=0.9, label=r'Model $f_{\mathrm{GW}}(t)$')

            if(bool(config.get("SPECTRO_DRAW_CONTOURS", True))):
                # --- Contour overlay: emphasize chirp ridge ---
                # Draw a few light contours near the top of the dynamic range.
                try:
                    SPECTRO_CONTOUR_STRENGTH = float(config.get("SPECTRO_CONTOUR_STRENGTH", 0.8))
                    levels = np.linspace(vmin + SPECTRO_CONTOUR_STRENGTH * (vmax - vmin), vmax, 6)
                    plt.contour(T, f_spec, logp, levels=levels, colors='w', linewidths=0.6, alpha=0.5)
                except Exception: pass

            if(bool(config.get("SPECTRO_DRAW_ARRIVE", True))):
                # --- Mark merger time (if known) ---
                plt.axvline(float(t_merge_obs), color='r', ls=':', lw=1.0, alpha=0.4, label=r'$t_\mathrm{detected}$')

            if(bool(config.get("SPECTRO_DRAW_MERGER", True))):
                # --- Mark merger time (if known) ---
                plt.axvline(float(t_merge_obs + delay + ALIGN_EPS), color='k', ls='--', lw=0.5, alpha=0.3, label=r'$t_\mathrm{merge}$')

            # Axes + labels
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [s]")
            plt.ylim(float(lowcut_obs) - 10.0, float(highcut_obs) + 50.0)
            plt.xlim(T[0], t_end_vis)

            plt.title(f"{title}: {use_whiten}Spectrogram – {name}")
            cbar = plt.colorbar()
            cbar.set_label("Whitened Power [dB, arbitrary units]")

            # Show legend only if we actually plotted overlays
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles: plt.legend(loc="upper left", frameon=True)

            plt.tight_layout()
            plt.savefig(f"{file_path}_GEN_Spectrogram_{name}.png", dpi=dpi)
            plt.close()


            print(f"✅ Finished Test: {title}, Detector:{name}.")


        # --- Final intrinsic GW frequency plot (detector-independent) ---
        if not strain_records: raise RuntimeError("No detector strain records found for frequency plot.")

        # Use first detector's strain length as reference timeline
        _ref_name, y_ref = next(iter(strain_records.items()))
        n_target = len(y_ref)

        # Canonical frequency track from main loop
        f_used = np.asarray(freq_record_obs, dtype=float).copy()

        # Match frequency track length to reference strain length
        if len(f_used) < n_target:
            f_used = np.pad(f_used, (0, n_target - len(f_used)), constant_values=f_rd_obs_eff)
        elif len(f_used) > n_target: f_used = f_used[:n_target]  # safety trim

        # Optional smooth approach to f_rd_obs_eff near the tail (plotting only, not physics)
        tailN = min(int(0.05 / dt_obs), max(len(f_used) // 20, 0))
        if tailN > 3:
            f0 = f_used[-tailN - 1]
            w = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, tailN)))
            f_used[-tailN:] = (1.0 - w) * f0 + w * f_rd_obs_eff

        # Time axis for intrinsic frequency track
        t_freq = np.arange(len(f_used)) * dt_obs

        # --- Amplitude-based mask: only trust f_GW(t) where signal is non-negligible ---
        if 'A_hist_use' in locals():
            amp = np.asarray(A_hist_use, dtype=float)
            if len(amp) < len(f_used): amp = np.pad(amp, (0, len(f_used) - len(amp)), mode="edge")
            elif len(amp) > len(f_used): amp = amp[:len(f_used)]
            amp_mask = (amp > 1e-3 * np.max(amp))
        else: amp_mask = np.ones_like(f_used, dtype=bool)

        plt.figure(figsize=(8, 4))

        # Analytic f_GW(t) from phase history (if available)
        if 'phase_hist_use' in locals():
            f_analytic = ensure_f_hist_from_phase(phase_hist_use, dt_obs, dtype=float)
            if len(f_analytic) < len(f_used):
                f_analytic = np.pad(f_analytic, (0, len(f_used) - len(f_analytic)), constant_values=f_rd_obs_eff)
            elif len(f_analytic) > len(f_used): f_analytic = f_analytic[:len(f_used)]

            f_analytic_plot = np.where(amp_mask, f_analytic, np.nan)
            plt.plot(t_freq, f_analytic_plot, linewidth=2.0, label=r'Analytic $f_{\mathrm{GW}}(t)$ from phase')

        # Numeric instantaneous track from main loop
        f_used_plot = np.where(amp_mask, f_used, np.nan)
        plt.plot(t_freq, f_used_plot, lw=1.0, alpha=0.8, label=r'Numerical $f_{\mathrm{GW}}(t)$ record')

        # Ringdown frequency reference
        plt.axhline(f_rd_obs_eff, color='red', lw=1.0, ls='--', label=fr'Ringdown $f_{{\mathrm{{rd}}}} = {f_rd_obs_eff:.1f}\ \mathrm{{Hz}}$')

        # Axes, limits, labels
        plt.xlabel("Time [s]")
        plt.ylabel(r'$f_{\mathrm{GW}}(t)$ [Hz]')
        plt.title(f"{title}: Intrinsic Gravitational-Wave Frequency Evolution (Detector-Independent)")
        plt.xlim(0.0, t_freq[-1] if len(t_freq) else 0.0)
        plt.ylim(0.0, max(1.1 * np.nanmax(f_used_plot), 10.0))
        plt.legend(loc="upper left", bbox_to_anchor=(0.12, -0.30, 1, 1), frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # More descriptive filename
        plt.savefig(f"{file_path}_GEN_InstFreq_Intrinsic.png", dpi=dpi)
        plt.close()


        # --- Instantaneous frequency from tapered, bandpassed signal (Hilbert) ---
        # y_ref: a composite strain-like timeseries (e.g. one detector or generator reference)
        abs_y = np.abs(y_ref)
        thr   = 0.02 * np.max(abs_y) + EPS_FLOOR
        mask  = abs_y > thr
        if not np.any(mask): mask[:] = True # fallback: if everything is tiny, treat all as valid to avoid all-zeros

        # Start from a working copy
        sig = y_ref.astype(dtype, copy=True)

        # Light cosine taper over low-amplitude regions instead of hard zeroing
        if np.any(~mask):
            off_idx = np.where(~mask)[0]
            w = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, len(off_idx))))
            # keep tiny but nonzero to avoid nasty Hilbert edge artifacts
            sig[off_idx] *= (1e-3 + 1e-3 * w)

        # Edge tapers near first/last "significant" region to control boundary effects
        tailN = min(int(0.05 / dt_obs), len(sig) // 20)
        if tailN > 3 and mask.any():
            i0 = int(np.argmax(mask))
            i1 = int(len(mask) - np.argmax(mask[::-1]) - 1)

            # Fade-in before i0
            if i0 > 0:
                n = min(tailN, i0)
                w = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
                sig[i0 - n:i0] *= w

            # Fade-out after i1
            if i1 < len(sig) - 1:
                n = min(tailN, (len(sig) - 1 - i1))
                w = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
                sig[i1 + 1:i1 + 1 + n] *= w[::-1]

        # Bandpass for Hilbert-based f(t) estimation (diagnostic/visual only)
        sig_bp = bandpass_filter(sig, lowcut_obs, highcut_obs, fs_obs, order=4)
        sig_bp = np.ascontiguousarray(sig_bp, dtype=dtype)

        # --- Frequency track ---
        f_track = np.asarray(freq_record_obs, dtype=float).copy()
        if len(f_track) != n: f_track = np.interp(t_array, np.arange(len(f_track)) * dt_obs, f_track)

        # --- Bandpass + taper BEFORE Hilbert (stabilizes Hilbert phase/frequency) ---
        nyq = 0.5 * fs_obs
        lo = max(float(lowcut_obs), 0.1)
        hi = min(float(highcut_obs), nyq * 0.98)

        if lo < hi and n > 8:
            bp_order=4
            b, a = butter(int(bp_order), [lo / nyq, hi / nyq], btype='band')
            y_bp = filtfilt(b, a, y_ref)
        else: y_bp = y_ref.copy()

        # Edge taper to suppress Hilbert edge artifacts
        taper_edge_sec=0.05
        n_edge = int(max(1, round(float(taper_edge_sec) / float(dt_obs))))
        if 2 * n_edge < n:
            w = np.ones(n, dtype=float)
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_edge))
            w[:n_edge] *= ramp
            w[-n_edge:] *= ramp[::-1]
            y_bp = y_bp * w

        # Apply same "high-SNR" mask concept to Hilbert freq for display
        analytic_signal = hilbert(y_bp)
        amp = np.abs(analytic_signal)
        mask = (amp / (amp.max() if amp.max() > 0 else 1.0)) > 0.05

        t_hilb, f_hilb = estimate_hilbert_inst_freq(y_ref, dt_obs, lowcut=lowcut_obs, highcut=highcut_obs)

        # Clamp outliers using median + MAD (only positive freqs)
        pos = f_hilb[f_hilb > 0.0]
        if len(pos):
            med = np.median(pos)
            mad = np.median(np.abs(pos - med)) + 1e-9
            hi  = med + 8.0 * mad
            hi  = min(hi, highcut_obs)
            #f_hilb = np.clip(f_hilb, lowcut_obs, hi)

        # Light smoothing to kill needle spikes (no physics impact, purely visual)
        if len(f_hilb) >= 13:
            win = 13 if len(f_hilb) >= 13 else (len(f_hilb) // 2 * 2 + 1)
            f_hilb = savgol_filter(f_hilb, window_length=win, polyorder=2, mode="interp")

        # Amplitude-based gate: trust f(t) only where signal amplitude is non-trivial
        if 'A_hist_use' in locals():
            amp = np.asarray(A_hist_use, dtype=float)
            if len(amp) < len(f_hilb): amp = np.pad(amp, (0, len(f_hilb) - len(amp)), mode="edge")
            elif len(amp) > len(f_hilb): amp = amp[:len(f_hilb)]
            amp_mask = (amp > 1e-3 * np.max(amp))
            f_hilb_plot = np.where(amp_mask, f_hilb, np.nan)
        else: f_hilb_plot = f_hilb

        # --- Model instantaneous frequency track from generator (freq_record) ---

        if len(f_track) < len(f_hilb): f_track = np.pad(f_track, (0, len(f_hilb) - len(f_track)), constant_values=f_rd_obs_eff)
        elif len(f_track) > len(f_hilb): f_track = f_track[:len(f_hilb)]

        f_model_plot = np.where(~np.isnan(f_hilb_plot), f_track, np.nan)

        # Time cutoff: up to end of data or merger + a few tau_rd_obs, whichever smaller
        if 't_merge_obs' in locals() and 'tau_rd_obs' in locals(): t_cut = min(t_hilb[-1], float(t_merge_obs + 6.0 * tau_rd_obs))
        else: t_cut = t_hilb[-1]

        plt.figure(figsize=(8, 4))

        ax = plt.gca()
    
        axins = inset_axes(ax, width="30%", height="30%", loc="upper left", bbox_to_anchor=(0.12, -0.30, 1, 1), bbox_transform=ax.transAxes)

        t_c = t_merge_obs  # merger time
        mask = t_hilb < t_c
        tc_minus_t = t_c - t_hilb[mask]
        axins.loglog(tc_minus_t, f_model_plot[mask], 'C1', lw=1.2)
        axins.set_xlabel(r'$t_c - t$ [s]', fontsize=8)
        axins.set_ylabel(r'$f(t)$ [Hz]', fontsize=8)
        axins.tick_params(axis='both', which='major', labelsize=8)
        axins.grid(True, which='both', ls=':')


        # Hilbert-derived instantaneous frequency (data-driven)
        ax.plot(t_hilb, f_hilb_plot, lw=1.0, alpha=0.8, label=r'Hilbert instantaneous $f_{\mathrm{GW}}(t)$')

        # Model PN+ringdown track from generator
        ax.plot(t_hilb, f_model_plot, lw=2.0, alpha=0.9, label=r'UMH analytic $f_{\mathrm{GW}}(t)$')

        # --- Annotate detector-weighted phase RMS ---
        if 'phase_weighted' in locals() and isinstance(phase_weighted, dict):
            try:
                comb_rms = phase_weighted["combined"]["weighted_rms_rad"]
                ax.text(0.02, 0.93, f"Weighted phase RMS ≈ {comb_rms:.3f} rad", transform=ax.transAxes, fontsize=10,
                         ha='left', va='top', bbox=dict(boxstyle="round,pad=0.2", alpha=0.25))
            except Exception: pass

        # Ringdown reference line
        ax.axhline(f_rd_obs_eff, color='C0', ls='--', label=fr'Ringdown $f_{{\mathrm{{rd}}}} = {f_rd_obs_eff:.1f}\ \mathrm{{Hz}}$')
        ax.axvline(t_merge_obs, color='k', ls='--', alpha=0.6, label=fr'$t_\mathrm{{merge}} = {t_merge_obs:.3f}\ \mathrm{{s}}$')

        # Axes + limits
        ax.set_title(f"{title}: Instantaneous GW Frequency – {_ref_name} (Hilbert vs UMH analytic)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r'$f_{\mathrm{GW}}(t)$ [Hz]')
        ax.set_xlim(0.0, t_cut)
        ax.set_ylim(0.0, max(1.3 * f_rd_obs_eff, 300.0))
        ax.grid(True, alpha=0.3)

        # Use model track where it exists and t_hilb < t_merge_obs
        mask_inset = (t_hilb < t_merge_obs) & np.isfinite(f_model_plot)
        tc_minus_t = t_merge_obs - t_hilb[mask_inset]
        f_inset = f_model_plot[mask_inset]
        # Only positive (t_merge_obs - t_hilb)
        pos = tc_minus_t > 0.0
        tc_minus_t = tc_minus_t[pos]
        f_inset = f_inset[pos]

        if len(tc_minus_t) > 0:
            axins.loglog(tc_minus_t, f_inset, lw=1.2)
            axins.set_xlabel(r'$t_c - t$ [s]', fontsize=8)
            axins.set_ylabel(r'$f_{\mathrm{GW}}(t)$ [Hz]', fontsize=8)
            axins.tick_params(axis='both', which='major', labelsize=8)
            axins.grid(True, which='both', ls=':')

        leg=ax.legend(loc="upper right", bbox_to_anchor=(0.94, 0.70), frameon=True) # None Found.
        leg.get_frame().set_alpha(0.4)
        plt.savefig(f"{file_path}_GEN_InstFreq_Hilbert_{_ref_name}.png", dpi=dpi)
        plt.close()

    print(f"✅ Finished Test: {title} Validated.")
    return True


def str2bool(s):
    if isinstance(s, bool): return s
    s = str(s).lower()
    if s in ("1", "true", "yes", "y", "on"): return True
    if s in ("0", "false", "no", "n", "off"): return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{s}' as bool.")

def cli_float(value: str) -> float:
    try: return float(value)
    except ValueError: raise ValueError(f"Invalid float value for argument: {value!r}")

# Main Entry Points of Script.
if __name__ == "__main__":
    overrides = {"profile": "replica_gw150914"}         # Used to Specify Override Profile to use.
    #overrides = {"profile": "replica_gw170814"}         # Used to Specify Override Profile to use.
    #if len(sys.argv) > 1:
    #    with open(sys.argv[1], "r") as f: overrides.update(json.load(f))
    #run_chirp_generator_test(overrides)


    parser = argparse.ArgumentParser(description="UMH LIGO Compiler – compare UMH chirp to LIGO data.")
    # Optional positional JSON overrides file
    parser.add_argument( "config_path", nargs="?", help="Optional JSON overrides file (positional).")
    # Optional named JSON overrides as well (--overrides)
    parser.add_argument("--overrides", dest="config_path_named", help="Optional JSON overrides file (named).")
    # Explicit override knobs
    parser.add_argument("--OUTPUT_FOLDER", type=str, help="Override output folder for plots/metadata.")
    
    parser.add_argument("--profile", help="Build in Profile to Use")

    parser.add_argument("--GENERATE_VISUAL_PREVIEWS", type=str2bool, help="Override whether to generate plots/visuals.")
    parser.add_argument("--GENERATE_NPZ", type=str2bool, help="Override whether to generate NPZ files.")

    parser.add_argument("--M1_solar_src", type=cli_float)
    parser.add_argument("--M2_solar_src", type=cli_float)
    parser.add_argument("--distance_Mpc", type=cli_float)
    parser.add_argument("--ra_deg",   type=cli_float)
    parser.add_argument("--dec_deg",  type=cli_float)
    parser.add_argument("--pol_psi_deg",  type=cli_float)
    parser.add_argument("--BINARY_IOTA_DEG",  type=cli_float)
    parser.add_argument("--SPIN_MODE", type=str)
    parser.add_argument("--spin1x", type=cli_float)
    parser.add_argument("--spin1y", type=cli_float)
    parser.add_argument("--spin1z", type=cli_float)
    parser.add_argument("--spin2x", type=cli_float)
    parser.add_argument("--spin2y", type=cli_float)
    parser.add_argument("--spin2z", type=cli_float)
    parser.add_argument("--chi1z", type=cli_float)
    parser.add_argument("--chi2z", type=cli_float)
    parser.add_argument("--REMNANT_APPROXIMANT", type=str)
    
    args = parser.parse_args()

    cfg_path = args.config_path_named or args.config_path
    if cfg_path is not None:
        with open(cfg_path, "r") as f: file_cfg = json.load(f)
        overrides.update(file_cfg)

    if args.profile is not None:          overrides["profile"]          = args.profile

    if args.OUTPUT_FOLDER is not None:    overrides["OUTPUT_FOLDER"] = args.OUTPUT_FOLDER
    if args.GENERATE_VISUAL_PREVIEWS is not None: overrides["GENERATE_VISUAL_PREVIEWS"] = args.GENERATE_VISUAL_PREVIEWS
    if args.GENERATE_NPZ is not None:     overrides["GENERATE_NPZ"]  = args.GENERATE_NPZ

    if args.M1_solar_src is not None: overrides["M1_solar_src"]  = args.M1_solar_src
    if args.M2_solar_src is not None: overrides["M2_solar_src"]  = args.M2_solar_src
    if args.distance_Mpc is not None: overrides["distance_Mpc"]  = args.distance_Mpc
    if args.ra_deg is not None:       overrides["ra_deg"]  = args.ra_deg
    if args.dec_deg is not None:      overrides["dec_deg"] = args.dec_deg
    if args.pol_psi_deg is not None:  overrides["pol_psi_deg"]   = args.pol_psi_deg
    if args.BINARY_IOTA_DEG is not None:  overrides["BINARY_IOTA_DEG"] = args.BINARY_IOTA_DEG   

    for key in ("SPIN_MODE", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "chi1z", "chi2z", "REMNANT_APPROXIMANT"):
        val = getattr(args, key, None)
        if val is not None: overrides[key] = val

    run_chirp_generator_test(overrides)