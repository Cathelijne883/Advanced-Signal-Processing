#%%
# ECG Holter Analysis – PAC Detection Pipeline
# TU Delft – Advanced Signal Processing
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from scipy.io import loadmat
from scipy import signal

#%%
# 1. Clinical Constants
# ---------------------------
LEADS = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

# %%
# 2. FUNCTION DEFINITIONS
# ==========================================================

# 
# Load and Clean ECG (Artifact Gating)
# ---------------------------
def read_and_clean_ecg_mat(path, plotresult=False):
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    ecg_raw = data['ecg'].sig[:, LEADS.index('II')]
    fs = data['ecg'].header.Sampling_Rate
    t0 = datetime.datetime(*data['ecg'].start_vec)

    t = pd.date_range(start=t0, periods=len(ecg_raw), freq=pd.Timedelta(seconds=1/fs))

    # Remove extreme telemetry dropouts and massive somatic artifacts
    valid_mask = (ecg_raw > -500) & (ecg_raw < 500)
    ecg_clean = ecg_raw[valid_mask]
    t_clean = t[valid_mask]

    print(f"Removed {len(ecg_raw) - len(ecg_clean)} corrupted samples.")
    return ecg_clean, fs, t_clean

#%%
# Clinical Bandpass Filter
# ---------------------------
def apply_clinical_bandpass(ecg, fs):
    # 15 Hz Lowpass blocks skeletal muscle (EMG) noise
    b_low, a_low = signal.butter(4, 15, 'low', fs=fs)
    # 0.5 Hz Highpass stabilizes respiratory baseline wander
    b_high, a_high = signal.butter(2, 0.5, 'high', fs=fs)

    filtered = signal.filtfilt(b_low, a_low, ecg)
    filtered = signal.filtfilt(b_high, a_high, filtered)
    return filtered

#%%
# R-Peak Detection (Pan-Tompkins Algorithm)
# ---------------------------
def detect_r_peaks(ecg_chunk, fs, plotresult=False, t_chunk=None):
    # 1. Derivative (Enhance steep QRS slopes)
    ecg_diff = np.diff(ecg_chunk, prepend=ecg_chunk[0])
    
    # 2. Square (Non-linear amplification)
    ecg_sq = ecg_diff ** 2
    
    # 3. Moving Window Integration (Smooth QRS envelope)
    window_len = int(0.12 * fs)
    window = np.ones(window_len) / window_len
    envelope = np.convolve(ecg_sq, window, mode='same')

    # 4. Adaptive Thresholding
    peak_height = np.mean(envelope) + 1.0 * np.std(envelope)
    dynamic_prominence = 0.5 * np.std(envelope)
    min_distance = int(0.3 * fs) # 300 ms refractory period

    locs_env, _ = signal.find_peaks(
        envelope, height=peak_height, distance=min_distance, prominence=dynamic_prominence
    )

    # 5. Refinement: Find exact R-peak apex in original ECG
    search_radius = int(0.08 * fs)
    locs_r = []
    for loc in locs_env:
        start = max(0, loc - search_radius)
        end = min(len(ecg_chunk), loc + search_radius)
        local_peak = np.argmax(ecg_chunk[start:end]) + start
        locs_r.append(local_peak)

    locs_r = np.unique(np.array(locs_r))
    return locs_r, envelope

#%%
# RR Interval Computation
# ---------------------------
def compute_rr_intervals(locs, t):
    rr = pd.Series(np.diff(t[locs])).dt.total_seconds().values
    valid = (rr > 0.25) & (rr < 2.0) # Physiological limits
    rr_clean = rr[valid]
    t_rr = t[locs[:-1]][valid]
    return rr_clean, t_rr

#%%
# Diagnostics: Heart Rate Tachogram
# ---------------------------
def plot_heart_rate(t_rr, RR, title):
    hr = 60 / RR
    plt.figure(figsize=(12,4))
    plt.plot(t_rr, hr, '.', markersize=1, color='tab:blue', alpha=0.5)
    plt.ylabel("Heart Rate (BPM)")
    plt.xlabel("Time")
    plt.title(title)
    plt.ylim(30,200)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.show()

#%%
# Diagnostics: QRS Regularity Analysis
# ---------------------------
def analyze_qrs_regularity(t_rr, RR):
    rr_series = pd.Series(RR, index=t_rr)
    rr_stats = rr_series.resample("10S").agg(["mean", "std", "count"])
    return rr_stats

def plot_rr_variability(rr_stats, title):
    plt.figure(figsize=(12,4))
    plt.plot(rr_stats.index, rr_stats["std"])
    plt.ylabel("RR std (s)")
    plt.xlabel("Time")
    plt.title(title)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.show()

#%%
# Pathology: PAC Classification
# ---------------------------
def detect_pac(RR, t_rr, locs, t):
    pac_locs = []
    window_size = 5

    for i in range(window_size, len(RR) - 1):
        local_median = np.median(RR[i-window_size:i])
        
        # Prematurity (< 85%) followed by compensatory pause (> 115%)
        if RR[i] < 0.85 * local_median and RR[i+1] > 1.15 * local_median:
            pac_time = t_rr[i+1]
            idx = np.where(t[locs] == pac_time)[0]
            if len(idx) > 0:
                pac_locs.append(locs[idx[0]])

    pac_locs = np.array(pac_locs)
    print(f"=========================================")
    print(f"--> CLINICAL PAC BURDEN DETECTED: {len(pac_locs)}")
    print(f"=========================================")
    return pac_locs

#%%
# Pathology Visualization: Auto-tracking ECG Plot
# ---------------------------
def plot_ecg_with_pac(ecg, t, locs, pac_locs, fs, pac_index=0, window_sec=10):
    if len(pac_locs) == 0:
        print("Geen PACs om te plotten!")
        return

    center_loc = pac_locs[pac_index]
    start = int(center_loc - (window_sec / 2) * fs)
    end = int(center_loc + (window_sec / 2) * fs)
    
    start = max(0, start)
    end = min(len(ecg), end)

    mask_qrs = (locs >= start) & (locs < end)
    mask_pac = (pac_locs >= start) & (pac_locs < end)

    plt.figure(figsize=(12, 4))
    plt.plot(t[start:end], ecg[start:end], linewidth=1, color='tab:blue', label="Filtered ECG")
    plt.plot(t[locs[mask_qrs]], ecg[locs[mask_qrs]], 'k*', markersize=6, label="Normal QRS")
    
    if np.any(mask_pac):
        plt.plot(t[pac_locs[mask_pac]], ecg[pac_locs[mask_pac]], 'ro', markersize=8, label="Ectopic PAC")

    plt.title(f"Diagnostic Tracing: PAC Validation (Index #{pac_index})")
    plt.ylabel("Amplitude (mV)")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    plt.show()

#%%
# 3. EXECUTION PIPELINE
# ==========================================================

# DEFINE PATHS (Ensure these are correct for your local machine)
path1 = r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs+PVCs.mat"
path2 = r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs.mat"
#%%
# ----------------------------------------------------------
# Recording 1: Complex Arrhythmia (PACs + PVCs)
# ----------------------------------------------------------
print("\n--- Initiating Diagnostic Run: Recording 1 ---")
ecg1, fs1, t1 = read_and_clean_ecg_mat(path1)
filtered1 = apply_clinical_bandpass(ecg1, fs1)
locs1, envelope1 = detect_r_peaks(filtered1, fs1, plotresult=False, t_chunk=t1)
RR1, t_rr1 = compute_rr_intervals(locs1, t1)

plot_heart_rate(t_rr1, RR1, "Ventricular Rate Profile – Recording 1")
rr_stats1 = analyze_qrs_regularity(t_rr1, RR1)
plot_rr_variability(rr_stats1, "Chronotropic Regularity – Recording 1")

pac_locs1 = detect_pac(RR1, t_rr1, locs1, t1)
plot_ecg_with_pac(filtered1, t1, locs1, pac_locs1, fs1, pac_index=0)


# ----------------------------------------------------------
# Recording 2: Isolated Atrial Ectopy (PACs only)
# ----------------------------------------------------------
print("\n--- Initiating Diagnostic Run: Recording 2 ---")
ecg2, fs2, t2 = read_and_clean_ecg_mat(path2)
filtered2 = apply_clinical_bandpass(ecg2, fs2)
locs2, envelope2 = detect_r_peaks(filtered2, fs2, plotresult=False, t_chunk=t2)
RR2, t_rr2 = compute_rr_intervals(locs2, t2)

plot_heart_rate(t_rr2, RR2, "Ventricular Rate Profile – Recording 2")
rr_stats2 = analyze_qrs_regularity(t_rr2, RR2)
plot_rr_variability(rr_stats2, "Chronotropic Regularity – Recording 2")

pac_locs2 = detect_pac(RR2, t_rr2, locs2, t2)
plot_ecg_with_pac(filtered2, t2, locs2, pac_locs2, fs2, pac_index=4)
# %%
