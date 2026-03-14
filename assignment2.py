#%%
# ECG Holter Analysis – Final Clinical Report Pipeline
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
# 1. CLINICAL CONSTANTS & CONFIGURATION and Path 

LEADS = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

# DEFINE PATHS 
path1 = r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs+PVCs.mat"
path2 = r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs.mat"

#%%
# 2. SIGNAL PROCESSING & DETECTION FUNCTIONS

def read_and_clean_ecg_mat(path):
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    ecg_raw = data['ecg'].sig[:, LEADS.index('II')]
    fs = data['ecg'].header.Sampling_Rate
    t0 = datetime.datetime(*data['ecg'].start_vec)
    t = pd.date_range(start=t0, periods=len(ecg_raw), freq=pd.Timedelta(seconds=1/fs))
    valid_mask = (ecg_raw > -500) & (ecg_raw < 500)
    return ecg_raw[valid_mask], fs, t[valid_mask]

def apply_clinical_bandpass(ecg, fs):
    b_low, a_low = signal.butter(4, 15, 'low', fs=fs)
    b_high, a_high = signal.butter(2, 0.5, 'high', fs=fs)
    filtered = signal.filtfilt(b_low, a_low, ecg)
    return signal.filtfilt(b_high, a_high, filtered)

def detect_r_peaks(ecg_chunk, fs):
    ecg_diff = np.diff(ecg_chunk, prepend=ecg_chunk[0])
    ecg_sq = ecg_diff ** 2
    window_len = int(0.12 * fs)
    envelope = np.convolve(ecg_sq, np.ones(window_len) / window_len, mode='same')

    peak_height = np.mean(envelope) + 1.0 * np.std(envelope)
    dynamic_prominence = 0.5 * np.std(envelope)
    locs_env, _ = signal.find_peaks(envelope, height=peak_height, distance=int(0.3 * fs), prominence=dynamic_prominence)

    search_radius = int(0.08 * fs)
    locs_r = [np.argmax(ecg_chunk[max(0, loc - search_radius):min(len(ecg_chunk), loc + search_radius)]) + max(0, loc - search_radius) for loc in locs_env]
    return np.unique(np.array(locs_r))

def compute_rr_intervals(locs, t):
    rr = pd.Series(np.diff(t[locs])).dt.total_seconds().values
    valid = (rr > 0.25) & (rr < 2.0) 
    return rr[valid], t[locs[:-1]][valid]

def detect_pac(RR, t_rr, locs, t):
    pac_locs = []
    for i in range(5, len(RR) - 1):
        local_median = np.median(RR[i-5:i])
        if RR[i] < 0.8 * local_median and RR[i+1] > 1.2 * local_median:
            idx = np.where(t[locs] == t_rr[i+1])[0]
            if len(idx) > 0: pac_locs.append(locs[idx[0]])
    return np.array(pac_locs)

def classify_pac_runs(pac_indices):
    if len(pac_indices) == 0: return [], [], []
    isolated, couplets, runs = [], [], []
    current_group = [pac_indices[0]]

    for i in range(1, len(pac_indices)):
        if pac_indices[i] == pac_indices[i-1] + 1:
            current_group.append(pac_indices[i])
        else:
            if len(current_group) == 1: isolated.append(current_group)
            elif len(current_group) == 2: couplets.append(current_group)
            else: runs.append(current_group)
            current_group = [pac_indices[i]]

    if len(current_group) == 1: isolated.append(current_group)
    elif len(current_group) == 2: couplets.append(current_group)
    else: runs.append(current_group)
    return isolated, couplets, runs

def matchednl(template, data):
    MM = len(template)
    hh = template[::-1]/np.linalg.norm(template)
    y = signal.lfilter(hh, 1, data)
    z = signal.lfilter(np.ones(MM), 1, np.square(data))
    return y/np.sqrt(z)

def classify_ectopic_beats_matched(ynon, ectopic_locs, fs, post_duration=0.20):
    confirmed_pacs, confirmed_pvcs = [], []
    delay_samples = int(post_duration * fs)
    search_window = int(0.05 * fs) 

    for loc in ectopic_locs:
        expected_peak_idx = loc + delay_samples
        if expected_peak_idx - search_window < 0 or expected_peak_idx + search_window >= len(ynon): continue
        if np.max(ynon[expected_peak_idx - search_window : expected_peak_idx + search_window]) >= 0.70:
            confirmed_pacs.append(loc)
        else:
            confirmed_pvcs.append(loc)

    print(f"Total premature complexes evaluated : {len(ectopic_locs)}")
    print(f"Supraventricular (PACs)           : {len(confirmed_pacs)}")
    print(f"Ventricular (PVCs)                : {len(confirmed_pvcs)}\n")
    return np.array(confirmed_pacs), np.array(confirmed_pvcs)

#%%
# 3. REPORT VISUALIZATION FUNCTIONS

def plot_20_sec_segment(ecg, t, fs, start_sec=1000, title="20 Second ECG Segment"):
    start = int(start_sec * fs)
    end = start + int(20 * fs)
    plt.figure(figsize=(12, 3))
    plt.plot(t[start:end], ecg[start:end], color='tab:blue', linewidth=1)
    plt.title(title)
    plt.ylabel("Amplitude (mV)")
    plt.xlabel("Time")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    plt.show()

def plot_single_validation(ecg, t, locs, target_locs, fs, title, marker_color, marker_label, window_sec=5):
    if len(target_locs) == 0: return
    center = target_locs[0] 
    start = max(0, int(center - (window_sec / 2) * fs))
    end = min(len(ecg), int(center + (window_sec / 2) * fs))

    m_qrs = (locs >= start) & (locs < end)
    m_target = (target_locs >= start) & (target_locs < end)

    plt.figure(figsize=(10, 3))
    plt.plot(t[start:end], ecg[start:end], color='tab:blue', linewidth=1)
    plt.plot(t[locs[m_qrs]], ecg[locs[m_qrs]], 'k*', markersize=6, label="Normal QRS")
    plt.plot(t[target_locs[m_target]], ecg[target_locs[m_target]], marker_color, markersize=8, label=marker_label)
    
    plt.title(title)
    plt.ylabel("Amplitude (mV)")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    plt.show()

def plot_heart_rate(t_rr, RR, title):
    """
    Genereert een Tachogram om de verandering in hartfrequentie over tijd te laten zien.
    """
    hr = 60 / RR
    plt.figure(figsize=(12, 3))
    plt.plot(t_rr, hr, '.', markersize=2, color='tab:blue', alpha=0.6)
    plt.ylabel("Heart Rate (BPM)")
    plt.xlabel("Time")
    plt.title(title)
    plt.ylim(30, 200) # Fysiologische grenzen
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()
    plt.show()

#%%
# 4. EXECUTION PIPELINE
# ==========================================================

# ----------------------------------------------------------
# RECORDING 1: Complex Arrhythmia (PACs + PVCs)
# ----------------------------------------------------------
print("\n" + "="*50)
print("--- CLINICAL REPORT: RECORDING 1 ---")
print("="*50)

ecg1, fs1, t1 = read_and_clean_ecg_mat(path1)
filtered1 = apply_clinical_bandpass(ecg1, fs1)
locs1 = detect_r_peaks(filtered1, fs1)
RR1, t_rr1 = compute_rr_intervals(locs1, t1)

# 1. Output Text / Statistics
raw_pac_locs1 = detect_pac(RR1, t_rr1, locs1, t1)
isolated_pacs1, pac_couplets1, pac_runs1 = classify_pac_runs(raw_pac_locs1)
print("\n--- Initial Rhythm Burden ---")
print(f"Total premature beats detected: {len(raw_pac_locs1)}")
print(f"Isolated instances: {len(isolated_pacs1)} | Couplets: {len(pac_couplets1)} | Runs: {len(pac_runs1)}\n")

# 2. Matched Filter Classification
pre, post = 0.20, 0.20
i0 = max(0, int(locs1[1] - pre * fs1))
i1 = min(len(filtered1), int(locs1[1] + post * fs1))
sinus_template = filtered1[i0:i1] # Save this template for Rec 2 as well

ynon1 = matchednl(sinus_template, filtered1)
true_pac_locs1, true_pvc_locs1 = classify_ectopic_beats_matched(ynon1, raw_pac_locs1, fs1, post_duration=post)

# 3. Plots for Recording 1
plot_20_sec_segment(filtered1, t1, fs1, start_sec=1000, title="Recording 1: 20-Second Filtered ECG Segment")
plot_single_validation(filtered1, t1, locs1, true_pac_locs1, fs1, "Recording 1: PAC Validation (Narrow Complex)", marker_color='ro', marker_label="PAC")
plot_single_validation(filtered1, t1, locs1, true_pvc_locs1, fs1, "Recording 1: PVC Validation (Wide/Bizarre Complex)", marker_color='s', marker_label="PVC") # 's' makes an orange square
plot_heart_rate(t_rr1, RR1, "Recording 1: Ventricular Rate Profile (BPM over Time)")
# ----------------------------------------------------------
# RECORDING 2: Isolated Atrial Ectopy (PACs only)
# ----------------------------------------------------------
print("\n" + "="*50)
print("--- CLINICAL REPORT: RECORDING 2 ---")
print("="*50)

ecg2, fs2, t2 = read_and_clean_ecg_mat(path2)
filtered2 = apply_clinical_bandpass(ecg2, fs2)
locs2 = detect_r_peaks(filtered2, fs2)
RR2, t_rr2 = compute_rr_intervals(locs2, t2)

# 1. Output Text / Statistics
raw_pac_locs2 = detect_pac(RR2, t_rr2, locs2, t2)
isolated_pacs2, pac_couplets2, pac_runs2 = classify_pac_runs(raw_pac_locs2)
print("\n--- Initial Rhythm Burden ---")
print(f"Total premature beats detected: {len(raw_pac_locs2)}")
print(f"Isolated instances: {len(isolated_pacs2)} | Couplets: {len(pac_couplets2)} | Runs: {len(pac_runs2)}\n")

# 2. Matched Filter Classification
ynon2 = matchednl(sinus_template, filtered2)
true_pac_locs2, true_pvc_locs2 = classify_ectopic_beats_matched(ynon2, raw_pac_locs2, fs2, post_duration=post)

# 3. Plots for Recording 2
plot_20_sec_segment(filtered2, t2, fs2, start_sec=2000, title="Recording 2: 20-Second Filtered ECG Segment")
plot_single_validation(filtered2, t2, locs2, true_pac_locs2, fs2, "Recording 2: PAC Validation (Narrow Complex)", marker_color='ro', marker_label="PAC")
plot_heart_rate(t_rr2, RR2, "Recording 2: Ventricular Rate Profile (BPM over Time)")
# NO PVC PLOT FOR RECORDING 2 AS REQUESTED
# %%
