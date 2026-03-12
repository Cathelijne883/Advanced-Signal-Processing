# ==========================================================
# ECG Holter Analysis – PAC Detection Pipeline
# TU Delft – Advanced Signal Processing
# ==========================================================
#%% 
# ---------------------------
# 1. Imports
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

from scipy.io import loadmat
from scipy import signal

#%%  Constants
# ---------------------------
LEADS = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

#%%  
# 3. FUNCTION DEFINITIONS
# ==========================================================

# ---------------------------
# Load and clean ECG
# ---------------------------
def read_and_clean_ecg_mat(path, plotresult=False):

    data = loadmat(path, squeeze_me=True, struct_as_record=False)

    ecg_raw = data['ecg'].sig[:, LEADS.index('II')]
    fs = data['ecg'].header.Sampling_Rate

    t0 = datetime.datetime(*data['ecg'].start_vec)

    t = pd.date_range(
        start=t0,
        periods=len(ecg_raw),
        freq=pd.Timedelta(seconds=1/fs)
    )

    # Remove extreme telemetry artifacts
    valid_mask = (ecg_raw > -500) & (ecg_raw < 500)

    ecg_clean = ecg_raw[valid_mask]
    t_clean = t[valid_mask]

    print(f"Removed {len(ecg_raw) - len(ecg_clean)} corrupted samples.")

    if plotresult:
        plt.figure(figsize=(12,4))
        plt.plot(t_clean, ecg_clean, linewidth=0.5)
        plt.title("Clean ECG signal")
        plt.ylabel("ECG (mV)")
        plt.xlabel("Time")
        plt.show()

    return ecg_clean, fs, t_clean

#%%
# Clinical bandpass filter
# ---------------------------
def apply_clinical_bandpass(ecg, fs):
    # De lage filter is aangepast naar 15 Hz (orde 4) om spierruis te blokkeren
    b_low, a_low = signal.butter(4, 15, 'low', fs=fs)
    b_high, a_high = signal.butter(2, 0.5, 'high', fs=fs)

    filtered = signal.filtfilt(b_low, a_low, ecg)
    filtered = signal.filtfilt(b_high, a_high, filtered)

    return filtered

#%% 
#  R peak detection
# ---------------------------
def detect_r_peaks(ecg_chunk, fs, plotresult=False, t_chunk=None):
    """
    Detect R-peaks using a simplified Pan-Tompkins approach:
    derivative -> square -> moving window integration -> peak detection
    """
    # 1. Derivative: emphasizes steep QRS slopes
    ecg_diff = np.diff(ecg_chunk, prepend=ecg_chunk[0])

    # 2. Square: makes all values positive and boosts large slopes
    ecg_sq = ecg_diff ** 2

    # 3. Moving window integration: smooth QRS energy envelope
    window_len = int(0.12 * fs)   # 120 ms
    window = np.ones(window_len) / window_len
    envelope = np.convolve(ecg_sq, window, mode='same')

    # 4. Adaptive thresholds on this chunk
    peak_height = np.mean(envelope) + 1.0 * np.std(envelope)
    dynamic_prominence = 0.5 * np.std(envelope)
    min_distance = int(0.3 * fs)

    locs_env, _ = signal.find_peaks(
        envelope,
        height=peak_height,
        distance=min_distance,
        prominence=dynamic_prominence
    )

    # 5. Refine: find actual R-peak in original ECG near each envelope peak
    search_radius = int(0.08 * fs)   # ±80 ms
    locs_r = []

    for loc in locs_env:
        start = max(0, loc - search_radius)
        end = min(len(ecg_chunk), loc + search_radius)

        local_peak = np.argmax(ecg_chunk[start:end]) + start
        locs_r.append(local_peak)

    locs_r = np.unique(np.array(locs_r))

    if plotresult and t_chunk is not None:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax[0].plot(t_chunk, ecg_chunk)
        ax[0].plot(t_chunk[locs_r], ecg_chunk[locs_r], 'r*')
        ax[0].set_title("Filtered ECG with detected R-peaks")
        ax[0].set_ylabel("ECG (mV)")
        ax[0].grid(True)

        ax[1].plot(t_chunk, envelope)
        ax[1].plot(t_chunk[locs_env], envelope[locs_env], 'ko')
        ax[1].set_title("QRS envelope")
        ax[1].set_ylabel("Energy")
        ax[1].set_xlabel("Time")
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    return locs_r, envelope

# Chunk-based detection
#  R peak detection (Full Pan-Tompkins Envelope)
# ---------------------------
# def detect_r_peaks_full_recording(ecg_filtered, fs, chunk_seconds=60):
    """
    True Pan-Tompkins QRS Detection Pipeline.
    Transforms the ECG into a smooth energy envelope, rendering 
    high-frequency muscle artifacts completely invisible to the detector.
    """
    # 1. Derivative (Enhances steep QRS slopes, ignores baseline)
    ecg_diff = np.diff(ecg_filtered, prepend=ecg_filtered[0])
    
    # 2. Squaring (Non-linear amplification of those steep slopes)
    ecg_sq = ecg_diff ** 2
    
    # 3. Moving Window Integration (The Artifact Crusher)
    # Averages out the jagged noise, turning QRS complexes into solid 'lumps'
    window_len = int(0.15 * fs) # 150 ms physiological QRS window
    window = np.ones(window_len) / window_len
    
    # np.convolve is exponentially faster than pd.Series.rolling
    envelope = np.convolve(ecg_sq, window, mode='same')
    
    # 4. Global Thresholding on the Smooth Envelope
    # Because the noise is now physically flattened out, std() is completely safe to use again.
    global_mean = np.mean(envelope)
    global_std = np.std(envelope)
    
    peak_height = global_mean + 1.0 * global_std
    dynamic_prominence = 0.5 * global_std

    chunk_samples = int(chunk_seconds * fs)
    total_samples = len(envelope)
    all_locs = []
    start = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        
        # We run the peak detector on the ENVELOPE, not the raw ECG!
        env_chunk = envelope[start:end]

        locs_chunk, _ = signal.find_peaks(
            env_chunk,
            height=peak_height,
            distance=int(0.3 * fs), # 300 ms refractory
            prominence=dynamic_prominence
        )
        
        all_locs.extend(locs_chunk + start)
        start += chunk_samples

    all_locs = np.array(all_locs)
    print("Detected beats:", len(all_locs))

    return all_locs

#%% 
# RR intervals
# ---------------------------
def compute_rr_intervals(locs, t):

    rr = pd.Series(np.diff(t[locs])).dt.total_seconds().values

    valid = (rr > 0.25) & (rr < 2.0)

    rr_clean = rr[valid]

    t_rr = t[locs[:-1]][valid]

    return rr_clean, t_rr

#%% 
#Heart rate plot
# ---------------------------
def plot_heart_rate(t_rr, RR, title):

    hr = 60 / RR

    plt.figure(figsize=(12,4))

    plt.plot(t_rr, hr, '.', markersize=1)

    plt.ylabel("Heart Rate (BPM)")
    plt.xlabel("Time")
    plt.title(title)

    plt.ylim(30,200)

    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.show()

#%% 
#  QRS regularity analysis
# ---------------------------
def analyze_qrs_regularity(t_rr, RR):

    rr_series = pd.Series(RR, index=t_rr)

    rr_stats = rr_series.resample("10S").agg([
        "mean",
        "std",
        "count"
    ])

    rr_stats["HR_mean"] = 60 / rr_stats["mean"]

    return rr_stats

#%% 
# Plot RR variability
# ---------------------------
def plot_rr_variability(rr_stats, title):

    plt.figure(figsize=(12,4))

    plt.plot(rr_stats.index, rr_stats["std"])

    plt.ylabel("RR std (s)")
    plt.xlabel("Time")

    plt.title(title)

    plt.grid(True)

    plt.show()

#%% 
# PAC detection
# ---------------------------

def detect_pac(RR, t_rr, locs, t):
    """
    Detecteert PACs en koppelt ze direct aan hun absolute locatie 
    in de originele ECG array om verschuivingen te voorkomen.
    """
    pac_locs = []
    window_size = 5

    for i in range(window_size, len(RR) - 1):
        local_median = np.median(RR[i-window_size:i])

        # RR[i] is het korte interval. De PAC-slag valt aan het EINDE hiervan.
        # Daarom kijken we naar t_rr[i+1] voor de tijd van de PAC.
        if RR[i] < 0.85 * local_median and RR[i+1] > 1.15 * local_median:
            
            # Pak de exacte timestamp van de prematuur gevallen slag
            pac_time = t_rr[i+1]
            
            # Zoek precies welke index in 'locs' bij deze timestamp hoort
            idx = np.where(t[locs] == pac_time)[0]
            if len(idx) > 0:
                pac_locs.append(locs[idx[0]])

    pac_locs = np.array(pac_locs)

    print(f"=========================================")
    print(f"--> TOTAAL AANTAL PACs GEVONDEN: {len(pac_locs)}")
    print(f"=========================================")

    return pac_locs

#%% 
#  QRS visual validation
# ---------------------------
def plot_qrs_check(ecg, t, locs, fs, start_sec=1000, duration=10):

    start = int(start_sec * fs)
    end = start + int(duration * fs)

    mask = (locs >= start) & (locs < end)

    plt.figure(figsize=(12,3))

    plt.plot(t[start:end], ecg[start:end])

    plt.plot(t[locs[mask]], ecg[locs[mask]], 'r*')

    plt.title("QRS detection check")

    plt.grid(True)

    plt.show()


# ----------------------------------------------------------
# Envelope check
# ----------------------------------------------------------
def plot_envelope_check(envelope, t, locs, fs, start_sec=1000, duration=10, title="Envelope check"):
    start = int(start_sec * fs)
    end = start + int(duration * fs)

    if end > len(envelope):
        end = len(envelope)

    mask = (locs >= start) & (locs < end)

    plt.figure(figsize=(12, 3))
    plt.plot(t[start:end], envelope[start:end], linewidth=1)
    plt.plot(t[locs[mask]], envelope[locs[mask]], 'ko', markersize=5)
    plt.title(title)
    plt.ylabel("Envelope")
    plt.xlabel("Time")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    plt.show()


#%%
# Plot RR intervals with PAC markers
# ---------------------------
def plot_rr_with_pac(t_rr, RR, pac_idx, title):

    plt.figure(figsize=(12,4))

    plt.plot(t_rr, RR, '.', markersize=2, label="RR interval")

    plt.plot(
        t_rr[pac_idx],
        RR[pac_idx],
        'ro',
        markersize=4,
        label="PAC"
    )

    plt.ylabel("RR interval (s)")
    plt.xlabel("Time")

    plt.title(title)

    plt.legend()

    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.show()

#%%
#%%
# Plot ECG with PAC markers

def plot_ecg_with_pac(ecg, t, locs, pac_locs, fs, pac_index=0, window_sec=10):
    """
    Richeert de grafiek automatisch op een gevonden PAC, in plaats van 
    op een willekeurige tijd.
    """
    if len(pac_locs) == 0:
        print("Geen PACs om te plotten!")
        return

    # Zoek de locatie van de specifieke PAC (standaard de eerste: index 0)
    center_loc = pac_locs[pac_index]
    
    # Bereken het window: 5 seconden vóór de PAC, en 5 seconden erna
    start = int(center_loc - (window_sec / 2) * fs)
    end = int(center_loc + (window_sec / 2) * fs)
    
    # Voorkom dat we buiten de data vallen
    start = max(0, start)
    end = min(len(ecg), end)

    # Filter de markers voor dit specifieke window
    mask_qrs = (locs >= start) & (locs < end)
    mask_pac = (pac_locs >= start) & (pac_locs < end)

    plt.figure(figsize=(12, 4))
    plt.plot(t[start:end], ecg[start:end], linewidth=1, color='tab:blue', label="Filtered ECG")
    
    # Plot QRS en PACs
    plt.plot(t[locs[mask_qrs]], ecg[locs[mask_qrs]], 'k*', markersize=6, label="QRS (Normaal)")
    
    if np.any(mask_pac):
        plt.plot(t[pac_locs[mask_pac]], ecg[pac_locs[mask_pac]], 'ro', markersize=8, label="PAC")

    plt.title(f"ECG Ingezoomd op PAC #{pac_index + 1}")
    plt.ylabel("ECG (mV)")
    plt.xlabel("Tijd")
    plt.grid(True)
    plt.legend(loc="upper right")
    
    # Zorg dat de x-as leesbaar blijft met uren, minuten en seconden
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    plt.show()
#%% 
# 4. EXECUTION PIPELINE paths 
# ==========================================================

# ---------------------------
# Define file paths
# ---------------------------
path1 = r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs+PVCs.mat"

path2 = r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs.mat"


#%% 
# uitvoeren

# Recording 1 
# ==========================================================

print("\nProcessing Recording 1")

ecg1, fs1, t1 = read_and_clean_ecg_mat(path1)

filtered1 = apply_clinical_bandpass(ecg1, fs1)

locs1, envelope1 = detect_r_peaks(filtered1, fs1, plotresult=False, t_chunk=t1)

RR1, t_rr1 = compute_rr_intervals(locs1, t1)

plot_heart_rate(t_rr1, RR1, "Heart Rate – Recording 1")

rr_stats1 = analyze_qrs_regularity(t_rr1, RR1)

plot_rr_variability(rr_stats1, "RR variability – Recording 1")

pac_idx1, t_pac1 = detect_pac(RR1, t_rr1)

plot_qrs_check(
    filtered1,
    t1,
    locs1,
    fs1,
    start_sec=1000,
    duration=10,
)

plot_envelope_check(
    envelope1,
    t1,
    locs1,
    fs1,
    start_sec=1000,
    duration=10,
)


# ==========================================================
# Recording 2
# ==========================================================

print("\nProcessing Recording 2")

ecg2, fs2, t2 = read_and_clean_ecg_mat(path2)

filtered2 = apply_clinical_bandpass(ecg2, fs2)

locs2, envelope2 = detect_r_peaks(
    filtered2,
    fs2,
    plotresult=False,
    t_chunk=t2,
)

RR2, t_rr2 = compute_rr_intervals(locs2, t2)

plot_heart_rate(t_rr2, RR2, "Heart Rate – Recording 2")

rr_stats2 = analyze_qrs_regularity(t_rr2, RR2)

plot_rr_variability(rr_stats2, "RR variability – Recording 2")

pac_idx2, t_pac2 = detect_pac(RR2, t_rr2)

plot_qrs_check(
    filtered2,
    t2,
    locs2,
    fs2,
    start_sec=1000,
    duration=10,
)

plot_envelope_check(
    envelope2,
    t2,
    locs2,
    fs2,
    start_sec=1000,
    duration=10,
)
# %%
# ==========================================================
# Recording 1 
# ==========================================================
print("\nProcessing Recording 1 (PACs + PVCs)")

ecg1, fs1, t1 = read_and_clean_ecg_mat(path1)
filtered1 = apply_clinical_bandpass(ecg1, fs1)
locs1, envelope1 = detect_r_peaks(filtered1, fs1, plotresult=False, t_chunk=t1)
RR1, t_rr1 = compute_rr_intervals(locs1, t1)

plot_heart_rate(t_rr1, RR1, "Heart Rate – Recording 1")
rr_stats1 = analyze_qrs_regularity(t_rr1, RR1)
plot_rr_variability(rr_stats1, "RR variability – Recording 1")

# --- HIER IS DE FIX VOOR RECORDING 1 ---
pac_locs1 = detect_pac(RR1, t_rr1, locs1, t1)
# Plot de 1e gevonden PAC (index 0)
plot_ecg_with_pac(filtered1, t1, locs1, pac_locs1, fs1, pac_index=0)


# ==========================================================
# Recording 2
# ==========================================================
print("\nProcessing Recording 2 (Only PACs)")

ecg2, fs2, t2 = read_and_clean_ecg_mat(path2)
filtered2 = apply_clinical_bandpass(ecg2, fs2)
locs2, envelope2 = detect_r_peaks(filtered2, fs2, plotresult=False, t_chunk=t2)
RR2, t_rr2 = compute_rr_intervals(locs2, t2)

plot_heart_rate(t_rr2, RR2, "Heart Rate – Recording 2")
rr_stats2 = analyze_qrs_regularity(t_rr2, RR2)
plot_rr_variability(rr_stats2, "RR variability – Recording 2")

# --- HIER IS DE FIX VOOR RECORDING 2 ---
pac_locs2 = detect_pac(RR2, t_rr2, locs2, t2)
# Plot de 5e gevonden PAC (index 4) om te kijken of die ook klopt
plot_ecg_with_pac(filtered2, t2, locs2, pac_locs2, fs2, pac_index=4)
# %%
