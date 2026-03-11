# Read Telemetry
# Advanced Signal Processing (TM12005)
# Made by: M.S. van Schie (m.vanschie@erasmusmc.nl) & M.M. de Boer (m.m.deboer@erasmusmc.nl)
#%% 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd


#%% Basis
leads = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

def read_ecg_mat(path, plotresult=True):
    # open datafile
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    ecg = data['ecg'].sig[:,leads.index('II')]

    fs = data['ecg'].header.Sampling_Rate
    t0 = datetime.datetime(*data['ecg'].start_vec)

    nSamples = data['ecg'].sig.shape[0]
    t = pd.date_range(
        start=t0,
        periods=nSamples,
        freq=pd.Timedelta(seconds=1/fs)
    )
    
    # Plot signal in time domain
    if plotresult:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t, ecg)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ECG (mV)")
        ax.set_title("Raw ECG Signal")
        plt.show()
    return ecg, fs, t


#%%

ecg1, fs1, t1 = read_ecg_mat(r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs+PVCs.mat", plotresult=True)
ecg2, fs2, t2 = read_ecg_mat(r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs.mat", plotresult=True)


# %% Zoom-functie (HIER neerzetten)

def plot_window_around_time(ecg, t, center_time="21:00:00", window_seconds=20, title="ECG zoom"):
    day = pd.Timestamp(t[0]).normalize()
    center = day + pd.to_timedelta(center_time)

    start = center - pd.to_timedelta(window_seconds/2, unit="s")
    end   = center + pd.to_timedelta(window_seconds/2, unit="s")

    mask = (t >= start) & (t <= end)
    """
    plt.figure(figsize=(15, 4))
    plt.plot(t[mask], ecg[mask], linewidth=1)
    plt.title(f"{title} | {start.time()}–{end.time()}")
    plt.ylabel("Amplitude (mV)")
    plt.xlabel("Time")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    plt.show()
    """

# %% Zoom-calls (HIER aanroepen)
plot_window_around_time(ecg1, t1, center_time="23:13:40", window_seconds=100,
                        title="004_Groenewoud_PACs+PVCs (Lead II)")

plot_window_around_time(ecg2, t2, center_time="11:14:40", window_seconds=100,
                        title="004_Groenewoud_PACs (Lead II)")
# %%
def plot_20sec_segment(ecg, fs, t, title="ECG Segment"):
    """
    Plots the first 20 seconds of an existing ECG array.
    """
    # Calculate samples for 20 seconds
    n_samples = int(100 * fs)
    
    # Slice the data
    ecg_slice = ecg[:n_samples]
    t_slice = t[:n_samples]
    
    # Create the plot
    plt.figure(figsize=(15, 4))
    plt.plot(t_slice, ecg_slice, color='darkred', linewidth=1)
    
    # Formatting
    plt.title(f"{title} (First 20 Seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.xlabel("Time")
    
    # Grid helps in identifying the duration of QRS complexes
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Formatting x-axis to show seconds clearly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.tight_layout()
    plt.show()

# Run the new plot for both datasets
plot_20sec_segment(ecg1, fs1, t1, title="004_Groenewoud_PACs+PVCs")
plot_20sec_segment(ecg2, fs2, t2, title="004_Groenewoud_PACs")

def filter_and_plot_clean_ecg(ecg_full, t_full, fs):
    # 1. Standard Peak Detection on the FULL signal first
    # This avoids index mismatching later
    ecg_sq = ecg_full**2
    
    # Adaptive height based on the full recording's 90th percentile
    peaks, _ = signal.find_peaks(
        ecg_sq, 
        height=np.percentile(ecg_sq, 90), 
        distance=int(0.3 * fs)
    )

    # 2. Calculate RR-intervals BEFORE filtering the gap
    # This uses the original time indices so they match the peaks
    rr_diffs = np.diff(t_full[peaks])
    rr_intervals = pd.Series(rr_diffs).dt.total_seconds().values
    
    # The time associated with each RR interval is the time of the first peak
    t_hr = t_full[peaks[:-1]]
    
    # 3. Apply the < 30 BPM Filter
    # This is where we "delete" the gap. 
    # If the gap is hours long, the RR interval will be > 2s, so it gets dropped.
    valid_mask = (rr_intervals > 60/220) & (rr_intervals < 60/30)
    
    hr_clean = 60 / rr_intervals[valid_mask]
    t_hr_clean = t_hr[valid_mask]

    # 4. Visualization
    plt.figure(figsize=(15, 6))
    
    # Scatter for individual beats
    plt.scatter(t_hr_clean, hr_clean, s=0.5, color='blue', alpha=0.1, label="Individual Beats")
    
    # Trend line (Moving Average)
    hr_series = pd.Series(hr_clean, index=t_hr_clean)
    moving_avg = hr_series.rolling(window=100, center=True, min_periods=1).mean()
    
    plt.plot(t_hr_clean, moving_avg, color='red', linewidth=1.5, label="Heart Rate Trend")

    # Layout and Formatting
    plt.title("Ventriculaire Frequentie Verloop (Gaps Filtered < 30 BPM)")
    plt.ylabel("Heart Rate (BPM)")
    plt.xlabel("Time of Day")
    plt.ylim(20, 210) 
    
    plt.axhline(100, color='orange', linestyle='--', alpha=0.4, label='Tachycardia')
    plt.axhline(60, color='green', linestyle='--', alpha=0.4, label='Bradycardia')
    
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.legend(loc='upper right', markerscale=10)
    
    plt.tight_layout()
    plt.show()

    return hr_clean, t_hr_clean

# Run the analysis
print("Analyzing Recording 1...")
hr_1, t_hr_1 = filter_and_plot_clean_ecg(ecg1, t1, fs1)

#%% verwijderen van datapunten
import numpy as np

def read_and_clean_ecg_mat(path, plotresult=True):
    # Open datafile
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    ecg_raw = data['ecg'].sig[:,leads.index('II')]

    fs = data['ecg'].header.Sampling_Rate
    t0 = datetime.datetime(*data['ecg'].start_vec)

    nSamples = data['ecg'].sig.shape[0]
    t = pd.date_range(
        start=t0,
        periods=nSamples,
        freq=pd.Timedelta(seconds=1/fs)
    )
    
    # --- DE FIX ---
    # Voor het PLOTTEN: We vervangen de kapotte data door 'NaN' (Not a Number).
    # Matplotlib tekent geen lijnen naar NaN. Hierdoor schaalt je Y-as weer normaal 
    # (bijv. -100 tot +200) én zie je een fysiek, leeg gat tussen 18:00 en 00:00!
    ecg_plot = ecg_raw.astype(float).copy()
    ecg_plot[ecg_raw <= -1000] = np.nan 

    if plotresult:
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(t, ecg_plot, linewidth=0.5, color='tab:blue')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_xlabel("Time")
        ax.set_ylabel("ECG (mV)")
        ax.set_title("Opgeschoond ECG Signaal (Artefacten en Gaten genegeerd)")
        plt.show()
    
    # Voor de DATA-ANALYSE: We geven alleen de échte samples terug.
    # Algoritmes zoals signal.find_peaks() crashen namelijk op 'NaN'.
    mask = ecg_raw > -1000
    ecg_clean = ecg_raw[mask]
    t_clean = t[mask]
    
    print(f"Er zijn {len(ecg_raw) - len(ecg_clean)} kapotte samples verwijderd.")
    
    return ecg_clean, fs, t_clean

# Laad de data opnieuw in met de nieuwe functie!
ecg1, fs1, t1 = read_and_clean_ecg_mat(r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs+PVCs.mat",  plotresult=True)
ecg2, fs2, t2 = read_and_clean_ecg_mat(r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs.mat",  plotresult=True)

#%% Filteren van Signaal
"""
Filteren van Signaal volgens colleges 

"""
# Kies testduur
test_seconds = 100

# Eerste 100 s van opname 1
n1 = int(test_seconds * fs1)
ecg1_short = ecg1[:n1]
t1_short = t1[:n1]

# Eerste 100 s van opname 2
n2 = int(test_seconds * fs2)
ecg2_short = ecg2[:n2]
t2_short = t2[:n2]

from scipy import signal

# Opname 1
b, a = signal.butter(8, 40, 'low', fs=fs1, output='ba')
b1, a1 = signal.butter(2, 0.5, 'high', fs=fs1, output='ba')

filtered_8_40hz1 = signal.filtfilt(b, a, ecg1_short)
filtered_2_05hz1 = signal.filtfilt(b1, a1, filtered_8_40hz1)

# Opname 2
bb, aa = signal.butter(8, 40, 'low', fs=fs2, output='ba')
b11, a11 = signal.butter(2, 0.5, 'high', fs=fs2, output='ba')

filtered_8_40hz2 = signal.filtfilt(bb, aa, ecg2_short)
filtered_2_05hz2 = signal.filtfilt(b11, a11, filtered_8_40hz2)

fig, ax = plt.subplots(4, 1, figsize=(12, 8))

ax[0].plot(t1_short, ecg1_short)
ax[0].set_title("Raw ECG 1 - first 100 s")

ax[1].plot(t1_short, filtered_2_05hz1)
ax[1].set_title("Filtered ECG 1 - first 100 s")

ax[2].plot(t2_short, ecg2_short)
ax[2].set_title("Raw ECG 2 - first 100 s")

ax[3].plot(t2_short, filtered_2_05hz2)
ax[3].set_title("Filtered ECG 2 - first 100 s")

for a in ax:
    a.set_xlabel("Time")
    a.set_ylabel("ECG (mV)")
    a.grid(True)

plt.tight_layout()
plt.show()

#%% Herkennen van de hartfrequentie dmv QRS
"""
Eerst Herkennen van QRS Complexen -> de noisen
Daarna frequentie QRS complexen plotten
Dan per tijdseenheid een gemiddelde stratificeren en pre-arterieel/ventriculair eruit halen"
- Per 20 slagen bijv. 
"""

#%% Herkennen van de hartfrequentie dmv QRS

# R-peak detectie en RR-interval analyse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal


def detect_r_peaks(ecg_filtered, fs, t, plotresult=False, title="ECG"):
    """
    Detecteert R-toppen in een gefilterd ECG-signaal.
    Gebruikt een adaptieve drempel, minimale afstand en prominence.
    """
    
    # Adaptieve thresholds
    peak_height = np.mean(ecg_filtered) + 1.0 * np.std(ecg_filtered)
    peak_prominence = 0.15
    min_distance = int(0.35 * fs)   # 350 ms
    
    # Peak detection
    locs, properties = signal.find_peaks(
        ecg_filtered,
        height=peak_height,
        distance=min_distance,
        prominence=peak_prominence
    )
    
    # RR-intervallen in seconden
    rr_intervals = pd.Series(np.diff(t[locs])).dt.total_seconds().values
    
    # Gemiddelden
    mean_rr = np.mean(rr_intervals)
    mean_hr = 60 / mean_rr
    
    if plotresult:
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        
        # Plot ECG + peaks
        ax[0].plot(t, ecg_filtered, linewidth=1)
        ax[0].plot(t[locs], ecg_filtered[locs], 'r*', markersize=8)
        ax[0].set_title(f"{title} - detected R-peaks")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("ECG (mV)")
        ax[0].grid(True)
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Plot RR intervals
        ax[1].plot(t[locs[:-1]], rr_intervals, '-o', markersize=4)
        ax[1].set_title(f"{title} - RR intervals")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("RR interval (s)")
        ax[1].grid(True)
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Tekst met gemiddelde waarden
        ax[1].text(
            0.98, 0.95,
            f"Mean RR = {mean_rr:.3f} s\nMean HR = {mean_hr:.1f} bpm",
            transform=ax[1].transAxes,
            ha='right',
            va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        plt.tight_layout()
        plt.show()
    
    return locs, rr_intervals, mean_rr, mean_hr

# Opname 1
locs, RR_intervals, mean_RR_interval, mean_heartrate = detect_r_peaks(
    filtered_2_05hz1,
    fs1,
    t1_short,
    plotresult=True,
    title="RR interval Groenewoud_PACs+PVCs"
)

# Opname 2
locs1, RR_intervals1, mean_RR_interval1, mean_heartrate1 = detect_r_peaks(
    filtered_2_05hz2,
    fs2,
    t2_short,
    plotresult=True,
    title="RR interval Groenewoud_PACs"
)

"""
Nu hebben we de QRS bepaald
Volgende stappen
- Checken of het ook klopt voor volgende secondes, nu alleen eerste 100
- Regularuteit bepalen van de QRS complexen per x aantal seconden, 10 of 100 ofzo?
- PVC's detecteren met een pvc vinden in het complex en die dan zo namaken?
- RR_prev, RR_next, qrs_width bepalen per hartslag, hiermee ook qrswidth maken voor pvc's




"""
# %%
from scipy import signal

def apply_clinical_bandpass(ecg_data, fs):
    """
    Applies a 0.5 - 15 Hz Butterworth filter. 
    15 Hz is the clinical gold standard for QRS detection as it 
    aggressively attenuates high-frequency skeletal muscle (EMG) noise.
    """
    # CHANGED: 40 Hz down to 15 Hz to crush motion artifacts
    b_low, a_low = signal.butter(4, 15, 'low', fs=fs, output='ba')
    
    b_high, a_high = signal.butter(2, 0.5, 'high', fs=fs, output='ba')

    filtered_low = signal.filtfilt(b_low, a_low, ecg_data)
    filtered_clinical = signal.filtfilt(b_high, a_high, filtered_low)
    
    return filtered_clinical

# --- 3. Diagnostics & Visualization ---
def load_and_preprocess_holter(path):
    """
    Extracts telemetry data and applies a strict physiological voltage gate 
    to eliminate both negative dropouts and extreme positive motion artifacts.
    """
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    ecg_raw = data['ecg'].sig[:, LEADS.index('II')]
    fs = data['ecg'].header.Sampling_Rate
    t0 = datetime.datetime(*data['ecg'].start_vec)
    
    t_full = pd.date_range(start=t0, periods=len(ecg_raw), freq=pd.Timedelta(seconds=1/fs))
    
    # CLINICAL FIX: Strict Voltage Gating
    # A normal QRS complex rarely exceeds +/- 5 mV. 
    # We remove anything outside +/- 500 mV to kill the massive artifact spikes entirely.
    valid_sensor_mask = (ecg_raw > -500) & (ecg_raw < 500)
    
    ecg_clean = ecg_raw[valid_sensor_mask]
    t_clean = t_full[valid_sensor_mask]
    
    print(f"File loaded. Excised {len(ecg_raw) - len(ecg_clean)} extreme artifact samples.")
    return ecg_clean, fs, t_clean

def analyze_full_holter_clinical(ecg_filtered, fs, t, title="24h Holter Analysis"):
    """
    Reverts to the highly successful detection algorithm, utilizing the 
    now cleanly gated telemetry data.
    """
    # 1. Ventricular Depolarization Detection
    # Because the extreme 5000mV spikes are gone, the std() is now purely physiological.
    peak_height = np.mean(ecg_filtered) + 2.5 * np.std(ecg_filtered)
    min_distance = int(0.3 * fs) 
    
    locs, properties = signal.find_peaks(
        ecg_filtered,
        height=peak_height,
        distance=min_distance,
        prominence=0.25
    )
    
    # 2. RR-Interval Computation
    rr_intervals = pd.Series(np.diff(t[locs])).dt.total_seconds().values
    
    # 3. Artifact Rejection
    valid_mask = (rr_intervals > 60/220) & (rr_intervals < 60/30)
    clean_rr = rr_intervals[valid_mask]
    heart_rates = 60 / clean_rr
    t_hr = t[locs[:-1]][valid_mask]
    
    # 4. Diagnostic Visualization
    plt.figure(figsize=(15, 6))
    
    plt.scatter(t_hr, heart_rates, s=1, color='blue', alpha=0.15, label="Instantaneous Rate")
    
    hr_series = pd.Series(heart_rates, index=t_hr)
    hr_resampled = hr_series.resample('10s').mean().interpolate(limit=6)
    moving_avg = hr_resampled.rolling(window=30, center=True, min_periods=1).mean()
    
    plt.plot(moving_avg.index, moving_avg.values, color='red', linewidth=1.5, label="Mean Trend")

    plt.title(f"Ventricular Rate Profile: {title}")
    plt.ylabel("Heart Rate (BPM)")
    plt.xlabel("Time of Day")
    plt.ylim(30, 200)
    plt.axhline(100, color='orange', linestyle='--', alpha=0.5, label='Tachycardia (>100)')
    plt.axhline(60, color='green', linestyle='--', alpha=0.5, label='Bradycardia (<60)')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', markerscale=10)
    plt.tight_layout()
    plt.show()
    
    return clean_rr, heart_rates, t_hr
# --- Execution Pipeline ---

# DEFINE PATHS
path_1 = r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs+PVCs.mat"
path_2 = r"/Users/cathelijnedeneke/Desktop/TM 2025:2026/Q3/Advanced signal processing /Erasmus deel/Assignment 2 /004_Groenewoud_PACs.mat"

# DIAGNOSTIC RUN 1
print("\n--- Initiating Diagnostic Run: Recording 1 ---")
# Utilizing your pre-existing clean extraction function
ecg1_clean, fs1, t1_clean = read_and_clean_ecg_mat(path_1, plotresult=False)
print("Applying physiological bandpass filter...")
filtered1 = apply_clinical_bandpass(ecg1_clean, fs1)
print("Computing tachogram...")
rr1, hr1, t_hr1 = analyze_full_holter_clinical(filtered1, fs1, t1_clean, "Recording 1 (PACs + PVCs)")

# DIAGNOSTIC RUN 2
print("\n--- Initiating Diagnostic Run: Recording 2 ---")
ecg2_clean, fs2, t2_clean = read_and_clean_ecg_mat(path_2, plotresult=False)
print("Applying physiological bandpass filter...")
filtered2 = apply_clinical_bandpass(ecg2_clean, fs2)
print("Computing tachogram...")
rr2, hr2, t_hr2 = analyze_full_holter_clinical(filtered2, fs2, t2_clean, "Recording 2 (PACs)")

#%% peak detection full code 

b, a = signal.butter(2, [5,15], btype='bandpass', fs=fs1)
ecg_qrs1 = signal.filtfilt(b,a,ecg1)


def detect_r_peaks_full_recording(ecg_filtered, fs, t, chunk_seconds=60):
    """
    Detect R-peaks over an entire ECG recording by processing chunks.
    This keeps computation reasonable for long recordings.
    """

    chunk_samples = int(chunk_seconds * fs)
    total_samples = len(ecg_filtered)

    all_locs = []

    start = 0

    while start < total_samples:

        end = min(start + chunk_samples, total_samples)

        ecg_chunk = ecg_filtered[start:end]
        t_chunk = t[start:end]

        # Detect peaks in this chunk
        locs_chunk, _, _, _ = detect_r_peaks(
            ecg_chunk,
            fs,
            t_chunk,
            plotresult=False
        )

        # Correct index because chunk starts later in signal
        locs_chunk = locs_chunk + start

        all_locs.extend(locs_chunk)

        start += chunk_samples

    all_locs = np.array(all_locs)

    # RR intervals
    rr_intervals = pd.Series(np.diff(t[all_locs])).dt.total_seconds().values

    mean_rr = np.mean(rr_intervals)
    mean_hr = 60 / mean_rr

    print(f"Detected beats: {len(all_locs)}")
    print(f"Mean HR: {mean_hr:.1f} bpm")

    return all_locs, rr_intervals
# Filter full ECG
b, a = signal.butter(8, 40, 'low', fs=fs1)
b1, a1 = signal.butter(2, 0.5, 'high', fs=fs1)

filtered_full1 = signal.filtfilt(b, a, ecg1)
filtered_full1 = signal.filtfilt(b1, a1, filtered_full1)

# Run full detection
locs_full1, RR_full1 = detect_r_peaks_full_recording(
    filtered_full1,
    fs1,
    t1,
    chunk_seconds=60
)

t_rr = t1[locs_full1[:-1]]
# Physiological RR interval limits
valid = (RR_full1 > 0.25) & (RR_full1 < 2.0)

RR_clean = RR_full1[valid]
t_rr = t1[locs_full1[:-1]][valid]

hr = 60 / RR_clean

plt.figure(figsize=(12,4))
plt.plot(t_rr, hr, '.', markersize=1)
plt.ylabel("Heart Rate (BPM)")
plt.xlabel("Time")
plt.title("Heart Rate over full recording")
plt.ylim(30,200)
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.show()

def plot_qrs_check(ecg, t, locs, start_sec=1000, duration=10):

    fs = int(1/(t[1]-t[0]).total_seconds())

    start = int(start_sec*fs)
    end = start + int(duration*fs)

    mask_peaks = (locs >= start) & (locs < end)

    plt.figure(figsize=(12,3))
    plt.plot(t[start:end], ecg[start:end])
    plt.plot(t[locs[mask_peaks]], ecg[locs[mask_peaks]], 'r*')
    plt.title("QRS detection check")
    plt.grid(True)
    plt.show()
    
plot_qrs_check(filtered_full1, t1, locs_full1, start_sec=2000)

#%% Calculate regularity 

def analyze_qrs_regularity(t_rr, RR):

    rr_series = pd.Series(RR, index=t_rr)

    # Resample per 10 seconds
    rr_stats = rr_series.resample("10S").agg([
        "mean",
        "std",
        "count"
    ])

    rr_stats["HR_mean"] = 60 / rr_stats["mean"]
    rr_stats["HR_std"] = 60 * rr_stats["std"] / (rr_stats["mean"]**2)

    return rr_stats

rr_stats = analyze_qrs_regularity(t_rr, RR_clean)

plt.figure(figsize=(12,4))
plt.plot(rr_stats.index, rr_stats["std"])
plt.ylabel("RR std (s)")
plt.xlabel("Time")
plt.title("QRS Regularity (RR variability)")
plt.grid(True)
plt.show()


#%% Clean rr

def clean_rr_intervals(locs, t, rr_intervals):

    # Physiological RR limits
    valid = (rr_intervals > 0.25) & (rr_intervals < 2.0)

    RR_clean = rr_intervals[valid]
    t_rr_clean = t[locs[:-1]][valid]

    return RR_clean, t_rr_clean

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

def analyze_qrs_regularity(t_rr, RR):

    rr_series = pd.Series(RR, index=t_rr)

    rr_stats = rr_series.resample("10S").agg([
        "mean",
        "std",
        "count"
    ])

    rr_stats["HR_mean"] = 60 / rr_stats["mean"]

    return rr_stats

def plot_rr_variability(rr_stats, title):

    plt.figure(figsize=(12,4))

    plt.plot(rr_stats.index, rr_stats["std"])

    plt.ylabel("RR std (s)")
    plt.xlabel("Time")
    plt.title(title)

    plt.grid(True)

    plt.show()

def detect_pac(RR, t_rr):

    median_rr = np.median(RR)

    pac_indices = []

    for i in range(len(RR)-1):

        if RR[i] < 0.8 * median_rr and RR[i+1] > 1.2 * median_rr:
            pac_indices.append(i)

    pac_indices = np.array(pac_indices)

    t_pac = t_rr[pac_indices]

    print("Detected PAC candidates:", len(pac_indices))

    return pac_indices, t_pac

RR_clean1, t_rr1 = clean_rr_intervals(locs_full1, t1, RR_full1)

plot_heart_rate(t_rr1, RR_clean1, "Heart Rate - Recording 1")

rr_stats1 = analyze_qrs_regularity(t_rr1, RR_clean1)

plot_rr_variability(rr_stats1, "RR Variability - Recording 1")

pac_idx1, t_pac1 = detect_pac(RR_clean1, t_rr1)

filtered_full2 = signal.filtfilt(bb, aa, ecg2)
filtered_full2 = signal.filtfilt(b11, a11, filtered_full2)

locs_full2, RR_full2 = detect_r_peaks_full_recording(
    filtered_full2,
    fs2,
    t2,
    chunk_seconds=60
)
RR_clean2, t_rr2 = clean_rr_intervals(locs_full2, t2, RR_full2)

plot_heart_rate(t_rr2, RR_clean2, "Heart Rate - Recording 2")

rr_stats2 = analyze_qrs_regularity(t_rr2, RR_clean2)

plot_rr_variability(rr_stats2, "RR Variability - Recording 2")

pac_idx2, t_pac2 = detect_pac(RR_clean2, t_rr2)

plt.figure(figsize=(12,4))

plt.plot(t_rr1, 60/RR_clean1, '.', markersize=1, label="HR")

plt.scatter(t_pac1, 60/RR_clean1[pac_idx1],
            color='red', s=10, label="PAC")

plt.legend()
plt.title("PAC candidates - Recording 1")

plt.show()
#%% Herkennen van PACs


#%% Herkennen van PVCs