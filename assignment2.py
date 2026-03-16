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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

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
def compute_rr_intervals(locs, t, max_rr=6.0):
    rr = pd.Series(np.diff(t[locs])).dt.total_seconds().values
    valid = (rr > 0.25) & (rr < max_rr) # Physiological limits
    rr_clean = rr[valid]
    t_rr = t[locs[:-1]][valid]
    return rr_clean, t_rr
#%% Diagnostics: Heart Rate Tachogram -> gaat HRV plotten
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

def compute_hrv_parameters(RR, t_rr):
    """
    Berekent klassieke HRV-parameters per 30s venster.
    RMSSD: gevoelig voor beat-to-beat variabiliteit (AF-marker)
    pNN50: percentage opeenvolgende RR-verschillen > 50ms
    """
    rr_series = pd.Series(RR, index=t_rr)
    
    def rmssd(x):
        if len(x) < 2: return np.nan
        return np.sqrt(np.mean(np.diff(x.values)**2))
    
    def pnn50(x):
        if len(x) < 2: return np.nan
        return np.mean(np.abs(np.diff(x.values)) > 0.05) * 100

    stats = rr_series.resample("30S").agg(
        sdnn=np.std,
        rmssd=rmssd,
        pnn50=pnn50,
        mean_rr="mean"
    )
    return stats

def plot_hrv_parameters(hrv_stats, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(hrv_stats.index, hrv_stats["sdnn"] * 1000, color='tab:blue')
    axes[0].set_ylabel("SDNN (ms)")
    axes[0].axhline(50, color='crimson', linestyle='--', alpha=0.5, label="Grens 50ms")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hrv_stats.index, hrv_stats["rmssd"] * 1000, color='tab:orange')
    axes[1].set_ylabel("RMSSD (ms)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(hrv_stats.index, hrv_stats["pnn50"], color='tab:green')
    axes[2].set_ylabel("pNN50 (%)")
    axes[2].set_xlabel("Tijd")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    fig.suptitle(title)
    plt.tight_layout()
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
# PAC detection (Met Isolatie-Filter)
def detect_pac(RR, t_rr, locs, t):
    pac_locs = []
    window_size = 5

    for i in range(window_size, len(RR) - 1):
        local_median = np.median(RR[i-window_size:i])
        if RR[i] > 2.5:  # dit zorgt ervoor dat als er een lange pauze is dat de slag erna geen pac is maar een escape ritme
            continue
        # Prematurity (< 85%) followed by compensatory pause (> 115%)
        if RR[i] < 0.8 * local_median and RR[i+1] > 1.2 * local_median:
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
# QRS Morphology: Duration Measurement
# ---------------------------
def measure_qrs_duration(ecg, locs, fs):
    """
    Meet de QRS-breedte rondom elk gedetecteerd R-piektop.
    Gebruikt een threshold van 20% van de lokale R-amplitude
    om het begin en einde van het QRS-complex te vinden.
    """
    durations = []
    search_ms = int(0.10 * fs)  # 100ms zoekvenster aan elke kant

    for loc in locs:
        start = max(0, loc - search_ms)
        end = min(len(ecg), loc + search_ms)
        segment = ecg[start:end]
        r_amp = ecg[loc]

        # Threshold: 20% van R-amplitude t.o.v. lokale baseline
        baseline = np.percentile(segment, 10)
        threshold = baseline + 0.20 * (r_amp - baseline)

        above = np.where(segment > threshold)[0]
        if len(above) < 2:
            durations.append(np.nan)
            continue

        qrs_width_samples = above[-1] - above[0]
        qrs_ms = (qrs_width_samples / fs) * 1000
        durations.append(qrs_ms)

    return np.array(durations)


#%%
# P-golf morfologie analyse
# ---------------------------
def analyze_p_wave(ecg, locs, fs):
    """
    Zoekt de P-golf in het venster 250–80 ms vóór elke R-piek.
    
    Normale P-golf: aanwezig, positief, amplitude 0.05–0.25 mV.
    Afwezig of negatief → verdacht voor PAC (ectopische focus)
    of AF (geen georganiseerde atriale activiteit).
    
    Geeft per slag terug:
      p_present  : bool   – P-golf boven drempel gevonden
      p_amplitude: float  – piekhoogte in mV (NaN als afwezig)
      p_positive : bool   – True als positief, False als negatief/bimodaal
    """
    results = []
    pre_start = int(0.25 * fs)   # 250 ms voor R
    pre_end   = int(0.08 * fs)   # 80 ms voor R (einde P-venster)

    for loc in locs:
        start = loc - pre_start
        end   = loc - pre_end

        if start < 0 or end <= start:
            results.append({
                "p_present": False,
                "p_amplitude": np.nan,
                "p_positive": False
            })
            continue

        segment = ecg[start:end]
        baseline = np.median(segment)
        peak_val = np.max(segment)
        trough_val = np.min(segment)

        # P-golf aanwezig als piek > 0.05 mV boven baseline
        amplitude = peak_val - baseline
        p_present = amplitude > 0.05

        # Positief als piek groter is dan dip
        p_positive = (peak_val - baseline) > (baseline - trough_val)

        results.append({
            "p_present":   p_present,
            "p_amplitude": round(float(amplitude), 4) if p_present else np.nan,
            "p_positive":  p_positive if p_present else False
        })

    return pd.DataFrame(results)


#%%
# T-golf inversie analyse
# ---------------------------
def analyze_t_wave(ecg, locs, fs):
    """
    Zoekt de T-golf in het venster 80–350 ms ná elke R-piek.
    
    Normale T-golf: positief in afleidingen I, II, V4-V6.
    Negatieve T-golf → typisch PVC-kenmerk (discordante repolarisatie),
    maar ook zichtbaar bij ischemie of LBBB.
    
    Geeft per slag terug:
      t_positive  : bool  – T-golf is positief
      t_amplitude : float – amplitude t.o.v. lokale baseline
      t_inverted  : bool  – True als negatief (klinisch relevant)
    """
    results = []
    post_start = int(0.08 * fs)   # 80 ms na R (voorbij ST-segment begin)
    post_end   = int(0.35 * fs)   # 350 ms na R

    for loc in locs:
        start = loc + post_start
        end   = loc + post_end

        if end >= len(ecg):
            results.append({
                "t_positive":  False,
                "t_amplitude": np.nan,
                "t_inverted":  False
            })
            continue

        segment  = ecg[start:end]
        baseline = np.median(ecg[max(0, loc - int(0.05 * fs)):loc])

        peak_pos = np.max(segment) - baseline
        peak_neg = baseline - np.min(segment)

        t_positive = peak_pos > peak_neg
        amplitude  = peak_pos if t_positive else -peak_neg
        t_inverted = not t_positive and peak_neg > 0.05

        results.append({
            "t_positive":  t_positive,
            "t_amplitude": round(float(amplitude), 4),
            "t_inverted":  t_inverted
        })

    return pd.DataFrame(results)

#%%
# Integreer morfologie in de ectopie-classificatie
# ---------------------------
def classify_ectopic_beats_full(RR, t_rr, locs, t, ecg, fs):
    """
    Uitbreiding van classify_ectopic_beats met P-golf en T-golf analyse.
    
    Classificatielogica per slag:
    
    PAC  : QRS smal (<120ms) + P aanwezig maar aberrant (negatief of
           vroeg) OF P afwezig met smal QRS
    PVC  : QRS breed (>=120ms) + T-golf geïnverteerd (discordant)
    Ambigue: QRS breed maar T-golf normaal → mogelijk aberrant
             geleid PAC, LBBB-slag of artefact
    """

    qrs_dur  = measure_qrs_duration(ecg, locs, fs)
    p_waves  = analyze_p_wave(ecg, locs, fs)
    t_waves  = analyze_t_wave(ecg, locs, fs)

    pac_locs       = []
    pvc_locs       = []
    ambiguous_locs = []
    window_size    = 5

    for i in range(window_size, len(RR) - 1):
        local_median = np.median(RR[i - window_size:i])
        if RR[i] >= 0.85 * local_median:
            continue

        beat_time = t_rr[i]
        idx = np.where(t[locs] == beat_time)[0]
        if len(idx) == 0:
            continue
        beat_idx = idx[0]

        qrs_ms     = qrs_dur[beat_idx]
        p_present  = p_waves.iloc[beat_idx]["p_present"]
        p_positive = p_waves.iloc[beat_idx]["p_positive"]
        t_inverted = t_waves.iloc[beat_idx]["t_inverted"]
        beat_loc   = locs[beat_idx]

        if np.isnan(qrs_ms):
            ambiguous_locs.append(beat_loc)
            continue

        if qrs_ms < 120:
            # Smal QRS → PAC, bevestigd als P afwezig of negatief
            pac_locs.append(beat_loc)

        elif qrs_ms >= 120 and t_inverted:
            # Breed QRS + discordante T → PVC
            pvc_locs.append(beat_loc)

        else:
            # Breed QRS maar T normaal → mogelijk aberrant PAC
            ambiguous_locs.append(beat_loc)

    pac_locs       = np.array(pac_locs)
    pvc_locs       = np.array(pvc_locs)
    ambiguous_locs = np.array(ambiguous_locs)

    # Samenvatting met morfologie breakdown
    n_pac_no_p  = sum(1 for i, l in enumerate(pac_locs)
                      if not p_waves.iloc[
                          np.where(locs == l)[0][0]]["p_present"])
    n_pac_neg_p = sum(1 for i, l in enumerate(pac_locs)
                      if p_waves.iloc[
                          np.where(locs == l)[0][0]]["p_present"]
                      and not p_waves.iloc[
                          np.where(locs == l)[0][0]]["p_positive"])

    print(f"=========================================")
    print(f"--> PACs totaal:             {len(pac_locs)}")
    print(f"    waarvan geen P-golf:     {n_pac_no_p}")
    print(f"    waarvan negatieve P:     {n_pac_neg_p}")
    print(f"--> PVCs totaal:             {len(pvc_locs)}")
    print(f"    waarvan T-geïnverteerd:  {len(pvc_locs)}")
    print(f"--> Ambigue slagen:          {len(ambiguous_locs)}")
    print(f"=========================================")

    return pac_locs, pvc_locs, ambiguous_locs

def compute_rr_without_pacs(locs, pac_locs, pvc_locs, t, max_rr=6.0):
    """
    Herberekent RR-reeks met PACs en PVCs verwijderd.
    Gebruikt voor clean AF-detectie zonder ectopie-ruis.
    """
    ectopic_set = set(np.concatenate([pac_locs, pvc_locs]))
    clean_locs = np.array([l for l in locs if l not in ectopic_set])
    
    rr = pd.Series(np.diff(t[clean_locs])).dt.total_seconds().values
    valid = (rr > 0.25) & (rr < max_rr)
    rr_clean = rr[valid]
    t_rr_clean = t[clean_locs[:-1]][valid]
    
    print(f"RR zonder ectopie: {len(rr_clean)} intervals "
          f"({len(locs) - len(clean_locs)} slagen verwijderd)")
    return rr_clean, t_rr_clean

#%%
# Visualisatie: PAC vs PVC op ECG-strip
# ---------------------------
def plot_ecg_pac_vs_pvc(ecg, t, locs, pac_locs, pvc_locs,
                         fs, event_locs=None, event_index=0, window_sec=10):
    """
    Plot een ECG-strip gecentreerd op een ectopische slag,
    met PACs (blauw) en PVCs (rood) apart gemarkeerd.
    """
    if event_locs is None:
        event_locs = np.sort(np.concatenate([pac_locs, pvc_locs]))
    if len(event_locs) == 0:
        print("Geen events om te plotten.")
        return

    center = event_locs[event_index]
    start = max(0, int(center - (window_sec / 2) * fs))
    end = min(len(ecg), int(center + (window_sec / 2) * fs))

    mask_qrs = (locs >= start) & (locs < end)
    mask_pac = (pac_locs >= start) & (pac_locs < end)
    mask_pvc = (pvc_locs >= start) & (pvc_locs < end)

    plt.figure(figsize=(14, 4))
    plt.plot(t[start:end], ecg[start:end],
             linewidth=0.8, color='tab:gray', alpha=0.8, label="Filtered ECG")
    plt.plot(t[locs[mask_qrs]], ecg[locs[mask_qrs]],
             'k*', markersize=5, label="Normal QRS")

    if np.any(mask_pac):
        plt.plot(t[pac_locs[mask_pac]], ecg[pac_locs[mask_pac]],
                 'bo', markersize=9, label="PAC (smal QRS)", zorder=5)
    if np.any(mask_pvc):
        plt.plot(t[pvc_locs[mask_pvc]], ecg[pvc_locs[mask_pvc]],
                 'r^', markersize=9, label="PVC (breed QRS)", zorder=5)

    plt.title(f"PAC vs PVC Classificatie (Event #{event_index})")
    plt.ylabel("Amplitude (mV)")
    plt.xlabel("Time")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    plt.show()


#%%
# Diagnostics: QRS-breedte histogram
# ---------------------------
def plot_qrs_width_histogram(ecg, locs, fs, title="QRS Duration Distribution"):
    """
    Histogram van alle QRS-breedtes. Handig om de 120ms grens
    te valideren en bimodaliteit (PAC + PVC populaties) te zien.
    """
    durations = measure_qrs_duration(ecg, locs, fs)
    durations = durations[~np.isnan(durations)]

    plt.figure(figsize=(8, 4))
    plt.hist(durations, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    plt.axvline(120, color='crimson', linestyle='--', linewidth=1.5,
                label='PAC/PVC grens (120 ms)')
    plt.xlabel("QRS Duur (ms)")
    plt.ylabel("Aantal slagen")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


#%%
# PVC Patroon Classificatie: Bigeminie & Trigeminie
# ---------------------------
def classify_pvc_patterns(locs, pvc_locs, min_run_length=4):
    """
    Detecteert bigeminie en trigeminie op basis van de positie
    van PVC-slagen ten opzichte van alle gedetecteerde R-pieken.

    Werkwijze:
    - Maak een binaire slag-sequentie: 0 = normaal, 1 = PVC
    - Schuif een venster over de sequentie
    - Bigeminie:  patroon [1,0] herhaalt zich >= min_run_length/2 keer
    - Trigeminie: patroon [1,0,0] herhaalt zich >= min_run_length/3 keer

    Parameters
    ----------
    locs         : alle R-piek indices (uit detect_r_peaks)
    pvc_locs     : PVC R-piek indices (uit classify_ectopic_beats)
    min_run_length: minimum aantal opeenvolgende slagen om als
                   patroon te tellen (default 4 voor bigeminie,
                   6 voor trigeminie — wordt intern aangepast)
    """
    if len(pvc_locs) == 0:
        print("Geen PVCs beschikbaar voor patroonanalyse.")
        return pd.DataFrame()

    # Stap 1: binaire sequentie bouwen
    pvc_set = set(pvc_locs)
    beat_labels = np.array([1 if loc in pvc_set else 0 for loc in locs])

    bigeminies  = []
    trigeminies = []

    n = len(beat_labels)

    # Stap 2: bigeminie scan — zoek [1,0] herhalingen
    i = 0
    while i < n - 1:
        if beat_labels[i] == 1 and beat_labels[i + 1] == 0:
            run_start = i
            j = i
            while j + 1 < n and beat_labels[j] == 1 and beat_labels[j + 1] == 0:
                j += 2
            run_length = j - run_start
            if run_length >= max(4, min_run_length):
                bigeminies.append({
                    "type": "Bigeminie",
                    "start_beat_idx": run_start,
                    "end_beat_idx": j,
                    "n_beats": run_length,
                    "n_pvc_in_run": run_length // 2,
                    "start_loc": locs[run_start],
                    "end_loc": locs[min(j, n - 1)]
                })
            i = j
        else:
            i += 1

    # Stap 3: trigeminie scan — zoek [1,0,0] herhalingen
    i = 0
    while i < n - 2:
        if (beat_labels[i] == 1 and
                beat_labels[i + 1] == 0 and
                beat_labels[i + 2] == 0):
            run_start = i
            j = i
            while (j + 2 < n and
                   beat_labels[j] == 1 and
                   beat_labels[j + 1] == 0 and
                   beat_labels[j + 2] == 0):
                j += 3
            run_length = j - run_start
            if run_length >= max(6, min_run_length):
                trigeminies.append({
                    "type": "Trigeminie",
                    "start_beat_idx": run_start,
                    "end_beat_idx": j,
                    "n_beats": run_length,
                    "n_pvc_in_run": run_length // 3,
                    "start_loc": locs[run_start],
                    "end_loc": locs[min(j, n - 1)]
                })
            i = j
        else:
            i += 1

    all_patterns = pd.DataFrame(bigeminies + trigeminies)

    print(f"=========================================")
    if all_patterns.empty:
        print("--> Geen bigeminie of trigeminie gevonden.")
    else:
        n_big = len(bigeminies)
        n_tri = len(trigeminies)
        print(f"--> BIGEMINIE episodes:   {n_big}")
        print(f"--> TRIGEMINIE episodes:  {n_tri}")
        for _, row in all_patterns.iterrows():
            print(f"    [{row['type']}] "
                  f"beat #{row['start_beat_idx']}–{row['end_beat_idx']} | "
                  f"{row['n_beats']} slagen | "
                  f"{row['n_pvc_in_run']} PVCs")
    print(f"=========================================")

    return all_patterns


#%% 
def plot_pvc_pattern(ecg, t, locs, pvc_locs, pattern_df, fs,
                     pattern_index=0, window_sec=12):
    """
    Plot een ECG-strip gecentreerd op een bigeminie- of trigeminie-episode.
    PVCs worden rood gemarkeerd, normale slagen zwart.
    """
    if pattern_df.empty:
        print("Geen patronen om te plotten.")
        return

    row = pattern_df.iloc[pattern_index]
    center_loc = row["start_loc"]

    start = max(0, int(center_loc - 1 * fs))
    end   = min(len(ecg), int(center_loc + (window_sec - 1) * fs))

    mask_all = (locs >= start) & (locs < end)
    mask_pvc = (pvc_locs >= start) & (pvc_locs < end)

    # Normale slagen = alle QRS minus de PVCs in dit venster
    pvc_set = set(pvc_locs)
    normal_in_window = np.array([l for l in locs[mask_all]
                                  if l not in pvc_set])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t[start:end], ecg[start:end],
            linewidth=0.8, color='tab:gray', alpha=0.8)

    if len(normal_in_window) > 0:
        ax.plot(t[normal_in_window], ecg[normal_in_window],
                'k*', markersize=6, label="Normaal")

    if np.any(mask_pvc):
        ax.plot(t[pvc_locs[mask_pvc]], ecg[pvc_locs[mask_pvc]],
                'r^', markersize=9, label="PVC", zorder=5)

    ax.set_title(f"{row['type']} – {row['n_pvc_in_run']} PVCs "
                 f"(Episode #{pattern_index})")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    plt.show()

#%% gaat p-golven vergelijken
def compare_p_wave_morphology(ecg, locs, pac_locs, fs, n_examples=5):
    """
    Vergelijkt P-golf morfologie tussen:
    - Normale sinusslagen
    - PAC-slagen
    
    Methode: gemiddeld P-golf template per groep +
    overlay plot voor visuele vergelijking.
    """
    pre_start = int(0.25 * fs)
    pre_end   = int(0.08 * fs)
    segment_len = pre_start - pre_end

    pac_set = set(pac_locs)

    normal_templates = []
    pac_templates    = []

    for loc in locs:
        start = loc - pre_start
        end   = loc - pre_end
        if start < 0 or end >= len(ecg):
            continue
        segment = ecg[start:end]
        # Normaliseer op baseline
        segment = segment - np.median(segment)

        if loc in pac_set:
            pac_templates.append(segment)
        else:
            normal_templates.append(segment)

    normal_templates = np.array(normal_templates)
    pac_templates    = np.array(pac_templates)

    t_axis = np.linspace(-250, -80, segment_len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Links: overlay van individuele P-golven
    ax = axes[0]
    for seg in normal_templates[:n_examples]:
        if len(seg) == segment_len:
            ax.plot(t_axis, seg, color='tab:blue', alpha=0.3, linewidth=0.8)
    for seg in pac_templates[:n_examples]:
        if len(seg) == segment_len:
            ax.plot(t_axis, seg, color='tab:red', alpha=0.3, linewidth=0.8)

    # Gemiddeld template
    if len(normal_templates) > 0:
        mean_normal = np.mean(
            [s for s in normal_templates if len(s) == segment_len], axis=0)
        ax.plot(t_axis, mean_normal, color='tab:blue',
                linewidth=2, label="Normaal gemiddeld")
    if len(pac_templates) > 0:
        mean_pac = np.mean(
            [s for s in pac_templates if len(s) == segment_len], axis=0)
        ax.plot(t_axis, mean_pac, color='tab:red',
                linewidth=2, label="PAC gemiddeld")

    ax.set_xlabel("Tijd voor R-piek (ms)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title("P-golf morfologie: normaal vs PAC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rechts: amplitude verdeling
    ax2 = axes[1]
    normal_amps = [np.max(s) - np.min(s)
                   for s in normal_templates if len(s) == segment_len]
    pac_amps    = [np.max(s) - np.min(s)
                   for s in pac_templates    if len(s) == segment_len]
    ax2.hist(normal_amps, bins=30, alpha=0.6,
             color='tab:blue', label="Normaal", edgecolor='white')
    ax2.hist(pac_amps,    bins=30, alpha=0.6,
             color='tab:red',  label="PAC",    edgecolor='white')
    ax2.set_xlabel("P-golf amplitude (mV)")
    ax2.set_ylabel("Aantal slagen")
    ax2.set_title("Amplitude verdeling P-golf")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return normal_templates, pac_templates


def cluster_pac_foci(ecg, locs, pac_locs, fs, t, correlation_threshold=0.75):
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist

    pre_start   = int(0.25 * fs)
    pre_end     = int(0.08 * fs)
    segment_len = pre_start - pre_end
    pac_set     = set(pac_locs)

    color_list = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange',
                  'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                  'tab:olive', 'tab:cyan']

    templates  = []
    valid_locs = []

    for loc in locs:
        if loc not in pac_set:
            continue
        start = loc - pre_start
        end   = loc - pre_end
        if start < 0 or end >= len(ecg):
            continue
        seg = ecg[start:end]
        if len(seg) != segment_len:
            continue
        seg = seg - np.median(seg)
        peak = np.max(np.abs(seg))
        if peak < 1e-8:
            continue
        seg = seg / peak
        if seg.shape == (segment_len,):
            templates.append(seg)
            valid_locs.append(loc)

    if len(templates) < 3:
        print("Te weinig bruikbare PAC templates voor clustering.")
        return None, None, None

    templates  = np.array(templates, dtype=np.float64)
    valid_locs = np.array(valid_locs)

    dist_condensed = pdist(templates, metric='correlation')
    Z = linkage(dist_condensed, method='ward')

    distance_cutoff = 1 - correlation_threshold
    cluster_labels  = fcluster(Z, t=distance_cutoff, criterion='distance')
    n_clusters      = len(np.unique(cluster_labels))

    print(f"=========================================")
    print(f"--> PAC FOCI ANALYSE")
    print(f"    Aantal PAC templates:  {len(templates)}")
    print(f"    Correlatie drempel:    {correlation_threshold}")
    print(f"    Aantal clusters:       {n_clusters}")
    print()
    for k in range(1, n_clusters + 1):
        idx_in_cluster = np.where(cluster_labels == k)[0]
        print(f"    Cluster {k} (focus {k}): {len(idx_in_cluster)} PACs")
    if n_clusters == 1:
        print(f"\n    --> UNIFOCAAL: alle PACs van één ectopische focus")
    elif n_clusters == 2:
        print(f"\n    --> BIFOCAAL: twee verschillende ectopische foci")
    else:
        print(f"\n    --> MULTIFOCAAL: {n_clusters} foci — verhoogd AF-risico")
    print(f"=========================================")

    t_axis = np.linspace(-250, -80, segment_len)

    # ── Figuur 1: templates per cluster ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax = axes[0]
    for k in range(1, n_clusters + 1):
        idx_k = np.where(cluster_labels == k)[0]
        cluster_templates = templates[idx_k]
        color = color_list[(k - 1) % len(color_list)]

        for seg in cluster_templates[:8]:
            ax.plot(t_axis, seg, color=color, alpha=0.15, linewidth=0.7)

        mean_t = np.mean(cluster_templates, axis=0)
        ax.plot(t_axis, mean_t, color=color, linewidth=2.5,
                label=f"Focus {k} (n={len(idx_k)})")

    ax.set_xlabel("Tijd voor R-piek (ms)")
    ax.set_ylabel("Genormaliseerde amplitude")
    ax.set_title("Gemiddeld P-golf template per focus")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(-120, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    # ── Figuur 2: dendrogram ──────────────────────────────
    ax2 = axes[1]
    dendrogram(
        Z,
        ax=ax2,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90,
        color_threshold=distance_cutoff * max(Z[:, 2]),
        above_threshold_color='gray'
    )
    ax2.axhline(distance_cutoff * max(Z[:, 2]),
                color='crimson', linestyle='--', linewidth=1,
                label=f"Knipdrempel (r={correlation_threshold})")
    ax2.set_xlabel("PAC index")
    ax2.set_ylabel("Ward afstand")
    ax2.set_title("Dendrogram – PAC morfologie clustering")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    # ── Figuur 3: tijdlijn van foci ───────────────────────
    fig2, ax3 = plt.subplots(figsize=(14, 3))
    for k in range(1, n_clusters + 1):
        idx_k  = np.where(cluster_labels == k)[0]
        locs_k = valid_locs[idx_k]
        color  = color_list[(k - 1) % len(color_list)]

        times_k = []
        for l in locs_k:
            l = int(l)
            if l < len(t):
                times_k.append(t[l])

        ax3.scatter(times_k, [k] * len(times_k),
                    color=color, s=20, alpha=0.7,
                    label=f"Focus {k}")

    ax3.set_yticks(range(1, n_clusters + 1))
    ax3.set_yticklabels([f"Focus {k}" for k in range(1, n_clusters + 1)])
    ax3.set_xlabel("Tijd")
    ax3.set_title("Tijdlijn: welke focus is actief wanneer?")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()
    plt.show()

    return cluster_labels, templates, valid_locs

#%%

def cluster_pac_morphology(ecg, locs, pac_locs, fs):
    """
    Groepeert PACs op basis van P-golf vormgelijkenis.
    Twee PACs met hoge correlatie (>0.85) komen waarschijnlijk
    van dezelfde ectopische focus.
    Lage correlatie tussen PACs → meerdere foci → hoger AF-risico.
    """
    pre_start   = int(0.25 * fs)
    pre_end     = int(0.08 * fs)
    segment_len = pre_start - pre_end
    pac_set     = set(pac_locs)

    templates = []
    valid_locs = []

    for loc in locs:
        if loc not in pac_set:
            continue
        start = loc - pre_start
        end   = loc - pre_end
        if start < 0 or end >= len(ecg):
            continue
        seg = ecg[start:end]
        if len(seg) != segment_len:
            continue
        seg = seg - np.median(seg)
        seg = seg / (np.max(np.abs(seg)) + 1e-8)  # normaliseer op vorm
        templates.append(seg)
        valid_locs.append(loc)

    if len(templates) < 2:
        print("Te weinig PAC templates voor clustering.")
        return

    templates = np.array(templates)

    # Correlatiematrix
    corr_matrix = np.corrcoef(templates)

    # Eenvoudige clustering: groepen op basis van drempel
    threshold = 0.80
    groups = []
    assigned = set()

    for i in range(len(templates)):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i + 1, len(templates)):
            if j not in assigned and corr_matrix[i, j] > threshold:
                group.append(j)
                assigned.add(j)
        groups.append(group)

    print(f"=========================================")
    print(f"--> PAC morfologie clusters: {len(groups)}")
    for k, g in enumerate(groups):
        print(f"    Cluster {k+1}: {len(g)} PACs")
    if len(groups) > 1:
        print(f"    --> Meerdere foci waarschijnlijk")
    print(f"=========================================")

    # Plot gemiddeld template per cluster
    t_axis = np.linspace(-250, -80, segment_len)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10.colors

    for k, group in enumerate(groups[:5]):  # max 5 clusters tonen
        mean_template = np.mean(templates[group], axis=0)
        ax.plot(t_axis, mean_template,
                color=colors[k], linewidth=2,
                label=f"Cluster {k+1} (n={len(group)})")

    ax.set_xlabel("Tijd voor R-piek (ms)")
    ax.set_ylabel("Genormaliseerde amplitude")
    ax.set_title("PAC P-golf morfologie per cluster")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return groups, templates, corr_matrix
#%%
# Pathologische AF Detectie
# ---------------------------
def detect_af_candidates(RR, t_rr, locs, t):
    """
    Klinische detectie van Atriumfibrilleren (AF) gebaseerd op
    absolute onregelmatigheid. Signaleert prematuriteit (< 80%) 
    zonder de aanwezigheid van een compensatoire pauze te eisen.
    """
    af_locs = []
    window_size = 5

    for i in range(window_size, len(RR) - 1):
        # Bepaal de klinische baseline (mediaan van de laatste 5 slagen)
        local_median = np.median(RR[i-window_size:i])
        
        # AF Criterium: Enkel prematuriteit (< 80%), GEEN bovengrens
        if RR[i] < 0.80 * local_median:
            
            # De AF-slag is de depolarisatie die dit korte interval afsluit
            af_time = t_rr[i+1]
            
            idx = np.where(t[locs] == af_time)[0]
            if len(idx) > 0:
                af_locs.append(locs[idx[0]])

    af_locs = np.array(af_locs)
    
    print(f"=========================================")
    print(f"--> POTENTIËLE AF SLAGEN GEVONDEN: {len(af_locs)}")
    print(f"=========================================")

    return af_locs



#def detect_af_episodes(RR, t_rr, min_duration_beats=6, irregularity_threshold=0.06):
    """
    Detecteert AF-episodes als aaneengesloten reeksen van
    onregelmatige slagen — niet als losse premature beats.

    Klinisch criterium: >= 6 opeenvolgende slagen waarbij
    de lokale RR-standaarddeviatie > 60 ms (0.06 s).
    """
    episodes = []
    in_episode = False
    episode_start = None
    window = 6

    for i in range(window, len(RR)):
        local_rr = RR[i - window:i]
        sd = np.std(local_rr)

        if sd > irregularity_threshold:
            if not in_episode:
                in_episode = True
                episode_start = i - window
        else:
            if in_episode:
                episodes.append({
                    "start_time": t_rr[episode_start],
                    "end_time":   t_rr[i],
                    "duration_beats": i - episode_start,
                    "mean_rr_sd": round(sd, 4)
                })
                in_episode = False

    if in_episode:
        episodes.append({
            "start_time": t_rr[episode_start],
            "end_time":   t_rr[-1],
            "duration_beats": len(RR) - episode_start,
            "mean_rr_sd": round(np.std(RR[episode_start:]), 4)
        })

    df = pd.DataFrame(episodes)
    print(f"=========================================")
    print(f"--> AF EPISODES GEVONDEN: {len(df)}")
    if not df.empty:
        for _, row in df.iterrows():
            print(f"    {row['start_time'].strftime('%H:%M:%S')} – "
                  f"{row['end_time'].strftime('%H:%M:%S')} | "
                  f"{row['duration_beats']} slagen")
    print(f"=========================================")
    return df


def detect_af_episodes_advanced(RR, t_rr, min_duration_beats=6):
    """
    Gebruikt drie complementaire irregulariteitsmarkers:
    - CV (coefficient of variation): normaliseerde spreiding
    - SampEn proxy: complexiteit van het RR-patroon  
    - RR-ratio test: verhouding opeenvolgende intervals
    
    AF scoort hoog op alle drie tegelijk.
    PACs scoren hoog op CV maar laag op complexiteit
    (ze zijn prematuur maar wel voorspelbaar).
    """
    episodes = []
    in_episode = False
    episode_start = None
    window = 10

    for i in range(window, len(RR)):
        w = RR[i - window:i]

        cv     = np.std(w) / np.mean(w)
        rr_diffs = np.abs(np.diff(w))
        mean_diff = np.mean(rr_diffs)

        # Proxy voor onvoorspelbaarheid: hoe vaak wisselt richting?
        direction_changes = np.sum(np.diff(np.sign(np.diff(w))) != 0)
        irregularity_score = cv * mean_diff * (1 + direction_changes / window)

        is_irregular = (cv > 0.08 and mean_diff > 0.04
                        and direction_changes >= 4)

        if is_irregular:
            if not in_episode:
                in_episode = True
                episode_start = i - window
        else:
            if in_episode:
                episodes.append({
                    "start_time":     t_rr[episode_start],
                    "end_time":       t_rr[i],
                    "duration_beats": i - episode_start,
                    "mean_cv":        round(cv, 4),
                    "mean_diff_rr":   round(mean_diff, 4)
                })
                in_episode = False

    if in_episode:
        episodes.append({
            "start_time":     t_rr[episode_start],
            "end_time":       t_rr[-1],
            "duration_beats": len(RR) - episode_start,
            "mean_cv":        round(np.std(RR[episode_start:]) /
                                    np.mean(RR[episode_start:]), 4),
            "mean_diff_rr":   round(np.mean(np.abs(np.diff(
                                    RR[episode_start:]))), 4)
        })

    df = pd.DataFrame(episodes)
    print(f"=========================================")
    print(f"--> AF EPISODES (advanced): {len(df)}")
    if not df.empty:
        for _, row in df.iterrows():
            print(f"    {row['start_time'].strftime('%H:%M:%S')} – "
                  f"{row['end_time'].strftime('%H:%M:%S')} | "
                  f"CV={row['mean_cv']:.3f}")
    print(f"=========================================")
    return df
#%%
# Sinus Arrest / Sinoatrial Block Detection
# ---------------------------
def detect_sinus_arrest(RR, t_rr, locs, t, pause_threshold_sec=2.0):
    """
    Detects sinus arrest and sinoatrial (SA) exit block.

    Sinus arrest: The SA node fails to fire entirely — the pause is NOT
    a multiple of the dominant PP interval.
    SA exit block (Type II): The SA node fires but conduction to atria
    fails — the pause IS approximately a multiple (2x, 3x) of the
    dominant PP interval.

    Clinical threshold: A pause > 2.0 s is pathological in awake adults.
    Pauses > 3.0 s often warrant pacemaker evaluation.
    """
    arrest_events = []
    window_size = 10  # beats for estimating dominant PP interval

    for i in range(window_size, len(RR)):
        if RR[i] > pause_threshold_sec:
            local_rr = RR[i - window_size:i]
            dominant_rr = np.median(local_rr)

            # Check if pause is a near-integer multiple of dominant RR
            # (within 15% tolerance — SA exit block signature)
            ratio = RR[i] / dominant_rr
            nearest_integer = round(ratio)
            is_sa_exit_block = (
                nearest_integer >= 2 and
                abs(ratio - nearest_integer) / nearest_integer < 0.15
            )

            arrest_type = "SA Exit Block" if is_sa_exit_block else "Sinus Arrest"

            # Locate the R-peak at the *end* of the pause
            pause_end_time = t_rr[i]
            idx = np.where(t[locs] == pause_end_time)[0]
            peak_loc = locs[idx[0]] if len(idx) > 0 else None

            arrest_events.append({
                "time": pause_end_time,
                "duration_sec": round(RR[i], 3),
                "dominant_rr": round(dominant_rr, 3),
                "ratio_to_dominant": round(ratio, 2),
                "type": arrest_type,
                "peak_loc": peak_loc
            })

    arrest_df = pd.DataFrame(arrest_events)

    print(f"=========================================")
    if len(arrest_df) == 0:
        print(f"--> No sinus pauses > {pause_threshold_sec}s detected.")
    else:
        print(f"--> SINUS PAUSES DETECTED: {len(arrest_df)}")
        for _, row in arrest_df.iterrows():
            print(f"    [{row['type']}] {row['time'].strftime('%H:%M:%S')} | "
                  f"Pause: {row['duration_sec']}s | "
                  f"Ratio: {row['ratio_to_dominant']}x dominant RR")
    print(f"=========================================")

    return arrest_df

#%% Sinus arrest herkennen
def plot_sinus_arrest(ecg, t, locs, arrest_df, fs, event_index=0, window_sec=15):
    """
    Plots an ECG strip centred on a detected sinus pause,
    with the pause duration annotated as a span.
    """
    if arrest_df.empty:
        print("No sinus arrest events to plot.")
        return

    event = arrest_df.iloc[event_index]
    if event["peak_loc"] is None:
        print("Could not locate R-peak for this event.")
        return

    center_loc = event["peak_loc"]
    start = max(0, int(center_loc - (window_sec / 2) * fs))
    end = min(len(ecg), int(center_loc + (window_sec / 2) * fs))

    mask_qrs = (locs >= start) & (locs < end)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t[start:end], ecg[start:end], linewidth=0.8, color='tab:blue', label="Filtered ECG")
    ax.plot(t[locs[mask_qrs]], ecg[locs[mask_qrs]], 'k*', markersize=5, label="QRS")

    # Annotate pause span
    pause_end = event["time"]
    pause_start = pause_end - pd.Timedelta(seconds=event["duration_sec"])
    y_top = ax.get_ylim()[1] * 0.85
    ax.annotate(
        "",
        xy=(pause_end, y_top),
        xytext=(pause_start, y_top),
        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5)
    )
    ax.text(
        pause_start + (pause_end - pause_start) / 2,
        y_top * 1.05,
        f"{event['type']}: {event['duration_sec']}s",
        ha='center', fontsize=9, color='crimson'
    )

    ax.set_title(f"Sinus Pause – {event['type']} (Event #{event_index})")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_xlabel("Time")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    plt.show()
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

def plot_irregularity_comparison(RR, t_rr, pac_locs, locs, t, title):
    """
    Toont in één figuur:
    - Boven: RR-interval tijdreeks met PAC-markers
    - Midden: rolling CV (AF-gevoelig)
    - Onder: rolling RMSSD (ook PAC-gevoelig, maar ander patroon)
    
    PACs: korte piek in CV + RMSSD, snel herstel
    AF:   aanhoudend verhoogde CV + RMSSD zonder herstel
    """
    window = 10
    cv_vals    = []
    rmssd_vals = []
    t_vals     = []

    for i in range(window, len(RR)):
        w = RR[i - window:i]
        cv_vals.append(np.std(w) / np.mean(w))
        rmssd_vals.append(np.sqrt(np.mean(np.diff(w)**2)) * 1000)
        t_vals.append(t_rr[i])

    t_vals     = np.array(t_vals)
    cv_vals    = np.array(cv_vals)
    rmssd_vals = np.array(rmssd_vals)

    pac_set = set(pac_locs)
    pac_times = [t[l] for l in pac_locs if l in set(locs)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # RR tijdreeks
    axes[0].plot(t_rr, RR * 1000, '.', markersize=2,
                 color='tab:blue', alpha=0.6)
    for pt in pac_times:
        axes[0].axvline(pt, color='red', alpha=0.3, linewidth=0.8)
    axes[0].set_ylabel("RR interval (ms)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # CV
    axes[1].plot(t_vals, cv_vals, color='tab:orange', linewidth=0.8)
    axes[1].axhline(0.08, color='crimson', linestyle='--',
                    linewidth=1, label="AF drempel (CV=0.08)")
    axes[1].set_ylabel("Coefficient of Variation")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # RMSSD
    axes[2].plot(t_vals, rmssd_vals, color='tab:green', linewidth=0.8)
    axes[2].axhline(50, color='crimson', linestyle='--',
                    linewidth=1, label="Grens 50ms")
    axes[2].set_ylabel("RMSSD (ms)")
    axes[2].set_xlabel("Tijd")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.show()
#%%
# Diagnostics: Poincaré Plot (Lorenz Plot)
# ---------------------------
def plot_poincare(RR, title="Poincaré Plot"):
    """
    Zet het huidige RR-interval (x-as) uit tegen het volgende RR-interval (y-as).
    Dit is de gouden standaard om AF van PACs te onderscheiden.
    """
    # rr_n is de huidige slag, rr_n1 is de volgende slag
    rr_n = RR[:-1]
    rr_n1 = RR[1:]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(rr_n, rr_n1, s=3, color='tab:red', alpha=0.3)
    
    # Teken de identiteitslijn (x = y). Een perfect regelmatig ritme ligt exact op deze lijn.
    min_val, max_val = min(RR), max(RR)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Perfect regelmatig (x=y)")
    
    plt.title(f"{title}\n(RR[i] vs RR[i+1])")
    plt.xlabel("Huidig RR Interval (s)")
    plt.ylabel("Volgend RR Interval (s)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%
def classify_pac_runs(pac_indices):
    """
    Classify isolated PACs, couplets, and runs.
    pac_indices are indices in the RR sequence.
    """
    if len(pac_indices) == 0:
        return [], [], []

    isolated = []
    couplets = []
    runs = []

    current_group = [pac_indices[0]]

    for i in range(1, len(pac_indices)):
        if pac_indices[i] == pac_indices[i-1] + 1:
            current_group.append(pac_indices[i])
        else:
            if len(current_group) == 1:
                isolated.append(current_group)
            elif len(current_group) == 2:
                couplets.append(current_group)
            else:
                runs.append(current_group)
            current_group = [pac_indices[i]]

    # laatste groep opslaan
    if len(current_group) == 1:
        isolated.append(current_group)
    elif len(current_group) == 2:
        couplets.append(current_group)
    else:
        runs.append(current_group)

    return isolated, couplets, runs

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

# 1. PAC Detectie
pac_locs1 = detect_pac(RR1, t_rr1, locs1, t1)
plot_ecg_with_pac(filtered1, t1, locs1, pac_locs1, fs1, pac_index=0)

isolated_pacs1, pac_couplets1, pac_runs1 = classify_pac_runs(pac_locs1)
print(f"Isolated PACs: {len(isolated_pacs1)}, Couplets: {len(pac_couplets1)}, Runs: {len(pac_runs1)}")

# PAC vs PVC classificatie
pac_locs1, pvc_locs1, ambiguous1 = classify_ectopic_beats_full(
    RR1, t_rr1, locs1, t1, filtered1, fs1
)
# PVC patroon classificatie
pvc_patterns1 = classify_pvc_patterns(locs1, pvc_locs1)

# Plot eerste gevonden patroon
if not pvc_patterns1.empty:
    plot_pvc_pattern(filtered1, t1, locs1, pvc_locs1,
                     pvc_patterns1, fs1, pattern_index=0)

# Histogram om grens te valideren
plot_qrs_width_histogram(filtered1, locs1, fs1,
                          "QRS Duration – Recording 1 (PACs + PVCs)")

# ECG-strip met beide types
plot_ecg_pac_vs_pvc(filtered1, t1, locs1,
                     pac_locs1, pvc_locs1, fs1, event_index=0)

# 2. AF Detectie (NIEUW)
af_locs1 = detect_af_candidates(RR1, t_rr1, locs1, t1)
# Plot de eerste AF slag (index 0) om het verschil met een PAC te zien
if len(af_locs1) > 0:
    plot_ecg_with_pac(filtered1, t1, locs1, af_locs1, fs1, pac_index=0)

#3. Detects Sinusarrest
arrests1 = detect_sinus_arrest(RR1, t_rr1, locs1, t1, pause_threshold_sec=2.0)
if not arrests1.empty:
    plot_sinus_arrest(filtered1, t1, locs1, arrests1, fs1, event_index=0)

# HRV parameters (vraag 3)
hrv1 = compute_hrv_parameters(RR1, t_rr1)
plot_hrv_parameters(hrv1, "HRV Parameters – Recording 1")

# RR zonder ectopie (vraag 5)
RR1_clean, t_rr1_clean = compute_rr_without_pacs(
    locs1, pac_locs1, pvc_locs1, t1)
#af_episodes1_clean = detect_af_episodes(RR1_clean, t_rr1_clean)

# P-golf morfologie vergelijking (vraag 8)
compare_p_wave_morphology(filtered1, locs1, pac_locs1, fs1)

# Advanced AF detectie
af_episodes1_adv = detect_af_episodes_advanced(RR1_clean, t_rr1_clean)

# PAC morfologie clustering (vraag 8 — meerdere foci)
cluster_pac_foci(filtered1, locs1, pac_locs1, fs1, t1)

# Vergelijkingsplot AF vs PAC irregulariteit (vraag 4)
plot_irregularity_comparison(RR1, t_rr1, pac_locs1, locs1, t1,
    "Irregulariteit: AF vs PAC – Recording 1")

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

# 1. PAC Detectie
pac_locs2 = detect_pac(RR2, t_rr2, locs2, t2)
plot_ecg_with_pac(filtered2, t2, locs2, pac_locs2, fs2, pac_index=4)

isolated_pacs2, pac_couplets2, pac_runs2 = classify_pac_runs(pac_locs2)
print(f"Isolated PACs: {len(isolated_pacs2)}, Couplets: {len(pac_couplets2)}, Runs: {len(pac_runs2)}")

# PAC vs PVC classificatie
pac_locs2, pvc_locs2, ambiguous2 = classify_ectopic_beats_full(
    RR2, t_rr2, locs2, t2, filtered2, fs2
)

# PVC patroon classificatie
pvc_patterns2 = classify_pvc_patterns(locs2, pvc_locs2)

# Plot eerste gevonden patroon
if not pvc_patterns2.empty:
    plot_pvc_pattern(filtered2, t2, locs2, pvc_locs2,
                     pvc_patterns2, fs2, pattern_index=0)
    
# Histogram om grens te valideren
plot_qrs_width_histogram(filtered2, locs2, fs2,
                          "QRS Duration – Recording 2 (PACs + PVCs)")

# ECG-strip met beide types
plot_ecg_pac_vs_pvc(filtered2, t2, locs2,
                     pac_locs2, pvc_locs2, fs2, event_index=0)

# 2. AF Detectie (NIEUW)

af_locs2 = detect_af_candidates(RR2, t_rr2, locs2, t2)
# Plot de eerste AF slag (index 0) om het verschil met een PAC te zien
if len(af_locs2) > 0:
    plot_ecg_with_pac(filtered2, t2, locs2, af_locs2, fs2, pac_index=0)

#3. Detects Sinusarrest
arrests2 = detect_sinus_arrest(RR2, t_rr2, locs2, t2, pause_threshold_sec=2.0)
if not arrests2.empty:
    plot_sinus_arrest(filtered2, t2, locs2, arrests2, fs2, event_index=0)

# HRV parameters (vraag 3)
hrv2 = compute_hrv_parameters(RR2, t_rr2)
plot_hrv_parameters(hrv2, "HRV Parameters – Recording 2")

# RR zonder ectopie (vraag 5)
RR2_clean, t_rr2_clean = compute_rr_without_pacs(
    locs2, pac_locs2, pvc_locs2, t2)
#af_episodes2_clean = detect_af_episodes(RR2_clean, t_rr2_clean)

# P-golf morfologie vergelijking (vraag 8)
compare_p_wave_morphology(filtered2, locs2, pac_locs2, fs2)

# Advanced AF detectie
af_episodes1_adv = detect_af_episodes_advanced(RR2_clean, t_rr2_clean)

# PAC morfologie clustering (vraag 8 — meerdere foci)
cluster_pac_foci(filtered2, locs2, pac_locs2, fs2, t2)

# Vergelijkingsplot AF vs PAC irregulariteit (vraag 4)
plot_irregularity_comparison(RR2, t_rr2, pac_locs2, locs2, t2,
    "Irregulariteit: AF vs PAC – Recording 2")
# %%
plot_poincare(RR1, "Poincaré Plot – Recording 1 (PACs + PVCs)")
# %%