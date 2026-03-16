# main_analysis.py
# ECG Holter Analysis – Clean Execution Pipeline
# ==========================================================

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all functions from your existing module
from v1503 import (
    read_and_clean_ecg_mat,
    apply_clinical_bandpass,
    detect_r_peaks,
    compute_rr_intervals,
    detect_pac,
    classify_ectopic_beats_full,
    classify_pvc_patterns,
    detect_sinus_arrest,
    detect_af_episodes_advanced,
    compute_rr_without_pacs,
    compute_hrv_parameters,
    cluster_pac_foci,
    # Plotting functions
    plot_heart_rate,
    plot_rr_variability,
    plot_hrv_parameters,
    plot_ecg_with_pac,
    plot_ecg_pac_vs_pvc,
    plot_pvc_pattern,
    plot_qrs_width_histogram,
    plot_sinus_arrest,
    plot_poincare,
    plot_irregularity_comparison,
    compare_p_wave_morphology,
    analyze_qrs_regularity
)

#%%
# ==========================================================
# CONFIGURATION
# ==========================================================

PATHS = {
    "recording_1": r"/path/to/004_Groenewoud_PACs+PVCs.mat",
    "recording_2": r"/path/to/004_Groenewoud_PACs.mat"
}

# Clinical thresholds (centralized for easy adjustment)
CONFIG = {
    "sinus_pause_threshold_sec": 2.0,
    "pvc_qrs_threshold_ms": 120,
    "af_cv_threshold": 0.08,
    "pac_prematurity_ratio": 0.80,
    "compensatory_pause_ratio": 1.20
}

#%%
# ==========================================================
# ANALYSIS PIPELINE FUNCTION
# ==========================================================

def run_full_analysis(path, recording_name, config=CONFIG, plot=True):
    """
    Complete analysis pipeline for a single recording.
    
    Returns a dictionary with all computed results.
    """
    results = {"name": recording_name}
    
    # ── 1. Load and Preprocess ────────────────────────────
    print(f"\n{'='*60}")
    print(f"ANALYZING: {recording_name}")
    print(f"{'='*60}")
    
    ecg, fs, t = read_and_clean_ecg_mat(path)
    filtered = apply_clinical_bandpass(ecg, fs)
    locs, _ = detect_r_peaks(filtered, fs)
    RR, t_rr = compute_rr_intervals(locs, t)
    
    results["fs"] = fs
    results["n_beats"] = len(locs)
    results["duration_hours"] = (t[-1] - t[0]).total_seconds() / 3600
    
    print(f"Duration: {results['duration_hours']:.2f} hours")
    print(f"Total beats: {results['n_beats']}")
    
    # ── 2. Ectopy Detection ───────────────────────────────
    print(f"\n--- Ectopy Analysis ---")
    
    pac_locs, pvc_locs, ambiguous_locs = classify_ectopic_beats_full(
        RR, t_rr, locs, t, filtered, fs
    )
    
    results["pac_count"] = len(pac_locs)
    results["pvc_count"] = len(pvc_locs)
    results["ambiguous_count"] = len(ambiguous_locs)
    results["pac_burden_pct"] = 100 * len(pac_locs) / len(locs)
    results["pvc_burden_pct"] = 100 * len(pvc_locs) / len(locs)
    
    # ── 3. PVC Pattern Analysis ───────────────────────────
    pvc_patterns = classify_pvc_patterns(locs, pvc_locs)
    results["bigeminy_episodes"] = len(pvc_patterns[pvc_patterns["type"] == "Bigeminie"])
    results["trigeminy_episodes"] = len(pvc_patterns[pvc_patterns["type"] == "Trigeminie"])
    
    # ── 4. Sinus Pause Detection ──────────────────────────
    print(f"\n--- Sinus Pause Analysis ---")
    arrests = detect_sinus_arrest(
        RR, t_rr, locs, t, 
        pause_threshold_sec=config["sinus_pause_threshold_sec"]
    )
    results["sinus_pauses"] = len(arrests)
    
    # ── 5. AF Episode Detection ───────────────────────────
    print(f"\n--- AF Episode Analysis ---")
    RR_clean, t_rr_clean = compute_rr_without_pacs(locs, pac_locs, pvc_locs, t)
    af_episodes = detect_af_episodes_advanced(RR_clean, t_rr_clean)
    results["af_episodes"] = len(af_episodes)
    
    # ── 6. HRV Analysis ───────────────────────────────────
    hrv_stats = compute_hrv_parameters(RR, t_rr)
    results["mean_hr"] = 60 / hrv_stats["mean_rr"].mean()
    results["mean_sdnn_ms"] = hrv_stats["sdnn"].mean() * 1000
    results["mean_rmssd_ms"] = hrv_stats["rmssd"].mean() * 1000
    
    # ── 7. PAC Focus Clustering ───────────────────────────
    print(f"\n--- PAC Morphology Clustering ---")
    cluster_labels, templates, valid_locs = cluster_pac_foci(
        filtered, locs, pac_locs, fs, t
    )
    if cluster_labels is not None:
        results["pac_foci_count"] = len(np.unique(cluster_labels))
    else:
        results["pac_foci_count"] = 0
    
    # ── 8. Visualization (optional) ───────────────────────
    if plot:
        print(f"\n--- Generating Plots ---")
        
        plot_heart_rate(t_rr, RR, f"Heart Rate – {recording_name}")
        
        rr_stats = analyze_qrs_regularity(t_rr, RR)
        plot_rr_variability(rr_stats, f"RR Variability – {recording_name}")
        
        plot_hrv_parameters(hrv_stats, f"HRV Parameters – {recording_name}")
        
        plot_qrs_width_histogram(filtered, locs, fs, 
            f"QRS Duration – {recording_name}")
        
        if len(pac_locs) > 0 or len(pvc_locs) > 0:
            plot_ecg_pac_vs_pvc(filtered, t, locs, pac_locs, pvc_locs, 
                fs, event_index=0)
        
        if not pvc_patterns.empty:
            plot_pvc_pattern(filtered, t, locs, pvc_locs, pvc_patterns, 
                fs, pattern_index=0)
        
        if not arrests.empty:
            plot_sinus_arrest(filtered, t, locs, arrests, fs, event_index=0)
        
        plot_poincare(RR, f"Poincaré Plot – {recording_name}")
        
        plot_irregularity_comparison(RR, t_rr, pac_locs, locs, t,
            f"Irregularity Analysis – {recording_name}")
        
        compare_p_wave_morphology(filtered, locs, pac_locs, fs)
    
    # ── Store raw data for further analysis ───────────────
    results["_raw"] = {
        "ecg": filtered,
        "t": t,
        "locs": locs,
        "RR": RR,
        "t_rr": t_rr,
        "pac_locs": pac_locs,
        "pvc_locs": pvc_locs,
        "arrests": arrests,
        "af_episodes": af_episodes,
        "hrv_stats": hrv_stats
    }
    
    return results

#%%
# ==========================================================
# SUMMARY REPORT
# ==========================================================

def print_summary(results):
    """Print a clinical summary of the analysis."""
    print(f"\n{'='*60}")
    print(f"CLINICAL SUMMARY: {results['name']}")
    print(f"{'='*60}")
    print(f"Recording duration:     {results['duration_hours']:.2f} hours")
    print(f"Total beats:            {results['n_beats']}")
    print(f"Mean heart rate:        {results['mean_hr']:.1f} bpm")
    print()
    print(f"PAC count:              {results['pac_count']} ({results['pac_burden_pct']:.2f}%)")
    print(f"PVC count:              {results['pvc_count']} ({results['pvc_burden_pct']:.2f}%)")
    print(f"Ambiguous beats:        {results['ambiguous_count']}")
    print()
    print(f"Bigeminy episodes:      {results['bigeminy_episodes']}")
    print(f"Trigeminy episodes:     {results['trigeminy_episodes']}")
    print(f"Sinus pauses (>2s):     {results['sinus_pauses']}")
    print(f"AF episodes:            {results['af_episodes']}")
    print()
    print(f"PAC foci detected:      {results['pac_foci_count']}")
    print(f"Mean SDNN:              {results['mean_sdnn_ms']:.1f} ms")
    print(f"Mean RMSSD:             {results['mean_rmssd_ms']:.1f} ms")
    print(f"{'='*60}")

#%%
# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    
    # Analyze Recording 1 (PACs + PVCs)
    results_1 = run_full_analysis(
        PATHS["recording_1"], 
        "Recording 1 (PACs + PVCs)",
        plot=True
    )
    print_summary(results_1)
    
    # Analyze Recording 2 (PACs only)
    results_2 = run_full_analysis(
        PATHS["recording_2"],
        "Recording 2 (PACs only)", 
        plot=True
    )
    print_summary(results_2)
    
    # ── Compare recordings ────────────────────────────────
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Metric':<25} {'Recording 1':>15} {'Recording 2':>15}")
    print("-"*60)
    for key in ["pac_count", "pvc_count", "sinus_pauses", "af_episodes", "pac_foci_count"]:
        print(f"{key:<25} {results_1[key]:>15} {results_2[key]:>15}")

#%%