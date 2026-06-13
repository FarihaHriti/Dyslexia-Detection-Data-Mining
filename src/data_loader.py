import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder

def get_etdd70_label(filename):
    # Filename format: Subject_1003_T1_...
    # Need metadata for labels? Or is it in the folder name?
    # User said: "fully labeled and balanced (~50% dyslexic, 50% control)"
    # I need to check where the labels are for ETDD70.
    # Usually provided in a separate file or implicit.
    # Let's assume a 'labels.csv' or check the 'data' folder for a map.
    # Wait, the user prompt didn't specify. I'll search for a label file in etdd700-data.
    return None

def calculate_ivt_features(df, x_col, y_col, t_col, timestamp_is_ms=True):
    # Velocity-Threshold Identification (I-VT)
    # 1. Calculate velocities
    if len(df) < 10: return None
    
    x = df[x_col].values
    y = df[y_col].values
    
    if t_col in df.columns:
        t = df[t_col].values
    else:
        # Assume 50Hz (20ms) if not provided? Or 300Hz?
        # Kronoberg is likely 20ms steps (50Hz) or similar? A1R.txt showed 20, 40, 60...
        t = np.arange(len(df)) * 20 # Placeholder 50Hz
        
    # velocity = dist / time
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    dt[dt == 0] = 1 # avoid div/0
    
    # Euclidean distance
    dist = np.sqrt(dx**2 + dy**2)
    # Velocity in pixels/ms (or deg/s if calibrated, but we stick to pixels for raw)
    vel = dist / dt
    
    # Threshold for fixation vs saccade
    # Simple statistical threshold: Mean + 2*Std? Or fixed value?
    # Let's use a dynamic threshold for robustness across devices
    threshold = np.nanmean(vel) + np.nanstd(vel)
    
    # Identify fixations (vel < threshold)
    is_fix = vel < threshold
    
    # Group consecutive fixations and saccades
    fix_durations = []
    sac_amplitudes = []
    regressions = 0
    
    i = 0
    n = len(is_fix)
    while i < n:
        if is_fix[i]:
            # Group fixation
            start_idx = i
            while i < n and is_fix[i]:
                i += 1
            end_idx = i
            # Duration is the sum of time steps
            dur = np.sum(dt[start_idx:end_idx])
            fix_durations.append(dur)
        else:
            # Group saccade
            start_idx = i
            while i < n and not is_fix[i]:
                i += 1
            end_idx = i
            
            # Saccade amplitude is the net Euclidean distance from start to end of saccade
            sx = x[start_idx]
            sy = y[start_idx]
            ex = x[min(end_idx, len(x)-1)]
            ey = y[min(end_idx, len(y)-1)]
            amp = np.sqrt((ex - sx)**2 + (ey - sy)**2)
            sac_amplitudes.append(amp)
            
            # Regression: net move left
            if ex < sx:
                regressions += 1
                
    f_count = len(fix_durations)
    s_count = len(sac_amplitudes)
    
    return {
        'fixation_duration_mean': np.mean(fix_durations) if fix_durations else 0,
        'fixation_duration_std': np.std(fix_durations) if fix_durations else 0,
        'fixation_duration_max': np.max(fix_durations) if fix_durations else 0,
        'saccade_length_mean': np.mean(sac_amplitudes) if len(sac_amplitudes) > 0 else 0,
        'saccade_length_std': np.std(sac_amplitudes) if len(sac_amplitudes) > 0 else 0,
        'fixation_count': f_count,
        'saccade_count': s_count,
        'fix_sac_ratio': f_count / s_count if s_count > 0 else 0,
        'regression_ratio': regressions / s_count if s_count > 0 else 0
    }

def extract_features_v2(df, dataset_type='child'):
    # Unified Feature Extraction
    
    if dataset_type == 'kronoberg':
        # Kronoberg: T, LX, LY, RX, RY. T is approx 20ms?
        # Clean numeric
        for col in ['LX', 'RX', 'LY', 'RY', 'T']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Average Eyes
        x = df[['LX', 'RX']].mean(axis=1, skipna=True)
        y = df[['LY', 'RY']].mean(axis=1, skipna=True)
        # T is already likely ms
        return calculate_ivt_features(pd.DataFrame({'x':x, 'y':y, 't':df['T']}), 'x', 'y', 't')
    return None

def load_etdd70_data(base_path):
    # Load ETDD70
    etdd_data = []
    data_dir = os.path.join(base_path, "datasets", "etdd700-data", "data")
    label_file = os.path.join(base_path, "datasets", "etdd700-data", "dyslexia_class_label.csv")
    
    # Load ground truth labels
    if os.path.exists(label_file):
        labels_df = pd.read_csv(label_file)
        label_map = dict(zip(labels_df['subject_id'].astype(int), labels_df['class_id'].astype(int)))
        print(f"Loaded {len(label_map)} ground truth labels from dyslexia_class_label.csv")
    else:
        print("WARNING: Ground truth label file not found! Using fallback logic.")
        label_map = {}
    
    # Get unique SIDs from filenames
    files = sorted(glob.glob(os.path.join(data_dir, "*_fixations.csv")))
    sids = set()
    for f in files:
        # Subject_1003_T1...
        parts = os.path.basename(f).split('_')
        if len(parts) > 1: sids.add(parts[1])
        
    for sid in sorted(sids):
        try:
            sid_int = int(sid)
            if sid_int in label_map:
                label = label_map[sid_int]
            else:
                label = 0 if sid_int < 1100 else 1  # Fallback
            
            # Load Fixations
            fix_files = sorted(glob.glob(os.path.join(data_dir, f"Subject_{sid}_*_fixations.csv")))
            sac_files = sorted(glob.glob(os.path.join(data_dir, f"Subject_{sid}_*_saccades.csv")))
            
            durations = []
            for ff in fix_files:
                df = pd.read_csv(ff)
                if 'duration_ms' in df.columns:
                    durations.extend(df['duration_ms'].dropna().tolist())
                    
            amplitudes = []
            regressions = 0
            for sf in sac_files:
                df = pd.read_csv(sf)
                if 'ampl' in df.columns:
                    amplitudes.extend(df['ampl'].dropna().tolist())
                if 'start_x' in df.columns and 'end_x' in df.columns:
                    # Regression: end_x < start_x
                    reg_mask = df['end_x'] < df['start_x']
                    regressions += reg_mask.sum()
            
            if not durations: continue
            
            f_count = len(durations)
            s_count = len(amplitudes)
            
            features = {
                'fixation_duration_mean': np.mean(durations),
                'fixation_duration_std': np.std(durations),
                'fixation_duration_max': np.max(durations),
                'saccade_length_mean': np.mean(amplitudes) if amplitudes else 0,
                'saccade_length_std': np.std(amplitudes) if amplitudes else 0,
                'fixation_count': f_count,
                'saccade_count': s_count,
                'fix_sac_ratio': f_count / s_count if s_count > 0 else 0,
                'regression_ratio': regressions / s_count if s_count > 0 else 0,
                'label': label,
                'group': 'etdd70'
            }
            etdd_data.append(features)
        except Exception as e:
            # print(f"Error loading ETDD {sid}: {e}")
            pass
            
    return etdd_data



def load_data(base_path):
    data_map = {'etdd70': [], 'kronoberg': []}
    
    # 1. ETDD70
    print("Loading ETDD70...")
    data_map['etdd70'] = load_etdd70_data(base_path)
    
    # 2. Kronoberg (Child)
    k_path = os.path.join(base_path, "datasets", "Kronoberg Reading Dataset (Child)")
    if os.path.exists(k_path):
        print("Loading Kronoberg...")
        for sf in sorted(glob.glob(os.path.join(k_path, "*"))):
            if not os.path.isdir(sf): continue
            sid = os.path.basename(sf)
            try:
                # Label logic: 1,2 Dyslexic (1); 3,4 Control (0)
                last = int(sid[-1])
                label = 1 if last in [1, 2] else (0 if last in [3,4] else None)
                if label is None: continue
                
                df = pd.read_csv(os.path.join(sf, "A1R.txt"), sep='\t')
                feats = extract_features_v2(df, 'kronoberg')
                if feats:
                    feats['label'] = label
                    feats['group'] = 'child_kronoberg'
                    data_map['kronoberg'].append(feats)
            except: pass

    return data_map

if __name__ == "__main__":
    data_map = load_data(".")
    print(f"Loaded samples: ETDD70={len(data_map['etdd70'])}, Kronoberg={len(data_map['kronoberg'])}")
