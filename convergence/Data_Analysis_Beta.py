import os
import re
import glob
import numpy as np
from numpy import pi, sin, cos
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# ---------------------------------------------------------
# Analytical Model (As requested)
# ---------------------------------------------------------
def model_noise(θ, α, β, Pt):
    # 1. Divide alpha by 2
    a = α / 2.0
    
    # 2. Probability assignment without dynamic re-scaling
    if Pt == 1.0:
        P_HH = P_HV = P_VH = P_VV = 0.0
    else:
        P_HH = 0.0030
        P_HV = 0.0138
        P_VH = 0.0278  # Using the corrected value
        P_VV = 0.0447
    
    # Pre-compute trigonometric values
    C_a = cos(a)
    S_a = sin(a)
    C_ab = cos(a + β)
    S_ab = sin(a + β)
    C_theta = cos(θ)
    
    # Term 1: Pt * (1/2) * [...]
    term1 = (Pt / 2.0) * (
        C_a**4 + S_a**4 + 2 * (S_a**2) * (C_a**2) * C_theta +
        C_ab**4 + S_ab**4 + 2 * (S_ab**2) * (C_ab**2) * C_theta
    )
    
    # Term 2: ((P_HH + P_VV)/2) * [...]
    term2 = ((P_HH + P_VV) / 2.0) * (
        C_a**4 + S_a**4 + C_ab**4 + S_ab**4
    )
    
    # Term 3: (P_HV + P_VH) * [...]
    term3 = (P_HV + P_VH) * (
        (C_a**2) * (S_a**2) + (C_ab**2) * (S_ab**2)
    )
    
    return term1 + term2 + term3

# ---------------------------------------------------------
# Data Processing Pipeline
# ---------------------------------------------------------
def process_simulation_files(directory: str = "."):
    """
    Scans the specified directory for simulation CSV files, parses the metadata
    from the filename, loads the data, and computes the R2 score.
    """
    # Regex to capture: alpha, Pt, rows, cols, states
    pattern = r"alpha_([\d\.]+)pi_Pt_([\d\.]+)_(\d+)x(\d+)_mesh_([\d\.E\+\-]+)_states"
    
    # We look for all csv files matching the new alpha pattern
    search_path = os.path.join(directory, "*alpha_*.csv")
    file_list = glob.glob(search_path)
    
    if not file_list:
        print(f"No files matching '*alpha_*.csv' found in {os.path.abspath(directory)}")
        return pd.DataFrame()

    results = []

    print(f"Found {len(file_list)} files to analyze. Processing...")
    

    for filepath in file_list:
        filename = os.path.basename(filepath)
        match = re.search(pattern, filename)
        
        if not match:
            print(f"Skipping {filename}: Could not parse parameters from name.")
            continue
            
        # Extract parameters from filename
        alpha_coeff = float(match.group(1)) # e.g., 0.25
        alpha_val = alpha_coeff * pi
        pt_val = float(match.group(2))
        states_val = float(match.group(5))
        
        # Load the experimental matrix data
        try:
            # Using pandas to gracefully handle potential trailing commas in the CSV
            df_exp = pd.read_csv(filepath, header=None)
            df_exp.dropna(axis=1, how='all', inplace=True) # Drop trailing empty columns
            experimental = df_exp.values
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        keys = ["N_states","α","β"]
        n_rows, n_cols = experimental.shape
        df_data_exp = pd.DataFrame([[states_val for k in range(n_rows)],
                                    [alpha_coeff for k in range(n_rows)],
                                    np.linspace(0,pi,n_cols),],
                            columns=keys)
        
        # add the dataframe to the bunh of data

        # Generate the theoretical model mesh dynamically based on extracted dimensions
        theta_vals = np.linspace(0, 2*pi, n_rows) # Range 0 to 2*pi
        beta_vals = np.linspace(0, pi, n_cols)    # Range 0 to pi
        
        theoretical = np.array([
            [model_noise(θ=t, α=alpha_val, β=b, Pt=pt_val) for b in beta_vals] 
            for t in theta_vals
        ])
        
