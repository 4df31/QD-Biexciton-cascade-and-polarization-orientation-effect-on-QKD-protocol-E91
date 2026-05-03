#   !/usr/bin/env python
# coding: utf-8

import os
import re
import glob
import numpy as np
from numpy import pi, sin, cos
import pandas as pd
from sklearn.metrics import r2_score
import plotly.express as px

# ---------------------------------------------------------
# Analytical Model (As requested)
# ---------------------------------------------------------
def analytical(theta: float, beta: float, alpha: float):
    return (
        cos(alpha)**4 + sin(alpha)**4 +
        2 * (sin(alpha)**2) * (cos(alpha)**2) * cos(theta) +
        cos(alpha+beta)**4 + sin(alpha+beta)**4 +
        2 * (sin(alpha+beta)**2) * (cos(alpha+beta)**2) * cos(theta)
    ) / 2

def model_noise(θ, α, β, Pt):
    return analytical(θ, β, α/2.0) * Pt + ((1 - Pt) / 2.0)

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
    
    # We look for all csv files starting with 'gpu_alpha'
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
            
        n_rows, n_cols = experimental.shape
        
        # Generate the theoretical model mesh dynamically based on extracted dimensions
        theta_vals = np.linspace(0, 2*pi, n_rows) # Range 0 to 2*pi
        beta_vals = np.linspace(0, pi, n_cols)    # Range 0 to pi
        
        theoretical = np.array([
            [model_noise(θ=t, α=alpha_val, β=b, Pt=pt_val) for b in beta_vals] 
            for t in theta_vals
        ])
        
        # Compute R2 score
        # We flatten both arrays to evaluate the 2D surface fit globally as a single scalar
        r2_val = r2_score(y_true=theoretical.flatten(), y_pred=experimental.flatten())
        
        # Append to our tracking list
        results.append({
            "Filename": filename,
            "Alpha_Coeff": alpha_coeff,
            "Alpha_Label": f"{alpha_coeff:.2f}π",
            "Pt": pt_val,
            "States": states_val,
            "Rows": n_rows,
            "Cols": n_cols,
            "R2_Score": r2_val
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def generate_scatter_plot(df: pd.DataFrame):
    if df.empty:
        print("No data available to plot.")
        return
        
    # Sort values so the legend displays cleanly
    df = df.sort_values(by=["Alpha_Coeff", "States"])
    
    # Create an interactive scatter plot using Plotly
    fig = px.scatter(
        df, 
        x="States", 
        y="R2_Score", 
        color="Alpha_Label",
        symbol="Alpha_Label",
        hover_data=["Rows", "Cols", "Pt"],
        #log_y=True,  # Logarithmic scale makes states like 1e3 to 4.5e4 much easier to read
        #log_x=False,
        title="Analytical Model Fidelity (R² Score) vs Number of States",
        labels={
            "States": "Number of Quantum States", 
            "R2_Score": "R² Correlation Score(Log Scale)",
            "Alpha_Label": "Alpha Parameter"
        },
        template="plotly_white"
    )
    
    # Enhance the visual styling of the markers
    fig.update_traces(
        marker=dict(size=12, opacity=0.6, line=dict(width=1.5, color='DarkSlateGrey'))
    )
    
    # Customize layout constraints for a premium scientific look
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=14),
        legend_title_text="Alpha Shift (α)",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=False),
        hovermode="closest"
    )
    
    # Save to interactive HTML and open in browser
    output_html = "R2_Analysis_Results.html"
    fig.write_html(output_html)
    print(f"\nPlot successfully generated and saved to: {output_html}")
    
    # Optional: If running in a Jupyter Notebook, fig.show() will render it inline
    # fig.show()


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Point this to the directory containing your output CSV files. 
    # Default is the current directory '.'
    # If your files are in the "Chen_based_sims" folder, change this to "./Chen_based_sims"
    data_directory = "./20" 
    
    print("Starting data analysis...")
    results_df = process_simulation_files(directory=data_directory)
    
    if not results_df.empty:
        print("\nAnalysis Summary:")
        print(results_df[["Alpha_Label", "States", "R2_Score"]].head(15).to_string(index=False))
        print("...")
        
        # Output results to a combined CSV for safekeeping
        results_df.to_csv("Aggregated_R2_Scores.csv", index=False)
        print("\nNumerical data saved to: Aggregated_R2_Scores.csv")
        
        # Generate visual Plotly map
        generate_scatter_plot(results_df)
    else:
        print("Data extraction failed. Please check the directory path and file names.")
