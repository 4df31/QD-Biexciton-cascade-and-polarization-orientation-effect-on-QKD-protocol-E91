#!/usr/bin/env python3
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go

def parse_filename(filename):
    """
    Extract physical parameters from the standard simulation filename structure.
    Example: CHSH_QS_alpha_0.25π_Pt_0.9107_100x100_mesh_5.0E+04_states_SIM_23_06_2026_17_18_18.csv
    """
    # Regex to capture alpha, Pt, and states
    match = re.search(r'alpha_(.*?)_Pt_(.*?)_\d+x\d+_mesh_(.*?)_states', filename)
    if match:
        alpha = match.group(1)
        pt = match.group(2)
        states = match.group(3)
        return {
            'alpha': alpha,
            'Pt': pt,
            'states': states
        }
    return None

def main():
    # Find all CSV files in the current directory
    csv_files = glob.glob("*.csv")
    large_files = []
    
    # Identify large files (100x100 mesh)
    for f in csv_files:
        try:
            with open(f, 'r') as file:
                lines = file.readlines()
                # Remove empty lines
                lines = [l for l in lines if l.strip()]
                if len(lines) >= 100:
                    first_line_cols = len(lines[0].split(','))
                    if first_line_cols >= 100:
                        large_files.append((f, len(lines), first_line_cols))
        except Exception as e:
            print(f"Error checking file {f}: {e}")
            
    if not large_files:
        print("No large (>= 100x100) CSV files found in the current directory.")
        return
        
    print(f"Found {len(large_files)} large CSV files to process.")
    
    for filepath, rows, cols in large_files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename} ({rows}x{cols})...")
        
        try:
            # Load the data matrix
            # z_data shape is (rows, cols)
            z_data = np.loadtxt(filepath, delimiter=',')
            
            # Since the outer loop is θ (x-axis, 0 to 2pi) and inner loop is β (y-axis, 0 to pi),
            # the CSV rows represent θ (x) and columns represent β (y).
            # For Plotly Heatmap:
            # - Columns of the 2D array correspond to the horizontal x-axis.
            # - Rows of the 2D array correspond to the vertical y-axis.
            # Therefore, we transpose the loaded data so that:
            # - The x-axis (columns of the heatmap) corresponds to θ (rows of the CSV).
            # - The y-axis (rows of the heatmap) corresponds to β (columns of the CSV).
            z_plot = z_data.T
            
            actual_rows, actual_cols = z_plot.shape
            
            # X-axis is θ (0 to 2pi), matching the columns of z_plot (length 100)
            x_vals = np.linspace(0, 2 * np.pi, actual_cols)
            # Y-axis is β (0 to pi), matching the rows of z_plot (length 100)
            y_vals = np.linspace(0, np.pi, actual_rows)
            
            # Parse parameters for a descriptive title
            params = parse_filename(filename)
            if params:
                title_text = (
                    f"<b>CHSH Bell Parameter Heatmap</b><br>"
                    f"<sup>α = {params['alpha']}, P<sub>t</sub> = {params['Pt']}, "
                    f"States = {params['states']}</sup>"
                )
            else:
                title_text = f"<b>CHSH Bell Parameter Heatmap</b><br><sup>File: {filename}</sup>"
                
            # Create the Heatmap figure
            fig = go.Figure(data=go.Heatmap(
                z=z_plot,
                x=x_vals,
                y=y_vals,
                colorscale='Viridis',
                colorbar=dict(
                    title=dict(
                        text='CHSH S-value',
                        side='right',
                        font=dict(size=14, family='Arial, sans-serif')
                    ),
                    tickfont=dict(size=12),
                    thickness=20,
                    len=0.9
                ),
                hovertemplate=(
                    '<b>θ (x-axis):</b> %{x:.3f} rad<br>'
                    '<b>β (y-axis):</b> %{y:.3f} rad<br>'
                    '<b>CHSH Value:</b> %{z:.5f}<extra></extra>'
                )
            ))
            
            # Style the layout to make it look professional and print-ready
            fig.update_layout(
                title=dict(
                    text=title_text,
                    x=0.5,
                    xanchor='center',
                    y=0.95,
                    font=dict(size=16, family='Arial, sans-serif', color='#2c3e50')
                ),
                xaxis=dict(
                    title=dict(
                        text='Angle θ (rad)',
                        font=dict(size=14, family='Arial, sans-serif')
                    ),
                    tickvals=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                    ticktext=['0', 'π/2', 'π', '3π/2', '2π'],
                    tickmode='array',
                    gridcolor='rgba(180, 180, 180, 0.3)',
                    zeroline=False,
                    mirror=True,
                    showline=True,
                    linecolor='#34495e',
                    linewidth=1.5
                ),
                yaxis=dict(
                    title=dict(
                        text='Angle β (rad)',
                        font=dict(size=14, family='Arial, sans-serif')
                    ),
                    tickvals=[0, np.pi/2, np.pi],
                    ticktext=['0', 'π/2', 'π'],
                    tickmode='array',
                    gridcolor='rgba(180, 180, 180, 0.3)',
                    zeroline=False,
                    mirror=True,
                    showline=True,
                    linecolor='#34495e',
                    linewidth=1.5
                ),
                # Set a clean square visual aspect ratio for the plot area
                width=750,
                height=600,
                template='plotly_white',
                margin=dict(l=70, r=80, t=90, b=70)
            )
            
            # Generate the output filename
            output_png = filepath.replace('.csv', '.png')
            
            # Save the figure as a high-resolution PNG image
            # scale=2 doubles the size for print-quality sharpness
            fig.write_image(output_png, format='png', scale=2)
            print(f"Successfully saved: {output_png}")
            
        except Exception as e:
            print(f"Failed to plot {filename}: {e}")

if __name__ == '__main__':
    main()
