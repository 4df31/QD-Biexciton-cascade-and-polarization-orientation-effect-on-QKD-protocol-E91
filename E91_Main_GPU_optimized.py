#!/usr/bin/env python
# coding: utf-8

import os
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from tqdm import tqdm
import numpy as np
from numpy import pi, sin, cos
from datetime import datetime
from sklearn.metrics import r2_score as r2
import pandas as pd

# ---------------------------------------------------------
# GPU ACCELERATION SETUP (CuPy)
# ---------------------------------------------------------
# We attempt to load CuPy for GPU tensor and array operations.
# If CuPy is not installed, we gracefully fallback to standard NumPy.
try:
    import cupy as cp
    GPU_ENABLED = True
    print("CUDA/CuPy successfully imported! GPU stochastic operations enabled.")
except ImportError:
    import numpy as cp
    GPU_ENABLED = False
    print("CuPy not found. Falling back to NumPy (CPU) for array operations.")
    print("To enable massive GPU vectorization, please run: 'pip install cupy'")


# ---------------------------------------------------------
# Analytical Model
# ---------------------------------------------------------
def model_noise(θ, β, α, Pt):
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


def correlation_computing(n_execs: int, n_phase: int, experiment: np.ndarray):
    theta = np.linspace(0, 2*pi, n_phase)
    beta = np.linspace(0, pi, n_execs)
    alpha = global_alpha
    
    theoretical = np.array([[model_noise(θ=t, β=b, α=alpha, Pt=0.9107) for b in beta] for t in theta])
    return r2(y_true=theoretical.flatten(), y_pred=experiment.flatten())


# ---------------------------------------------------------
# Core Optimization: Stochastic Statevector Bypass
# ---------------------------------------------------------
def local_run(n_states: int, angle_1_2: float):
    """
    Instead of using AerSimulator (which causes massive CPU bottlenecks for 
    small 2-qubit circuits), this uses a Stochastic Statevector Bypass.
    It calculates the exact ideal probability of the state, then delegates
    the massive finite-shot sampling directly to the GPU via CuPy's binomial.
    """
    # 1. High-speed GPU Array creation and filtering
    # Probabilities based on QST experiments on QD XX-X-0 cascade
    probs = cp.array([0.9107, 0.0030, 0.0138, 0.0278, 0.0447])
    
    states = cp.random.choice(5, size=n_states, p=probs)
    basis_alice = cp.random.choice(cp.array([0, 1, 2]), size=n_states)
    basis_bob = cp.random.choice(cp.array([1, 2, 3]), size=n_states)
    
    # 2. Filter target conditions ('11' or '22') directly in VRAM
    mask_11 = (basis_alice == 1) & (basis_bob == 1)
    mask_22 = (basis_alice == 2) & (basis_bob == 2)
    
    # Extract the exact shot counts needed for each base state
    counts_11 = [int(cp.sum(mask_11 & (states == s))) for s in range(5)]
    counts_22 = [int(cp.sum(mask_22 & (states == s))) for s in range(5)]
    
    total_same_basis = sum(counts_11) + sum(counts_22)
    if total_same_basis == 0:
        return 0.0
        
    total_anticorr = 0
    
    # 3. Simulate grouping without Aer Simulator overhead
    for st in range(5):
        shots_11 = counts_11[st]
        shots_22 = counts_22[st]
        
        if shots_11 > 0 or shots_22 > 0:
            # Build base state preparation (NO Measurement mappings needed)
            #st = 0 # comment for noise based simulation
            q = QuantumCircuit(2)
            if st == 0:
                q.h(0); q.rz(relative_phase, 0); q.cx(0, 1) # Singlet with phase
            elif st == 1:
                pass # |00>
            elif st == 2:
                q.x(1) # |01>
            elif st == 3:
                q.x(0) # |10>
            elif st == 4:
                q.x(0); q.x(1) # |11> 
            
            # Execute counts for basis combination '11'
            if shots_11 > 0:
                q_11 = q.copy()
                q_11.ry(global_alpha, 0)
                q_11.ry(global_alpha, 1)
                
                # Fast analytical probability extraction
                probs_dict = Statevector(q_11).probabilities_dict()
                p_target = probs_dict.get('00', 0.0) + probs_dict.get('11', 0.0)
                
                # GPU-accelerated finite-shot noise simulation
                total_anticorr += int(cp.random.binomial(shots_11, p_target))
                
            # Execute counts for basis combination '22'
            if shots_22 > 0:
                angle_22 = global_alpha + angle_1_2
                q_22 = q.copy()
                q_22.ry(angle_22, 0)
                q_22.ry(angle_22, 1)
                
                # Fast analytical probability extraction
                probs_dict = Statevector(q_22).probabilities_dict()
                p_target = probs_dict.get('00', 0.0) + probs_dict.get('11', 0.0)
                
                # GPU-accelerated finite-shot noise simulation
                total_anticorr += int(cp.random.binomial(shots_22, p_target))

    return total_anticorr / total_same_basis

# ---------------------------------------------------------
# Execution Handlers
# ---------------------------------------------------------
def main(n_states: int, n_execs: int):
    fidelity = list()
    for k in range(n_execs):
        fidelity.append(local_run(n_states=n_states, angle_1_2=angle_between_analyzers[k]))
    return np.array(fidelity)
    
def save_info_to_csv(column: list, backend_name, time_str):
    path_name = f"./Results and Plots/Paper/Fig_6/"
    os.makedirs(path_name, exist_ok=True) # Ensure directory exists
    file_name = f"{backend_name}_{time_str}.csv"
    string = ",".join(map(str, column))
    with open(path_name+file_name, "a") as file:
        file.write(string + "\n")
    return

def run_mode():
    while True:
        print("single point execution? Y/N")
        run_opt_bool = input().lower() 
        if run_opt_bool == 'y':
            return True
        elif run_opt_bool == 'n':
            return False

# ---------------------------------------------------------
# Main Block
# ---------------------------------------------------------
if __name__ == '__main__':
    single_point = run_mode()
    
    global angle_between_analyzers
    global relative_phase
    global global_alpha

    if single_point:
        states_list = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
        angle_between_analyzers = [pi/2]
        relative_phase = pi
        
        performance_parameter = np.zeros(len(states_list)) 
        time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        
        for k, states in enumerate(states_list):
            print(f"states = {states:8d}")
            out = main(n_states=states, n_execs=1)
            performance_parameter[k] = out[0]
            save_info_to_csv(column=out, backend_name=f"NOISY_single_point_SV_SIM", time_str=time_now)
    else:
        #states_list = [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000, 10000, 5000, 1000]  # run 2
        states_list = [50000]  
        r2_values = list()
        alpha_values = np.array([k*pi/4.0 for k in range(6)], dtype=np.float64)

        for alpha in alpha_values:
            global_alpha = alpha
            print(f"alpha = {global_alpha/pi:.2f} * pi")
            for states in states_list:
                execs = int(1e2) # steps of angle β between coincident analyzers
                n_phase = int(1e2) # steps of FSS parameter theta
            
                phase = np.linspace(0, 2*pi, n_phase) # θ_FSS
                angle_between_analyzers = np.linspace(0, 2*pi, execs) # β
            
                time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                mesh = list()
                
                for rel_phase in tqdm(phase, desc=f"progress ({states} states): "):
                    relative_phase = rel_phase # phase between bell pair state
                    
                    out = main(n_states=states, n_execs=execs)
                    mesh.append(out)
                    # Pt = 0.9107 for Chens based data
                    save_info_to_csv(column=out, backend_name=f"P_Corr_QS_alpha_{global_alpha/pi:.2f}pi_Pt_0.9107_{execs}x{n_phase}_mesh_{states:1.1E}_states_SIM", time_str=time_now)
                    
                mesh = np.array(mesh, dtype=np.float64)
                r2_values.append(correlation_computing(n_execs=execs, n_phase=n_phase, experiment=mesh))
                
            r2_filename = f"R2_Pt_0.9107_P_Corr_QS_{execs}x{n_phase}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.csv"
            with open("./Results and Plots/Paper/Fig_6/" + r2_filename, "w") as file:
                file.write("'n_states','r2'\n")
                for n, r in zip(states_list, r2_values):
                    file.write(f"{n},{r}\n")
