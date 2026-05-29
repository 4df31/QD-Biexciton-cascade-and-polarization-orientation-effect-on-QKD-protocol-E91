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
# Theoretical Models
# ---------------------------------------------------------
def model_noise(θ, α, β, Pt=1.0):
    """ Analytical model for correlation probability """
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


def analytical_CHSH(theta_FSS, alpha, beta, Pt=1.0):
    """ Analytical model for CHSH quantity """
    if Pt == 1.0:
        P_HH = P_HV = P_VH = P_VV = 0.0
    else:
        P_HH = 0.0030
        P_HV = 0.0138
        P_VH = 0.0278
        P_VV = 0.0447

    # Qiskit's ry(angle) applies a Bloch rotation corresponding to 2 * physical angle
    # The CHSH equations expect 2*phi_a terms, which exactly match the Bloch angles.
    C_0 = cos(0)
    C_1 = cos(alpha)
    C_2 = cos(alpha + beta)
    C_3 = cos(3 * pi / 4)

    S_0 = sin(0)
    S_1 = sin(alpha)
    S_2 = sin(alpha + beta)
    S_3 = sin(3 * pi / 4)

    C_theta = cos(theta_FSS)

    term1 = (Pt + P_HH + P_VV - P_HV - P_VH) * (
        C_0 * (C_1 - C_3) + C_2 * (C_3 + C_1)
    )
    term2 = Pt * C_theta * (
        S_0 * (S_1 - S_3) + S_2 * (S_3 - S_1)
    )

    return term1 + term2


def correlation_computing(n_execs: int, n_phase: int, experiment: np.ndarray, calc_type: str, fixed_beta=None):
    theta = np.linspace(0, 2*pi, n_phase)
    alpha = global_alpha
    Pt_val = 0.9107
    
    if n_execs > 1 and fixed_beta is None:
        # Full Mesh Mode (2D)
        beta = np.linspace(0, pi, n_execs)
        if calc_type == 'corr':
            theoretical = np.array([[model_noise(θ=t, β=b, α=alpha, Pt=Pt_val) for b in beta] for t in theta])
        else:
            theoretical = np.array([[analytical_CHSH(theta_FSS=t, alpha=alpha, beta=b, Pt=Pt_val) for b in beta] for t in theta])
    else:
        # Fixed Alpha/Beta Mode (1D Sweep)
        beta_val = fixed_beta if fixed_beta is not None else 0.0
        if calc_type == 'corr':
            theoretical = np.array([model_noise(θ=t, α=alpha, β=beta_val, Pt=Pt_val) for t in theta])
        else:
            theoretical = np.array([analytical_CHSH(theta_FSS=t, alpha=alpha, beta=beta_val, Pt=Pt_val) for t in theta])
            
    return r2(y_true=theoretical.flatten(), y_pred=experiment.flatten())


# ---------------------------------------------------------
# Core Optimization: Stochastic Statevector Bypass
# ---------------------------------------------------------
def local_run(n_states: int, angle_1_2: float, calc_type: str):
    probs = cp.array([0.9107, 0.0030, 0.0138, 0.0278, 0.0447])
    
    states = cp.random.choice(5, size=n_states, p=probs)
    basis_alice = cp.random.choice(cp.array([0, 1, 2]), size=n_states)
    basis_bob = cp.random.choice(cp.array([1, 2, 3]), size=n_states)
    
    def get_p_corr(counts_list, angle_a, angle_b):
        """ Helper to compute absolute correlation hits for a specific basis combination. """
        p_corr_total = 0
        for st in range(5):
            shots = counts_list[st]
            if shots > 0:
                q = QuantumCircuit(2)
                if st == 0:
                    q.h(0); q.rz(relative_phase, 0); q.cx(0, 1) # Singlet with phase
                elif st == 1: pass # |00>
                elif st == 2: q.x(1) # |01>
                elif st == 3: q.x(0) # |10>
                elif st == 4: q.x(0); q.x(1) # |11> 
                
                q.ry(angle_a, 0)
                q.ry(angle_b, 1)
                
                probs_dict = Statevector(q).probabilities_dict()
                p_target = probs_dict.get('00', 0.0) + probs_dict.get('11', 0.0)
                p_corr_total += int(cp.random.binomial(shots, p_target))
        return p_corr_total

    if calc_type == 'corr':
        mask_11 = (basis_alice == 1) & (basis_bob == 1)
        mask_22 = (basis_alice == 2) & (basis_bob == 2)
        
        counts_11 = [int(cp.sum(mask_11 & (states == s))) for s in range(5)]
        counts_22 = [int(cp.sum(mask_22 & (states == s))) for s in range(5)]
        
        total_same_basis = sum(counts_11) + sum(counts_22)
        if total_same_basis == 0:
            return 0.0
            
        corr_11 = get_p_corr(counts_11, global_alpha, global_alpha)
        corr_22 = get_p_corr(counts_22, global_alpha + angle_1_2, global_alpha + angle_1_2)
        
        return (corr_11 + corr_22) / total_same_basis
        
    elif calc_type == 'chsh':
        mask_01 = (basis_alice == 0) & (basis_bob == 1)
        mask_23 = (basis_alice == 2) & (basis_bob == 3)
        mask_03 = (basis_alice == 0) & (basis_bob == 3)
        mask_21 = (basis_alice == 2) & (basis_bob == 1)
        
        counts_01 = [int(cp.sum(mask_01 & (states == s))) for s in range(5)]
        counts_23 = [int(cp.sum(mask_23 & (states == s))) for s in range(5)]
        counts_03 = [int(cp.sum(mask_03 & (states == s))) for s in range(5)]
        counts_21 = [int(cp.sum(mask_21 & (states == s))) for s in range(5)]
        
        def get_E_from_counts(counts_list, angle_a, angle_b):
            total_shots = sum(counts_list)
            if total_shots == 0: return 0.0
            p_corr = get_p_corr(counts_list, angle_a, angle_b)
            return (2.0 * p_corr - total_shots) / total_shots

        # Compute each E(phi_a, phi_b)
        E_01 = get_E_from_counts(counts_01, 0, global_alpha)
        E_23 = get_E_from_counts(counts_23, global_alpha + angle_1_2, 3 * pi / 8)
        E_03 = get_E_from_counts(counts_03, 0, 3 * pi / 8)
        E_21 = get_E_from_counts(counts_21, global_alpha + angle_1_2, global_alpha)
        
        return E_01 + E_23 - E_03 + E_21


# ---------------------------------------------------------
# Execution Handlers
# ---------------------------------------------------------
def main(n_states: int, n_execs: int, calc_type: str):
    fidelity = list()
    for k in range(n_execs):
        fidelity.append(local_run(n_states=n_states, angle_1_2=angle_between_analyzers[k], calc_type=calc_type))
    return np.array(fidelity)
    
def save_info_to_csv(column: list, backend_name, time_str):
    path_name = f"./Results and Plots/Paper/Fig_7/"
    os.makedirs(path_name, exist_ok=True) 
    file_name = f"{backend_name}_{time_str}.csv"
    string = ",".join(map(str, column))
    with open(path_name+file_name, "a") as file:
        file.write(string + "\n")
    return

def get_run_settings():
    settings = {}
    print("\n--- Quantum Simulation Configuration ---")
    while True:
        c_type = input("Compute CHSH Quantity or Correlation Probability? (chsh/corr): ").strip().lower()
        if c_type in ['chsh', 'corr']:
            settings['calc_type'] = c_type
            break
            
    while True:
        print("\nExecution Mode:")
        print("  (1) Single point execution")
        print("  (2) Fixed Alpha & Beta (1D Sweep over Theta)")
        print("  (3) Full Parameter Mesh (Sweep Beta and Theta)")
        mode = input("Select mode (1/2/3): ").strip()
        if mode in ['1', '2', '3']:
            settings['run_mode'] = int(mode)
            break
            
    if settings['run_mode'] == 2:
        print("\nFixed Parameter Configuration:")
        settings['alpha_pi'] = float(input("  Enter fixed alpha (in multiples of pi, e.g., 0.25): "))
        settings['beta_pi'] = float(input("  Enter fixed beta (in multiples of pi, e.g., 0.5): "))
        
    return settings


# ---------------------------------------------------------
# Main Block
# ---------------------------------------------------------
if __name__ == '__main__':
    settings = get_run_settings()
    calc_type = settings['calc_type']
    run_mode = settings['run_mode']
    
    global angle_between_analyzers
    global relative_phase
    global global_alpha

    if run_mode == 1:
        states_list = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
        angle_between_analyzers = [pi/2]
        relative_phase = pi
        global_alpha = 0.25 * pi # Arbitrary default for single point evaluation
        
        performance_parameter = np.zeros(len(states_list)) 
        time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        
        print(f"\n--- Running Single Point ({calc_type.upper()}) ---")
        for k, states in enumerate(states_list):
            print(f"states = {states:8d}")
            out = main(n_states=states, n_execs=1, calc_type=calc_type)
            performance_parameter[k] = out[0]
            save_info_to_csv(column=out, backend_name=f"{calc_type.upper()}_single_point_SV_SIM", time_str=time_now)
            
    elif run_mode == 2:
        # Fixed Alpha & Beta Setup (1D Sweep)
        global_alpha = settings['alpha_pi'] * pi
        fixed_beta = settings['beta_pi'] * pi
        angle_between_analyzers = [fixed_beta]
        
        states_list = [15000]
        n_phase = int(1e2)
        phase = np.linspace(0, 2*pi, n_phase)
        r2_values = list()
        
        print(f"\n--- Running Fixed Mode ({calc_type.upper()}) ---")
        print(f"Alpha = {settings['alpha_pi']}π, Beta = {settings['beta_pi']}π")
        
        for states in states_list:
            time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            mesh = list()
            
            for rel_phase in tqdm(phase, desc=f"progress ({states} states): "):
                relative_phase = rel_phase 
                out = main(n_states=states, n_execs=1, calc_type=calc_type)
                mesh.append(out[0]) # Since n_execs=1, unpack the first result
                
            # Save 1D experimental output
            save_info_to_csv(column=mesh, backend_name=f"{calc_type.upper()}_fixed_a{settings['alpha_pi']}_b{settings['beta_pi']}_Pt_0.9107_{n_phase}_points_{states:1.1E}_states_SIM", time_str=time_now)
            
            mesh = np.array(mesh, dtype=np.float64)
            r2_val = correlation_computing(n_execs=1, n_phase=n_phase, experiment=mesh, calc_type=calc_type, fixed_beta=fixed_beta)
            r2_values.append(r2_val)
            
        r2_filename = f"R2_Pt_0.9107_{calc_type.upper()}_fixed_{n_phase}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.csv"
        with open("./Results and Plots/Paper/Fig_7/" + r2_filename, "w") as file:
            file.write("'n_states','r2'\n")
            for n, r in zip(states_list, r2_values):
                file.write(f"{n},{r}\n")
                
    elif run_mode == 3:
        # Full Parameter Mesh setup
        states_list = [50000]  
        r2_values = list()
        alpha_values = np.array([k*pi/4.0 for k in range(6)], dtype=np.float64)

        print(f"\n--- Running Full Mesh ({calc_type.upper()}) ---")
        for alpha in alpha_values:
            global_alpha = alpha
            print(f"alpha = {global_alpha/pi:.2f} * pi")
            for states in states_list:
                execs = int(1e2) 
                n_phase = int(1e2) 
            
                phase = np.linspace(0, 2*pi, n_phase) 
                angle_between_analyzers = np.linspace(0, pi, execs) # Fixed boundary alignment bug mapping 0->pi 
            
                time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                mesh = list()
                
                for rel_phase in tqdm(phase, desc=f"progress ({states} states): "):
                    relative_phase = rel_phase 
                    
                    out = main(n_states=states, n_execs=execs, calc_type=calc_type)
                    mesh.append(out)
                    save_info_to_csv(column=out, backend_name=f"{calc_type.upper()}_QS_alpha_{global_alpha/pi:.2f}pi_Pt_0.9107_{execs}x{n_phase}_mesh_{states:1.1E}_states_SIM", time_str=time_now)
                    
                mesh = np.array(mesh, dtype=np.float64)
                r2_values.append(correlation_computing(n_execs=execs, n_phase=n_phase, experiment=mesh, calc_type=calc_type))
                
            r2_filename = f"R2_Pt_0.9107_{calc_type.upper()}_QS_{execs}x{n_phase}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.csv"
            with open("./Results and Plots/Paper/Fig_6/" + r2_filename, "w") as file:
                file.write("'n_states','r2'\n")
                for n, r in zip(states_list, r2_values):
                    file.write(f"{n},{r}\n")
