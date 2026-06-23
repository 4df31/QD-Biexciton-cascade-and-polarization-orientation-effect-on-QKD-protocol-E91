#!/usr/bin/env python
# coding: utf-8

# ----
# Import Libraries
# ----
import os
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from tqdm import tqdm
import numpy as np
from numpy import  sin, cos, pi as π
from datetime import datetime
from sklearn.metrics import r2_score as r2

# ----
# GPU Acceleration via CuPy
# ----
try:
    import cupy as cp
    GPU_ENABLED = True
    print("CUDA/CuPy successfully imported! GPU stochastic operations enabled.")
except ImportError:
    import numpy as cp
    GPU_ENABLED = False
    print("CuPy not found. Falling back to NumPy (CPU).")

# ----
# Theoretical Models
# ----

# Correlation Probability Model with Noise

def model_noise(θ, α, β, Pt=1.0):
    if Pt == 1.0:
        P_HH = P_HV = P_VH = P_VV = 0.0
    else:
        P_HH = 0.0030
        P_HV = 0.0138
        P_VH = 0.0278
        P_VV = 0.0447

    C_a  = cos(α);      S_a  = sin(α)
    C_ab = cos(α + β);  S_ab = sin(α + β)
    C_theta = cos(θ)

    term1 = (Pt / 2.0) * (
        C_a**4 + S_a**4 + 2*(S_a**2)*(C_a**2)*C_theta +
        C_ab**4 + S_ab**4 + 2*(S_ab**2)*(C_ab**2)*C_theta
    )
    term2 = ((P_HH + P_VV) / 2.0) * (C_a**4 + S_a**4 + C_ab**4 + S_ab**4)
    term3 = (P_HV + P_VH) * ((C_a**2)*(S_a**2) + (C_ab**2)*(S_ab**2))

    return term1 + term2 + term3

# Analytical CHSH Model

def analytical_CHSH(theta_FSS, α, β, Pt=1.0):
    if Pt == 1.0:
        P_HH = P_HV = P_VH = P_VV = 0.0
    else:
        P_HH = 0.0030
        P_HV = 0.0138
        P_VH = 0.0278
        P_VV = 0.0447

    C_0 = cos(0);           S_0 = sin(0)
    C_1 = cos(α);           S_1 = sin(α)
    C_2 = cos(α + β);       S_2 = sin(α + β)
    C_3 = cos(3 * π / 4);  S_3 = sin(3 * π / 4)
    C_theta = cos(theta_FSS)

    term1 = (Pt + P_HH + P_VV - P_HV - P_VH) * (
        C_0 * (C_1 - C_3) + C_2 * (C_3 + C_1)
    )
    term2 = Pt * C_theta * (
        S_0 * (S_1 - S_3) + S_2 * (S_3 - S_1)
    )
    return term1 + term2



def correlation_computing(n_execs: int, n_phase: int, experiment: np.ndarray, calc_type: str, fixed_beta=None):
    theta = np.linspace(0, 2*π, n_phase)
    α = global_alpha
    Pt_val = 0.9107

    if n_execs > 1 and fixed_beta is None:
        β = np.linspace(0, π, n_execs)
        if calc_type == 'corr':
            theoretical = np.array([[model_noise(θ=t, β=b, α=α, Pt=Pt_val) for b in β] for t in theta])
        else:
            theoretical = np.array([[analytical_CHSH(theta_FSS=t, α=α, β=b, Pt=Pt_val) for b in β] for t in theta])
    else:
        beta_val = fixed_beta if fixed_beta is not None else 0.0
        if calc_type == 'corr':
            theoretical = np.array([model_noise(θ=t, α=α, β=beta_val, Pt=Pt_val) for t in theta])
        else:
            theoretical = np.array([analytical_CHSH(theta_FSS=t, α=α, β=beta_val, Pt=Pt_val) for t in theta])

    return r2(y_true=theoretical.flatten(), y_pred=experiment.flatten())


# ----
# Core: Stochastic Statevector Bypass
# ----
def local_run(n_states: int, angle_1_2: float, calc_type: str):
    probs = cp.array([0.9107, 0.0030, 0.0138, 0.0278, 0.0447])

    states      = cp.random.choice(5, size=n_states, p=probs)
    basis_alice = cp.random.choice(cp.array([0, 1, 2]), size=n_states)
    basis_bob   = cp.random.choice(cp.array([1, 2, 3]), size=n_states)

    def build_base_circuit(st):
        q = QuantumCircuit(2)
        st = 0  # ucomment for PT \neq to 1
        if st == 0:
            q.h(0); q.rz(relative_phase, 0); q.cx(0, 1)
        elif st == 1: pass
        elif st == 2: q.x(1)
        elif st == 3: q.x(0)
        elif st == 4: q.x(0); q.x(1)
        return q

    def get_p_corr(counts_list, angle_a, angle_b):
        total = 0
        for st in range(5):
            shots = counts_list[st]
            if shots > 0:
                q = build_base_circuit(st)
                q.ry(angle_a, 0); q.ry(angle_b, 1)
                pd = Statevector(q).probabilities_dict()
                p = float(np.clip(pd.get('00', 0.0) + pd.get('11', 0.0), 0.0, 1.0))
                total += int(cp.random.binomial(shots, p))
        return total

    def get_p_ant_corr(counts_list, angle_a, angle_b):
        total = 0
        for st in range(5):
            shots = counts_list[st]
            if shots > 0:
                q = build_base_circuit(st)
                q.ry(angle_a, 0); q.ry(angle_b, 1)
                pd = Statevector(q).probabilities_dict()
                p = float(np.clip(pd.get('01', 0.0) + pd.get('10', 0.0), 0.0, 1.0))
                total += int(cp.random.binomial(shots, p))
        return total

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

        def get_E(counts_list, angle_a, angle_b):
            total = sum(counts_list)
            if total == 0: return 0.0
            return (get_p_corr(counts_list, angle_a, angle_b) -
                    get_p_ant_corr(counts_list, angle_a, angle_b)) / total

        E_01 = get_E(counts_01, 0, global_alpha)
        E_23 = get_E(counts_23, global_alpha + angle_1_2, 3 * π / 4)
        E_03 = get_E(counts_03, 0, 3 * π / 4)
        E_21 = get_E(counts_21, global_alpha + angle_1_2, global_alpha)

        return E_01 + E_23 - E_03 + E_21


# ----
# Execution Handlers
# ----
def main(n_states: int, n_execs: int, calc_type: str):
    return np.array([
        local_run(n_states=n_states, angle_1_2=angle_between_analyzers[k], calc_type=calc_type)
        for k in range(n_execs)
    ])


def save_info_to_csv(column: list, backend_name: str, time_str: str, calc_type: str):
    subfolder = "Pcorr/" if calc_type == 'corr' else "CHSH/"
    path_name = f"./{subfolder}"
    os.makedirs(path_name, exist_ok=True)
    file_name = f"{backend_name}_{time_str}.csv"
    with open(path_name + file_name, "a") as f:
        f.write(",".join(map(str, column)) + "\n")


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
        print("  (1) Fixed Alpha & Beta  — 1D Sweep over Theta")
        print("  (2) Full Parameter Mesh — Sweep Beta and Theta")
        mode = input("Select mode (1/2): ").strip()
        if mode in ['1', '2']:
            settings['run_mode'] = int(mode)
            break

    if settings['run_mode'] == 1:
        print("\nFixed Parameter Configuration:")
        settings['alpha_π'] = float(input("  Enter fixed α (multiples of π, e.g. 0.25): "))
        settings['beta_π']  = float(input("  Enter fixed β (multiples of π, e.g. 0.5): "))

    return settings


# ----
# Main Block
# ----
if __name__ == '__main__':
    settings  = get_run_settings()
    calc_type = settings['calc_type']
    run_mode  = settings['run_mode']

    global angle_between_analyzers
    global relative_phase
    global global_alpha

    if run_mode == 1:
        # Fixed Alpha & Beta — 1D Sweep
        global_alpha = settings['alpha_π'] * π
        fixed_beta   = settings['beta_π']  * π
        angle_between_analyzers = [fixed_beta]

        states_list = [15000]
        n_phase     = int(1e2)
        phase       = np.linspace(0, 2*π, n_phase)
        r2_values   = []

        print(f"\n--- Running Fixed Mode ({calc_type.upper()}) ---")
        print(f"Alpha = {settings['alpha_π']}π,  Beta = {settings['beta_π']}π")

        for states in states_list:
            time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            mesh = []

            for rel_phase in tqdm(phase, desc=f"progress ({states} states):"):
                relative_phase = rel_phase
                out = main(n_states=states, n_execs=1, calc_type=calc_type)
                mesh.append(out[0])

            save_info_to_csv(
                column=mesh,
                backend_name=f"{calc_type.upper()}_fixed_a{settings['alpha_π']}_b{settings['beta_π']}_Pt_0.9107_{n_phase}_points_{states:1.1E}_states_SIM",
                time_str=time_now,
                calc_type=calc_type
            )

            mesh    = np.array(mesh, dtype=np.float64)
            r2_val  = correlation_computing(n_execs=1, n_phase=n_phase, experiment=mesh,
                                            calc_type=calc_type, fixed_beta=fixed_beta)
            r2_values.append(r2_val)

        subfolder = "Pcorr/" if calc_type == 'corr' else "CHSH/"
        r2_filename = f"R2_Pt_0.9107_{calc_type.upper()}_fixed_{n_phase}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.csv"
        with open(f"./Results and Plots/Paper/{subfolder}" + r2_filename, "w") as f:
            f.write("'n_states','r2'\n")
            for n, r in zip(states_list, r2_values):
                f.write(f"{n},{r}\n")

    elif run_mode == 2:
        # Full Parameter Mesh — Sweep Beta and Theta
        states_list   = [50000]
        r2_values     = []
        alpha_values  = np.array([k * π / 4.0 for k in range(6)], dtype=np.float64)

        print(f"\n--- Running Full Mesh ({calc_type.upper()}) ---")

        for α in alpha_values:
            global_alpha = α
            print(f"α = {global_alpha/π:.2f} * π")

            for states in states_list:
                execs   = int(1e2)
                n_phase = int(1e2)

                phase                   = np.linspace(0, 2*π, n_phase)
                angle_between_analyzers = np.linspace(0, π, execs)

                time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                mesh = []

                for rel_phase in tqdm(phase, desc=f"progress ({states} states):"):
                    relative_phase = rel_phase
                    out = main(n_states=states, n_execs=execs, calc_type=calc_type)
                    mesh.append(out)
                    save_info_to_csv(
                        column=out,
                        backend_name=f"{calc_type.upper()}_QS_alpha_{global_alpha/π:.2f}π_Pt_0.9107_{execs}x{n_phase}_mesh_{states:1.1E}_states_SIM",
                        time_str=time_now,
                        calc_type=calc_type
                    )