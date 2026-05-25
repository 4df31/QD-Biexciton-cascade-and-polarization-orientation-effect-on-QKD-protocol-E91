#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import cpu_count
from qiskit_aer import AerSimulator
from qiskit import*
from tqdm import tqdm
import numpy as np
from numpy import pi,sin,cos
from numpy.random import randint,choice
from qiskit.quantum_info import Statevector
from multiprocessing import Pool
from datetime import datetime
from sklearn.metrics import r2_score as r2
import pandas as pd


# In[2]:


def parallel(vector:list, func=None, use_gpu:bool=False):
    if use_gpu:
        # Utilize Qiskit's native batch processing for optimal GPU acceleration
        simulator = AerSimulator(device='GPU')
        circ = transpile(vector, simulator)
        results = simulator.run(circ, shots=1).result().get_counts()
        
        # get_counts() returns a single dict if only 1 circuit was run, 
        # but returns a list of dicts for multiple batched circuits.
        if isinstance(results, dict):
            return [list(results.keys())[0]]
        return [list(res.keys())[0] for res in results]
    else:
        # Standard CPU multiprocessing
        with Pool(int(cpu_count())) as p:
            return p.map(func, vector)


# ## Quantum Circuits Functions

# In[3]:


def circuit_singlet_creator(angle_between_analyzers:float):
    """
        ----------------------------------------------
        ----- REMEBER TO MODIFY THIS ALPHA VALUE -----
        ----------------------------------------------
    """
    global_shift = global_alpha # this angel will deppend on main loop
    #global_shift = 0*pi/4 # alpha angle between analizer 0 and 1 for alice, angle between x and analizer 0 for bob
    # ---------------------------------------
    q = QuantumCircuit(2,2)
    # Noise indroduction based on QST experiments on QD XX-X-0 cascade
    state = choice(a=[0,1,2,3,4],p=[0.9107,0.0030,0.0138,0.0278,0.0447]) # Chen's based probabilities (diff(ρ_PURE,ρ_QST))
    # Pt = 0.9107
    if state == 0:
        q.h(0); q.rz(relative_phase,0); q.cx(0,1) # singlet state with phase
    elif state == 1:
        q.barrier()#q.z(0);q.z(1) # |00>
    elif state == 2:
        q.x(1); # |01>
    elif state == 3:
        q.x(0) # |10>
    elif state == 4:
        q.x(1);q.x(1) # |11>
    # Basis selection
    
    basis_alice = choice([0,1,2]); basis_bob = choice([1,2,3])
    basis_list = {0:0,1:global_shift,2:(global_shift+angle_between_analyzers),3:3*pi/8}
    q.ry(basis_list[basis_alice],0); q.ry(basis_list[basis_bob],1)
    # Measure
    q.measure(0,0); q.measure(1,1)
    basis = str(basis_alice)+str(basis_bob)
    
    return q,basis
        


# In[4]:


def circuit_list_vector_creator(n_states:int,angle_between_analyzers:float):
    # Use standard CPU multiprocessing for circuit string generation
    circs_basis_vector = parallel([angle_between_analyzers for k in range(n_states)], func=circuit_singlet_creator, use_gpu=False)
    return circs_basis_vector


# In[5]:


def exec(q:QuantumCircuit):
    simulator = AerSimulator()
    circ = transpile(q, simulator)
    result = simulator.run(circ,shots=1).result().get_counts()
    return list(dict(result).keys())[0]


# ## Post Execution Functions

# In[6]:


def info_selector(circs_basis_vector:list):
    N = len(circs_basis_vector)
    circuits = list()
    basis=list()
    for k in range(N):
        circuits.append(circs_basis_vector[k][0])
        basis.append(circs_basis_vector[k][1])
    return [circuits,basis]

def probability_computation(basis_vector:list,result_vector:list):
    N_same_basis = 0
    N_anticorrelation = 0
    N_states = len(basis_vector)
    for k in range(N_states):
        if basis_vector[k] == '11' or basis_vector[k] == '22':
            N_same_basis += 1
            if result_vector[k] == '00' or result_vector[k] == '11':
                N_anticorrelation += 1
            else:
                pass
        else:
            pass
    return N_anticorrelation/N_same_basis


# ## Run Function

# In[7]:


def local_run (n_states:int,angle_1_2:float):
    info = circuit_list_vector_creator(n_states=n_states,angle_between_analyzers=angle_1_2)
    circuit_list, basis_list = info_selector(info)
    del info
    
    # Send the batch of circuits natively to the GPU
    result = parallel(vector=circuit_list, func=None, use_gpu=True)
    result = [result[k] for k in range(len(result))]
    
    del circuit_list
    fidelity = probability_computation(basis_vector=basis_list,result_vector=result)    
    return fidelity


# # Analytical Model

# In[8]:


def analytical(theta:float,beta:float,alpha:float):
    return (\
        cos(alpha)**4 + sin(alpha)**4 +\
        2 * (sin(alpha)**2) * (cos(alpha)**2) * cos(theta) + \
        cos(alpha+beta)**4 + sin(alpha+beta)**4 + \
        2 * (sin(alpha+beta)**2) * (cos(alpha+beta)**2) * cos(theta)\
        )/2  

def model_noise(θ,β,α,Pt):
    # α/2 for the change between spin and polarization representation
    return analytical(θ,β,α/2)*Pt + ((1-Pt)/2.0)


# In[9]:


def correlation_computing(n_execs:int,n_phase:int,experiment:np.ndarray):
    theta = np.linspace(0,2*pi,n_phase)
    beta = np.linspace(0,pi,n_execs)
    """
        ----------------------------------------------
        ----- REMEBER TO MODIFY THIS ALPHA VALUE -----
        ----------------------------------------------
    """
    alpha = global_alpha
    theoretical = np.array([[model_noise(θ=t,β=b,α=alpha,Pt=0.9107) for t in theta] for b in beta])
    return  r2(y_true=theoretical,y_pred=experiment)
    


# In[10]:


def main(n_states:int,n_execs:int):
    fidelity = list()
    for k in range(n_execs):
        fidelity.append(local_run(n_states=n_states,angle_1_2=angle_between_analyzers[k]))
    return np.array(fidelity)
    
def save_info_to_csv(column:list,backend_name,time_str):
    path_name = f"./Chen_based_sims/"
    file_name = f"{backend_name}_{time_str}.csv"
    string = ""
    for data in column:
        string += f"{data},"
    with open(path_name+file_name, "a") as file:
        file.write(string + "\n")
    return


# In[ ]:


def run_mode():
    print("single point execution? Y/N")
    run_opt_bool = input().lower() # .lower() handles both 'y' and 'Y' automatically!
    
    if run_opt_bool == 'y':
        return True
    elif run_opt_bool == 'n':
        return False
    else:
        return run_mode() 

if __name__ == __main__:
   
    single_point = run_mode()
    
    
    global angle_between_analyzers
    global relative_phase
    global global_alpha
    
    if single_point:
        #states_list = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]
        state_list = 15000
    
        angle_between_analyzers = [pi/8]
        relative_phase = np.linspace(pi)
        
        performance_parameter = np.zeros(len(states_list)) 
        time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        
        for k, states in enumerate(states_list):
            print(f"states = {states:8d}")
            out = main(n_states=states, n_execs=1)
            performance_parameter[k] = out[0]
            save_info_to_csv(column=out, backend_name=f"NOISY_single_point_AER_SIM", time_str=time_now)
    else:
        # FIX: Uncommented states_list so the loop below actually has data to run on
        #states_list = [45000, 30000, 25000, 10000, 5000]  # run 1
        #states_list = [450, 300, 250, 100, 50]  # run 1
        #states_list = [40000, 35000, 20000, 15000, 1000]  # run 2
        state_list = [15000]
        r2_values = list()
        #alpha_values = np.array([k*pi/4.0 for k in range(6)],dtype=np.float64)
        alpha_values = np.array([pi/4],dtype=np.float64)
    
        for alpha in alpha_values:
            global_alpha = alpha
            print(f"alpha = {global_alpha/pi} *pi")
            for states in states_list:
                execs = int(20) # steps of angle β between coincident analyzers beta parameter between 0 and π
                n_phase = int(20) # steps of FSS parameter theta between 0 and 2 pi
            
                phase = np.linspace(0, 2*pi, n_phase) # θ_FSS
                #angle_between_analyzers = np.linspace(0, 2*pi, execs) # β
                anlge_between_analyzers = np.array(np.pi/4,dtype=np.float64)
            
                time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                mesh = list()
                
                for rel_phase in tqdm(phase, desc="progress: "):
                    
                    relative_phase = rel_phase # phase between bell pair state
                    #---------------------------------------------------
                    out = main(n_states=states, n_execs=execs)
                    mesh.append(out)
                    save_info_to_csv(column=out, backend_name=f"alpha_{global_alpha/pi}pi_Pt_0.9107_{len(angle_between_analyzers)}x{len(alpha_values)}_mesh_{states:1.1E}_states_SIM", time_str=time_now)
                    #---------------------------------------------------
                    
                mesh = np.array(mesh, dtype=np.float64)
            
                r2_values.append(correlation_computing(n_execs=execs, n_phase=n_phase, experiment=mesh))
                
            r2_filename = f"R2_Pt_0.9107_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.csv"
            with open("./Chen_based_sims/"+r2_filename, "w") as file:
                file.write("'n_states','r2'\n")
                for n, r in zip(states_list, r2_values):
                    file.write(f"{n},{r}\n")
        """
        states_list = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]
        n_phase = 30
        angle_between_analyzers = [pi/2]
        phase = np.linspace(0, 2*pi, n_phase)
        
        for k, states in enumerate(states_list):
            time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            print(f"states = {states:8d}")
            performance_parameter = list()
            for rel_phase in tqdm(phase, desc="progress: "):
                relative_phase = rel_phase
                out = main(n_states=states, n_execs=1)
                performance_parameter.append(out[0])
                save_info_to_csv(column=out, backend_name=f"NOISY_{states}_FSS_parametrized_AER_SIM", time_str=time_now)
        """
