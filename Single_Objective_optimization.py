
import csv
import sys
import os
import numpy as np
import pandas as pd
import datetime

# ✅ Global simulation time settings
SIM_TIME = 1806
PRE_TIME = 260

os.chdir(r"C:/Fariba_2025/Ca_OpenCOR_Calibration_2025")
sys.path.insert(0, os.getcwd())

print("🚀 Script started running!")

from scipy.optimize import minimize

from Functions.model_utils import SimulationManager
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

param_names = [
    'SMC_Par/alpha0', 'SMC_Par/alpha1',
    'SMC_Par/V0', 'SMC_Par/V1',
    'SMC_Par/k_ryr0', 'SMC_Par/k_ryr1', 'SMC_Par/k_ryr2', 'SMC_Par/k_ryr3',
    'SMC_Par/Vm', 'SMC_Par/km', 'SMC_Par/gca',
    'SMC_Par/F', 'SMC_Par/R', 'SMC_Par/T',
    'SMC_Par/Jer', 'SMC_Par/Ve', 'SMC_Par/Ke',
    'SMC_Par/Vp', 'SMC_Par/Kp', 'SMC_Par/gamma',
    'SMC_Par/Ca_SR', 'model_parameters1/Ca_in_SMC',
    'SMC_Par/delta_SMC', 'SMC_Par/k_RyR'
]

output_names = ['model_parameters1/Ca_in_SMC', 'SMC_Par/Ca_in_SMC_dig']
output_type = 'max_Ca'

try:
    working_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    working_dir = os.getcwd()

##working_dir = os.path.join(os.path.dirname(__file__))

model_path = os.path.abspath(os.path.join(working_dir, "Model", "Main_Coupled_SMC_Model.cellml"))

# Setting the output directory
output_file_path = os.path.join(working_dir, "outputs", "Single_objective", "run2")

if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

sim_manager = SimulationManager(cal_param_names=param_names, tau=0.0, sim_time=SIM_TIME, pre_time=PRE_TIME)

param_init_vals = sim_manager.get_init_param_vals()
print(f"Initial parameter values: {param_init_vals}")

# Define custom multipliers for min and max bound per parameter
lower_multipliers = [
    0.1, 0.1,           # alpha0, alpha1
    1, 0.7,             # V0, V1
    0.01, 0.1, 0.2, 0.2,# k_ryr0, k_ryr1, k_ryr2, k_ryr3 
    0.7, 0.1, 0.1,      # Vm, km, gca
    1, 1, 1,            # F, R, T
    0.1, 0.1, 0.1,      # Jer, Ve, Ke
    0.1, 0.1, 0.1,      # Vp, Kp, gamma
    1, 1,                # Ca_SR, Ca_in_SMC
    0.1, 0.1            # delta_SMC, k_RyR
]

upper_multipliers = [
    2, 2,              # alpha0, alpha1
    1, 1.34,           # V0, V1
    10, 10, 5, 5,      # k_ryr0, k_ryr1, k_ryr2, k_ryr3
    1.34, 20, 10,      # Vm, km, gca
    1, 1, 1,           # F, R, T
    2, 20, 20,         # Jer, Ve, Ke
    5, 5, 10,          # Vp, Kp, gamma
    1, 1,               # Ca_SR, Ca_in_SMC
    10, 10             # delta_SMC, k_RyR
]

param_bounds = [
    (min(l * val, u * val), max(l * val, u * val))
    for val, l, u in zip(param_init_vals, lower_multipliers, upper_multipliers)
]

def optimization_cost(params):
    outputs, _ = sim_manager.run_and_get_results(params)
    Ca_in_SMC = np.squeeze(outputs[0])
    Ca_in_SMC_dig = np.squeeze(outputs[1])

    if not np.all(np.isfinite(Ca_in_SMC)) or not np.all(np.isfinite(Ca_in_SMC_dig)):
        print("⚠️ Invalid output encountered.")
        return 1e6  # Penalize invalid outputs

    if np.any(np.abs(Ca_in_SMC) > 1e3):  # Too spiky
        print("⚠️ High output encountered.")
        return 1e4

    cost = np.mean((Ca_in_SMC - Ca_in_SMC_dig) ** 2)
    print(f"🔄 Cost: {cost:.6f}, Params: {params}")
    return cost

# Run the optimizer
options = {'disp': True, 'maxiter': 100}

res = minimize(
    optimization_cost,
    param_init_vals,
    method='Powell',  # Slower but more robust to bad gradients
    bounds=param_bounds,
    options={'disp': True, 'maxiter': 100}
)

optimal_params = res.x

print(f"Optimal parameters: {optimal_params}")

csv_file = os.path.join(output_file_path, "optimized_parameters.csv")
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Prepare simplified parameter names
param_names_simple = [name.split('/')[-1] for name in param_names]

# Compute bounds
min_bounds = [min(l * val, u * val) for val, l, u in zip(param_init_vals, lower_multipliers, upper_multipliers)]
max_bounds = [max(l * val, u * val) for val, l, u in zip(param_init_vals, lower_multipliers, upper_multipliers)]

# Prepare row as dictionary
row_data = {
    'run_id': run_timestamp,
    'sim_time': sim_manager.sim_time,
    'pre_time': sim_manager.pre_time,
    'final_cost': res.fun,
}

# Add parameters
for name, val in zip(param_names_simple, optimal_params):
    row_data[name] = val

# Add bounds
for i, (min_val, max_val) in enumerate(zip(min_bounds, max_bounds), start=1):
    row_data[f'min_bound_{i}'] = min_val
    row_data[f'max_bound_{i}'] = max_val

# Create dataframe for current run
current_run_df = pd.DataFrame([row_data])

# Append or create file
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
    merged_df = pd.concat([existing_df, current_run_df], ignore_index=True)
else:
    merged_df = current_run_df

# Save file
merged_df.to_csv(csv_file, index=False)


# Run the simulation with optimal parameters
outputs, t = sim_manager.run_and_get_results(optimal_params)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_plot_path = os.path.join(output_file_path, f"ACh_concentration_plot_{timestamp}.png")

sim_manager.plot_model_out_and_experimental_data(outputs, t, output_plot_path)
