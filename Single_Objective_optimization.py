
import csv
import sys
import os
import numpy as np
import pandas as pd
import datetime

# ✅ Global simulation time settings
SIM_TIME = 2200
PRE_TIME = 0

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
    'SMC_Par/delta_SMC', 'SMC_Par/k_RyR'
]

output_names = ['model_parameters1/Ca_in_SMC']

working_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.abspath(os.path.join(working_dir, "Model", "Main_Coupled_SMC_Model.cellml"))

# Setting the output directory
output_file_path = "outputs/Single_objective/run2/"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

sim_manager = SimulationManager(cal_param_names=param_names, tau=0.0, sim_time=SIM_TIME, pre_time=PRE_TIME)

# Load experimental CSV data
csv_exp_file = r"C:\\Fariba_2025\\Ca_OpenCOR_Calibration_2025\\Experimental Data\\Exp_data_Exp5_fig4_Ca_uM_Base_200sec.csv"
sim_manager.load_experimental_data(csv_exp_file)

param_init_vals = sim_manager.get_init_param_vals()
print(f"Initial parameter values: {param_init_vals}")

# Define custom multipliers for min and max bound per parameter
lower_multipliers = [
    0.1, 0.1,           # alpha0, alpha1
    1, 0.7,             # V0, V1
    0.1, 0.1, 0.1, 0.1, # k_ryr0, k_ryr1, k_ryr2, k_ryr3
    0.7, 0.1, 0.1,      # Vm, km, gca
    1, 1, 1,            # F, R, T
    0.1, 0.1, 0.1,      # Jer, Ve, Ke
    0.1, 0.1, 0.1,      # Vp, Kp, gamma
    0.1, 0.1            # delta_SMC, k_RyR
]

upper_multipliers = [
    2, 10,             # alpha0, alpha1
    1, 1.34,           # V0, V1
    10, 10, 10, 10,    # k_ryr0, k_ryr1, k_ryr2, k_ryr3
    1.2, 10, 10,       # Vm, km, gca
    1, 1, 1,           # F, R, T
    2, 10, 10,         # Jer, Ve, Ke
    10, 10, 10,        # Vp, Kp, gamma
    10, 10             # delta_SMC, k_RyR
]
param_bounds = [
    (min(l * val, u * val), max(l * val, u * val))
    for val, l, u in zip(param_init_vals, lower_multipliers, upper_multipliers)
]

def optimization_cost(params):
    outputs, t = sim_manager.run_and_get_results(params)
    Ca_in_SMC = np.squeeze(outputs[0])   # model output

    # Match model time points exactly with experimental times
    matched_model_vals = []
    matched_exp_vals = []
    unmatched_times = []

    for i, et in enumerate(sim_manager.exp_times):
        idx = np.where(t == et)[0]
        if len(idx) == 0:
            unmatched_times.append(et)
        else:
            matched_model_vals.append(Ca_in_SMC[idx[0]])
            matched_exp_vals.append(sim_manager.exp_values[i])

    if len(unmatched_times) > 0:
        print(f"⚠️ Warning: {len(unmatched_times)} experimental times not found in model times.")
        print("Unmatched times:")
        for ut in unmatched_times:
            print(f"   {ut:.3f} sec")
    if not np.all(np.isfinite(matched_model_vals)):
        print("⚠️ Invalid output encountered.")
        return 1e6  # Penalize invalid outputs

    if np.any(np.abs(matched_model_vals) > 1e3):  # Too spiky
        print("⚠️ High output encountered.")
        return 1e4
    if len(matched_model_vals) == 0:
        print("❌ No experimental times matched model times. Cannot compute cost.")
        return 1e6  # big penalty

    # cost = MSE between *matched* model and experimental data
    cost = np.mean((np.array(matched_model_vals) - np.array(matched_exp_vals)) ** 2)
    print(f"🔄 Cost: {cost:.6f}, Params: {params}")

    # Save step results to run2 folder as CSV
    step_file = os.path.join(output_file_path, "optimization_progress.csv")
    step_row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cost": cost,
        **{name.split('/')[-1]: val for name, val in zip(param_names, params)}
    }
    step_df = pd.DataFrame([step_row])

    if os.path.exists(step_file):
        step_df.to_csv(step_file, mode="a", header=False, index=False)
    else:
        step_df.to_csv(step_file, index=False)

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

csv_output_file = os.path.join(output_file_path, "optimized_parameters.csv")
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Prepare simplified parameter names
param_names_simple = [name.split('/')[-1] for name in param_names]

# Compute bounds
min_bounds = [b[0] for b in param_bounds]
max_bounds = [b[1] for b in param_bounds]

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
if os.path.exists(csv_output_file):
    existing_df = pd.read_csv(csv_output_file)
    merged_df = pd.concat([existing_df, current_run_df], ignore_index=True)
else:
    merged_df = current_run_df

# Save file
merged_df.to_csv(csv_output_file, index=False)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_plot_path_full = os.path.join(output_file_path, f"Ca_concentration_plot_{timestamp}.png")

# ✅ Full or custom-time simulation for plotting
plot_sim_time = 2000   # change this to whatever total simulation time you want
plot_pre_time = 0      # change if you want a pre-time offset

sim_manager.sim_time = plot_sim_time
sim_manager.pre_time = plot_pre_time

outputs_full, t_full = sim_manager.run_and_get_results(optimal_params)
sim_manager.plot_model_out_and_experimental_data(outputs_full, t_full, output_plot_path_full)
