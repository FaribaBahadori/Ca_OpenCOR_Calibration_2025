import sys, csv, os, numpy as np, pandas as pd, datetime, warnings

from scipy.optimize import minimize
from Functions.model_utils import SimulationManager
from Functions.plot_eqs import plot_eqs
warnings.filterwarnings("ignore", category=UserWarning)

# Shows which Python program is running this script
print("Python executable:", sys.executable)

#from time import sleep 
## print("largest integer Python can handle:",sys.maxsize)
# Global simulation time settings
SIM_TIME = 2000
PRE_TIME = 2
run_counter = 0
# Physiological steady-state ranges (for filtering)
Ca_IN_RANGE = (0.05, 0.3)     # μM, cytosolic [Ca2+] range
Ca_SR_RANGE = (10, 100)          # μM, SR [Ca2+] range

os.chdir(r"C:/Fariba_2025/Ca_OpenCOR_Calibration_2025")
sys.path.insert(0, os.getcwd())
## print("Current working directory:", os.getcwd())
print("Calib_Script started running!")
#sleep(1000)

param_names = [
    'SMC_Par/l_1','SMC_Par/l_2','SMC_Par/l_3','SMC_Par/l_4','SMC_Par/l_5',
    'SMC_Par/l_m1','SMC_Par/l_m2','SMC_Par/l_m3','SMC_Par/l_m4','SMC_Par/l_m5',
    'SMC_Par/p_agonist','SMC_Par/Ca_E',
    'SMC_Par/t1_KCL','SMC_Par/t2_KCL',
    'SMC_Par/alpha0', 'SMC_Par/alpha1', 'SMC_Par/alpha2',
    'SMC_Par/V0', 'SMC_Par/V1',
    'SMC_Par/k_ryr0', 'SMC_Par/k_ryr1', 'SMC_Par/k_ryr2', 'SMC_Par/k_ryr3',
    'SMC_Par/Vm', 'SMC_Par/km', 'SMC_Par/gca',
    'SMC_Par/F', 'SMC_Par/R', 'SMC_Par/T',
    'SMC_Par/Jer', 'SMC_Par/Ve', 'SMC_Par/Ke', 
    'SMC_Par/Vp', 'SMC_Par/Kp', 'SMC_Par/gamma',
    'SMC_Par/delta_SMC', 'SMC_Par/k_RyR', 'SMC_Par/k_ipr'
]
vari_names=['SMC_Par/Ca_in_SMC0', 'SMC_Par/Ca_SR0', 'SMC_Par/y0']

output_names = ['SMC_Par/Ca_in_SMC']

working_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.abspath(os.path.join(working_dir, "Model", "Main_Coupled_SMC_Model.cellml"))


# Add timestamps for organized output folders

run_timestamp     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # run-level timestamp

# Base output directory for this session

output_run_path  = os.path.join(output_base_path, "run")
output_SS_path   = os.path.join(output_base_path, "SS_run")
plot_path        = os.path.join(output_base_path, "plots")
SSplot_path        = os.path.join(output_base_path, "SS_plots")

# Make sure all paths exist
os.makedirs(output_run_path, exist_ok=True)
os.makedirs(output_SS_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)
os.makedirs(SSplot_path, exist_ok=True)

sim_manager = SimulationManager(cal_param_names=param_names, cal_var_names=vari_names, tau=0.0, sim_time=SIM_TIME, pre_time=PRE_TIME)
print('t2')
#---------------------------------------------
'''
# Access each state individually
init_Ca_in_SMC = sim_manager.sim_object.data.states()['SMC_Par/Ca_in_SMC']
init_Ca_SR = sim_manager.sim_object.data.states()['SMC_Par/Ca_SR']
init_y = sim_manager.sim_object.data.states()['SMC_Par/y']
print("Initial values:Ca_in_SMC:", init_Ca_in_SMC, "Ca_SR:", init_Ca_SR, "y:", init_y)
'''
#---------------------------------------------
# Load experimental CSV data
csv_exp_file = r"C:\\Fariba_2025\\Ca_OpenCOR_Calibration_2025\\Experimental Data\\Exp_data_Exp5_fig4_Ca_uM_Base_2sec.csv"
sim_manager.load_experimental_data(csv_exp_file)
# Get initial parameter values from the model and print them
if 'param_init_vals' not in globals():
    param_init_vals = sim_manager.get_init_param_vals(sim_manager.call_param_names)
    print("parameter values not provided, using model defaults.")
param_names = sim_manager.call_param_names
'''
## print("Initial parameter values:")
for name, val in zip(param_names, param_init_vals):
    print(f"{name}: {val}", end="  ")
print()  # move to next line after printing all


# Get initial variable values from the model and print them
vari_init_vals  = sim_manager.get_init_param_vals(sim_manager.call_var_names)
vari_names = sim_manager.call_var_names
## print("Initial vari values:")
for name, val in zip(vari_names,vari_init_vals):
    print(f"{name}: {val}", end="  ")
print()  # move to next line after printing all
'''
# Define custom multipliers for min and max bound per parameter
lower_multipliers = [
    1,1,1,1,1,          # l_1, l_2, l_3, l_4', l_5
    1,1,1,1,1,          # l_m1, l_m2, l_m3, l_m4, l_m5
    1,1,                # p_agonist, Ca_E
    1,1,                # t1_KCL, t2_KCL
    1, 1, 1,            # alpha0, alpha1, alpha2
    1, 1,               # V0, V1
    1, 1, 1, 1,         # k_ryr0, k_ryr1, k_ryr2, k_ryr3
    1, 1, 1,            # Vm, km, gca
    1, 1, 1,            # F, R, T
    1, 1, 1,            # Jer, Ve, Ke
    1, 1, 1,            # Vp, Kp, gamma
    1, 1, 1             # delta_SMC, k_RyR, k_ipr
]

upper_multipliers = [
    1,1,1,1,1,         # l_1, l_2, l_3, l_4', l_5
    1,1,1,1,1,         # l_m1, l_m2, l_m3, l_m4, l_m5
    1,1,               # p_agonist, Ca_E
    1,1,               # t1_KCL, t2_KCL
    1, 1, 1,           # alpha0, alpha1, alpha2
    1, 1,              # V0, V1
    1, 1, 1, 1,        # k_ryr0, k_ryr1, k_ryr2, k_ryr3
    1, 1, 1,           # Vm, km, gca
    1, 1, 1,           # F, R, T
    1, 1, 1,           # Jer, Ve, Ke
    1, 1, 1,          # Vp, Kp, gamma
    1, 1, 1          # delta_SMC, k_RyR, k_ipr
]
param_bounds = [
    (min(l * val, u * val), max(l * val, u * val))
    for val, l, u in zip(param_init_vals, lower_multipliers, upper_multipliers)
]
# optimization_cost function returns cost for given params
def optimization_cost(params):
    
    global csv_progress_file
    global run_counter
    run_counter += 1
    print(f"\n--- Run #{run_counter} ---")   # <-- shows which parameter set is being tried

    # Run model to get steady-state values ---
    outputs, t, Ca_in_SMC_val, Ca_SR_val, y_val = sim_manager.run_and_get_results(params, SS_vals=True)

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
        print(f"Warning: {len(unmatched_times)} experimental times not found in model times.")
        print("Unmatched times:")
        for ut in unmatched_times:
            print(f"   {ut:.3f} sec")
    if not np.all(np.isfinite(matched_model_vals)):
        print("! Invalid output encountered.")
        return 1e6  # Penalize invalid outputs

    if np.any(np.abs(matched_model_vals) > 1e3):  
        print("High output encountered.")
        return 1e4
    if len(matched_model_vals) == 0:
        print("No experimental times matched model times. Cannot compute cost.")
        return 1e6  # big penalty

    # cost = MSE between *matched* model and experimental data
    sim_vals = np.array(matched_model_vals)
    exp_vals = np.array(matched_exp_vals)
    mse = np.mean((np.array(sim_vals) - exp_vals) ** 2)
    # Smoothness penalty: large if simulation oscillates too much
    smooth_penalty = np.var(np.diff(sim_vals))
    # λ controls how strongly to penalize oscillations (try 0.1–10)
    lambda_smooth = 1.0

    ## cost = mse + lambda_smooth * smooth_penalty
    cost = mse*(1+smooth_penalty*0)

    ## print(f" Cost: {cost:.6f}, Params: {params}")
    print(f" Cost: {cost:.6f}")
    
    #
    # === Save per-run results to progress CSV ===
    step_row = {
        "run_counter": run_counter,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cost": cost,
        "SS_Ca_in_SMC0": Ca_in_SMC_val,
        "SS_Ca_SR0": Ca_SR_val,
        "SS_y0": y_val,
    }

    # Add all current parameter values
    for name, val in zip(param_names, params):
        step_row[name.split('/')[-1]] = val

    step_df = pd.DataFrame([step_row])

    # Append or create the CSV file
    if os.path.exists(csv_progress_file):
        step_df.to_csv(csv_progress_file, mode="a", header=False, index=False)
    else:
        step_df.to_csv(csv_progress_file, index=False)
    #

    # Save step results to run2 folder as CSV
    ##step_file = os.path.join(output_file_path, "optimization_progress.csv")
    ##step_row = {
    ##    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ##    "cost": cost,
    ##    "SS_Ca_in_SMC0": Ca_in_SMC_val, "SS_Ca_SR0": Ca_SR_val, "SS_y0": y_val,
    ##    **{name.split('/')[-1]: val for name, val in zip(param_names, params)}
    ##}
    ##step_df = pd.DataFrame([step_row])

    ##if os.path.exists(step_file):
    ##    step_df.to_csv(step_file, mode="a", header=False, index=False)
    ##else:
    ##    step_df.to_csv(step_file, index=False)

    return cost
csv_progress_file = os.path.join(output_run_path, "optimized_progress.csv")
# Run the optimizer
res = minimize(
    optimization_cost,
    param_init_vals,
    method='Powell',  # Slower but more robust to bad gradients
    bounds=param_bounds,
    options={'disp': True, 'maxiter': 100}
)
optimal_params = res.x

print(f"Optimal parameters: {optimal_params}")

csv_output_file   = os.path.join(output_run_path, "optimized_parameters.csv")
csv_output_SSfile = os.path.join(output_SS_path, "SS_optimized_parameters.csv")
#csv_progress_file = os.path.join(output_run_path, "optimized_progress.csv")


# Prepare simplified parameter names
param_names_simple = [name.split('/')[-1] for name in param_names]

# Compute bounds
min_bounds = [b[0] for b in param_bounds]
max_bounds = [b[1] for b in param_bounds]

# Only add the steady-state values (Ca_in_SMC_val, Ca_SR_val, y_val) into row_data in the next lines
yy, t, Ca_in_SMC_val, Ca_SR_val, y_val = sim_manager.run_and_get_results(optimal_params, SS_vals=True) 
# Prepare row as dictionary
row_data = {
    'run_id': run_timestamp,
    'run_counter': run_counter,
 ##   'sim_time': sim_manager.sim_time,
 ##   'pre_time': sim_manager.pre_time,
    'final_cost': res.fun,
    'SS_Ca_in_SMC0': Ca_in_SMC_val,
    'SS_Ca_SR0': Ca_SR_val,
    'SS_y0': y_val,
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

# --- Check if steady-state values fall in physiological ranges ---
in_SS_range = (Ca_IN_RANGE[0] <= Ca_in_SMC_val <= Ca_IN_RANGE[1]) and \
               (Ca_SR_RANGE[0] <= Ca_SR_val <= Ca_SR_RANGE[1])

if in_SS_range:
    print(f"Steady-state OK: Ca_cyt={Ca_in_SMC_val:.3f} μM, Ca_SR={Ca_SR_val:.1f} μM")
    # Append or create file
    if os.path.exists(csv_output_SSfile):
        SS_existing_df = pd.read_csv(csv_output_SSfile)
        SS_merged_df = pd.concat([SS_existing_df, current_run_df], ignore_index=True)
    else:
        SS_merged_df = current_run_df

    # Save file
    SS_merged_df.to_csv(csv_output_SSfile, index=False)
else:
    print(f"Steady-state out of range: Ca_cyt={Ca_in_SMC_val:.3f} μM, Ca_SR={Ca_SR_val:.1f} μM")

plot_filename     = f"[Ca]_plot_{run_timestamp}.png"
plot_SS_filename  = f"SS_[Ca]_plot_{run_timestamp}.png"

output_plot_path_full    = os.path.join(plot_path, plot_filename)
output_SSplot_path_full  = os.path.join(SSplot_path, plot_SS_filename)

# Full or custom-time simulation for plotting
plot_sim_time = 2000   #total simulation time
plot_pre_time = 0      # a pre-time offset

sim_manager.sim_time = plot_sim_time
sim_manager.pre_time = plot_pre_time

outputs_full, t_full = sim_manager.run_and_get_results(optimal_params, SS_vals=False)

sim_manager.plot_model_out_and_experimental_data(outputs_full, t_full, output_plot_path_full)
if in_SS_range:
    sim_manager.plot_model_out_and_experimental_data(outputs_full, t_full, output_SSplot_path_full)

"""# --- Prepare folder for plots ---
plot_folder = os.path.join(os.path.dirname(output_file_path), "run22")
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# --- Prepare input dictionaries for plot_eqs ---
optimal_params_dict = {name: val for name, val in zip(param_names, optimal_params)}
vari_init_vals = sim_manager.get_init_param_vals(sim_manager.call_var_names)
vari_init_vals_dict = {name: val for name, val in zip(sim_manager.call_var_names, vari_init_vals)}

# --- Call plot_eqs for optimized parameter set ---
print(f"Current params: {optimal_params_dict}, Cost (MSE): {res.fun:.6f}")
plot_eqs(optimal_params_dict, vari_init_vals_dict, plot_folder)"""