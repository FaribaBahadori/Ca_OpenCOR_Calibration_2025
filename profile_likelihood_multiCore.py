from scipy.optimize import minimize
import os
from Functions.model_utils import SimulationManager
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

param_names = ['SMC_Par/delta_SMC', 'SMC_Par/k_RyR']      # This list can be appended
output_names = ['model_parameters1/Ca_in_SMC', 'SMC_Par/Ca_in_SMC_dig']

working_dir = os.path.join(os.path.dirname(__file__))
model_path = os.path.join(working_dir, "Models/Main_Coupled_SMC_Model.cellml")

# Setting the output directory
output_file_path = "outputs/profile_likelihood_multiCore/run8/"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)


sim_manager = SimulationManager(cal_param_names=param_names)
param_init_vals = sim_manager.get_init_param_vals()
param_vals_mins = [0.25 * var for var in param_init_vals]
param_vals_maxs = [5 * var for var in param_init_vals]
param_bounds = [(min_val, max_val) for min_val, max_val in zip(param_vals_mins, param_vals_maxs)]

experimental_data = sim_manager.get_expermental_data()
ground_truth = sim_manager.get_ground_truth()

print(f"Initial parameter values: {param_init_vals}")

def optimization_cost(params, z_hat):
    return sim_manager.cost_function(params, z_hat, verbose=True)

options = {'gtol': 1e-6, 'ftol': 1e-6, 'disp': True}

# res = minimize(optimization_cost, param_init_vals, args=[ground_truth], method='L-BFGS-B',
#                    options=options, bounds=param_bounds)
    
# theta_res = res.x
# best_fun = res.fun

theta_res = [3.37037037e-06, 9.07519967e-05, 2.49817614e-05] #, Best function value: 5.807651467819674e-06
best_fun = 5.807651467819674e-06

print(f"Optimal parameters: {theta_res}, Best function value: {best_fun}")

#    Compute profile likelihood
samples_per_profile = 40
min_factor = [0.5, 0.3, 0.5]
max_factor = [1.5, 1.8, 1.5]

#    Iterating over each variable and each sample of the profile
profiles = np.zeros( (len(theta_res),samples_per_profile) )
ranges = np.zeros( (len(theta_res),samples_per_profile) )

def F_likelihood(params, z_hat, param_idx, current_value):
    return sim_manager.Likelihood_cost_function(params, z_hat, param_idx, current_value, verbose=False)

def compute_profile_likelihood_sample(param_idx, current_value, theta_res, param_vals_mins, param_vals_maxs, ground_truth, options):
    theta_init = np.delete(theta_res, param_idx)
    bounds = [(param_vals_mins[i], param_vals_maxs[i]) for i in range(len(theta_res)) if i != param_idx]

    res = minimize(F_likelihood, theta_init, args=(ground_truth, param_idx, current_value), method='L-BFGS-B', options=options, bounds=bounds)
    return current_value, res.fun

# Outer loop: iterate over each parameter
for param_idx in [0, 1, 2]: #range(1, len(theta_res)-1):
    param_value = theta_res[param_idx]
    min_range = min_factor[param_idx] * param_value
    max_range = max_factor[param_idx] * param_value

    param_range = np.linspace(min_range, max_range, samples_per_profile)
    ranges[param_idx, :] = param_range

    profile_values = np.zeros(samples_per_profile)

    # Use ProcessPoolExecutor to parallelize across profile points
    num_cores = 18
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(compute_profile_likelihood_sample, param_idx, val, theta_res, param_vals_mins, param_vals_maxs, ground_truth, options)
            for val in param_range
        ]

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f'Param: {param_names[param_idx]}')):
            current_value, cost = future.result()
            idx_sample = np.where(param_range == current_value)[0][0]
            profile_values[idx_sample] = cost
            print(f"Profile likelihood for parameter {param_names[param_idx]} - value {current_value}: {cost}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, profile_values, label='Profile Likelihood')
    plt.xlabel(param_names[param_idx])
    # plt.ylim(-0.05, 0.3)
    plt.ylabel('Cost')

    plt.title(f'Profile likelihood for {param_names[param_idx]}')
    plt.plot(theta_res[param_idx], best_fun, 'rx', label='Best Fit')
    threshold_cost = 0.01
    plt.axhline(y=threshold_cost, color='purple', linestyle='-.', linewidth=1, label=f'Threshold Cost ({threshold_cost:.4f})')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_file_path, f'profile_likelihood_{param_names[param_idx].replace("/", "_")}.png'))
    plt.close()