import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
# Ensure Python can find sympy when running via OpenCOR
sympy_path = os.path.expandvars(r"%APPDATA%\Python\Python312\site-packages")
if sympy_path not in sys.path:
    sys.path.insert(0, sympy_path)
# Ensuring Python can find helper modules
# ----------------------------
# 1. Adding OpenCor_Py folder inside this project (to path for OpenCOR Python)
sys.path.append(os.path.join(os.path.dirname(__file__), "OpenCor_Py"))
from .OpenCor_Py.opencor_helper import SimulationHelper  #Importing helper modules from local OpenCor_Py
# from Functions.OpenCor_Py.opencor_helper import SimulationHelper (Importing helper modules from Functions folder, if needed)

# 2. Adding the project root to sys.path (to find Functions folder modules), so Python finds Functions and steady_state_solver
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------------------
# Importing helper modules
from Functions.steady_state_solver_test1 import steady_state_smct

# Matplotlib backend for saving figures without display
matplotlib.use('Agg')

# Define stimulus parameters
class SimulationManager:
    def __init__(self,
                 model_path="./Models/Main_Coupled_SMC_Model.cellml",
                 dt=1,
                 sim_time=2000,
                 pre_time=0,
                 solver_info=None,
                 tau=0,
                 output_names = ['SMC_Par/Ca_in_SMC'],
                 cal_param_names=['SMC_Par/delta_SMC', 'SMC_Par/k_RyR'
                                  ],
                 cal_var_names=['SMC_Par/Ca_in_SMC0', 'SMC_Par/Ca_SR0', 'SMC_Par/y0'],
                 feature_names=['max_Ca', 'time_constant_to_max', 'time_to_return_to_baseline']):

        # Default solver parameters if not provided
        if solver_info is None:
            solver_info = {'MaximumStep': 0.1, 'MaximumNumberOfSteps': 5000}

        self.pre_time = pre_time
        self.sim_time = sim_time + pre_time
        self.output_names = output_names
        self.call_param_names = cal_param_names
        self.call_var_names = cal_var_names
        self.feature_names = feature_names
        
        self.sim_object = SimulationHelper(model_path, dt, sim_time, solver_info=solver_info, pre_time=pre_time)
        
        
        self.stim_params_names = ['Environment/tau']
        self.stim_param_vals = [tau]
        self.sim_object.set_param_vals(self.stim_params_names, self.stim_param_vals)
        ############################
       # def init_experimental_data(self, verbose=False):
        #    csv_file_path = 'Experimental Data/Exp_data_Exp5_fig4_Ca_uM_Base_200sec.csv'
        #    experimental_data_df = pd.read_csv(csv_file_path)    
        ############################
    def load_experimental_data(self, csv_Exp_file):
        times = []
        values = []
        with open(csv_Exp_file, 'r') as f:
            next(f)  # skip header
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split(',')  # split by comma
                times.append(float(parts[0]))
                values.append(float(parts[1]))
        self.exp_times = np.array(times)
        self.exp_values = np.array(values)
        print(f" Loaded {len(times)} experimental points from {csv_Exp_file}")


    def run_and_get_results(self, param_vals, SS_vals=False):
        print('test 1')

        self.sim_object.set_param_vals(self.call_param_names, param_vals)
        print('test 2')

        # Get initial values directly from the model (before running simulation)
        vari_init_vals_list = self.sim_object.get_init_param_vals(self.call_var_names)
        vari_init_vals = dict(zip([name.split('/')[-1] for name in self.call_var_names], vari_init_vals_list))


        Ca_in_SMC_val = vari_init_vals['Ca_in_SMC0']
        Ca_SR_val     = vari_init_vals['Ca_SR0']
        y_val         = vari_init_vals['y0']
        print(f'test 3: vari_init_vals={vari_init_vals}')


        # Get initial parameter values as a list
        param_vals_list = self.sim_object.get_init_param_vals(self.call_param_names)

        # Convert to dictionary: {'SMC_Par/delta_SMC': 0.05, ...}
        param_vals_dict = dict(zip(self.call_param_names, param_vals_list))

        # Update initial values using steady-state function
        Ca_in_SMC_val, Ca_SR_val, y_val = steady_state_smct(
            params=param_vals_dict,
            vari_init_vals={'Ca_in_SMC0': Ca_in_SMC_val, 'Ca_SR0': Ca_SR_val, 'y0': y_val}, return_extra=False
        )

        # Update the simulation object with these steady-state values
        self.sim_object.set_param_vals(
            self.call_var_names,
            [Ca_in_SMC_val, Ca_SR_val, y_val]
         )
        self.sim_object.reset_states()  # reset simulation with these states
        
        
        # Run simulation
        success = self.sim_object.run()

        if success:

            yy = self.sim_object.get_results(self.output_names)
            t = self.sim_object.tSim - self.pre_time
        else:
            print(f"Simulation failed to run with parameters = {param_vals}.")
            yy = np.zeros((len(self.output_names), self.sim_time // self.sim_object.dt))
            t = np.arange(0, self.sim_time, self.sim_object.dt) - self.pre_time

        self.sim_object.reset_and_clear()
        if SS_vals:
            return  yy, t, Ca_in_SMC_val, Ca_SR_val, y_val
        else:
            return yy, t

    def feature_extraction(self, var_data, t):
        outputs = np.squeeze(var_data) * 1000
        z1 = np.max(outputs, axis=0)
        z2_idx = np.argmax(outputs)
        if z2_idx > len(t) - 1:
            z2_idx = len(t) - 1
        z2 = t[z2_idx] / 1000

        z3_idx = np.where(np.array(outputs[z2_idx:]) <= 0.01 * z1)[0]
        if len(z3_idx) == 0:
            slope = (outputs[-1] - outputs[-2]) / (t[-1] - t[-2])
            dt = (0.01*z1 - outputs[-1]) / slope
            z3 = (t[-1]+dt) / 1000 - z2
        else:
            z3_idx = z3_idx[0] + z2_idx
            z3 = t[z3_idx] / 1000 - z2

        return [z1, z2, z3]

    # Cost function for optimisation
    def cost_function(self, param_vals_current, z_hat=None, verbose=True):
        sim_results, t = self.run_and_get_results(param_vals_current, SS_vals=False)
        Ca_in_SMC = np.squeeze(sim_results[0])      # model output

        # --- REPLACE HERE ---
        matched_indices = []
        unmatched_times = []

        for i, et in enumerate(self.exp_times):
            idx = np.where(t == et)[0]
            if len(idx) == 0:
                unmatched_times.append(et)  # To store missing times
            else:
                matched_indices.append(idx[0])

        if len(unmatched_times) > 0:
            print(f" Warning: {len(unmatched_times)} experimental times not found in model times: {unmatched_times}")

        Ca_in_SMC_matched = [Ca_in_SMC[i] for i in matched_indices]
        exp_values_matched = [self.exp_values[i] for i, et in enumerate(self.exp_times) if et in t]

        mse = np.mean((np.array(Ca_in_SMC_matched) - np.array(exp_values_matched))**2)

        if verbose:
            print(f"Cost (MSE): {mse:.6f}")
            ## print(f"Current params: {param_vals_current}, Cost (MSE): {mse:.6f}")
        return mse

    def Likelihood_cost_function(self, param_vals_current, z_hat, param_idx, current_value, verbose=True):
        theta_full = np.insert(param_vals_current, param_idx, current_value)
        if verbose:
            print(f"Running simulation with parameters: {theta_full}")

        outputs, t = self.run_and_get_results(theta_full, SS_vals=False)
        z_hat = np.atleast_1d(z_hat)
        z_model = self.feature_extraction(outputs, t)

        cost = ((z_model[0] - z_hat[0]) / z_hat[0])**2 + ((z_model[1] - z_hat[1]) / z_hat[1])**2 + ((z_model[2] - z_hat[2]) / z_hat[2])**2

        if verbose:
            print(f"Current parameters: {theta_full}, Model output: {z_model}, Ground truth: {z_hat}, cost: {cost}")

        return cost
    def get_init_param_vals(self, init_names):
        return self.sim_object.get_init_param_vals(init_names)  


    def plot_model_out_and_experimental_data(self, var_data, t, output_file_path):
        Ca_in_SMC = np.squeeze(var_data[0])

        plt.figure(figsize=(10, 6))
        plt.plot(t, Ca_in_SMC, label='Model Output: Ca_in_SMC', color='blue')
        plt.plot(self.exp_times, self.exp_values, 'ro', label='Experimental Data')  # CSV data
        plt.xlabel("Time (s)")
        plt.ylabel("Ca Concentration (uM)")
        plt.legend()
        plt.grid()
        plt.title("Model Output vs Experimental Data")
        plt.savefig(output_file_path)
        plt.close()


