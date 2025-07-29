import numpy as np
import pandas as pd

from Functions.OpenCor_Py.opencor_helper import SimulationHelper
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations

# Define stimulus parameters
class SimulationManager:
    def __init__(self,
                 model_path="./Models/Main_Coupled_SMC_Model.cellml",
                 dt=1,
                 sim_time=1800,
                 pre_time=266,
                 solver_info=None,
                 tau=0,
                 output_names = ['model_parameters1/Ca_in_SMC', 'SMC_Par/Ca_in_SMC_dig'],
                 cal_param_names=['SMC_Par/delta_SMC', 'SMC_Par/k_RyR'],
                 feature_names=['max_Ca', 'time_constant_to_max', 'time_to_return_to_baseline']):

        if solver_info is None:
            solver_info = {'MaximumStep': 0.1, 'MaximumNumberOfSteps': 5000}

        self.pre_time = pre_time
        self.sim_time = sim_time + pre_time
        self.output_names = output_names
        self.call_param_names = cal_param_names
        self.feature_names = feature_names

        self.sim_object = SimulationHelper(model_path, dt, sim_time, solver_info=solver_info, pre_time=pre_time)

        self.stim_params_names = ['Environment/tau']
        self.stim_param_vals = [tau]
        self.sim_object.set_param_vals(self.stim_params_names, self.stim_param_vals)

    def run_and_get_results(self, param_vals):
        self.sim_object.set_param_vals(self.call_param_names, param_vals)
        self.sim_object.reset_states()
        out = self.sim_object.run()

        if out:
            y = self.sim_object.get_results(self.output_names)
            t = self.sim_object.tSim - self.pre_time
        else:
            print(f"Simulation failed to run with parameters = {param_vals}.")
            y = np.zeros((len(self.output_names), self.sim_time // self.sim_object.dt))
            t = np.arange(0, self.sim_time, self.sim_object.dt) - self.pre_time

        self.sim_object.reset_and_clear()
        return y, t

    def feature_extraction(self, outputs, t):
        outputs = np.squeeze(outputs) * 1000
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

    def cost_function(self, param_vals_current, z_hat=None, verbose=True):
        outputs, t = self.run_and_get_results(param_vals_current)
        Ca_in_SMC = np.squeeze(outputs[0])
        Ca_in_SMC_dig = np.squeeze(outputs[1])
        mse = np.mean((Ca_in_SMC - Ca_in_SMC_dig)**2)
        if verbose:
            print(f"Current params: {param_vals_current}, Cost (MSE): {mse:.6f}")
        return mse

    def Likelihood_cost_function(self, param_vals_current, z_hat, param_idx, current_value, verbose=True):
        theta_full = np.insert(param_vals_current, param_idx, current_value)
        if verbose:
            print(f"Running simulation with parameters: {theta_full}")

        outputs, t = self.run_and_get_results(theta_full)
        z_hat = np.atleast_1d(z_hat)
        z_model = self.feature_extraction(outputs, t)

        cost = ((z_model[0] - z_hat[0]) / z_hat[0])**2 + ((z_model[1] - z_hat[1]) / z_hat[1])**2 + ((z_model[2] - z_hat[2]) / z_hat[2])**2

        if verbose:
            print(f"Current parameters: {theta_full}, Model output: {z_model}, Ground truth: {z_hat}, cost: {cost}")

        return cost

    def get_init_param_vals(self):
        return self.sim_object.get_init_param_vals(self.call_param_names)


    def plot_model_out_and_experimental_data(self, outputs, t, output_file_path):
        Ca_in_SMC = np.squeeze(outputs[0])
        Ca_in_SMC_dig = np.squeeze(outputs[1])

        plt.figure(figsize=(10, 6))
        plt.plot(t, Ca_in_SMC, label='Model Output: Ca_in_SMC', color='blue')
        plt.plot(t, Ca_in_SMC_dig, label='Synthetic Reference: Ca_in_SMC_dig', color='red', linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Ca Concentration (uM)")
        plt.legend()
        plt.grid()
        plt.title("Model Output vs Synthetic Experimental Data")
        plt.savefig(output_file_path)
        plt.close()
