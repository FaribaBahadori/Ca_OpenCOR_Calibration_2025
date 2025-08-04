How to Run:

To run the code, you'll need OpenCOR installed on your computer. I'm using the latest version (0.8.3), which you can download [from this link].

Here's a quick summary of the steps I followed:

Install OpenCOR (v0.8.3).
Install VS Code and open the main Python file: "Single_Objective_optimization.py"
Install missing libraries: When running the code, I received errors about missing libraries. I installed each one by typing pip install library_name in the terminal inside VS Code. I think some of the libraries I used are: csv, sys, os, numpy, pandas, and datetime.
Run the code: In the VS Code terminal, I ran this command: Executing task: 
 C:\Users\fbir042\codes\OpenCOR\pythonshell.bat C:\Fariba_2025\Ca_OpenCOR_Calibration_2025\Single_Objective_optimization.py
The first path points to the pythonshell.bat file in my OpenCOR folder on my computer.
The second path is where the Python script is located on my computer.

********************************************************************88


Run the main script: Single_Objective_optimization.py
 It performs parameter calibration using the CellML model.

Output plots and results are saved in:
Ca_OpenCOR_Calibration_2025/outputs/Single_objective/run2/

After each run, a row is added to a CSV file in the same folder.
 This includes the optimized cost, updated parameters, and their bounds.
Example plots and CSV outputs from previous runs are also provided in the folder.



