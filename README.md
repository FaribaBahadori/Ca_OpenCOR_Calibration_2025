2025_08_04

How to Run:

To run the code, you'll need OpenCOR installed on your computer. I'm using the latest version (0.8.3), which you can download: https://opencor.ws/downloads/

Here's a quick summary of the steps I followed:

1. Install OpenCOR (v0.8.3).
2. Install VS Code and open the main Python file: "Single_Objective_optimization.py"
3. Install missing libraries: When running the code, I received errors about missing libraries. I installed each one by typing pip install library_name in the terminal inside VS Code. I think some of the libraries I used are: csv, sys, os, numpy, pandas, and datetime.
4. Run the code: In the VS Code terminal, I ran this command: Executing task:
   
   C:\Users\fbir042\codes\OpenCOR\pythonshell.bat C:\Fariba_2025\Ca_OpenCOR_Calibration_2025\Single_Objective_optimization.py

  - The first path points to the pythonshell.bat file in my OpenCOR folder on my computer.
  - The second path is where the Python script is located on my computer.
  - You can adjust these paths based on where the files are saved on your computer.
To run the command, I navigated to the folder containing pythonshell.bat by typing the commands below, pressing Enter after each line:

PS C:\Users\fbir042> cd codes

PS C:\Users\fbir042\codes> cd OpenCOR

PS C:\Users\fbir042\codes\OpenCOR>

Then I ran the script by typing:

.\pythonshell.bat C:\Fariba_2025\Ca_OpenCOR_Calibration_2025\Single_Objective_optimization.py

5. The terminal should then output something like this:
   
?? Script started running!

Initial parameter values: [0.05, 0.25, -60.0, -30.0, 0.0072, 0.334, 0.5, 38.0, -50.0, 12.0, 16.0, 96485.0, 8345.0, 310.0, 0.1, 4.5, 0.1, 4.5, 0.4, 5.5, 27.5, 0.116477, 0.05, 5.0]

?? Cost: 0.114601, Params: [ 5.00000e-02  2.50000e-01 -6.00000e+01 -3.00000e+01  7.20000e-03

  3.34000e-01  5.00000e-01  3.80000e+01 -5.00000e+01  1.20000e+01
  1.60000e+01  9.64850e+04  8.34500e+03  3.10000e+02  1.00000e-01
  4.50000e+00  1.00000e-01  4.50000e+00  4.00000e-01  5.50000e+00
  2.75000e+01  1.16477e-01  5.00000e-02  5.00000e+00]...
 
********************************************************************

2025_07_29

Run the main script: Single_Objective_optimization.py
 It performs parameter calibration using the CellML model.

Output plots and results are saved in:
Ca_OpenCOR_Calibration_2025/outputs/Single_objective/run2/

After each run, a row is added to a CSV file in the same folder.
 This includes the optimized cost, updated parameters, and their bounds.
Example plots and CSV outputs from previous runs are also provided in the folder.



