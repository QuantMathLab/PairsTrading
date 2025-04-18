Description of files sent with the project.
PDF files
1)	TS Rahul Yadav REPORT.pdf
Project report in PDF format

Files required to run the code
•	Main.py
This is the main file that needs to run to get the final outputs. It will ask for the location in which all these files are saved so that it can import all the necessary files.
•	LinearRegresion.py
This is the file in which numerical methods are coded from first principles and then imported in ‘Main.py’ and further used
•	Backtesting.py
This is the file in which backtesting trading signals and performance-related functions are written. This file is called a module in ‘Main.py’ and is further used.
•	download_data.py
This file downloads data from Yahoo Finance. This file is called a module in ‘Main.py’ and is further used.
•	India 10-Year Bond Yield Historical Data.csv
Data used for risk-free rate calculation.
Output Folders
1)	Output
It saves all the intermediate and final results including Input data, output of stationarity, cointegration, ECM, and backtesting in Excel files. Also, it contains an Excel file in which the results of each pair are presented in an Excel workbook. It contains result for all the investigated pairs
2)	Charts
It contains charts of the backtesting performance of all the pairs.
3)	Results of finally selected pairs are saved like above with different values of Z. 
e.g. Output_Z_0.7 contains the result of the final selected pair with Z = 0.7
and Charts_Z_0.7 contains backtesting performance charts.

Instructions to run the code to generate the final output
Make sure that ‘Files required to run the code’ are saved in one folder. 
Open ‘Main.py’ and at lines 54 to 62 specify the stocks required to be run accordingly.
At line 251 specify the value of Z in the variable k in the code.
Then run the ‘Main.py’ file. It will ask for the file path. Give the file path where all the required files are saved. Code will run and create output folders as described above save results in Excel files and create charts.
