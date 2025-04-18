# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 11:42:17 2022

@author: yrahu
"""

#import module with numerical method
data_file_loc = input('Enter folder location where all files are saved :\n') #D:\CQF\FinalProject_Code - Jan 2022
import os
import sys 
sys.path.append(os.path.abspath(data_file_loc))
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns	
from itertools import combinations
import numpy as np
import pandas as pd
import warnings


### Self coded modules########
import LinearRegresion_v4 as lr
import download_data as d
import Backtesting_v1 as bt


# Suppress warnings
warnings.filterwarnings("ignore")

#get file location and import data
#data_file_loc = input('Enter folder location of Final_Data.xlsx :\n') #D:\CQF\FinalProject_Code\Data\
#excel_file = pd.ExcelFile(data_file_loc + '\\Final_Data.xlsx')

#get 10 year bond yield as risk free rate
risk_free_rate = pd.read_csv(data_file_loc + '\\India 10-Year Bond Yield Historical Data.csv')
risk_free_rate['Date'] = pd.to_datetime(risk_free_rate['Date'])
risk_free_rate = risk_free_rate.set_index(risk_free_rate['Date'], drop = True)

#create save location at same location if does not exist
save_loc = data_file_loc + "\\Output_Z_1"
if not os.path.exists(save_loc):
    os.makedirs(save_loc)
    
#create save location at same location for charts if does not exist
save_loc_chart = data_file_loc + "\\Charts_Z_1\\"
if not os.path.exists(save_loc_chart):
    os.makedirs(save_loc_chart)

writer = pd.ExcelWriter(save_loc + "\\Output_Z_1.xlsx")
#get required data
#specify stocks 
#stocks = ['ADANIPORTS.NS','ASIANPAINT.NS','AXISBANK.NS','BAJAJ-AUTO.NS','BAJFINANCE.NS','BAJAJFINSV.NS','BPCL.NS','BHARTIARTL.NS','BRITANNIA.NS','CIPLA.NS','COALINDIA.NS','DIVISLAB.NS','DRREDDY.NS','EICHERMOT.NS','GRASIM.NS','HCLTECH.NS','HDFCBANK.NS','HDFCLIFE.NS','HEROMOTOCO.NS','HINDALCO.NS','HINDUNILVR.NS','HDFC.NS','ICICIBANK.NS','ITC.NS','IOC.NS','INDUSINDBK.NS','INFY.NS','JSWSTEEL.NS','KOTAKBANK.NS','LT.NS','M&M.NS','MARUTI.NS','NTPC.NS','NESTLEIND.NS','ONGC.NS','POWERGRID.NS','RELIANCE.NS','SBILIFE.NS','SHREECEM.NS','SBIN.NS','SUNPHARMA.NS','TCS.NS','TATACONSUM.NS','TATAMOTORS.NS','TATASTEEL.NS','TECHM.NS','TITAN.NS','UPL.NS','ULTRACEMCO.NS','WIPRO.NS']
stocks = ['BAJFINANCE.NS','BAJAJFINSV.NS']
#specify start and end date to determine model
start_train = '2021-01-01'
end_train = '2022-06-30'
#specify start and end date to determine model
start_test = '2022-07-01'
end_test = '2023-12-30'

#download all data
#data contains only close prices and master_data contains all the remaining attributes
data, master_data = d.download_data(stocks, start_train, end_test)

# Print columns with blanks
columns_with_blanks = data.columns[data.isna().sum() > 50].tolist()
print("Stocks having na values > 50:", columns_with_blanks)
# Remove columns with blanks
data = data.drop(columns=columns_with_blanks)
# Remove rows with na
data = data.dropna()

writer_data = pd.ExcelWriter(save_loc + "\\Input_Data_Z_1.xlsx")
data.to_excel(writer_data, sheet_name = 'Close_Prices_Data') 
master_data.to_excel(writer_data, sheet_name = 'Master_Date') 
writer_data.save()
#Sort data by dates
data.reset_index(inplace=True)
data = data.sort_values(by='Date')

#splirt data intotrain and test
data_train = data[(data['Date'] >= start_train) & (data['Date'] <= end_train)]
data_test = data[(data['Date'] >= start_test) & (data['Date'] <= end_test)]

#create list and dictionaries to hold results
adf_results = {}
coint_results = {}
coint_stock_pairs = []
stationary_stocks = []
non_stationary_stocks = []
vece_results = []


#remove stocks for which historical data is not present
stocks = [x for x in stocks if x not in columns_with_blanks]

#check which stocks are not stationary in train period
for stock in stocks:
    #run ADF test
    adf_result = lr.ADF_test(data_train[stock],1)
    adf_results[stock] = adf_result
    if adf_result[0] < adf_result[1]['5% critical value:']:
        stock_statinary = "Yes"
        print ('{}'.format(stock) + " is statinary \n")
        stationary_stocks.append(stock)
    else:
        print ('{}'.format(stock) + " is not statinary \n")
        non_stationary_stocks.append(stock)
        
#convert all stionarity reslts in data frmae and export 
df_statinarity_results = pd.DataFrame.from_dict(adf_results, orient= 'index')  
df_statinarity_results = df_statinarity_results.reset_index()
df_statinarity_results.columns = ['Ticker','Test_Statistic','Critical Valus','AIC','BIC','No of lags Used']  
#export result
df_statinarity_results.to_excel(writer, sheet_name = 'stationarity_results', index = False) 

#plot heat map from staionarity results
# Create a DataFrame
heatmap_data = df_statinarity_results[['Ticker','Test_Statistic']]

# Pivot the DataFrame to create a matrix for the heatmap
heatmap_data = heatmap_data.pivot_table(index='Ticker', columns=None, values='Test_Statistic')

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".4f", cbar_kws={'label': 'Test_Statistic'})
plt.title('Stationarity Test Statistic heat map')
plt.show()

#make pairs of non-stationary stocks        
stock_pairs = list(combinations(non_stationary_stocks,2))

#check for co-integration of all pairs
for stock_pair in stock_pairs:
    #check if both x and y are same
    if stock_pair[0] == stock_pair[1]:
        pass
    else:
        #run ADF test
        model_op = lr.ols_linear_regression(data_train[stock_pair[0]],data_train[stock_pair[1]])
        residuals = model_op[2]
        #run ADF test
        coint_result = lr.ADF_test(residuals,1)
        coint_results[stock_pair] = coint_result
        if coint_result[0] < coint_result[1]['5% critical value:']:
            print ('{}'.format(stock_pair) + " stock pair is co-integrated \n")
            coint_stock_pairs.append(stock_pair)
        else:
            print ('{}'.format(stock_pair) + " stock pair is not co-integrated \n")
            
#conver all cointegration reslts in data frmae and export 
df_coint_results = pd.DataFrame.from_dict(coint_results, orient= 'index')  
df_coint_results = df_coint_results.reset_index()
df_coint_results.columns = ['TickerPairs','Test_Statistic','Critical Valus','AIC','BIC','No of lags Used']  
#export result
df_coint_results.to_excel(writer, sheet_name = 'co_intergration_results', index = False) 

# =============================================================================
# # plot as a heatmap of coint results
# heatmap_coint = df_coint_results[['TickerPairs','Test_Statistic']]
# #test_statistics_rounded  = np.round(heatmap_coint['Test_statistic'], decimals=1)
# # Extract tickers from pairs
# heatmap_coint['Ticker1'] = heatmap_coint['TickerPairs'].apply(lambda x: x[0])
# heatmap_coint['Ticker2'] = heatmap_coint['TickerPairs'].apply(lambda x: x[1])
# heatmap_coint
# # Create a pivot table for the heatmap
# heatmap_coint = heatmap_coint.pivot(index='Ticker1', columns='Ticker2', values='Test_Statistic')
# heatmap_coint.to_excel(writer, sheet_name = 'heatmap_coint') 
# # Set up the matplotlib figure
# plt.figure(figsize=(12, 8))
# 
# # Create a lower triangular heatmap
# mask = np.triu(np.ones_like(heatmap_data, dtype=bool))
# # Adjusted the layout for better visibility
# sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='coolwarm', mask=mask, linewidths=.5, cbar_kws={"shrink": 0.75, "aspect": 20})
# plt.title('Co-integration Test Statistics Heatmap (Lower Triangular)')
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# plt.show()
# =============================================================================

#run on all co-integrating pairs vece to get the dominant equation

for stock_pair in coint_stock_pairs:
    vece_result = lr.vece(data_train[stock_pair[0]],data_train[stock_pair[1]])
    vece_results.append(vece_result)

y_x_relationship = pd.DataFrame()    
#export all results to excel
for vece_result in vece_results:
    #export y and x
    final_op = [{'selected_y' : vece_result[0], 'selected_x' : vece_result[1]}]
    final_op = pd.DataFrame.from_records(final_op)
    y_x_relationship = y_x_relationship.append(final_op)
    if vece_result[0] == 'Not selected because Mean reversion is not happening':
        pass
    else:
        final_op.to_excel(writer, sheet_name = vece_result[0] + "_" + vece_result[1],startcol = 0, index = False)
        #export both vecm tests done
        row = 0
        for df in vece_result[2]:
            df.to_excel(writer, sheet_name = vece_result[0] + "_" + vece_result[1],startcol = 4, startrow = row, index = False)
            row = row + 5
        row = 0
        for df in vece_result[3]:
            df.to_excel(writer, sheet_name = vece_result[0] + "_" + vece_result[1],startcol = 13, startrow = row, index = False)
            row = row + 5


#writer1 = pd.ExcelWriter(save_loc + "\\y_x_relationship.xlsx")
writer_y_x = pd.ExcelWriter(save_loc + "\\y_x_relationship.xlsx")
y_x_relationship.to_excel(writer, sheet_name = 'y_x_relationship', index = False)
writer_y_x.save()

#fit OU process to all
ou_results = pd.DataFrame()
for lst in y_x_relationship.values.tolist():
    if lst[0] == 'Not selected because Mean reversion is not happening':
        pass
    else:
        ou_result = lr.fit_ou_process(data_train[lst[0]], data_train[lst[1]])
        ou_result = pd.DataFrame.from_records([ou_result])
        ou_results = ou_results.append(ou_result)
        ou_results = ou_results.reset_index(drop = True)

writer_ou = pd.ExcelWriter(save_loc + "\\ou_results.xlsx")
ou_results.to_excel(writer_ou, sheet_name = 'ou_results')
writer_ou.save()

#create a dictionary to hold all the bt results
bt_results = {}
#ou_results1 = ou_results.head(2)

#Generate signals and backtest results
for lst in ou_results.values.tolist():
    #print(lst)
    if lst[0] == 'Not selected because Mean reversion is not happening':
        pass
    else:
        #get required data from the ou_results dataframe
        
        y_ticker = lst[0]
        x_ticker = lst[1]
        intercept = lst[2]
        b_coint  = lst[3]
        mu_e = lst[7]
        sigma_eq = lst[9]
        half_life = lst[10]
        k = 1
        capital = 100000
        
        #get the closing prices of two tickers
        #print(y_ticker)
        #print(x_ticker)
        df = data_test[['Date', y_ticker, x_ticker]]
        #call the function to get backtesting returns
        bt_result = bt.backtest(df, y_ticker, x_ticker, intercept, b_coint, mu_e, sigma_eq, k, capital, save_loc_chart)
        
        #append the result to a dictionary
        bt_results[(y_ticker+ "_" + x_ticker).replace(".NS","")] = bt_result
        
        #bt_result.to_excel(writer, sheet_name = (y_ticker+ "_" + x_ticker).replace(".NS",""), index = False)

writer.save()
#ou_results.to_excel(writer, sheet_name = 'ou_results', index = False)

#performance of backtesting stratergy

#dowload nifty 50 data
nifty_50_data = yf.download("^NSEI", start=start_test, end=end_test)

benchmark_index = nifty_50_data[['Close']]
benchmark_index['nifty_daily_return'] = nifty_50_data['Close'].pct_change()

#backtset performance
#create new dataframe to hold all returns
bt_performance1 = {}
bt_performance2 = {}

for bt_result in bt_results:
    print(f"Runnning bt performance results for {bt_result}.")
    #get tickers names
    tickers = bt_result.split('_')
    y_ticker = f"{tickers[0]}.NS"
    x_ticker = f"{tickers[1]}.NS"
    #create blank df to store bt performance results
    returns = pd.DataFrame()
    returns['Returns'] = bt_results[bt_result]['Total_Assets'].pct_change()
    # Replace inf and -inf with zero
    returns['Returns'] = returns['Returns'].replace([np.inf, -np.inf], 0)
    #number of trades columnn
    returns['n_trades'] = bt_results[bt_result]['n_trades']
    #set index to date
    returns = returns.set_index(data_test['Date'])
    #drop na
    returns = returns.dropna()
    
    #merge with risk free rate df
    returns = pd.merge(returns, risk_free_rate[['Price']], how = 'left', left_index = True, right_index = True)
    returns.rename(columns = {'Price' : 'risk_free_rate'}, inplace = True)
    # Forward fill NaN values in the 'risk_free_rate' column
    returns['risk_free_rate'] = returns['risk_free_rate'].fillna(method='ffill')
    
    #covert rfr to daily and in absolute number
    returns['risk_free_rate'] = returns['risk_free_rate'] / 100.0
    returns['risk_free_rate_daily'] = (1 + returns['risk_free_rate'] ) ** (1/252) - 1
    
    #merge nifty index return 
    returns = pd.merge(returns, benchmark_index, how = 'left', left_index = True, right_index = True)
    
    #run bt performance
    bt_perf, returns = bt.create_tear_sheet(returns, save_loc_chart, y_ticker, x_ticker)
    
    #append the result to a dictionary
    bt_performance1[(y_ticker+ "_" + x_ticker).replace(".NS","")] = bt_perf
    bt_performance2[(y_ticker+ "_" + x_ticker).replace(".NS","")] = returns
    
    #break;
    

#save summary of backtesting results of all the tickers
bt_performance = pd.DataFrame(bt_performance1)
#Export the results
writer_bt_perf = pd.ExcelWriter(save_loc + "\\bt_performance.xlsx")
bt_performance.to_excel(writer_bt_perf, sheet_name = 'bt_performance')
writer_bt_perf.save()   

#export all the results in separate sheets
for bt_result in bt_results:
    #get tickers names
    tickers = bt_result.split('_')
    y_ticker = f"{tickers[0]}.NS"
    x_ticker = f"{tickers[1]}.NS"
    #create excel sheet
    save_loc_pairs = data_file_loc + "\\Output\Pairs_Results"
    if not os.path.exists(save_loc_pairs):
        os.makedirs(save_loc_pairs)
    writer_bt = pd.ExcelWriter(save_loc_pairs + "\\" + bt_result + ".xlsx")
    #export staionarity results
    df = df_statinarity_results[(df_statinarity_results['Ticker'] == y_ticker) | (df_statinarity_results['Ticker'] == x_ticker)]
    df.to_excel(writer_bt, sheet_name = 'staionarity_results', index = False)
    #export cointegration results
    for a in combinations([y_ticker,y_ticker], 2):
        df = df_coint_results[df_coint_results['TickerPairs'] == a]
        df.to_excel(writer_bt, sheet_name = 'cointegration_results', index = False)
    #export ou process results
    df = ou_results[(ou_results['y_ticker'] == y_ticker) & (ou_results['x_ticker'] == x_ticker)]
    df.to_excel(writer_bt, sheet_name = 'ou_results', index = False)
    #export vece results
    for vece_result in vece_results:
        if (y_ticker == vece_result[0]) & (x_ticker == vece_result[1]):
            final_op = pd.DataFrame({'selected_y' : [vece_result[0]], 'selected_x' : [vece_result[1]]})
            final_op.to_excel(writer_bt, sheet_name = 'VECE_Results',startcol = 0, index = False)
            #export both vecm tests done
            row = 0
            for df in vece_result[2]:
                df.to_excel(writer_bt, sheet_name = 'VECE_Results',startcol = 4, startrow = row, index = False)
                row = row + 5
            row = 0
            for df in vece_result[3]:
                df.to_excel(writer_bt, sheet_name = 'VECE_Results',startcol = 13, startrow = row, index = False)
                row = row + 5
    #export bt_results
    bt_results[bt_result].to_excel(writer_bt, sheet_name = 'bt_returns')
    #export bt_performance results
    df = pd.DataFrame(list(bt_performance1[bt_result].items()), columns=['Key', 'Value']).set_index('Key')
    df.to_excel(writer_bt, sheet_name = 'bt_performance')
    bt_performance2[bt_result].to_excel(writer_bt, sheet_name = 'final_returns')
    
    #save writer
    writer_bt.save()
    writer_bt.close()
    

    
    
    
                                 



