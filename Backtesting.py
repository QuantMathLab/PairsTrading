# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 00:52:54 2023

@author: yrahu
"""

#Import libraries
import LinearRegresion as lr
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from matplotlib.ticker import FuncFormatter


import numpy as np
import pandas as pd


def backtest(data_test, y_ticker, x_ticker, intercept, b_coint, mu_e, sigma_eq, k, capital, save_loc_chart):
    #get the spread
    data_test['Spread'] = data_test[y_ticker] - b_coint * data_test[x_ticker] - intercept
    
    # Plots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))

    # Plot closing prices
    print(data_test[y_ticker].shape)
    data_test[y_ticker].plot(ax=axes[0], color='blue', label=y_ticker)
    data_test[x_ticker].plot(ax=axes[0], color='purple', label=x_ticker)
    axes[0].set_title('Closing prices')
    axes[0].legend()
    
    # Plot spread chart
    data_test['Spread'].plot(ax=axes[1], color='red', label='Spread')
    axes[1].axhline(y=mu_e + k * sigma_eq, color='purple', label='mu_e +' + str(k) + ' * sigma_eq',  linestyle='--')
    axes[1].axhline(y=mu_e - k * sigma_eq, color='purple', label='mu_e -' + str(k) + ' * sigma_eq',  linestyle='--')
    axes[1].axhline(y=mu_e, color='g', label='mu_e', linestyle='--')
    axes[1].set_title(f"Spread {y_ticker} - {x_ticker}")
    axes[1].legend()
    
    # Save the figure to a file (e.g., in PNG format)
    chart_file_path = save_loc_chart + f"Spread_{y_ticker}_{x_ticker}.png"
    plt.savefig(chart_file_path, format='png')
    #show the plots
    plt.show()
    
    #reset dataframe
    data_test = data_test.reset_index(drop = True)
    
    initial_capital = capital
    
    y_ticker_holdings = round(initial_capital /(b_coint * data_test[x_ticker][0] + data_test[y_ticker][0] ))
    x_ticker_holdings = round(b_coint * y_ticker_holdings)
    Cash = initial_capital
    
    
    #shift the spread 
    #This is done so that signal received from the closing price of yesterday is used to make the decision today
    data_test['Shifted_Spread'] = data_test['Spread'].shift(1)
    #create new required columns
    data_test['Signal'] = 'NA'
    data_test['y_stock_holdings'] = 0
    data_test['x_stock_holdings'] = 0
    data_test['Cash'] = 0
    data_test['Total_Assets'] = 0
    
    #Assign at point zero Cash and Total_Assets
    data_test.loc[0, 'Cash'] = Cash
    data_test.loc[0, 'Total_Assets'] = initial_capital
    
    #create column to count the number of transactions (trades) done
    data_test['n_trades'] = 0
    n_trades = 0
    for i in range(1, len(data_test)):
        #get the bounds
        upper_bound =  mu_e + k * sigma_eq
        lower_bound =  mu_e - k * sigma_eq
        # 3 cases for the first day
        if (data_test.loc[i, 'Shifted_Spread'] > upper_bound) & (data_test.loc[i-1,'Signal']  == 'NA'):
            
            data_test.loc[i,'Signal'] = 'Short_the_Spread'
            data_test.loc[i,'y_stock_holdings'] = -data_test.loc[i, y_ticker] * y_ticker_holdings 
            data_test.loc[i,'x_stock_holdings'] = data_test.loc[i, x_ticker]  * x_ticker_holdings
            data_test.loc[i,'Cash']  = data_test.loc[i-1,'Cash'] - data_test.loc[i,'y_stock_holdings'] - data_test.loc[i,'x_stock_holdings'] 
            data_test.loc[i, 'Total_Assets'] = data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] + data_test.loc[i,'Cash'] 
            
            #change trade count
            n_trades = n_trades + 1
            data_test.loc[i, 'n_trades'] = n_trades
            
        elif (data_test.loc[i, 'Shifted_Spread'] < lower_bound) &  (data_test.loc[i-1,'Signal']  == 'NA'):
            
            data_test.loc[i,'Signal'] = 'Long_the_Spread'
            data_test.loc[i,'y_stock_holdings'] = data_test.loc[i, y_ticker] * y_ticker_holdings 
            data_test.loc[i,'x_stock_holdings'] = -data_test.loc[i, x_ticker]  * x_ticker_holdings
            data_test.loc[i,'Cash']  = data_test.loc[i-1,'Cash']- data_test.loc[i,'y_stock_holdings'] - data_test.loc[i,'x_stock_holdings'] 
            data_test.loc[i, 'Total_Assets'] = data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] + data_test.loc[i,'Cash'] 
            
            #change trade count
            n_trades = n_trades + 1
            data_test.loc[i, 'n_trades'] = n_trades
                        
            
        elif (data_test.loc[i, 'Shifted_Spread'] <= upper_bound) & (data_test.loc[i, 'Shifted_Spread'] >= lower_bound) \
                & (data_test.loc[i-1,'Signal']  == 'NA'):
                    
            data_test.loc[i,'Signal']  = 'No_Signal'
            data_test.loc[i,'y_stock_holdings'] = 0
            data_test.loc[i,'x_stock_holdings'] = 0
            data_test.loc[i,'Cash']  = data_test.loc[i-1,'Cash']
            data_test.loc[i, 'Total_Assets'] = data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] + data_test.loc[i,'Cash'] 
            
            #change trade count
            data_test.loc[i, 'n_trades'] = n_trades 
    
        # 3 cases for signal Short_the_Spread on prev day
        elif (data_test.loc[i, 'Shifted_Spread'] > mu_e) & (data_test.loc[i-1,'Signal']  == 'Short_the_Spread'):
            
            data_test.loc[i,'Signal']  = 'Short_the_Spread'
            data_test.loc[i,'y_stock_holdings'] = -data_test.loc[i, y_ticker] * y_ticker_holdings 
            data_test.loc[i,'x_stock_holdings'] = data_test.loc[i, x_ticker]  * x_ticker_holdings
            data_test.loc[i,'Cash']  = data_test.loc[i-1,'Cash'] 
            data_test.loc[i, 'Total_Assets'] = data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] + data_test.loc[i,'Cash'] 
            #change trade count
            data_test.loc[i, 'n_trades'] = n_trades
    
            
        elif (data_test.loc[i, 'Shifted_Spread'] <= mu_e)  & (data_test.loc[i, 'Shifted_Spread'] >= lower_bound) \
                & (data_test.loc[i-1,'Signal']  == 'Short_the_Spread'):
            
            data_test.loc[i,'Signal']  = 'No_Signal'
            data_test.loc[i,'y_stock_holdings'] = 0
            data_test.loc[i,'x_stock_holdings'] = 0
            data_test.loc[i,'Cash'] = data_test.loc[i-1,'Total_Assets']
            data_test.loc[i, 'Total_Assets'] =  data_test.loc[i-1,'Total_Assets']
            #change trade count
            n_trades = n_trades + 1
            data_test.loc[i, 'n_trades'] = n_trades
            
            
        elif (data_test.loc[i, 'Shifted_Spread'] < lower_bound) & (data_test.loc[i-1,'Signal']  == 'Short_the_Spread'):
                    
            data_test.loc[i,'Signal']  = 'Long_the_Spread'
            data_test.loc[i,'y_stock_holdings'] = data_test.loc[i, y_ticker] * y_ticker_holdings 
            data_test.loc[i,'x_stock_holdings'] = -data_test.loc[i, x_ticker]  * x_ticker_holdings
            data_test.loc[i,'Cash']  = data_test.loc[i-1,'Total_Assets']- data_test.loc[i,'y_stock_holdings'] - data_test.loc[i,'x_stock_holdings'] 
            data_test.loc[i, 'Total_Assets'] = data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] + data_test.loc[i,'Cash'] 
            #change trade count
            n_trades = n_trades + 2
            data_test.loc[i, 'n_trades'] = n_trades
    
        # 3 cases for signal Long_the_Spread on prev day    
        elif (data_test.loc[i, 'Shifted_Spread'] < mu_e) & (data_test.loc[i-1,'Signal']  == 'Long_the_Spread'):
            
            data_test.loc[i,'Signal']  = 'Long_the_Spread'
            data_test.loc[i,'y_stock_holdings'] = data_test.loc[i, y_ticker] * y_ticker_holdings 
            data_test.loc[i,'x_stock_holdings'] = -data_test.loc[i, x_ticker]  * x_ticker_holdings
            data_test.loc[i,'Cash']  = data_test.loc[i-1,'Cash'] 
            data_test.loc[i, 'Total_Assets'] = data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] + data_test.loc[i,'Cash']       
            #change trade count
            data_test.loc[i, 'n_trades'] = n_trades
            
        elif (data_test.loc[i, 'Shifted_Spread'] >= mu_e)  & (data_test.loc[i, 'Shifted_Spread'] <= upper_bound) \
                & (data_test.loc[i-1,'Signal']  == 'Long_the_Spread'):
            
            data_test.loc[i,'Signal']  = 'No_Signal'
            data_test.loc[i,'y_stock_holdings'] = 0
            data_test.loc[i,'x_stock_holdings'] = 0
            data_test.loc[i,'Cash'] = data_test.loc[i-1,'Total_Assets']
            data_test.loc[i, 'Total_Assets'] =  data_test.loc[i-1,'Total_Assets']
            #change trade count
            n_trades = n_trades + 1
            data_test.loc[i, 'n_trades'] = n_trades
            
        elif (data_test.loc[i, 'Shifted_Spread'] < lower_bound) & (data_test.loc[i-1,'Signal']  == 'Long_the_Spread'):
                    
            data_test.loc[i,'Signal']  = 'Short_the_Spread'
            data_test.loc[i,'y_stock_holdings'] = -data_test.loc[i, y_ticker] * y_ticker_holdings 
            data_test.loc[i,'x_stock_holdings'] = data_test.loc[i, x_ticker]  * x_ticker_holdings
            data_test.loc[i,'Cash']  = data_test.loc[i-1,'Total_Assets'] + data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] 
            #change trade count
            n_trades = n_trades + 2
            data_test.loc[i, 'n_trades'] = n_trades
            
        # 3 cases for signal No_Signal on prev day     
        elif (data_test.loc[i, 'Shifted_Spread']  > upper_bound) & (data_test.loc[i-1,'Signal']  == 'No_Signal'):
            
            data_test.loc[i,'Signal']  = 'Short_the_Spread'
            data_test.loc[i,'y_stock_holdings'] = -data_test.loc[i, y_ticker] * y_ticker_holdings 
            data_test.loc[i,'x_stock_holdings'] = data_test.loc[i, x_ticker]  * x_ticker_holdings
            data_test.loc[i,'Cash']  = data_test.loc[i-1,'Cash'] - data_test.loc[i,'y_stock_holdings'] - data_test.loc[i,'x_stock_holdings'] 
            data_test.loc[i, 'Total_Assets'] = data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] + data_test.loc[i,'Cash'] 
            #change trade count
            n_trades = n_trades + 1
            data_test.loc[i, 'n_trades'] = n_trades
            
            
        elif (data_test.loc[i, 'Shifted_Spread'] <= upper_bound)  & (data_test.loc[i, 'Shifted_Spread'] >= lower_bound) \
                & (data_test.loc[i-1,'Signal']  == 'No_Signal'):
            
            data_test.loc[i,'Signal']  = 'No_Signal'
            data_test.loc[i,'y_stock_holdings'] = 0
            data_test.loc[i,'x_stock_holdings'] = 0
            data_test.loc[i,'Cash'] = data_test.loc[i-1,'Total_Assets']
            data_test.loc[i, 'Total_Assets'] = data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] + data_test.loc[i,'Cash'] 
            #change trade count
            data_test.loc[i, 'n_trades'] = n_trades
            
            
        elif (data_test.loc[i, 'Shifted_Spread'] < lower_bound) & (data_test.loc[i-1,'Signal']  == 'No_Signal'):
                    
            data_test.loc[i,'Signal']  = 'Long_the_Spread'
            data_test.loc[i,'y_stock_holdings'] = data_test.loc[i, y_ticker] * y_ticker_holdings 
            data_test.loc[i,'x_stock_holdings'] = -data_test.loc[i, x_ticker]  * x_ticker_holdings
            data_test.loc[i,'Cash']  = data_test.loc[i-1,'Cash'] - data_test.loc[i,'y_stock_holdings'] - data_test.loc[i,'x_stock_holdings']
            data_test.loc[i, 'Total_Assets'] = data_test.loc[i,'y_stock_holdings'] + data_test.loc[i,'x_stock_holdings'] + data_test.loc[i,'Cash'] 
            #change trade count
            n_trades = n_trades + 1
            data_test.loc[i, 'n_trades'] = n_trades
            
            
        else:
            
            pass
    #write dataframe
    return data_test



# Calculate rolling beta over a 3-month window
def calculate_rolling_beta(data, rolling_months = 3):
    #counter = 0
    betas = []
    window = rolling_months * 21
    for i in range(len(data) - window + 1):
        #print(counter)
        #counter = counter + 1
        window_data = data.iloc[i:i+window]
        #print(window_data)
        y = window_data['Returns']
        X = window_data['nifty_daily_return']

        try:
            model_op = lr.ols_linear_regression(y,X)

            beta = model_op[0]['Coefficients'][1]
        except Exception as e:
            # Handle the exception
            # print(f"An error occurred: {e}")
            beta = 0
            
        betas.append(beta)

    return pd.Series(betas, index=data.index[window-1:])

def calculate_metrics(returns, rolling_months = 3):
    #get start date and End date
    start_date = returns.index[0].strftime('%d-%b-%Y')
    end_date = returns.index[-1].strftime('%d-%b-%Y')
    
    # Calculate cumulative returns
    returns['Cumulative Returns'] = (1 + returns['Returns']).cumprod() - 1

    # Calculate drawdowns
    cumulative_returns = returns['Cumulative Returns']
    previous_peaks = cumulative_returns.expanding(min_periods=1).max()
    returns['drawdowns'] = cumulative_returns - previous_peaks
    
    max_drawdown = returns['drawdowns'].min()

    # Calculate annualized returns (simple annualization)
    total_return = cumulative_returns.iloc[-1] + 1
    n_years = len(returns) / 252  # Assuming 252 trading days in a year
    annualized_returns = total_return ** (1 / n_years) - 1
    
    # Calculate rolling annualized returns (simple annualization)
    returns['Rolling_Cumulative_Returns'] = ((1 + returns['Returns']).rolling(window=rolling_months * 21).apply(lambda x: x.prod()) - 1)
    rolling_annualized_returns = (returns['Rolling_Cumulative_Returns'] + 1) ** (1 / (rolling_months/12)) - 1
    returns['rolling_annualized_returns'] = rolling_annualized_returns
    
    #Calculte Annual volatility
    annual_volatility  = returns['Returns'].std() * np.sqrt(252)
    
    #Calculte rolling Annual volatility
    returns['rolling_annual_volatility'] = returns['Returns'].rolling(window=rolling_months * 21).std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    daily_excess_returns = returns['Returns'] - returns['risk_free_rate_daily']
    sharp_ratio = (daily_excess_returns.mean() / daily_excess_returns.std()) * np.sqrt(252)
    
    # Calculate rolling Sharpe ratio
    returns['rolling_excess_mean'] = daily_excess_returns.rolling(window=rolling_months * 21).mean()
    returns['rolling_excess_sd'] = daily_excess_returns.rolling(window=rolling_months * 21).std()
    
    
    returns['rolling_sharpe_ratio'] = np.where( returns['rolling_excess_sd'] > 1e-4,
        returns['rolling_excess_mean']/
        returns['rolling_excess_sd'] , 0
    ) * np.sqrt(252)
    
    #calculte value at risk at 99 percentile
    VaR_99_Percentile = np.percentile(returns['Returns'], 1)
    
    # Calculate skewness and kurtosis
    skewness = skew(returns['Returns'])
    kurt = kurtosis(returns['Returns'])
    
    #Calculte beta with the index
    # Calculate covariance matrix
    cov_matrix = np.cov(returns['Returns'], returns['nifty_daily_return'])

    # Calculate beta
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    #calculate rolling betas
    betas = calculate_rolling_beta(returns, rolling_months = 3)
    betas = betas.to_frame(name = 'Rolling_Betas')
    
    #merge betas with returns series
    returns = pd.merge(returns, betas, how = 'left', left_index = True, right_index = True)
    #returns.to_csv('D:\CQF\Jun 2023 Project\Code\Output\op.csv')
    
    #number of trade happened during the period

    n_trades = returns['n_trades']
    n_trades = n_trades.iloc[-1]
    
    #get all results in summary
    backtest_results = {'Start Date' : start_date, 'End Date' : end_date, 'Total Days' : len(returns),
                        'Annual Return' : annualized_returns, 'Cumulative Return' : total_return,
                        'Annual Volatility' : annual_volatility, 'Sharp Ratio' : sharp_ratio,
                        'Beta with the benchmark index' : beta, 'Rolling Months' : rolling_months,
                        'Max Drawdown' : max_drawdown, 'Skew' : skewness, 'Kurtosis' : kurt,
                        'Numbers trades done ' : n_trades, 'Daily VaR at 99th Percentile' : VaR_99_Percentile}

    return backtest_results, returns


# Function to format y-labels as percentages
def percentage_formatter(x, pos):
    return f'{x:.0%}'

def create_tear_sheet(returns, save_loc_chart, y_ticker, x_ticker):
    backtest_results, returns = calculate_metrics(returns)

    # Plotting
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 10))

    # Plot cumulative returns
    returns['Cumulative Returns'].plot(ax=axes[0,0], color='blue', label='Cumulative Returns')
    axes[0,0].set_title('Cumulative Returns')
    axes[0,0].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    axes[0,0].legend()
    
    
    # Plot drawdowns
    returns['drawdowns'].plot(ax=axes[0,1], color='red', title='Drawdowns')
    axes[0,1].fill_between(returns['drawdowns'].index, returns['drawdowns'], color='red', alpha=0.3)

    # Plot rolling annualized returns
    returns['rolling_annualized_returns'].plot(ax=axes[1,0], color='purple', label='Rolling ' + str(backtest_results['Rolling Months']) + ' month Annualized Returns')
    axes[1,0].axhline(y = backtest_results['Annual Return'], color = 'b', linestyle='--', linewidth=2, label = 'Average Annualised return')
    axes[1,0].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    axes[1,0].set_title('Annualised return')
    axes[1,0].legend()

    # Plot rolling Sharpe ratio
    returns['rolling_sharpe_ratio'].plot(ax=axes[1,1], color='green', label='Rolling ' + str(backtest_results['Rolling Months']) + ' Sharp Ratio')
    axes[1,1].axhline(y = backtest_results['Sharp Ratio'], color = 'b', linestyle='--', linewidth=2, label = 'Average Sharp Ratio')
    axes[1,1].set_title('Sharp Ratio')
    axes[1,1].legend()
    
    # Plot rolling Annual Volatility
    returns['rolling_annual_volatility'].plot(ax=axes[2,0], color='blue', label='Rolling ' + str(backtest_results['Rolling Months']) + ' month annual volatility')
    axes[2,0].axhline(y = backtest_results['Annual Volatility'], color = 'b', linestyle='--', linewidth=2, label = 'Average annual volatility')
    axes[2,0].set_title('Annual Volatility')
    axes[2,0].legend()
    
    # histogram of rolling Annual Volatility
    returns['Returns'][returns['Returns'] > backtest_results['Daily VaR at 99th Percentile']].hist(ax=axes[2,1], 
                                                                            color='blue', bins = 20, alpha = 0.8)
    # Highlight returns below -2% in a different color
    axes[2,1].hist(returns['Returns'][returns['Returns'] <= backtest_results['Daily VaR at 99th Percentile']], bins=5, 
                   color='red', alpha=0.7)
    # Add a horizontal line at -2%
    axes[2,1].axvline(backtest_results['Daily VaR at 99th Percentile'], color='red', linestyle='--', 
                      linewidth=2, label='Threshold: VaR')
    axes[2,1].xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    axes[2,1].set_xlabel('Daily Returns')
    axes[2,1].set_title('Histogram of Daily Returns')
    axes[2,1].legend()
    
    # Plot returns
    returns['Returns'].plot(ax=axes[3,0], color='purple', label='Daily Returns')
    axes[3,0].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    axes[3,0].set_title('Daily Returns')
    axes[3,0].legend()
    
    # Plot rolling betas
    returns['Rolling_Betas'].plot(ax=axes[3,1], color='green', label='Rolling ' + str(backtest_results['Rolling Months']) + ' month betas')
    axes[3,1].axhline(y = backtest_results['Beta with the benchmark index'], color = 'b', linestyle='--', linewidth=2, label = 'Average beta')
    axes[3,1].set_title('Beta with the benchmark index')
    axes[3,1].legend()
    
    # Set a common x-axis title
    fig.suptitle(f"{y_ticker} {x_ticker} Backtesting Performance Metrics Over Time", fontsize=16)

    plt.tight_layout()
       
    # Save the figure to a file (e.g., in PNG format)
    chart_file_path = save_loc_chart + f"bt_perf_{y_ticker}_{x_ticker}.png"
    plt.savefig(chart_file_path, format='png')
    #show plot
    plt.show()
    
    return backtest_results, returns
