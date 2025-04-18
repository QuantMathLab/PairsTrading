# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:33:53 2023

@author: yrahu
"""

import numpy as np
import pandas as pd
import scipy.stats
import math


#Function to get diagonal elements of squar matrix
def get_diagonal(m):
    rows, columns = m.shape
    diag = np.ones(rows)
    for i in range(rows):
        for j in range(columns):
            if (i==j):
                diag[i] = m[i][j]
    return diag               

def ols_linear_regression(y,x,constant = "Y"):
    #create empty dataframe to hold results
    lr_results = pd.DataFrame()
    
    #get names of x and y variables
    #if x has more than 1 dependent variable. It becomes dataframe hence require different processiong 
    if type(x) == pd.DataFrame: 
        x_col_names = x.columns
        #convert x and y into numpy arrays
        x = x.values
        if constant == "Y":
            x_col_names = x_col_names.insert(0,'Intercept')
            #create column vector with 1
            rows, columns = x.shape
            b0 = np.full((rows, ), 1)
            #incert tis row at the top of x matrix
            x = np.insert(x, 0, b0, 1) 
    #if x as only one dependent varaible, it is series and becoms 1-d array, needs to converted to matrix form
    else: 
        x_col_names = x.name
       
        if constant == "Y":
            x_col_names = ['Intercept',x_col_names]
            #convert x and y into numpy arrays
            x = x.values
            #convert 1-d array to 2-d array of row * 1
            x = np.reshape(x,(len(x),1))
            #create column vector with 1
            rows = len(x)
            b0 = np.full((rows, ), 1)
            #incert tis row at the top of x matrix
            x = np.insert(x, 0, b0, 1) 

    y_col_name = y.name
    
    #solution for OLS parameter estimation (Xt*X)-1 *Xt*Y
    betas =  np.linalg.inv(x.T @ x) @ x.T @y
    
    #calculate residuals e = y - b * X_t
    residuals = pd.DataFrame()
    e = y - betas @ x.T
    residuals['et'] =  e
    
    #calculate varaition in residuals
    row, columns = x.shape
    #This gives full varaince-covariance matrix of beatas, here column alreday has + 1 to no of varaibles
    sigma_2 = e @ e.T / (row-columns)
    #Get diagonal elements in above to get varaince of betas
    sigma_2 = (sigma_2)
    
    #calculte standard error of estimates σ * (X_t * X)-1
    betas_sd = sigma_2 * np.linalg.inv(x.T @ x)
    #get diagonal elements which has varaince of beats
    betas_sd = get_diagonal(betas_sd)
    #take squar root to get standard error
    betas_sd = np.sqrt(betas_sd)
  
    #get t stastic for all the betas
    betas_t_value = betas/betas_sd
    
    #get p value for all the betas
    betas_p_value = scipy.stats.t.sf(abs(betas_t_value), df= row - columns)*2

    #calculre r-squared
    TSS = np.sum((y- np.mean(y)) * (y- np.mean(y))) #total sum of squares
    y_hat = x @ betas # predictions
    RSS = np.sum((y- y_hat) * (y- y_hat)) #residual sum of squares
    r_squared = 1 - RSS/TSS
    
    #calculre adj. r-squared
    adj_r_squared = 1 - ((1 - r_squared) * (row -1) / (row-columns)) #here column alreday has + 1 to no of varaibles
    
    #calculate log likelihood
    RSS = np.sum(np.square(np.array(residuals['et']))) 
    log_likelihood = - row/2 * math.log(2 * math.pi) - row/2 * math.log(RSS/row) - row/2
    #calculte AIC from log likelihood
    AIC = 2 * (columns)  - 2 * (log_likelihood)
    #calculte AIC from log likelihood
    BIC = - 2 * (log_likelihood) + (columns) * math.log(row)
    
    #collect all results
    lr_results['X'] = x_col_names
    lr_results['Coefficients'] = betas.tolist()
    lr_results['Std Err'] = betas_sd.tolist()
    lr_results['t Statistic'] = betas_t_value.tolist()
    lr_results['P Value'] = betas_p_value.tolist()

    #collect overall results of regression
    overall_result = {'Dep. Variable' : [y_col_name], 'R-squared' : [r_squared],'Adj. R-squared' : [adj_r_squared],
                      'Log Likelihood' : [log_likelihood], 'AIC' : [AIC] , 'BIC' : [BIC]}
    overall_result = pd.DataFrame.from_dict(overall_result)
    final_result = [lr_results,overall_result,residuals]
    return final_result

def ADF_test(df,lags):
    #df = residuals
    #create copy of it to work on
    df = df.copy(deep = True)
    if type(df) == pd.Series: 
        df = df.to_frame()
        df.columns = ['et']
    #get differences of e. asumed that e is sorted in descending order of dates
    df['del_et'] = df['et'].diff()
    df['et_minus_1'] = df['et'].shift(1)
    #get lag variables
    for i in range(0,lags):
        df['del_et' + str(i+1) ] = df['del_et'].shift(i+1)
    #drop na
    df = df.dropna()
    del_ed = df['del_et'] 
    df = df.drop(['et','del_et'], 1)
    #get number of samples from row count
    row,column = df.shape
    #run regression on required values
    final_result = ols_linear_regression(del_ed,df)
    betas = final_result[0]
    AIC = final_result[1].iloc[0,4]
    BIC = final_result[1].iloc[0,5]
    test_statistic =  betas.loc[betas['X'] == 'et_minus_1', 't Statistic'].iloc[0]
    p_value_test_statistic = MacKinnon_p_value(row)
    final_result = [test_statistic,p_value_test_statistic,AIC,BIC,lags]
    return final_result
    

def ADF_test_maxlags(df,maxlags):
    if (maxlags > 10):
        maxlags = 10
    for i in range(maxlags):
        if i == 0:
            adf_result_prev = ADF_test(df,i)
            #print(adf_result_prev)
        else:
            adf_result_now = ADF_test(df,i)
            if adf_result_now[2] > adf_result_prev[2]:
                #print(adf_result_now)
                break
            else:
                adf_result_prev = adf_result_now
    
    return adf_result_prev
    
            
            
def MacKinnon_p_value(N):
    # array of Critical Values for No Trend Case from MacKinnon, 2010
    critical_values_table = np.array([[-3.43035	,-3.1175,-2.86154,-2.56677],
                       [-6.5393,-4.53235,-2.8903,-1.5384],
                       [-16.786,-9.8824,-4.234,	-2.809],
                       [-79.433,-57.7669,-40.04,	0]])
    #Calculate critical values
    critical_values = {}
    values = ['1%','2.5%','5%','10%']
    for value,i in zip(values,range(4)):
        critical_values[value + ' critical value:'] = critical_values_table[0][i] + critical_values_table[1][i] / N + critical_values_table[2][i] / (N*N) + critical_values_table[3][i] / (N*N*N)
    return critical_values

def Engle_Granger_Test(y,x):
    y_name = y.name
    x_name = x.name
    x_statinary = "No"
    y_statinary = "No"
    print("\n Results of  " + y_name + " and " + x_name + " :\n")
    #Ceck if both time series have unit root
    x_adf_result = ADF_test(x, 1)
    print("Results of ADF test for " + x_name + ":\n")
    print(x_adf_result)
    if x_adf_result[0] < x_adf_result[1]['5% critical value:']:
        x_statinary = "Yes"
        print (x.name + " is statinary \n")
    else:
        print (x.name + " is not statinary \n")
    y_adf_result = ADF_test(y, 1)
    print("Results of ADF test for " + y_name + ":\n")
    print(y_adf_result)
    if y_adf_result[0] < y_adf_result[1]['5% critical value:']:
        y_statinary = "Yes"
        print (y.name + " is statinary \n")
    else:
        print (y.name + " is not statinary \n")
    #check if both are not statinary and then proceed
    if (y_statinary == "No" and x_statinary == "No"):
        #run regression of x and y to get residuals
        model_op = ols_linear_regression(y,x)
        #run adf test on residulas to check if it statinary
        residuals = model_op[2]
        residuals_adf_result = ADF_test(residuals,1)
        #check if stationary
        if residuals_adf_result[0] < residuals_adf_result[1]['5% critical value:']:
            print(y_name + " name " + x_name + " are not co-integrated" )
            print("Results of ADF test on residuals is :\n")
        else:
            print(y_name + " name " + x_name + " are co-integrated" )
        
        print(residuals_adf_result)
    else:
        print ("No co-ntegration test is required as both y and x are not stationary")

def vece(stock1,stock2):
    stock1_name = stock1.name
    stock2_name = stock2.name
    
    #get difference of order one from both stocks
    stock1_diff =  stock1.diff().dropna()
    stock1_diff.reset_index(inplace = True, drop = True)
    stock2_diff =  stock2.diff().dropna()
    stock2_diff.reset_index(inplace = True, drop = True)

    #run y = stock1_diff on x = stock2_diff first and get residuals
    e = ols_linear_regression(stock1, stock2)[2]
    et_minus_1 = e.shift(1).dropna()
    et_minus_1.reset_index(inplace = True, drop = True)
    #now run vece 
    x = pd.DataFrame()
    x['stock2_diff'] = stock2_diff
    x['et_minus_1'] = et_minus_1
    result1 = ols_linear_regression(stock1_diff, x)
    p_value1 = result1[0].iloc[2,4]
    
    #run y = stock2_diff on x = stock1_diff first and get residuals
    e = ols_linear_regression(stock2, stock1)[2]
    et_minus_1 = e.shift(1).dropna()
    et_minus_1.reset_index(inplace = True, drop = True)
    #now run vece 
    x = pd.DataFrame()
    x['stock1_diff'] = stock1_diff
    x['et_minus_1'] = et_minus_1
    result2 = ols_linear_regression(stock2_diff, x)
    p_value2 = result2[0].iloc[2,4]
    
    #Determine which equation is more significant
    y = ''
    x = ''
    
    if p_value1 <= p_value2:
        if p_value1 < 0.05:
            print("y = " + stock1_name + ' and x =' + stock2_name + ' is more significant vector error corection equation')
            y = stock1.name
            x = stock2.name      
        else:
            print("Mean reversion is not happening")
            y = 'Not selected because Mean reversion is not happening'
            x = 'Not selected because Mean reversion is not happening'
    elif p_value1 > p_value2:
        if p_value2 < 0.05:
            print("y = " + stock2_name + ' and x =' + stock1_name + ' is more significant vector error corection equation')
            y = stock2.name
            x = stock1.name
        else:
            print("Mean reversion is not happening")
            y = 'Not selected because Mean reversion is not happening'
            x = 'Not selected because Mean reversion is not happening'
        
    return [y,x,result1,result2]

def fit_ou_process(y,x,delta_t = 1):
    et = pd.DataFrame()
    result1 = ols_linear_regression(y, x)
    #get beta_coint and intercept
    intercept = result1[0].iloc[0,1]
    b_coint = result1[0].iloc[1,1]
    #get residuals and run regression
    et['et'] = result1[2]['et']
    et['et_minus_1'] = et.shift(1)
    et = et.dropna()
    et.reset_index(inplace = True, drop = True)
    #run regression of rt and et-1
    result2 = ols_linear_regression(et['et'], et['et_minus_1'])
    #parameters C and B from above results
    C = result2[0].iloc[0,1]
    B = result2[0].iloc[1,1]
    #calculte OU process parameters
    theta = - np.log(B) / delta_t
    mu_e = C / (1-B)
    et_new = result2[2]['et']
    N = len(et_new)
    SE = math.sqrt(np.sum(et_new * et_new)/(N-3)) #standard error
    sigma_ou =  SE * math.sqrt( (- 2 * np.log(B)) / ( (1 -  B * B) * delta_t) )
    sigma_eq = sigma_ou / math.sqrt(2*theta)
    half_life = np.log(2) / theta
    #write for final op
    final_op = {'y_ticker': y.name , 'x_ticker' : x.name, 'intercept' : intercept, 'b_coint' : b_coint ,'C' : C, 'B' : B, 'theta' : theta, 'mu_e' : mu_e , 'sigma_ou' : sigma_ou , 'sigma_eq' : sigma_eq ,
                'half_life' : half_life}
    return final_op
    
                       
                         