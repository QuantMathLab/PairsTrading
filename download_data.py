# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:34:16 2023

@author: yrahu
"""



import pandas as pd
import yfinance as yf


def download_data(stocks, start_date, end_date):
    #specify stocks 
    #stocks = ['ADANIPORTS.NS','ASIANPAINT.NS','AXISBANK.NS','BAJAJ-AUTO.NS','BAJFINANCE.NS','BAJAJFINSV.NS','BPCL.NS','BHARTIARTL.NS','BRITANNIA.NS','CIPLA.NS','COALINDIA.NS','DIVISLAB.NS','DRREDDY.NS','EICHERMOT.NS','GRASIM.NS','HCLTECH.NS','HDFCBANK.NS','HDFCLIFE.NS','HEROMOTOCO.NS','HINDALCO.NS','HINDUNILVR.NS','HDFC.NS','ICICIBANK.NS','ITC.NS','IOC.NS','INDUSINDBK.NS','INFY.NS','JSWSTEEL.NS','KOTAKBANK.NS','LT.NS','M&M.NS','MARUTI.NS','NTPC.NS','NESTLEIND.NS','ONGC.NS','POWERGRID.NS','RELIANCE.NS','SBILIFE.NS','SHREECEM.NS','SBIN.NS','SUNPHARMA.NS','TCS.NS','TATACONSUM.NS','TATAMOTORS.NS','TATASTEEL.NS','TECHM.NS','TITAN.NS','UPL.NS','ULTRACEMCO.NS','WIPRO.NS']
            
    #retriving data
    data = pd.DataFrame()
    master_data = pd.DataFrame()
    #Import data
    for stock in stocks:
        prices = yf.download(stock, start_date, end_date)
        prices['Ticker'] = stock
        master_data = master_data.append(prices)
        data[stock] = prices['Close']
    
    return data, master_data
    
# writer = pd.ExcelWriter(r"D:\CQF\FinalProject_Code\Data\Final_Data.xlsx", engine = 'xlsxwriter')
# data.to_excel(writer, sheet_name = 'data')
# master_data.to_excel(writer, sheet_name = 'master_data')

# writer.save()


