# -*- coding: utf-8 -*-
"""
Created on Thu Dec 2 17:20:14 2020

@author: William
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# 2010-2019 stock prices gathered from macrotrends.net
amd_csv = 'Stocks\\amd10year.csv'
apple_csv = 'Stocks\\apple10year.csv'
nvidia_csv = 'Stocks\\nvidia10year.csv'
tesla_csv = 'Stocks\\tesla10year.csv'

# read the csv file with pandas, and resets the index to the years, for each stock
df_amd = pd.read_csv(amd_csv)
df_amd.index = ['2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010']
df_apple = pd.read_csv(apple_csv)
df_apple.index = ['2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010']
df_nvidia = pd.read_csv(nvidia_csv)
df_nvidia.index = ['2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010']
df_tesla = pd.read_csv(tesla_csv)
df_tesla.index = ['2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010']

# Plot all four stock, to visualy see the progession of each stock
plt.plot(df_amd['Average Stock Price'], label='AMD' )
plt.plot(df_apple['Average Stock Price'], label='Apple')
plt.plot(df_nvidia['Average Stock Price'], label='Nvidia')
plt.plot(df_tesla['Average Stock Price'], label='Tesla')
plt.title('AMD, Apple, Nvidia, and Tesla Average Stock Price')
plt.legend(loc=1)
plt.ylabel('Price')
plt.grid(True)
plt.xlabel('Date')

plt.show()

# Plots graph of AMD's average stock price and associated Linear Regression
Y = df_amd['Average Stock Price'].values.reshape(-1,1) 
X = df_amd['Year'].values.reshape(-1,1)
lr = LinearRegression()
lr.fit(X,Y)
y_pred = lr.predict(X)
plt.plot(X,Y , label='Average Stock Price')
plt.plot(X,y_pred, label='Linear Regression', color='red')
plt.title('AMD Linear Regression')
plt.legend(loc=1)
plt.ylabel('Price')
plt.legend(loc=2)
plt.grid(True)
plt.xlabel('Date')
plt.show()

# Plots graph of Apple's average stock price and associated Linear Regression
Y = df_apple['Average Stock Price'].values.reshape(-1,1) 
X = df_apple['Year'].values.reshape(-1,1)
lr = LinearRegression()
lr.fit(X,Y)
y_pred = lr.predict(X)
plt.plot(X,Y , label='Average Stock Price')
plt.plot(X,y_pred, label='Linear Regression', color='red')
plt.title('Apple Linear Regression')
plt.legend(loc=1)
plt.ylabel('Price')
plt.legend(loc=2)
plt.grid(True)
plt.xlabel('Date')
plt.show()

# Plots graph of Nvidia's average stock price and associated Linear Regression
Y = df_nvidia['Average Stock Price'].values.reshape(-1,1) 
X = df_nvidia['Year'].values.reshape(-1,1)
lr = LinearRegression()
lr.fit(X,Y)
y_pred = lr.predict(X)
plt.plot(X,Y , label='Average Stock Price')
plt.plot(X,y_pred, label='Linear Regression', color='red')
plt.title('Nvidia Linear Regression')
plt.legend(loc=1)
plt.ylabel('Price')
plt.legend(loc=2)
plt.grid(True)
plt.xlabel('Date')
plt.show()

# Plots graph of tesla's average stock price and associated Linear Regression
Y = df_tesla['Average Stock Price'].values.reshape(-1,1) 
X = df_tesla['Year'].values.reshape(-1,1)
lr = LinearRegression()
lr.fit(X,Y)
y_pred = lr.predict(X)
plt.plot(X,Y , label='Average Stock Price')
plt.plot(X,y_pred, label='Linear Regression', color='red')
plt.title('Tesla Linear Regression')
plt.legend(loc=1)
plt.ylabel('Price')
plt.legend(loc=2)
plt.grid(True)
plt.xlabel('Date')
plt.show()


# The above plot was for me to practice and utilize Linear Regression. I will now solve the problem with the values given.
# The values given are simply theoretical and do not relate to the above values at all.
file_name = 'EC225problem\\EC225problem.csv'

df = pd.read_csv(file_name)

data_irr = []
data_npv = []
data_pinv = []

# gets the annual rate of return for each of the stocks
for row in range(len(df)):
    nper = 10
    pmt = df.iloc[row, 2]
    pv = df.iloc[row, 1]
    fv = df.iloc[row, 3]
    irr = np.rate(nper, pmt, -pv, fv) * 100
    npv = np.pv(0.04, nper, -pmt, -fv, 0) - pv
    pinv = npv/pv
    data_irr.append(irr)
    data_npv.append(npv)
    data_pinv.append(pinv)

# creates new row for each of the problems
df['Annual Rate of Return in %'] = data_irr
df['NPV at 4%'] = data_npv
df['NPV per investment'] = data_pinv

print(df['Annual Rate of Return in %'])
print(df['NPV at 4%'])
print(df['NPV per investment'])




