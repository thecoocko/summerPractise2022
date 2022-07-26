import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import itertools
import statsmodels.api as sm
import statsmodels.tsa as ts
import warnings

df = pd.read_csv('../Foreign_Exchange_Rates.csv')
print(f"DATASET SHAPE: {df.shape}")
print(df)

CURRENCY_DATA = df[['Time Serie', 'TAIWAN - NEW TAIWAN DOLLAR/US$']]
print(CURRENCY_DATA.shape)
for i in range(1,len(CURRENCY_DATA.columns)):
    CURRENCY_DATA[CURRENCY_DATA.columns[i]] = CURRENCY_DATA[CURRENCY_DATA.columns[i]].replace('ND',0).astype('float')

def defaultPlot():
    plt.figure(figsize=(16,8))
    plt.plot(CURRENCY_DATA)
    plt.show()

def autoAndPartcorrelation():
    _ = plt.figure(figsize=(18,10))
    ax_1 = plt.subplot(2,1,1)
    _ = sm.graphics.tsa.plot_acf(CURRENCY_DATA.values.squeeze(), lags=50, ax=ax_1)
    ax_2 = plt.subplot(2,1,2)
    _ = sm.graphics.tsa.plot_pacf(CURRENCY_DATA.values.squeeze(), lags=50, ax=ax_2)
    plt.show()

def isStationary():
    test =  sm.tsa.stattools.adfuller(CURRENCY_DATA)
    print(f"adf, p-value, c-values: { test[0]}, {test[1]}, {test[4]}")
    if test[0]>test[4]['5%']:
        print('non-stationary series')
    else:
        print('stationary series')

def differenceSeries():
    difference_row = CURRENCY_DATA.diff(periods=2).dropna()
    _ = plt.figure(figsize=(15,10))
    _ = plt.plot(difference_row)
    test = sm.tsa.stattools.adfuller(difference_row)
    print(f"adf, p-value, c-values: { test[0]}, {test[1]}, {test[4]}")
    plt.show()
    isStationary()
    _ = plt.figure(figsize=(18,10))
    ax_1 = plt.subplot(2,1,1)
    _ = sm.graphics.tsa.plot_acf(difference_row.values.squeeze(), lags=50, ax=ax_1)
    ax_2 = plt.subplot(2,1,2)
    _ = sm.graphics.tsa.plot_pacf(difference_row.values.squeeze(), lags=50, ax=ax_2)

defaultPlot()
autoAndPartcorrelation()
isStationary()
differenceSeries()
