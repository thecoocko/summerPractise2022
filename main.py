import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import warnings

df = pd.read_csv('../Foreign_Exchange_Rates.csv',index_col='Time Serie',parse_dates=True)
for i in range(len(df[df.columns[0]])):
    if df[df.columns[0]][i] == 'ND':
        df[df.columns[0]][i] = df[df.columns[0]][i-1]

df[df.columns[0]] = df[df.columns[0]].astype('float')
df.to_csv('../Foreign_Exchange_RatesClear.csv',index='Time Serie')

train, test= train_test_split(df ,test_size=0.2, shuffle=False)
train = train.sort_values(by='Time Serie')
test = test.sort_values(by='Time Serie')
print(train)
print(test)

def defaultPlot():
    train.plot()
    plt.show()

def autoAndPartcorrelation():
    ax_1 = plt.subplot(2,1,1)
    _ = sm.graphics.tsa.plot_acf(train.values.squeeze(), lags=50, ax=ax_1)
    ax_2 = plt.subplot(2,1,2)
    _ = sm.graphics.tsa.plot_pacf(train.values.squeeze(), lags=50, ax=ax_2)
    
    plt.show()

def isStationary():
    _ = sm.tsa.seasonal_decompose(train).plot()
    plt.show()
    test =  sm.tsa.stattools.adfuller(train)
    print(f"adf, p-value, c-values: { test[0]}, {test[1]}, {test[4]}")
    if test[0]>test[4]['5%']:
        print('non-stationary series')
    else:
        print('stationary series')

def differenceSeries():
    difference_row = train.diff(periods=1).dropna()
    _ = sm.tsa.seasonal_decompose(difference_row).plot()
    _ = plt.plot(difference_row)
    test = sm.tsa.stattools.adfuller(difference_row)
    print(f"adf, p-value, c-values: { test[0]}, {test[1]}, {test[4]}")
    plt.show()
    return difference_row

def differencePlot():

    difference_row = differenceSeries()
    ax_1 = plt.subplot(2,1,1)
    _ = sm.graphics.tsa.plot_acf(difference_row.values.squeeze(), lags=50, ax=ax_1)
    ax_2 = plt.subplot(2,1,2)
    _ = sm.graphics.tsa.plot_pacf(difference_row.values.squeeze(), lags=50, ax=ax_2)
    plt.show()

def model():
    d = 1
    ps = list(range(1))
    qs = list(range(1))
    parameters_list = list(itertools.product(ps, qs))
    global results 
    results = []
    best_aic = float("inf")
    for param in parameters_list:
        try:
            model=sm.tsa.statespace.SARIMAX(train, order=(param[0], d, param[1]), seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
        except ValueError:
            print('wrong paramaters for model:', param)
            continue
        aic = model.aic
        
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
    warnings.filterwarnings('default')
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    print(result_table.sort_values(by = 'aic', ascending=True))
    print(best_model.summary())
    

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plotARIMA(series, model, n_steps, d=1, plot_intervals=True, alpha=0.2):
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    data['arima_model'][:d] = np.NaN
    forecast = model.get_forecast(steps=n_steps)
    model_predictions = data.arima_model.append(forecast.predicted_mean)
    error = mean_absolute_percentage_error(data['actual'][d:], data['arima_model'][d:])
    plt.title(f"Mean Absolute Percentage Error: {error}")    
    plt.plot(model_predictions, color='r', label="model")
    plt.plot(data.actual, label="actual")
    
    plt.legend()
    plt.grid(True)
    plt.show()
   

defaultPlot()
autoAndPartcorrelation()
isStationary()
differenceSeries()
differencePlot()
model() 
selected_modelBest = sm.tsa.statespace.SARIMAX(test, order=(5, 1, 9), seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
selected_modelSecond = sm.tsa.statespace.SARIMAX(test, order=(8, 1, 9), seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
selected_modelThird = sm.tsa.statespace.SARIMAX(test, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
selected_modelWorst = sm.tsa.statespace.SARIMAX(test, order=(11, 1, 9), seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
plotARIMA(test, selected_modelBest, 30, alpha=0.5)
plotARIMA(test, selected_modelSecond,30, alpha=0.5)
plotARIMA(test, selected_modelThird, 30, alpha=0.5)
plotARIMA(test, selected_modelWorst, 30, alpha=0.5)

