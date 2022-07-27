import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm

df = pd.read_csv('../Foreign_Exchange_Rates.csv',index_col='Time Serie',parse_dates=True)

for i in range(len(df[df.columns[0]])):
    if df[df.columns[0]][i] == 'ND':
        df[df.columns[0]][i] = df[df.columns[0]][i-1]

df[df.columns[0]] = df[df.columns[0]].astype('float')

def defaultPlot():
    df.plot()
    plt.show()

def autoAndPartcorrelation():
    ax_1 = plt.subplot(2,1,1)
    _ = sm.graphics.tsa.plot_acf(df.values.squeeze(), lags=50, ax=ax_1)
    ax_2 = plt.subplot(2,1,2)
    _ = sm.graphics.tsa.plot_pacf(df.values.squeeze(), lags=50, ax=ax_2)
    
    plt.show()

def isStationary():
    _ = sm.tsa.seasonal_decompose(df).plot()
    plt.show()
    test =  sm.tsa.stattools.adfuller(df)
    print(f"adf, p-value, c-values: { test[0]}, {test[1]}, {test[4]}")
    if test[0]>test[4]['5%']:
        print('non-stationary series')
    else:
        print('stationary series')

def differenceSeries():
    difference_row = df.diff(periods=2).dropna()
    _ = sm.tsa.seasonal_decompose(difference_row).plot()
    _ = plt.plot(difference_row)
    test = sm.tsa.stattools.adfuller(difference_row)
    print(f"adf, p-value, c-values: { test[0]}, {test[1]}, {test[4]}")
    plt.show()
    return difference_row

def model():
    difference_row = differenceSeries()
    ax_1 = plt.subplot(2,1,1)
    _ = sm.graphics.tsa.plot_acf(difference_row.values.squeeze(), lags=50, ax=ax_1)
    ax_2 = plt.subplot(2,1,2)
    _ = sm.graphics.tsa.plot_pacf(difference_row.values.squeeze(), lags=50, ax=ax_2)
    d = 1
    ps = list(range(12))
    qs = list(range(12))
    parameters_list = list(itertools.product(ps, qs))
    results = []
    best_aic = float("inf")
    for param in parameters_list:
        try:
            model=sm.tsa.statespace.SARIMAX(df, order=(param[0], d, param[1]), seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
        except ValueError:
            print('wrong paramaters for model:', param)
            continue
        aic = model.aic
        
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
    
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
    plt.figure(figsize=(15, 7))
    plt.title(f"Mean Absolute Percentage Error: {error}")    
    plt.plot(model_predictions, color='r', label="model")
    plt.plot(data.actual, label="actual")
    plt.show()


defaultPlot()
autoAndPartcorrelation()
isStationary()
differenceSeries()
model() 
selected_model = sm.tsa.statespace.SARIMAX(df, order=(5, 1, 9), seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
plotARIMA(df, selected_model, 30, alpha=0.5)

