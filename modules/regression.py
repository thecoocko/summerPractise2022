from modules.imports import *
from ..main import df


def defaultPlot():
    plt.figure(figsize=(16,8))
    plt.plot(df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'])
    plt.show()

def autoAndPartcorrelation():
    _ = plt.figure(figsize=(18,10))
    ax_1 = plt.subplot(2,1,1)
    _ = sm.graphics.tsa.plot_acf(df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].values.squeeze(), lags=50, ax=ax_1)
    ax_2 = plt.subplot(2,1,2)
    _ = sm.graphics.tsa.plot_pacf(df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].values.squeeze(), lags=50, ax=ax_2)
    plt.show()

def isStationar():

    print(f"adf, p-value, c-values: { sm.tsa.stattools.adfuller(df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'])[0]}, {sm.tsa.stattools.adfuller(df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'])[1]}, {sm.tsa.stattools.adfuller(df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'])[4]}")
    