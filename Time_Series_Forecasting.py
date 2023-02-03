# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:38:24 2019

@author: Pranay.choudekar
"""

# =============================================================================
# TIME SERIES FORECASTING IN PYTHON
# =============================================================================



# =============================================================================
# SETTING THE ENVOIRNMENT
# =============================================================================
import os # working directory
import warnings # Ensure ignore /no warnings are displayed 
#import itertools
import numpy as np # Data Processing
import matplotlib.pyplot as plt # Visualization
import pandas as pd # Data Processing 
import statsmodels.api as sm # Forecasting
import matplotlib
from pylab import rcParams # Setting up the chart elements/visualization
from statsmodels.tsa.stattools import adfuller # Stationarity
from numpy import log # Stationarity
from pmdarima.arima.utils import ndiffs # Stationarity/Differncing 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # ACF and PACF Plots to determine the order of MA and AR components
from statsmodels.tsa.arima_model import ARIMA # Applying ARIMA
#import pmdarima as pm # Applying ARIMA

#pip install pmdarima -- > Install pmdarima in the local repository

warnings.filterwarnings("ignore")

#Customization of the Plots created---------------->
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# =============================================================================
# OBJECTIVE : FORECASTING FURNTIURE SALES FOR THE RETAIL STORE
# =============================================================================

os.chdir(R'C:\Documents\IVY NOTES\PYTHON\Time Series')
os.getcwd()

df = pd.read_excel("Superstore.xls")
furniture = df.loc[df['Category'] == 'Furniture']#Filtering for Furniture Category

#Understanding the data 
print(furniture['Order Date'].min())# min date 
print(furniture['Order Date'].max())# max date

#-----> TS --- More amount of sample data ---> better is the prediction capability
#-----> TS --- Ideal minimum requiremt -----> 2- 3 years of data   


# =============================================================================
# 1. Data Pre-Processing
# =============================================================================

#Checking Missing Values
furniture.isnull().sum()
# rolling mean withing pandas for derving moving average 

#Arranging the Data chronoligcally
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()


#Indexing with Time Series
furniture = furniture.set_index('Order Date')
furniture.index

#Work at the Average Monthly Sales
y = furniture['Sales'].resample('MS').mean()
y.plot(figsize=(15, 6))
plt.show()


# =============================================================================
# 2. Decomposing the Data: Trend, Seasonal and Irregular Component 
# =============================================================================
rcParams['figure.figsize'] = 8, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()


# =============================================================================
# 3. Checking the Stationarity of the Model
# =============================================================================
y_1 = y.reset_index()
#result = adfuller(y_1.Sales.dropna())
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])

# More Visualization
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':70})

fig, axes = plt.subplots(3, 2, sharex=True)

#Original Series
#fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(y_1.Sales); axes[0, 0].set_title('Original Series')
plot_acf(y_1.Sales, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(y_1.Sales.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(y_1.Sales.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(y_1.Sales.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(y_1.Sales.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

result_1 = adfuller(y_1.Sales.diff().dropna())
print('ADF Statistic: %f' % result_1[0])
print('p-value: %f' % result_1[1])# p-value is low --> reject the Ho (Data is Non - Stationary), accept Ha: Data is stationary 



# =============================================================================
# 4. Finding the AR Term of Model
# =============================================================================

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':70})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(y_1.Sales.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1))
plot_pacf(y_1.Sales.diff().dropna(), ax=axes[1])

plt.show()


# =============================================================================
# 5. Finding the MA Term of Model
# =============================================================================

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(y_1.Sales.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(y_1.Sales.diff().dropna(), ax=axes[1])

plt.show()


# =============================================================================
# 5. Fittting the SARIMA Model
# =============================================================================

#Seasonal Autoregressive Integrated Moving Average, SARIMA or Seasonal ARIMA, 
#is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component.


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),#Non Seasonal Component
                                seasonal_order=(1, 1, 0, 12),#Seasonal Components
                                enforce_stationarity=False,# Not Enforcing Stationarity, 
                                enforce_invertibility=False)#Models do converge, Model is estimatble
results = mod.fit()
print(results.summary().tables[1])

#-----------> Iterate on different combinations of : p | d | q (Both Seasonal & Non Seasonal)
#-----------> Choose the model with the lowest MAPE


#Trend Elements (Non Seasonal Components)
#p: Trend autoregression order.
#d: Trend difference order.
#q: Trend moving average order.

#Seasonal Elements
#P: Seasonal autoregressive order.
#D: Seasonal difference order.
#Q: Seasonal moving average order.
#m: The number of time steps for a single seasonal period.


# =============================================================================
# 6. Validating the Forecast
# =============================================================================

pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})

forecast_accuracy(y_forecasted, y_truth)


# =============================================================================
# Applying Auto - ARIMA
# =============================================================================
from pmdarima.arima import auto_arima

v_arima = y[0:36,]#subsetting data till 2016[ 3years]
auto_mod = auto_arima(v_arima,start_p=0,start_q=0,max_p=6,max_q=6,start_P=0,
                      start_Q=0,max_P=6,max_Q=6,m=12,seasonal =True, trace = True,n_fits=10,stepwise=True)


auto_mod.summary()
Prediction_Auto_Arima = pd.DataFrame({'y_forecast':auto_mod.predict(n_periods =12)}).reset_index(drop=True)
y_truth_df = pd.DataFrame({'y_truth':y_truth.reset_index(drop=True)})
Prediction_Auto_Arima=pd.concat([Prediction_Auto_Arima,y_truth_df],axis=1)
mape_auto_arima = np.mean(np.abs(Prediction_Auto_Arima.iloc[:,0] - Prediction_Auto_Arima.iloc[:,1])/np.abs(Prediction_Auto_Arima.iloc[:,0]))  # MAPE
print('mape from Auto - Arima-->' + str(float(mape_auto_arima)))