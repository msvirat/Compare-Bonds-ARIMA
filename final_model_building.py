# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 10:36:22 2021

@author: Sathiya vigraman M
"""""""


import pandas as pd#library to handle dataframe
import numpy as np#library for scientific computing
import matplotlib.pyplot as plt#library for plots
import requests#library for request from URLs
from bs4 import BeautifulSoup as bs#library for scraping
import investpy#library for scraping
from nsepy import get_history#library for scraping
from sqlalchemy import *#library for SQL
import pymysql #library for SQL
import warnings#library for skip warning
from datetime import date#library for handle DateTime format
from dateutil.relativedelta import *#library for handle DateTime format
from datetime import timedelta#library for handle DateTime format
from statistics import stdev#library for find standard deviation
import scipy.stats as stats#library for statistical functions
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf#library for ACF & PACF plots
from statsmodels.tsa.seasonal import seasonal_decompose#library for seasonal plot
from pmdarima import auto_arima#library for finding p, q, d
from statsmodels.tsa.stattools import adfuller#library for finding d
from statsmodels.tsa.arima_model import ARIMA#library for ARIMA model
from sklearn.metrics import mean_squared_error, mean_absolute_error#library for finding error in model
import pickle#library for Pickle model into pkl file

#ignore warnings
warnings.filterwarnings('ignore')

#change working dic. (pickle in the same folder where file saved)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)




#------Functions-------------

#Some Basic EDA & Plottings
def eda(data):
    print("Mean: ",np.mean(data))
    print("Median: ",np.median(data))
    print("Mode: ",stats.mode(data))
    print("Standard Deviation: ",stdev(data))
    print("Skewness: ",data.skew(axis=0))
    print("Kurtosis: ",data.kurt(axis = 0))


#QQ plot
def qqplot(data):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 15})
    stats.probplot(data,dist="norm", plot=plt)
    plt.xlabel('Quantile')
    plt.ylabel('Price')
    plt.title("Normal Q-Q plot")
    plt.show()


#Plot the Autocorrelation Function & Plot the partial autocorrelation function
def acf_pacf(data):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(data,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(data,ax=ax2)



#Decomposition Plot
def decomposition(data):
    decomposition = seasonal_decompose(data,freq=1,model = 'multiplicative')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(1)
    plt.subplot(411)
    plt.plot(data, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.ylabel('Price')
    plt.xlabel('Day')


#Histogram 
def histogram(data):
    plt.hist(data)
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.title('Histogram of Data')
    plt.show()


#Function for Miss Dates
def imputation(data):
    dummy = []
    r = pd.date_range(start = data.index.min(), end = data.index.max(), freq = 'D')
    dummy = data.reindex(r).fillna(' ').rename_axis('Date').reset_index()
    dummy = dummy.replace(' ',np.nan)
    dummy = dummy.ffill()
    dummy.set_index('Date', inplace=True)
    acf_pacf(data)
    acf_pacf(dummy)
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(data, color = 'green', label = 'Original Price')
    plt.plot(dummy, color = 'blue', label = 'Price after imputation')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    return dummy

#to find p, d, q
def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    

# Forecast for test data
def forecast_test(fitted, train, test):
    fc, se, conf = fitted.forecast(len(test), alpha=0.05)  # 95% confidence
    fc_series_test = pd.Series(fc, index = test.index)
    lower_series_test = pd.Series(conf[:, 0], index=test.index)
    upper_series_test = pd.Series(conf[:, 1], index=test.index)
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='Training Data')
    plt.plot(test, color = 'blue', label='Actual Price')
    plt.plot(fc_series_test, color = 'orange',label='Predicted Price')
    plt.fill_between(lower_series_test.index, lower_series_test, upper_series_test, color='k', alpha=.10)
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    #Errors - ARIMA model
    mse = mean_squared_error(test, fc_series_test)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(test, fc_series_test)
    print('MAE: '+str(mae))
    rmse = mean_squared_error(test, fc_series_test, squared=False)
    print('RMSE: '+str(rmse))



# Forecast for main data

def forecast_main(fitted, data):
    fc, se, conf = fitted.forecast(len(pd.DataFrame(pd.date_range(start = pd.to_datetime('today').date(), end = pd.to_datetime('today').date() + relativedelta(years =+ 5)))), alpha=0.05)  # 95% confidence
    fc_series = pd.DataFrame(fc, index = pd.date_range(start = pd.to_datetime('today').date(), periods = len(pd.DataFrame(pd.date_range(start = pd.to_datetime('today').date(), end = pd.to_datetime('today').date() + relativedelta(years =+ 5)))), freq='D'), columns = ['forecast'])
    lower_series = pd.Series(conf[:, 0], index = pd.date_range(start = pd.to_datetime('today').date(), periods = len(pd.DataFrame(pd.date_range(start = pd.to_datetime('today').date(), end = pd.to_datetime('today').date() + relativedelta(years =+ 5)))), freq='D'))
    upper_series = pd.Series(conf[:, 1], index = pd.date_range(start = pd.to_datetime('today').date(), periods = len(pd.DataFrame(pd.date_range(start = pd.to_datetime('today').date(), end = pd.to_datetime('today').date() + relativedelta(years =+ 5)))), freq='D'))
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(data, color = 'blue', label='Actual Price')
    plt.plot(fc_series, color = 'orange',label='Predicted Price')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    return fc_series



#-------SGB------------


#Scrapping live data from URL
df_SGB = pd.DataFrame()

for j in range(15, 22):
    for i in range (1, 13):
        data_frame = []
        url = 'https://www.livechennai.com/get_goldrate_history.asp?monthno='+str(i)+'&yearno=20'+str(j)
        response = requests.get(url)
        soup = bs(response.content, 'html.parser')
        tbl = soup.findAll('table', attrs = {'class', 'table-price'})
        data_frame = pd.read_html(str(tbl), header = 0)[0]
        df_SGB = pd.concat([df_SGB, data_frame], axis = 0)


#SQL

connection=create_engine('mysql+pymysql://root:@localhost/sgb')
try:
    df.to_sql(name = 'sgb', con = connection, if_exists = 'replace', index = False)
    df=pd.read_sql("sgb",connection)
except:
    pass


del response, i, j, soup, tbl, data_frame, url , connection


#Shape of the data
print(df_SGB.shape)

#First and last row of data
df_SGB.head()
df_SGB.tail()

#Trim the data
df_SGB = df_SGB.iloc[:, 0:2]

#Rename the data column
df_SGB.columns = ['Date', 'pure_gold']

#Change Date colunm to DateTime format 
df_SGB['Date']= pd.to_datetime(df_SGB['Date'])

#Set Date Colunm as Index
df_SGB.set_index('Date', inplace=True)


#EDA & Plots before imputation
eda(df_SGB['pure_gold'])
qqplot(df_SGB['pure_gold'])
histogram(df_SGB)
decomposition(df_SGB['pure_gold'])
acf_pacf(df_SGB['pure_gold'])



# imputation missing dates
df_SGB = imputation(df_SGB)


#EDA & Plots after imputation
eda(df_SGB['pure_gold'])
qqplot(df_SGB['pure_gold'])
histogram(df_SGB)
decomposition(df_SGB['pure_gold'])
acf_pacf(df_SGB['pure_gold'])


#Split Data into Train and Test
df_SGB_train = df_SGB.loc[:'2020-12-31', :]#loc last is inclusive
df_SGB_test = df_SGB.loc['2021-1-1':, :]



#Auto ARIMA - to find p, d, q
model_auto = auto_arima(df_SGB.dropna(), seasonal = False)
model_auto.summary()


#ARIMA for training data

arima_model = ARIMA(df_SGB_train, order=model_auto.order, freq = 'D')
fitted_SGB1 = arima_model.fit(disp = -1)
print(fitted_SGB1.summary())


# Forecast for test data
forecast_test(fitted_SGB1, df_SGB_train, df_SGB_test)



#ARIMA ---- Main model ---- SGB

arima_model = ARIMA(df_SGB.dropna(), order=model_auto.order, freq='d')
fitted_SGB = arima_model.fit(disp = -1)
print(fitted_SGB.summary())


SGB_forecast = forecast_main(fitted_SGB, df_SGB)


#Pickle
pickle.dump(fitted_SGB, open('SGB.pkl', 'wb'))




del arima_model, df_SGB_test, df_SGB_train, fitted_SGB, fitted_SGB1, model_auto



#-----------SBI------------------



#Scraping data from investpy
search_result = investpy.search_quotes(text='Sbi Life - Bond Fund', products = ['funds'], countries=['INDIA'], n_results = 1)
historical_data = search_result.retrieve_historical_data(from_date='01/01/2015', to_date = str(date.today().strftime('%d/%m/%Y')))
print(historical_data.tail())
historical_data.describe()



#SQL
connection=create_engine('mysql+pymysql://root:@localhost/sbi')
try:
    df.to_sql(name = 'sbi', con = connection, if_exists = 'replace', index = False)
    df=pd.read_sql("sbi",connection)
except:
    pass


#Trimming data
df_SBI = pd.DataFrame(historical_data, columns=["Close"])

#EDA & plot before imputation 
eda(df_SBI['Close'])
qqplot(df_SBI['Close'])
histogram(df_SBI)
decomposition(df_SBI['Close'])
acf_pacf(df_SBI['Close'])

# Handling missing dates
df_SBI = imputation(df_SBI)

#EDA & plot after imputation 
eda(df_SBI['Close'])
qqplot(df_SBI['Close'])
histogram(df_SBI)
decomposition(df_SBI['Close'])
acf_pacf(df_SBI['Close'])


del historical_data, search_result


#train, test data
train_data = df_SBI.loc[:'2021-02-28',:]
test_data =  df_SBI.loc['2021-03-01':, :]   


#AutoARIMA for bestfit
model_autoARIMA = auto_arima(df_SBI, test='adf', seasonal=False)
print(model_autoARIMA.summary())


#model building - train data
model = ARIMA(train_data, order=model_autoARIMA.order, freq = 'D')   #best fit value from autoarima
fitted = model.fit(disp=-1)
print(fitted.summary())


#forecast-test data
forecast_test(fitted, train_data, test_data)



#Main model building

model = ARIMA(df_SBI, order=model_autoARIMA.order, freq = 'D')   #best fit value from autoarima
fitted_SBI = model.fit(disp=-1)
print(fitted_SBI.summary())

#forecast main data
SBI_forecast = forecast_main(fitted_SBI, df_SBI)


#pickle file
pickle.dump(fitted_SBI, open('SBI.pkl', 'wb'))


del fitted, fitted_SBI, model, model_autoARIMA, test_data, train_data, connection




#--------------IRFC---------------



#Scrapping data

df_IRFC = get_history(symbol="IRFC",series = "N1",start=date(2015,1,1),end=date.today())
df_IRFC = df_IRFC.reset_index()
df_IRFC.drop(['Symbol','Series','Turnover',"%Deliverble"],axis =1,inplace =True)


#SQL

connection=create_engine('mysql+pymysql://root:@localhost/irfc')
try:
    df.to_sql(name = 'irfcc', con = connection, if_exists = 'replace', index = False)
    df=pd.read_sql("irfcc",connection)
except:
    pass



#Convert Data column as DateTime format
df_IRFC['Date']= pd.to_datetime(df_IRFC['Date'])


#Set Date Colunm as Index
df_IRFC.set_index('Date', inplace=True)


#Trimming Data 
df_IRFC = pd.DataFrame(df_IRFC['Close'])

#EDA & Plotting before imputation
eda(df_IRFC['Close'])
qqplot(df_IRFC['Close'])
histogram(df_IRFC)
decomposition(df_IRFC['Close'])
acf_pacf(df_IRFC['Close'])

#Handling missing date
df_IRFC = imputation(df_IRFC)


#EDA & Plotting after imputation
eda(df_IRFC['Close'])
qqplot(df_IRFC['Close'])
histogram(df_IRFC)
decomposition(df_IRFC['Close'])
acf_pacf(df_IRFC['Close'])


#Adfuller for d value
adfuller_test(df_IRFC)

#d = 0

#ACF & PACF for finding p & q

acf_pacf(df_IRFC['Close'])

#p = 1
#q = 1


#Splitting Data
df_IRFC_train = df_IRFC.loc[:'2020-12-31', :]#loc last is inclusive
df_IRFC_test = df_IRFC.loc['2021-01-01':, :]



### Training model ARIMA
arima_model = ARIMA(df_IRFC_train, order=(1, 0, 1), freq = 'D')
fitted_IRFC = arima_model.fit(disp=-1)
print(fitted_IRFC.summary())

#Forecast - test data
forecast_test(fitted_IRFC, df_IRFC_train, df_IRFC_test)



#Main Forecast
arima_model = ARIMA(df_IRFC, order=(1, 0, 1), freq = 'D')
fitted_IRFC = arima_model.fit(disp=-1)
print(fitted_IRFC.summary())

#Forecast - main data
IRFC_forecast = forecast_main(fitted_IRFC, df_IRFC)

#pickle
pickle.dump(fitted_IRFC, open('IRFC.pkl', 'wb'))


del arima_model, df_IRFC_train, df_IRFC_test, fitted_IRFC, connection
