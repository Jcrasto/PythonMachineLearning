import pandas
import os
import matplotlib.pyplot as plt
import numpy
import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import math


def plot_curve(covid,country,threshold):
    top = covid[covid['Country/Region'].isin(topcases)].groupby(
        ['ObservationDate', 'Country/Region']).sum().sort_values(['ObservationDate', 'Confirmed'],
                                                                 ascending=False)
    top = top[top['Confirmed'] > threshold]
    top['Confirmed'] = top['Confirmed'] / max(top['Confirmed'])
    y_data = np.array(top[::-1].xs(country, level=1)['Confirmed'])
    x_data = np.array([i for i in range(len(y_data))])
    log_y_data = np.log(y_data)
    curve_fit = np.polyfit(x_data, log_y_data, 1)
    y = np.exp(curve_fit[1]) * np.exp(curve_fit[0]*x_data)
    plt.plot(x_data, y_data, "o")
    plt.plot(x_data, y)
    plt.title(country)
    plt.show()
    return()

def fit_curve(top,country):
    y_data = np.array(top[::-1].xs(country, level=1)['Confirmed'])
    x_data = np.array([i for i in range(len(y_data))])
    log_y_data = np.log(y_data)
    curve_fit = np.polyfit(x_data, log_y_data, 1)
    y = np.exp(curve_fit[1]) * np.exp(curve_fit[0]*x_data)
    return(math.sqrt(mean_squared_error(y_data,y)))

if __name__ == '__main__':
    covid = pandas.read_csv(os.environ['DATADIR'] + '/covid_data/covid_19_data.csv')
    covid['ObservationDate']= pandas.to_datetime(covid['ObservationDate'])
    topcases = list(covid[covid['ObservationDate'] == covid['ObservationDate'].max()].groupby(['Country/Region']).sum().sort_values('Confirmed',ascending=False).head(10).index)
    for country in topcases:
        min_rmse = float('inf')
        threshold = 0
        print(country)
        for i in range(50):
            top = covid[covid['Country/Region'].isin(topcases)].groupby(
                ['ObservationDate', 'Country/Region']).sum().sort_values(['ObservationDate', 'Confirmed'],
                                                                         ascending=False)
            top = top[top['Confirmed'] > i]
            top['Confirmed'] = top['Confirmed'] / max(top['Confirmed'])
            rmse = fit_curve(top,country)
            if rmse < min_rmse:
                threshold = i
                min_rmse = rmse
        print(min_rmse)
        print(threshold)
        #plot_curve(covid, country, threshold)