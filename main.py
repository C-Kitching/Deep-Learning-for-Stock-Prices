# HEADER FILE
# Name: Christopher Robert Kitching
# Email: christopher.kitching@manchester.ac.uk
# Last edited: 04/10/22
# Title: Deep Learning for Stock Prices
# Description: 

# Imports
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from alpha_vantage.timeseries import TimeSeries

print("All libraries loaded")

def get_config():
    """Set config for the file

    Returns:
        config (dict{}): config for file
    """

    config = {
        "alpha_vantage": {
            "key": "demo", 
            "symbol": "IBM",
            "outputsize": "full",
            "key_adjusted_close": "5. adjusted close",
        },
        "data": {
            "window_size": 20,
            "train_split_size": 0.80,
        }, 
        "plots": {
            "xticks_interval": 90, # show a date every 90 days
            "color_actual": "#001f3f",
            "color_train": "#3D9970",
            "color_val": "#0074D9",
            "color_pred_train": "#3D9970",
            "color_pred_val": "#0074D9",
            "color_pred_test": "#FF4136",
        },
        "model": {
            "input_size": 1, # since we are only using 1 feature, close price
            "num_lstm_layers": 2,
            "lstm_size": 32,
            "dropout": 0.2,
        },
        "training": {
            "device": "cpu", # "cuda" or "cpu"
            "batch_size": 64,
            "num_epoch": 100,
            "learning_rate": 0.01,
            "scheduler_step_size": 40,
        }
    }

    return config

def download_data(config):
    """Download stock data from API in reverse chronological order

    Args:
        config (dict{}): config for file

    Returns:
        date_data (str[]): array of dates
        data_close_price (float[]): array of close price values
        num_data_points (int): number of data points in sample
        display_data_range (str): time period over which dates occur
    """

    # get full IBM data
    ts = TimeSeries(key='demo')
    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], 
                      outputsize=config["alpha_vantage"]["outputsize"])

    # extract the dates and put in time order
    data_date = [date for date in data.keys()]
    data_date.reverse()

    # get close price of stock
    data_close_price = [float(data[date][config["alpha_vantage"]
                       ["key_adjusted_close"]]) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    # calculate number of data points
    num_data_points = len(data_date)

    # set the date range
    display_date_range = ("from " + data_date[0] + " to "  
                          + data_date[num_data_points-1])

    # print number of data points 
    print("Number data points", num_data_points, display_date_range)

    # return data 
    return data_date, data_close_price, num_data_points, display_date_range

def plot_time_series_data(data_date, data_close_price, num_data_points, 
                          display_date_range, config):
    """Plot time series data for a stock

    Args:
        date_data (str[]): array of dates
        data_close_price (float[]): array of close price values
        num_data_points (int): number of data points in sample
        display_data_range (str): time period over which dates occur
        config (dict{}): configuration for file
    """

    # format figure
    fig = figure(figsize=(25, 5), dpi=80) 
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] 
              + ", " + display_date_range)
    plt.grid(visible = None, which = 'major', axis = 'y', linestyle = '--')

    # plot data
    plt.plot(data_date, data_close_price, 
             color = config["plots"]["color_actual"]) 

    # format xticks nicely
    xticks = [""] * num_data_points
    for i in range(num_data_points):
        if((i%config["plots"]["xticks_interval"] == 0 
           and (num_data_points-i) > config["plots"]["xticks_interval"]) 
           or i == num_data_points-1):
            xticks[i] = data_date[i]
        else:
            xticks[i] = None
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation = 'vertical')

    plt.show() # show graph

class Normalizer():
    """Class to normalise data
    """

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        """Normalise input data

        Args:
            x (float[]): input data to normalise

        Returns:
            nromalised_x (float[]): normalised data
        """
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalised_x = (x - self.mu)/self.sd
        return normalised_x

    def inverse_transform(self, x):
        """Inverse normalisation

        Args:
            x (float[]): data to unnormalise

        Returns:
            (float[]): unnormalised data
        """
        return (x*self.sd) + self.mu

def prepare_data_x(x, window_size):
    """Given a data set x of size N and a target window size of w,
    perform a sliding window transformation
    Start at the first element and slice an array of size w and store
    Move to the next element and slice a array of size w and store
    Continue until reaching the -w element of x, at which point the 
    most recent slice will end exactly at x[-1]
    Output will have shape (N - w + 1, w)

    Args:
        x (float[]): data to perform sliding window transformation on
        window_size (int): size of window

    Returns:
        output (float[][]): data after perfoming sliding window transform
    """
    # number of data arrays we're splitting data into
    n_row = x.shape[0] - window_size + 1

    # perform the sliding window transformation
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), 
             strides=(x.strides[0], x.strides[0]))

    # output the results, splitting the last array off, as this is the data
    # we will ultimately use to predict the next days value
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    """Get the target values for the sliding window data
    I.e the sliding window transformation turned the data into arrays like
    x[:w], x[1:w+1], ..., x[-w:]
    We are using these to predict the next value of the time series, i.e the 
    corresponding target values are 
    x[w], x[w+1], ..., x[-1]
    So we put all these target values into one array with x[w:]

    Args:
        x (float[]): data to perform transformation on
        window_size (int): size of sliding window transformation that was 
                           performed on x

    Returns:
        output (float[]): target values of sliding window transformation
    """
    output = x[window_size:]
    return output

def plot_time_series_data_split_into_train_and_val(
    data_date, to_plot_data_y_train, to_plot_data_y_val, 
    num_data_points, config):
    """Plot time series data divided into training and validation dats

    Args:
        date_data (str[]): array of dates
        to_ploy_data_y_train (float[]): training data
        to_plot_data_y_val (float[]): validation data
        num_data_points (int): number of data points in sample
        config (dict{}): configuration for file
    """

    # format graph
    fig = figure(figsize=(25, 5), dpi = 80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.title("Daily close prices for " + config["alpha_vantage"]["symbol"] 
              + " - showing training and validation data")
    plt.grid(visible = None, which = 'major', axis = 'y', linestyle = '--')

    # plot data
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", 
             color = config["plots"]["color_train"])
    plt.plot(data_date, to_plot_data_y_val, label = "Prices (validation)", 
             color = config["plots"]["color_val"])

    # format xticks nicely
    xticks = [""] * num_data_points
    for i in range(num_data_points):
        if((i%config["plots"]["xticks_interval"] == 0 
           and (num_data_points-i) > config["plots"]["xticks_interval"]) 
           or i == num_data_points-1):
            xticks[i] = data_date[i]
        else:
            xticks[i] = None
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    
    plt.legend() # add legend
    plt.show() # show graph



def main():
    """Main function
    """

    # get configuration for file
    config = get_config()

    # download the data
    data_date, data_close_price, num_data_points, display_date_range \
    = download_data(config)

    # plot time series data for IBM
    #plot_time_series_data(data_date, data_close_price, num_data_points, 
    #                       display_date_range, config)

    # Normalise the data for LSTM
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    # perform sliding window transformation on the data
    # x is the data we will feed into the model, i.e previous 20 days of prices
    # y is the target values, i.e price on 21st day
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, 
                                           window_size = 
                                           config["data"]["window_size"])
    data_y = prepare_data_y(normalized_data_close_price, 
                            window_size=config["data"]["window_size"])

    # split data into trainning and validation data
    split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    # declare empty arrays that will hold plotting data
    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val = np.zeros(num_data_points)

    # seperate the data into training and validation to distinguish in plot
    # note that we have to unnormalise the data for plotting
    # note that we also remove the first 20 values of the data
    # this is because the window we are using in the model is of size 20
    # so we won't be able to predict those values, doing so would require
    # previous time values which we do not have i.e x[-1:19] is needed to
    # predict x[19]
    to_plot_data_y_train[config["data"]["window_size"] : 
                         split_index + config["data"]["window_size"]] \
                         = scaler.inverse_transform(data_y_train)
    to_plot_data_y_val[split_index + config["data"]["window_size"] :] \
                       = scaler.inverse_transform(data_y_val)

    # replace zero values in the data with None
    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, 
                                    to_plot_data_y_train)
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, 
                                  to_plot_data_y_val)

    
    # plot graph of training and validation data split
    plot_time_series_data_split_into_train_and_val(
        data_date, to_plot_data_y_train, to_plot_data_y_val, 
        num_data_points, config)


    



# run file
if __name__ == "__main__":
    main()