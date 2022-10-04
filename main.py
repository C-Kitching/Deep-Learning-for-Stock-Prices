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
    """Download stock data from API

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

def plot_time_series_data(config):
    """Plot time series data for a stock

    Args:
        config (dict{}): configuration for file
    """

    # download the data
    data_date, data_close_price, num_data_points, display_date_range \
    = download_data(config)
    
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

    # show graph
    plt.show()


def main():
    """Main function
    """

    # get configuration for file
    config = get_config()

    # plot time series data for IBM
    plot_time_series_data(config)


    



# run file
if __name__ == "__main__":
    main()