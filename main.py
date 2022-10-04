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

def main():
    """Main funciton
    """

    print("All libraries loaded")


# run file
if __name__ == "__main__":
    main()
