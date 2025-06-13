# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:12 2022
Code to grab historical stock data for the purposes of training an AI/ML time series model
@author: Brendan

To Do List:
2) Generalize the plotting steps
3) Add stock volume as 2nd input
4) Add S&P or other index as 3rd input
5) Add sentiment analysis
6) Add MACD
7) Improve screen output
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from alpha_vantage.timeseries import TimeSeries 

# Configuration parameters reference endpoint here: https://www.alphavantage.co/documentation/#dailyadj

#Stock data API information
AVAPIKEY = "PYEDBBG84BI5QBN2" # Original free key WQ3ADFCTZHK0PJHZ"
SYMBOL = "IBM" #Stock symbol you want to train the model on
OUTPUTSIZE = "full" #accepts compact or full use compact for testing, full for actual training and run.
DATATYPE = "json" #accepts json or csv
WINDOWSIZE = 20 #Used to define the running average length applied to the data to train the model
TRAINSPLIT = 0.80 #This is the percentage of data you want to use for training, the rest will be used for validation

#Plot parameters
TICKINT = 180 # show a date every 90 days
CLR_ACT = "#001f3f"
CLR_TRAIN = "#3D9970"
CLR_VAL = "#0074D9"
CLR_PRED_TRN = "#3D9970"
CLR_PRED_VAL = "#0074D9"
CLR_PRED_TEST = "#FF4136"

#Model parameters
FEATURES = 1 # since we are only using 1 feature, close price for now, but will add volume shortly
LTSM_LAYERS = 2
LTSM_SIZE = 32
DROPOUT = 0.2
DEVICE = "cuda"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.01
STEP_SIZE = 50

print("All libraries loaded")
print(f"Pulling {OUTPUTSIZE} data for {SYMBOL}")

#Normalize class to norm and reverse norm datasets must be passed a list with no null values
class Normalizer():
    def __init__(self):
        self.mean = None
        self.stdev = None
    
    def norm(self, x):    
        self.mean = np.mean(x, axis = (0), keepdims=True)
        self.stdev = np.std(x, axis = (0), keepdims = True)
        norm_data = (x - self.mean)/self.stdev
        return norm_data
    
    def rev_norm(self, x):
        return (x*self.stdev) + self.mean

#Custom dataset class to feed to the pytorch dataloader.  
#Must have init, len, and getitem.  Init is only run once on instantiation of course.
#len returns the number of samples in our dataset.  Of course, x and y are the same length so we only reference x
#getitem loads and returns a sample from the dataset at the given index (idx).
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        #We need to put the x data into the form of (L,N,Hin) where L is sequence length, N is batch size and Hin is input size if batch_first = false which is default
        #or (N, L, Hin) if batch_first=true as chosen currently
        x = np.expand_dims(x, 2) 
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])    

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]

#Function to call time series data from Alpha Vantage
def get_data():
    ts = TimeSeries(key = AVAPIKEY)
    data, meta_data = ts.get_daily_adjusted(SYMBOL, outputsize = OUTPUTSIZE)
    print( f"Data pull for {SYMBOL} was successful resulting in {len(data)} datapoints")
    return(data, meta_data)

#Function to plot stock timeseries data
def showplot(data_date, data_close_price, title):
    #Setup info
    fig = figure(figsize=(25, 20), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color= CLR_ACT)
    font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 22}

    matplotlib.rc('font', **font)    
    matplotlib.rc('xtick', labelsize = 20)
    matplotlib.rc('ytick', labelsize = 20)
    
    #Calculate tickmark frequency
    xticks = []
    display_date_range = "from" + data_date[0] + " to " + data_date[len(data_date)-1]
    for i in range(len(data_date)):
        if i%TICKINT==0 and len(data_date)-i > TICKINT or i==len(data_date)-1:
            xticks.append(data_date[i])
        else:
            xticks.append(None)
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')

    #More format information
    plt.axis('tight')
    plt.title(title + display_date_range)
    plt.grid(visible=None, which='major', axis='y', linestyle='-')
    plt.show()
    
def run_epoch(dataloader, is_training=False):
    epoch_loss= 0
    
    if is_training:
        model.train()
    else:
        model.eval()
    
    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()
        print("idx is:", idx)
        batchsize = x.shape[0]
        print(batchsize)
        
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        out = model(x)
        print("Out: ", out)
        loss = criterion(out.contiguous(),y.contiguous())
        print("loss: ", loss)
        
        if is_training:
            loss.backward()
            optimizer.step()
        
        epoch_loss += (loss.detach().item() / batchsize)
            
    lr = scheduler.get_last_lr()[0]
    
    return(epoch_loss, lr)

if __name__ == "__main__":
    
    #Pull stock data from API
    #Separate out necessary data; Alpha Vantage returns data recent to oldest so that has to be corrected.
    data, meta_data = get_data()
    date_info = [date for date in data.keys()]
    date_info.reverse()
    data_close = [data[date]['5. adjusted close'] for date in data.keys()]
    data_close.reverse()
    #Convert closing data from string to float
    data_close = [float(i) for i in data_close]

    #Plot original stock information
    title = "Daily closing price for " + SYMBOL + ", "     
    showplot(date_info, data_close, title)
    
    #Instantiate the normalizer / reverse class
    xform = Normalizer()
    
    norm_close = xform.norm(data_close)

    #Plot normalized stock data 
    title = "Daily normalized closing price for " + SYMBOL + ", "        
    showplot(date_info, norm_close, title)
    
    #This takes an input array of length X and a desired sequence length of Y and step size of Z
    #It creates an output array of shape (FEATURE, X-Y, Y)
    #where it creates overlapping steps of 0, 1, 2, ...  to Y-1
    #                                         1, 2, ...  to Y
    #                                            2, ...  to Y, Y+1 for a step size of 1
    
    #                                      0, 1, 2, ...  to Y-1
    #                                            2, ...  to Y+1, Y+2 for a step size of 2
    # etc., etc.
    step_size = 1
    n_steps = int((len(norm_close)-WINDOWSIZE)/step_size)
    print("Original array length is: ", len(norm_close), "\n")
    print("Length of new array will be: ", n_steps, "\n")
    xdata = np.zeros(shape=(n_steps, WINDOWSIZE))
    for j in range(n_steps):
        for k in range(WINDOWSIZE):
            xdata[j][k] = norm_close[k+j*step_size]
    
    data_x_unseen = norm_close[-WINDOWSIZE:]
    
    #Here we get the expected output to pair up with the xdata
    #For xdata, we took time series input data into repeating, overlapping chunks of WINDOWSIZE.
    #For ydata, we need one data point to correspond to each WINDOWSIZE chunk that is the next day's target (WINDOWSIZE+1)
    ydata = norm_close[WINDOWSIZE:]

    # split the dataset
    split_index = int(ydata.shape[0]*TRAINSPLIT)
    data_x_train = xdata[:split_index]
    data_x_val = xdata[split_index:]
    data_y_train = ydata[:split_index]
    data_y_val = ydata[split_index:]    

    #Define training and validation datasets in the expected class for pytorch dataloader
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape, "\n")
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape, "\n") 
    
    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(input_size=FEATURES, hidden_layer_size=LTSM_SIZE, num_layers=LTSM_LAYERS, output_size=1, dropout=DROPOUT)
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)

    for epoch in range(EPOCHS):
        loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader)
        scheduler.step()
    
        print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
                  .format(epoch+1, EPOCHS, loss_train, loss_val, lr_train))
    print("\n")
    
    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize

    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    # predict on the validation data, to see how the model does

    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(DEVICE)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    # prepare data for plotting

    to_plot_data_y_train_pred = np.zeros(len(date_info))
    to_plot_data_y_val_pred = np.zeros(len(date_info))
    
    to_plot_data_y_train_pred[WINDOWSIZE:split_index+WINDOWSIZE] = xform.rev_norm(predicted_train)
    to_plot_data_y_val_pred[split_index+WINDOWSIZE:] = xform.rev_norm(predicted_val)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

    # plots

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(date_info, data_close, label="Actual prices", color=CLR_ACT)
    plt.plot(date_info, to_plot_data_y_train_pred, label="Predicted prices (train)", color=CLR_PRED_TRN)
    plt.plot(date_info, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=CLR_PRED_VAL)
    plt.title("Compare predicted prices to actual prices")
    xticks = [date_info[i] if ((i%TICKINT==0 and (len(date_info)-i) > TICKINT) or i==len(date_info)-1) else None for i in range(len(date_info))] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()
    
    model.eval()

    x = torch.tensor(data_x_unseen).float().to(DEVICE).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()

    # prepare plots

    plot_range = 10
    to_plot_data_y_val = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)

    to_plot_data_y_val[:plot_range-1] = xform.rev_norm(data_y_val)[-plot_range+1:]
    to_plot_data_y_val_pred[:plot_range-1] = xform.rev_norm(predicted_val)[-plot_range+1:]

    to_plot_data_y_test_pred[plot_range-1] = xform.rev_norm(prediction)
    
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    # plot

    plot_date_test = date_info[-plot_range+1:]
    plot_date_test.append("tomorrow")

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=CLR_ACT)
plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=CLR_PRED_TRN)
plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=CLR_PRED_VAL)
plt.title("Predicted close price of the next trading day")
plt.grid(visible=None, which='major', axis='y', linestyle='--')
plt.legend()
plt.show()

print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))