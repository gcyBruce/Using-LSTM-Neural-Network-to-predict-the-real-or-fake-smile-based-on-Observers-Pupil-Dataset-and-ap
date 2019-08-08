#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:29:18 2019

@author: gongchaoyun
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import Imputer


#read the data for Assignment 1
df=pd.read_excel('Asian_Observers_Pupil_Data.xlsx',header = 2)
data = np.array(df)

#red the data for Assignment 2. data_r is the real smile data, data_f is the fake smile data.
#data_r = np.full([2168,174], np.nan)
#data_f = np.full([2168,196], np.nan)
data_r = np.zeros([2168,174])
data_f = np.zeros([2168,196])
df_lr = pd.read_excel('./smiles_v2/LEPD.xlsx',header = 2,sheetname=[0,1,2,3,4,5,6,7,8])
df_rr = pd.read_excel('./smiles_v2/REPD.xlsx',header = 2,sheetname=[0,1,2,3,4,5,6,7,8])
df_lf = pd.read_excel('./smiles_v2/LEPD.xlsx',header = 2,sheetname=[9,10,11,12,13,14,15,16,17,18])
df_rf = pd.read_excel('./smiles_v2/REPD.xlsx',header = 2,sheetname=[9,10,11,12,13,14,15,16,17,18])


#drop the p3 data, because p3 is not from Asia
#and supplement data for Nan (the lost data),copy the data from the before previous value for Nan, if no previous value, using 0 instead
for i in range(9):
    df_lr[i] = df_lr[i].drop(['p3'],axis=1)
    df_rr[i] = df_rr[i].drop(['p3'],axis=1)
    df_lr[i] = df_lr[i].fillna(method = 'ffill', axis = 0)
    df_rr[i] = df_rr[i].fillna(method = 'ffill', axis = 0)
    df_lr[i] = df_lr[i].fillna(0)
    df_rr[i] = df_rr[i].fillna(0)
for i in range(10):
    df_lf[i+9] = df_lf[i+9].drop(['p3'],axis=1)
    df_rf[i+9] = df_rf[i+9].drop(['p3'],axis=1)
    df_lf[i+9] = df_lf[i+9].fillna(method = 'ffill', axis = 0)
    df_rf[i+9] = df_rf[i+9].fillna(method = 'ffill', axis = 0)
    df_lf[i+9] = df_lf[i+9].fillna(0)
    df_rf[i+9] = df_rf[i+9].fillna(0)

#merge the data to matrix data_r and data_f, and transform the type to numpy
col_total = 20
for i in range(9):
    if i == 0 :
        data_r[:1151,0:10] = np.array(df_lr[i].iloc[:,:10])
        data_r[:1151,10:20] = np.array(df_rr[i].iloc[:,:10])
    else:
        row_l,col_l = df_lr[i].shape
        row_r,col_r = df_rr[i].shape
        data_r[:row_l,col_total:(col_total+col_l)] = np.array(df_lr[i])
        data_r[:row_r,(col_total+col_l):(col_total+col_l+col_r)] = np.array(df_rr[i])
        col_total = col_total + col_l + col_r

col_total = 0
for i in range(10):
    i = i+9
    row_l,col_l = df_lf[i].shape
    row_r,col_r = df_rf[i].shape
    data_f[:row_l,col_total:(col_total+col_l)] = np.array(df_lf[i])
    data_f[:row_r,(col_total+col_l):(col_total+col_l+col_r)] = np.array(df_rf[i])
    col_total = col_total + col_l + col_r


'''
imp_f  = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)
imp_f.fit(data_f)
imp_f.transform(data_f)

imp_r  = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)
imp_r.fit(data_r)
imp_r.transform(data_r)
'''
#split the data to train data and test data. 300 (150 real, 150 fake) for train, 70 (24 real, 46 fake) for test.
data_train = np.zeros([2168,300])
label_train = np.zeros([300,1])

for i in range(300):
    if i%2 == 0:
        data_train[:,i] = data_r[:,int(i/2)]
        label_train[i] = 1
    elif i%2 == 1:
        data_train[:,i] = data_f[:,int((i-1)/2)]
        label_train[i] = 0

#data_train = np.concatenate((data_r[:,:150],data_f[:,:150]),axis=1)
#label_train = np.concatenate((np.ones([150,1]),np.zeros([150,1])),axis=0)   
data_test = np.concatenate((data_r[:,150:],data_f[:,150:]),axis=1)
label_test = np.concatenate((np.ones([24,1]),np.zeros([46,1])),axis=0)   



# identify hyperparameters
smile = [0,1]
seq_length = 1168 # number of steps to unroll the RNN for
batch_size = 5
learning_rate = 0.001
num_epochs = 50
hidden_size = 10*seq_length # size of hidden layer of neurons
hidden_size2 = 200
hidden_size3 = 20
loss_train = []
loss_test = []
loss_test2 = []
accuracy_train = []
accuracy_test = []
accuracy_test2 = []

#Define a LSTM neural network 
class _smileLSTM(nn.Module):
	def __init__(self):
		super(_smileLSTM, self).__init__()
		self.LSTM = nn.LSTM(1, 10, 1)
		self.FC1 = nn.Linear(hidden_size, hidden_size2)
		self.FC2 = nn.Linear(hidden_size2,hidden_size3)
		self.FC3 = nn.Linear(hidden_size3,len(smile))

	def forward(self, input, seq_len):
		input = input.view(input.data.shape[0], batch_size, 1)#input.data.shape[1])
		output, hc = self.LSTM(input.float())
		output = output.view(-1,hidden_size)
		output = self.FC1(output)
		output = torch.sigmoid(output)
		output = F.dropout(output, training=self.training)
		output = self.FC2(output)
		output = torch.sigmoid(output)
		output = F.dropout(output, training=self.training)
		output = self.FC3(output)
		return output.view(-1,len(smile))

#build four patterns to prune network by applying distinctiveness 
smileLSTM1 = _smileLSTM()
smileLSTM2 = _smileLSTM()
smileLSTM3 = _smileLSTM()
smileLSTM4 = _smileLSTM()
#identify CrossEntropy as loss function
criterion = nn.CrossEntropyLoss()

#identify Stochastic gradient descent to optimize model
optimizer = optim.SGD(smileLSTM1.parameters(), lr=learning_rate)
optimizer2 = optim.SGD(smileLSTM2.parameters(), lr=learning_rate)
optimizer3 = optim.SGD(smileLSTM3.parameters(), lr=learning_rate)
optimizer4 = optim.SGD(smileLSTM4.parameters(), lr=learning_rate)

#training the neural network    
def train1(epoch):
    smileLSTM1.train()
    correct = 0.0
    train_loss = 0
    for batch_idx in range(int(75/batch_size)):
        inputs = data_train[:seq_length,(batch_size*batch_idx):(batch_size*batch_idx+batch_size)]
        #inputs = [data_train[m,[batch_idx]] for m in range(seq_length)]
        input = torch.from_numpy(np.array(inputs).reshape(seq_length,batch_size)).float()
        targets = [int(label_train[batch_size*batch_idx+i]) for i in range(batch_size)]
        targets = torch.from_numpy(np.array(targets)).long()

        optimizer.zero_grad()
        output = smileLSTM1(input,seq_length)
        #print(output,targets)
        loss = criterion(output, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx+1,int(75/batch_size),
                100. * (batch_idx+1) * batch_size / 75, loss.item()))
        _, predicted = torch.max(output, 1)
        
        # calculate and print accuracy
        correct = sum(predicted.data.numpy() == targets.data.numpy())+correct
    accuracy = 100*correct/75
    train_loss /= int(75/batch_size)
    print('\nTrain set: Average loss: {:.4f}\n'.format(
        train_loss))
    print('\nTrain set: Accuracy: {:.4f}%\n'.format(
        accuracy))
    loss_train.append(train_loss)
    accuracy_train.append(accuracy)

def train2(epoch):
    smileLSTM2.train()
    train_loss = 0
    for batch_idx in range(int(75/batch_size)):
        batch_idx = batch_idx + int(75/batch_size)
        inputs = data_train[:seq_length,(batch_size*batch_idx):(batch_size*batch_idx+batch_size)]
        #inputs = [data_train[m,[batch_idx]] for m in range(seq_length)]
        input = torch.from_numpy(np.array(inputs).reshape(seq_length,batch_size)).float()
        targets = [int(label_train[batch_size*batch_idx+i]) for i in range(batch_size)]
        targets = torch.from_numpy(np.array(targets)).long()
        optimizer2.zero_grad()
        output = smileLSTM2(input,seq_length)
       
        loss = criterion(output, targets)
        train_loss += loss.item() 
        loss.backward()
        optimizer2.step()

def train3(epoch):
    smileLSTM3.train()
    train_loss = 0
    for batch_idx in range(int(75/batch_size)):
        batch_idx = batch_idx + int(150/batch_size)
        inputs = data_train[:seq_length,(batch_size*batch_idx):(batch_size*batch_idx+batch_size)]
        #inputs = [data_train[m,[batch_idx]] for m in range(seq_length)]
        input = torch.from_numpy(np.array(inputs).reshape(seq_length,batch_size)).float()
        targets = [int(label_train[batch_size*batch_idx+i]) for i in range(batch_size)]
        targets = torch.from_numpy(np.array(targets)).long()
        optimizer3.zero_grad()
        output = smileLSTM3(input,seq_length)
       
        loss = criterion(output, targets)
        train_loss += loss.item() 
        loss.backward()
        optimizer3.step()

def train4(epoch):
    smileLSTM4.train()
    train_loss = 0
    for batch_idx in range(int(75/batch_size)):
        batch_idx = batch_idx + int(225/batch_size)
        inputs = data_train[:seq_length,(batch_size*batch_idx):(batch_size*batch_idx+batch_size)]
        #inputs = [data_train[m,[batch_idx]] for m in range(seq_length)]
        input = torch.from_numpy(np.array(inputs).reshape(seq_length,batch_size)).float()
        targets = [int(label_train[batch_size*batch_idx+i]) for i in range(batch_size)]
        targets = torch.from_numpy(np.array(targets)).long()
        optimizer4.zero_grad()
        output = smileLSTM4(input,seq_length)
       
        loss = criterion(output, targets)
        train_loss += loss.item() 
        loss.backward()
        optimizer4.step()        

#apply test data to test the train1 model
def test():
    smileLSTM1.eval()
    correct = 0.0
    test_loss = 0
    for batch_idx in range(int(70/batch_size)):
        inputs = data_test[:seq_length,(batch_size*batch_idx):(batch_size*batch_idx+batch_size)]
        input = torch.from_numpy(np.array(inputs).reshape(seq_length,batch_size)).float()
        targets = [int(label_test[batch_size*batch_idx+i]) for i in range(batch_size)]
        targets = torch.from_numpy(np.array(targets)).long()

        output = smileLSTM1(input,seq_length)
        
        loss = criterion(output, targets)
        test_loss += loss.item()
        _, predicted = torch.max(output, 1)
        # calculate and print accuracy
        correct = sum(predicted.data.numpy() == targets.data.numpy())+correct
    accuracy = 100*correct/70

    test_loss /= int(70/batch_size)
    print('\n\Test : Average loss: {:.4f}\n'.format(
        test_loss))
    print('\nTest set: Accuracy: {:.4f}%\n'.format(
        accuracy))
    loss_test.append(test_loss)
    accuracy_test.append(accuracy)

# after pruning, get the performence of new model by using test data  
def test_after_pruning():
    smileLSTM1.eval()
    correct = 0.0
    test_loss = 0
    for batch_idx in range(int(70/batch_size)):
        inputs = data_test[:seq_length,(batch_size*batch_idx):(batch_size*batch_idx+batch_size)]
        input = torch.from_numpy(np.array(inputs).reshape(seq_length,batch_size)).float()
        targets = [int(label_test[batch_size*batch_idx+i]) for i in range(batch_size)]
        targets = torch.from_numpy(np.array(targets)).long()

        output = smileLSTM1(input,seq_length)

        
        loss = criterion(output, targets)
        test_loss += loss.item()
        _, predicted = torch.max(output, 1)
        # calculate and print accuracy
        correct = sum(predicted.data.numpy() == targets.data.numpy())+correct
    accuracy = 100*correct/70

    test_loss /= 70
    print('\n\Test_after_pruning : Average loss: {:.4f}\n'.format(
        test_loss))
    print('\nTest set_after_pruning: Accuracy: {:.4f}%\n'.format(
        accuracy))
    loss_test2.append(test_loss)
    accuracy_test2.append(accuracy)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))      

# distinctiveness algorithm
def distinctiveness(smileLSTM1,smileLSTM2,smileLSTM3,smileLSTM4,):
    # get the weights of Full connect layer 2 for four patterns
    for name, param in smileLSTM1.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
        if name =='FC2.weight':
            param1 = np.array(param.data)
    for name, param in smileLSTM2.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
        if name =='FC2.weight':
            param2 = np.array(param.data)
    for name, param in smileLSTM3.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
        if name =='FC2.weight':
            param3 = np.array(param.data)
    for name, param in smileLSTM4.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
        if name =='FC2.weight':
            param4 = np.array(param.data)
# apply sigmoid activition function for weights
    param1 = sigmoid(param1)
    param2 = sigmoid(param2)    
    param3 = sigmoid(param3)
    param4 = sigmoid(param4)
    
    param = np.vstack((param1.reshape(1,-1),param2.reshape(1,-1),param1.reshape(1,-1),param1.reshape(1,-1)))
    param = param - 0.5
# compare the similarity of weights vectors of four patterns in pattern space by pairs. if angle is smaller than 15 or larger than 165, delate the parameters.  
    for i in range((len(param[0])-1)):
        for j in range((len(param[0])-1-i)):
            x = param[:,i]
            y = param[:,j+i]
            num = float(x.T @ y)  
            denom = np.linalg.norm(x) * np.linalg.norm(y)  
            cos_angle = num / denom 
            angle=np.arccos(cos_angle)
            angle2=angle*360/2/np.pi
            #print(angle2)
            if angle2 < 15 or angle > 165:
                param[:,i] = 0

    return param



# main function
for epoch in range(1, num_epochs + 1):
    train1(epoch)
    test()
    train2(epoch)
    train3(epoch)
    train4(epoch)
param = distinctiveness(smileLSTM1,smileLSTM2,smileLSTM3,smileLSTM4)
new_param = param.ravel()[np.flatnonzero(param)].reshape(4,-1)
new_FC2weight = torch.from_numpy(param[0].reshape(hidden_size3,hidden_size2))
for name, param in smileLSTM1.named_parameters():
    if name =='FC2.weight':
        param.data = new_FC2weight
test_after_pruning()

