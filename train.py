#!/usr/bin/env python
# coding: utf-8

import numpy as np
#import torch
#from torch import nn, optim
from comet_ml import Experiment
from sklearn.datasets import load_digits

def dot(x, w):
    return np.dot(x, w)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = a.max() * 1e-4
    exp_a = np.exp(a - c)
    sum_exp_a = exp_a.sum(axis=1)
    sum_exp_a = sum_exp_a.reshape(1, sum_exp_a.size).transpose(1,0)
    value = exp_a / sum_exp_a
    return value

def cross_entropy_error(x, y):
    if x.ndim == 1:
        x = x.reshape(1, x.size)
        y = y.reshape(1, y.size)
    batch_size = x.shape[0]
    value = -np.sum(np.log(x[np.arange(batch_size), y] + 1e-7)) / batch_size
    return value

def numerical_diff(f, x, i):
    h = 1e-4
    h_vec = np.zeros_like(x)
    h_vec[i] = h
    return (f(x + h_vec) - f(x - h_vec)) / (2*h)

def numerical_diff2(f, x, i, j):
    h = 1e-4
    h_vec = np.zeros_like(x)
    h_vec[i, j] = h
    return (f(x + h_vec) - f(x - h_vec)) / (2*h)

def numerical_gradient(f, x):
    grad = np.zeros_like(x).astype(np.longdouble)
    n, m = x.shape
    for i in range(n):
        for j in range(m):
            grad[i, j] = numerical_diff2(f, x, i, j)
    return grad

def main():
    digits = load_digits()
    x = digits.data
    y = digits.target
    w = np.random.randn(x.shape[1], 10) * 1e-4
    #x = torch.DoubleTensor(x)
    #y = torch.Tensor(y)
    #x = x.cuda()
    #y = y.cuda()
    #net = nn.Linear(x.size()[1], 10)
    #loss_fn = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.01)
    #net = net.cuda()
    
    losses = []
    grads = []
    train_size = x.shape[0]
    batch_count = 18
    batch_size = train_size // batch_count
    gamma = 1e-3
    epoch = 30
    
    hyper_params = {"learning_rate": gamma, "epoch": epoch, "batch_count": batch_count, "batch_size": batch_size}
    experiment = Experiment()
    experiment.log_parameters(hyper_params)
    for param, value in hyper_params.items():
        print(f'{param}: {value}')

    for epoc in range(epoch):
        train_loss = 0.
        for batch in range(batch_count):
            #optimizer.zero_grad()
            #y_pred = net(x)
            #loss = loss_fn(y_pred, y)
            #loss.backward()
            #optimizer.step()
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x[batch_mask]
            y_batch = y[batch_mask]
            loss = lambda w: cross_entropy_error(softmax(dot(x_batch, w)), y_batch)
            dw = numerical_gradient(loss, w)
            w = w - gamma * dw
            batch_train_loss = loss(w)
            train_loss += batch_train_loss
            print(f'[{epoc+1} : {batch+1}/{batch_count}] batch train loss: {batch_train_loss}')
            experiment.log_metric('batch_train_loss', batch_train_loss.astype(np.float64), step=epoc*batch)
        losses.append(train_loss)
        print(f'[{epoc+1}] total train loss: {train_loss}')
        experiment.log_metric('train_loss', train_loss.astype(np.float64), step=epoc*batch)
    
#    %matplotlib inline
#    from matplotlib import pyplot as plt
#    plt.plot(losses)
    
    y_pred = np.argmax(dot(x,w), axis=1)
    correct = (y_pred == y).sum()
    train_accuracy = correct / len(y)
    print(f'train accuracy: {train_accuracy}')
    experiment.log_metric('train_accuracy', train_accuracy)

main()
