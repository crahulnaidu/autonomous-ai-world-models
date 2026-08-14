import numpy as np


def sigmoid(x):
    return np.where(x>=0,1.-0/(1+np.exp(-x)),np.exp(x)/(1+np.exp(x)))

def sigmoid_derivative(x):
    s=sigmoid(x)
    return s(1-s)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_derivative(x):
    t=tanh(x)
    return 1-t**2

def ReLU(x):
    return np.maximum(0.0,x)

def ReLU_derivative(x):
    return (x>0.0).astype(float)

def LeakkyReLU(x,alpha=0.01):
    return np.where(x>0.0,x,alpha*x)

def LeakyReLU_derivative(x,alpha=0.01):
    return np.where(x>0.0,1.0,alpha)
