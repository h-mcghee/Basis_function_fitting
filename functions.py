import numpy as np

def smooth(y,smooth_window):
    return np.convolve(y, np.ones((smooth_window,))/smooth_window, mode='same')

def prepare_data(file_path,smooth_window):
    data = np.genfromtxt(file_path)
    delay = data[0,1:]
    energy = data[1:,0]
    matrix = data[1:,1:]
    matrix = np.apply_along_axis(smooth,0,matrix,smooth_window)
    return delay,energy,matrix

def exp(x,A,tau):
    return A*np.exp(-x / tau)