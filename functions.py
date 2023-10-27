import numpy as np

def smooth(y,smooth):
    return np.convolve(y, np.ones((smooth,))/smooth, mode='same')