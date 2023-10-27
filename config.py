import numpy as np
import matplotlib.pyplot as plt

from lmfit import Model
from lmfit.models import GaussianModel
from lmfit import Parameters
import config
from functions import smooth
import os

# import data matrix

data_path = 'test_data/para_matrix.txt'

matrix = np.genfromtxt(data_path)

delay = matrix[0,1:]
energy = matrix[1:,0]
z = matrix[1:,1:]

smooth = 1

#Define model

gs = np.genfromtxt('/Users/harrymcghee/Dropbox (UCL)/4_PROJECTS/5_PALRIXS/3_BEAMTIME/2_SCRIPTS/6_POST/TXT/pNA_gs.txt')
gs = np.interp(energy,gs[:,0],gs[:,1])


def bkg(x,c):
    return c*gs[np.where(x==x)]

model = GaussianModel(prefix = 'N2_')+ GaussianModel(prefix = 'g1_') + GaussianModel(prefix = 'g2_') + GaussianModel(prefix = 'g3_') + GaussianModel(prefix = 'g4_')- Model(bkg)


#set initial parameters

params = Parameters()


params.add('N2_amplitude', value=0.4, min = 0, vary = False)
params.add('N2_center', value=429.07,vary = False)
params.add('N2_sigma', value=0.4,max = 1,vary = False)

params.add('g1_amplitude', value=0.7, min = 0,vary = True)
params.add('g1_center', value=429.07,vary = False)
params.add('g1_sigma', value=0.4,max = 1,vary = True)

params.add('g2_amplitude', value=0.25,min = 0)
params.add('g2_center', value=430)
params.add('g2_sigma', value=0.6,max = 1,vary = True)

params.add('g3_amplitude', value=0.25,min = 0,)
params.add('g3_center', value=433.5)
params.add('g3_sigma', value=0.6,max = 1,vary = True)

params.add('g4_amplitude', value=0.25,min = 0)
params.add('g4_center', value=427.5, vary = False)
params.add('g4_sigma', value=0.4,max = 1,vary = False)


params.add('c', value=1,vary = False)

