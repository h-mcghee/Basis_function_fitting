import numpy as np
import matplotlib.pyplot as plt

from lmfit import Model
from lmfit.models import GaussianModel
from lmfit import Parameters
import config
from functions import smooth, prepare_data,exp
import os

# import data matrix

data_path = 'test_data/para_matrix.txt'

# set smoothing window

smooth = 3

# save pdf?

save = True
outfile = 'output_files/PAL/fit.pdf'

# get energy axis to interpolate basis functions (leave this)
energy = prepare_data(data_path,smooth)[1]

#add any basis functions here

gs = np.genfromtxt('test_data/pNA_gs_calib.txt')
gs = np.interp(energy,gs[:,0],gs[:,1])

aniline = np.genfromtxt('/Users/harrymcghee/Dropbox (UCL)/4_PROJECTS/5_PALRIXS/3_BEAMTIME/2_SCRIPTS/6_POST/lit_spectra/aniline.csv',delimiter = ',')
aniline = np.interp(energy,aniline[:,0],aniline[:,1])

NO = np.genfromtxt('/Users/harrymcghee/Dropbox (UCL)/4_PROJECTS/5_PALRIXS/3_BEAMTIME/2_SCRIPTS/6_POST/lit_spectra/NO.csv',delimiter = ',')
NO = np.interp(energy,NO[:,0],NO[:,1])

NO2 = np.genfromtxt('/Users/harrymcghee/Dropbox (UCL)/4_PROJECTS/5_PALRIXS/3_BEAMTIME/2_SCRIPTS/6_POST/lit_spectra/NO2.csv',delimiter = ',')
NO2 = np.interp(energy,NO2[:,0],NO2[:,1])

gs_pump = np.genfromtxt('test_data/pNA_gs_calib.txt')
gs_pump = np.interp(energy,gs_pump[:,0]-4.67,gs_pump[:,1])


def gs_fit(x,amplitude):
    return amplitude*gs[np.where(x==x)]

def aniline_fit(x,amplitude):
    return amplitude*aniline[np.where(x==x)]

def NO_fit(x,amplitude):
    return amplitude*NO[np.where(x==x)]

def NO2_fit(x,amplitude):
    return amplitude*NO2[np.where(x==x)]

def gs_pump_fit(x,amplitude):
    return amplitude*NO2[np.where(x==x)]


# model = GaussianModel(prefix = 'g1_') + GaussianModel(prefix = 'g2_')  + GaussianModel(prefix = 'g4_') - Model(bkg,prefix = 'gs_')
# model = GaussianModel(prefix = 'g1_') + GaussianModel(prefix = 'g2_') +GaussianModel(prefix = 'g3_') - Model(bkg,prefix = 'gs_')

model = Model(NO_fit,prefix = 'NO_') - Model(gs_fit,prefix = 'gs_') + Model(NO2_fit,prefix = 'NO2_') + GaussianModel(prefix = 'g1_') + GaussianModel(prefix = 'g2_') + GaussianModel(prefix = 'g3_') 

#set initial parameters

params = Parameters()

params.add('gs_amplitude', value=0.1,vary = True)
params.add('NO_amplitude', value=0.1,vary = True)
params.add('NO2_amplitude', value=0.1,vary = True)

params.add('g1_amplitude', value=0.1,min = 0,vary = True)
params.add('g1_center', value=400.7,min = 400.3,max = 401,vary = True)
params.add('g1_sigma', value=0.2,min = 0.1,max = 0.5,vary = True)

params.add('g2_amplitude', value=0.1,min = 0,vary = True)
params.add('g2_center', value=402.7,min = 402.3,max = 403,vary = True)
params.add('g2_sigma', value=0.2,min = 0.1,max = 0.5,vary = True)

params.add('g3_amplitude', value=0.1,min = 0,vary = True)
params.add('g3_center', value=405,min = 404.7,max = 405.3,vary = True)
params.add('g3_sigma', value=0.2,min = 0.1,max = 0.5,vary = True)



# params.add('gs_pump_amplitude', value=0.1,vary = True)



# impose any kinetic models based on idx(after inspection usually)
def update_params(params,delay):
    params['NO_amplitude'].value = exp(delay,-0.02,32)+0.02 
    params['NO_amplitude'].vary = False
    params['NO2_amplitude'].value = exp(delay,-0.013,32)+0.013 
    params['NO2_amplitude'].vary = False
    # params['gs_amplitude'].value = exp(delay,0.03,0.42) + 0.03
    # params['gs_amplitude'].vary = False
    # params['NO2_amplitude'].value = exp(delay,0.01,32)
    # params['NO2_amplitude'].vary = False

    return params


param_names = model.param_names
param_names = [name.split('_')[0] for name in param_names]
param_names = np.unique(param_names)



