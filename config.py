import numpy as np
import matplotlib.pyplot as plt

from lmfit import Model
from lmfit.models import GaussianModel
from lmfit import Parameters
import config
from functions import smooth, prepare_data
import os

# import data matrix

data_path = 'test_data/para_matrix.txt'

# set smoothing window

smooth = 3

# save pdf?

save = True
outfile = 'output_files/test.pdf'

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


def gs_fit(x,amplitude):
    return amplitude*gs[np.where(x==x)]

def aniline_fit(x,amplitude):
    return amplitude*aniline[np.where(x==x)]

def NO_fit(x,amplitude):
    return amplitude*NO[np.where(x==x)]

def NO2_fit(x,amplitude):
    return amplitude*NO2[np.where(x==x)]


# model = GaussianModel(prefix = 'g1_') + GaussianModel(prefix = 'g2_')  + GaussianModel(prefix = 'g4_') - Model(bkg,prefix = 'gs_')
# model = GaussianModel(prefix = 'g1_') + GaussianModel(prefix = 'g2_') +GaussianModel(prefix = 'g3_') - Model(bkg,prefix = 'gs_')

model =  Model(NO_fit,prefix = 'NO_') + Model(NO2_fit,prefix = 'NO2_') - Model(gs_fit,prefix = 'gs_')

#set initial parameters

params = Parameters()

params.add('gs_amplitude', value=0.1,vary = True)
# params.add('aniline_amplitude', value=0.1,vary = True)
params.add('NO_amplitude', value=0.1,vary = True)
params.add('NO2_amplitude', value=0.1,vary = True)


# params.add('gs_amplitude', value=0.1,vary = True)

# params.add('g1_amplitude', value=0.025,min = 0)
# params.add('g1_center', value=399.65, vary = False)
# params.add('g1_sigma', value=0.4,min = 0.2,max = 0.6,vary = True)

# params.add('g2_amplitude', value=0.07, min = 0,vary = True)
# params.add('g2_center', value=400.65,vary = False)
# params.add('g2_sigma', value=0.4,min = 0.2,max = 0.6,vary = True)

# params.add('g3_amplitude', value = 0.1, min = 0,vary = True)
# params.add('g3_center', value=403,vary = False)
# params.add('g3_sigma', value=0.4,min = 0.2,max = 0.6,vary = True)

# params.add('g4_amplitude', expr = 'g1_amplitude', min = 0,vary = True)
# params.add('g4_center', value=405,vary = False)
# params.add('g4_sigma', value=0.4,max = 1,expr = 'g1_sigma',vary = True)


param_names = model.param_names
param_names = [name.split('_')[0] for name in param_names]
param_names = np.unique(param_names)



