import numpy as np
import matplotlib.pyplot as plt
import math

from lmfit import Model
from lmfit.models import GaussianModel
from lmfit import Parameters
import config
from functions import smooth
import os
import matplotlib.gridspec as gridspec

plt.style.use('style')


#Import data from config file

matrix = np.genfromtxt(config.data_path)

delay = matrix[0,1:]
energy = matrix[1:,0]
z = matrix[1:,1:]

#Apply any smoothing or truncation

z = [smooth(x,config.smooth) for x in z]
z = np.array(z)

#Plot heat map

plt.pcolormesh(delay,energy,z)
plt.xscale('symlog')
plt.xlabel('Delay (ps)')
plt.ylabel('Energy (eV)')

#import model from config file

model = config.model
params = config.params
param_names = config.param_names

#plot
fit_matrix = []

vars_dict = {}
for name in param_names:
    vars_dict[name] = []


gs = gridspec.GridSpec(math.ceil(len(delay)/2), 2)
plt.figure(figsize=(10,len(delay)*2))

for idx in range(0,len(delay)):

    y = z[:,idx]
    y = y/np.max(abs(y))

    result = model.fit(y, params, x=energy)
    comps = result.eval_components(x=energy)

    # Create a new subplot on the grid
    ax = plt.subplot(gs[idx // 2, idx % 2])

    # Plot the data on the subplot
    ax.plot(energy,y,'o',markersize = 2)
    ax.plot(energy,result.best_fit,color = 'red')
    ax.set_title('{} ps'.format(delay[idx]))
    for i in comps:
        ax.fill_between(energy,0,comps[i],alpha = 0.2,label = i)
    
    fit_matrix.append(result.best_fit)

    for name in param_names:
        if name == 'c':
            vars_dict[name].append(result.params[name].value)
        else:
            vars_dict[name].append(result.params[name+'_amplitude'].value)
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

plt.figure()
plt.pcolormesh(delay,energy,np.array(fit_matrix).T)
plt.xscale('symlog')
plt.xlabel('Delay (ps)')
plt.ylabel('Energy (eV)')
plt.show()

plt.figure(figsize = (10,15))
gs = gridspec.GridSpec(6, 2)

for j,i in enumerate([N2,g1,g2,g3,g4,c]):
    ax = plt.subplot(gs[j, :])
    ax.plot(delay,i,'-x',markersize = 2,label = param_names[j])
    ax.set_xscale('symlog',linthresh = 10)
    ax.set_xlabel('Delay (ps)')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.legend()

plt.tight_layout()

