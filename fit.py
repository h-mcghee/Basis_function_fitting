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

#plot

N2 = []
g1 = []
g2 = []
g3 = []
g4 = []
c = []

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
    
    N2.append(result.params['N2_amplitude'].value)
    g1.append(result.params['g1_amplitude'].value)
    g2.append(result.params['g2_amplitude'].value)
    g3.append(result.params['g3_amplitude'].value)
    g4.append(result.params['g4_amplitude'].value)
    c.append(result.params['c'].value)
    
    ax.legend()

# Show the plot
plt.tight_layout()
plt.show()


