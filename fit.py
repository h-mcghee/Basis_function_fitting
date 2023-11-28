import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.backends.backend_pdf


from lmfit import Model
from lmfit.models import GaussianModel
from lmfit import Parameters
import importlib
from functions import smooth, prepare_data
import os
import matplotlib.gridspec as gridspec

#reload config file so kernal doesn't have to be restarted 
config_file = 'config' #make ths sysargv?
config = importlib.import_module(config_file)
config = importlib.reload(config)

#set plot style

plt.style.use('style') 

#Import data from config file

delay,energy,z = prepare_data(config.data_path,config.smooth)

#Import model from config file

model = config.model
params = config.params
param_names = config.param_names

#plot


fit_matrix = []
vars_dict = {}
for name in param_names:
    vars_dict[name] = []

if config.save:
    pdf = matplotlib.backends.backend_pdf.PdfPages(config.outfile)

for idx in range(0,len(delay)):

    y = z[:,idx]

    result = model.fit(y, params, x=energy)
    comps = result.eval_components(x=energy)
    fig = plt.figure(figsize = (3,2))

    plt.plot(energy,y,'o',markersize = 2)
    plt.plot(energy,result.best_fit,color = 'red')
    plt.title('{} ps'.format(delay[idx]))
    for i in comps:
        plt.fill_between(energy,0,comps[i],alpha = 0.2,label = i)
    
    fit_matrix.append(result.best_fit)

    for name in param_names:
        vars_dict[name].append(result.params[name+'_amplitude'].value)
    
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(ncol = 2,loc = 'lower left')
    plt.ylim(-0.05,0.05)
    plt.tight_layout()

    if config.save:
        pdf.savefig(fig)

fit_matrix = np.array(fit_matrix).T
plt.tight_layout()
plt.show()


#colormaps

titles = ['raw data','fitted data','residual']  
for l,i in enumerate([z,fit_matrix,z-fit_matrix]):
    fig = plt.figure(figsize = (3,2))
    plt.pcolormesh(delay,energy,i)
    plt.colorbar()
    plt.xlabel('Delay / ps')
    plt.ylabel('Energy / eV')
    plt.title(titles[l])
    plt.show()
    if config.save:
        pdf.savefig(fig)



# fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize = (6,10))

# ax1.pcolormesh(delay,energy,z)
# ax1.set_title('raw data')

# ax2.pcolormesh(delay,energy,np.array(fit_matrix).T)
# ax2.set_title('fitted data')

# ax3.pcolormesh(delay,energy,z-np.array(fit_matrix).T)
# ax3.set_title('residual')

# for i in [ax1,ax2,ax3]:
#     # i.set_xscale('symlog')
#     i.set_xlabel('Delay / ps')
#     i.set_ylabel('Energy / eV')
# plt.tight_layout()
# plt.show()

# Plot kinetic components
fig = plt.figure(figsize = (3,len(param_names)*2))
gs = gridspec.GridSpec(len(param_names), 1)

for j,i in enumerate(param_names):
    ax = plt.subplot(gs[j, :])
    ax.plot(delay,vars_dict[i],'-x',label = i)
    ax.set_xscale('symlog',linthresh = 10)
    ax.set_xlabel('Delay (ps)')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.legend()
    
plt.tight_layout()
if config.save:
    pdf.savefig(fig)
    pdf.close()

#check no kinetic behaviour of widths / centres
#save out to test fits

# x = delay
# y = vars_dict['g2']
# np.savetxt('g2.txt',np.c_[x,y])

