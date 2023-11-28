import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.backends.backend_pdf


from lmfit import Model
from lmfit.models import GaussianModel
from lmfit import Parameters
import importlib
from functions import smooth, prepare_data, exp
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker



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

    params = config.update_params(params,delay[idx])

    result = model.fit(y, params, x=energy)
    comps = result.eval_components(x=energy)

    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2],hspace = 0)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1],sharex = ax1)

    ax1.plot(energy,y,'o',markersize = 2)
    ax1.plot(energy,result.best_fit,color = 'red')
    ax2.plot(energy,y-result.best_fit,color = 'green')
    ax1.set_title('{} ps'.format(delay[idx]))
    for i in comps:
        ax1.fill_between(energy,0,comps[i],alpha = 0.2,label = i)
    
    fit_matrix.append(result.best_fit)

    for name in param_names:
        vars_dict[name].append(result.params[name+'_amplitude'].value)
    
    ax1.set_ylabel('Intensity (a.u.)') 
    ax2.set_ylim(ax1.get_ylim())
    ax2.axhline(0,color = 'black',linestyle = '--')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Residual (a.u.)')
    ax1.legend(ncol=2,loc = 'lower left',fontsize = 8)

    
    plt.tight_layout()

    if config.save:
        pdf.savefig(fig)

fit_matrix = np.array(fit_matrix).T
plt.tight_layout()
plt.show()


#colormaps

titles = ['raw data','fitted data','residual']  
fig,axs = plt.subplots(3,figsize = (4,10))

global_vmin = min(np.min(z), np.min(fit_matrix), np.min(z-fit_matrix))
global_vmax = max(np.max(z), np.max(fit_matrix), np.max(z-fit_matrix))

for l,i in enumerate([z,fit_matrix,z-fit_matrix]):
    mesh = axs[l].pcolormesh(delay,energy,i,vmin=global_vmin, vmax=global_vmax)
    # plt.colorbar()
    axs[l].set_xlabel('Delay / ps')
    axs[l].set_ylabel('Energy / eV')
    axs[l].set_title(titles[l])
    fig.colorbar(mesh,ax=axs[l])
plt.tight_layout()
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
fig = plt.figure(figsize = (4,len(param_names)*2))
gs = gridspec.GridSpec(len(param_names), 1)

linthresh = 100

for j,i in enumerate(param_names):
    ax = plt.subplot(gs[j, :])
    ax.plot(delay,vars_dict[i],'-x',label = i)
    ax.set_xscale('symlog',linthresh = linthresh)
    ax.set_xlabel('Delay (ps)')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.legend()
    ax.axvline(linthresh,color = 'black',linestyle = '--',alpha = 0.5)
    
    linear_ticks = np.arange(0, linthresh, linthresh/10)  # Adjust the range and step to your needs
    log_ticks = np.arange(linthresh,10000,100)  # Adjust the range and step to your needs
    all_ticks = np.concatenate((linear_ticks, log_ticks))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(all_ticks))
    
plt.tight_layout()
if config.save:
    pdf.savefig(fig)
    pdf.close()

#check no kinetic behaviour of widths / centres
#save out to test fits

# x = delay
# y = vars_dict['g2']
# np.savetxt('g2.txt',np.c_[x,y])

