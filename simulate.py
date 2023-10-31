import numpy as np
import matplotlib.pyplot as plt
import scipy

from lmfit import Model
from lmfit.models import GaussianModel
from lmfit import Parameters

plt.style.use('style')

#make 3 component gaussian model

def exp(x, tau):
    return np.exp(-x / tau)

def conv(x,t0,sigma, *args):
    num_exponentials = len(args) // 2
    num_params = len(args)

    if num_params % 2 != 0 or num_exponentials == 0:
        raise ValueError("The number of parameters must be a multiple of 2 (A, tau) for each exponential.")

    # sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    result = np.zeros_like(x)

    for i in range(num_exponentials):
        A = args[2*i]
        tau = args[2*i + 1]
        exp_term = A / 2 * np.exp((-1 / tau) * (x - t0)) * np.exp((sigma**2) / (2 * tau**2)) * \
           (1 + scipy.special.erf((x - t0 - ((sigma **2) / tau)) / (np.sqrt(2.0) * sigma)))
        result += exp_term

    return result

delay = np.linspace(-150,1500,17)
energy = np.linspace(0, 20, 100)

# need time constant and kinetic model for each component
# if parallel decay, then same equation for each component
#this is essentially constructing the D matrix as per GKA
t0 = 0
irf = 50
tau1 = 500
tau2 = 500
tau3 = 400

mat = np.zeros((len(energy),len(delay)))

for i,d in enumerate(delay):

    model = GaussianModel(prefix = 'g1_') + GaussianModel(prefix = 'g2_') - GaussianModel(prefix = 'g3_')
    params = Parameters()

    params.add('g1_amplitude', value=conv(d,t0,irf,1,tau1))
    params.add('g1_center', value=10)
    params.add('g1_sigma', value=1)
    params.add('g2_amplitude', value=conv(d,t0,irf,1,tau2))
    params.add('g2_center', value=15)
    params.add('g2_sigma', value=0.5)
    params.add('g3_amplitude', value=conv(d,t0,irf,1,tau3))
    params.add('g3_center', value=12.5)
    params.add('g3_sigma', value=0.8)
    
    y = model.eval(params=params, x=energy)
    mat[:,i] = y

#generate noise matrix
SNR = 40
noise_std_dev = np.max(mat)/ SNR
noise = np.random.normal(scale=noise_std_dev, size=mat.shape)

plt.pcolormesh(delay,energy,mat+noise)
plt.xlabel('delay / fs')
plt.ylabel('energy / eV')
plt.colorbar()

plt.figure()
for i in (mat+noise).T:
    plt.plot(energy,i)