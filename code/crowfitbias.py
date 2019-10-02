import warnings
warnings.filterwarnings("ignore")

import numpy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from pmesh.pm import ParticleMesh
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from nbodykit.lab import BigFileMesh, BigFileCatalog, FFTPower
from nbodykit.cosmology import Planck15, EHPower, Cosmology

import tools
import sys, os
sys.path.append('./utils/')
import za
import features as ft
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)



def fitbias(ph, spectra, binit=[1, 0, 0, 0, 1], k=None, kmax=None):


    if k is not None and kmax is not None:
        ik = np.where(k > kmax)[0][0]
    else: ik = len(ph)
    tomin = lambda b: sum( (ph - tools.getmodel(spectra, [1] + list(b[:-1])) - b[-1]) [:ik]**2 ) #added shot noise
    rep = minimize(tomin, binit, method='Nelder-Mead', options={'maxfev':10000})
    #rep = minimize(tomin, binit,  options={'maxfev':10000})
    return rep



if __name__=="__main__":



    subf = '/crow-1024/'
    try: os.makedirs('./output/%s'%subf)
    except Exception as e: print(e)
    try: os.makedirs('./figs/%s'%subf)
    except Exception as e: print(e)

    zadisp = False
    
    bs, nc = 3200, 1024
    seed = 9200
    aa = 1.0000
    zz = 1/aa-1
    Rsm = 0

    
    sn = 1e2
    print(sn)


    dpath = '/global/cscratch1/sd/yfeng1/m3127/desi/6144-%d-40eae2464/'%(seed)
    lpath = '/global/cscratch1/sd/chmodi/m3127/crowcanyon/N%d-T0/S%d/'%(nc, seed)
    ldpath = '/global/cscratch1/sd/chmodi/m3127/crowcanyon/N%d-T0/S%d/'%(2048, seed)
    ofolder = lpath + '/spectra/'
    odfolder = ldpath + '/spectra/'

    header = '1, b1, b2, bg, bk'
    spectra = np.loadtxt(ofolder  + 'spectra2-z000-R0.txt').T
    k, spectra = spectra[0], spectra[1:]
    
    numd = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    num = [int(bs**3 * i) for i in numd]
    
    header = header.split(',')
    iv = len(header)
    bvec = [1, 1, 1, 1, 1]
    model = tools.getmodel(spectra, bvec)
    
    fig, ax = plt.subplots(1, iv, figsize=(15, 4), sharex=True)
    counter = 0

    for i in range(iv):
        for j in range(i, iv):
            ax[i].plot(k, spectra[counter], '-C%d'%j, label=header[j])
            ax[i].plot(k, -spectra[counter], '--C%d'%j)
            counter += 1
            ax[i].set_title(header[i])
        
    for axis in ax:
        #axis.plot(k, ph, 'k', label='Halo')
        axis.plot(k, model, 'k--', label='Model')
        axis.set_xlabel('k (h/Mpc)', fontsize=12)
        ax[0].set_ylabel('$P_{ab}$', fontsize=12)
        axis.legend(fontsize=12)
        axis.loglog()
    plt.tight_layout()
    plt.savefig('figs/%s/S%d-spectra-z%03d-R%d.png'%(subf, seed, zz*100, Rsm))
    plt.close()

    plt.figure()
    for j in range(len(numd)):

        ph = np.loadtxt(odfolder  + 'ph-%05d.txt'%(numd[j]*1e5)).T
        kh, ph = ph[0], ph[1]
        if nc !=2048: ph = np.interp(k, kh, ph)
        plt.plot(k, ph, label='%.2e'%numd[j])
    plt.plot(k, spectra[1], 'k', label='1 shift')
    plt.loglog()
    plt.legend()
    plt.grid(which='both')
    plt.savefig('figs/%s/S%d-ph-z%03d.png'%(subf, seed, zz*100))
        
    fig, axar = plt.subplots(1, len(numd), figsize=(15, 4), sharex=True, sharey=True)
    for j in range(len(numd)):

        ph = np.loadtxt(odfolder  + 'ph-%05d.txt'%(numd[j]*1e5)).T
        kh, ph = ph[0], ph[1]
        if nc !=2048: ph = np.interp(k, kh, ph)

        b1 = (ph/spectra[1])[5:10].mean()**0.5
        print(b1)
        for ik, kmax in enumerate([0.1, 0.3, 0.5, 0.8, 1.0]):

            binit = [b1-1, 0, 0, 0, 1]
            rep = fitbias(ph, spectra, k=k, kmax=kmax, binit=binit)
            print(rep)
            bvec = [1] + list(rep.x[:-1])

            model = tools.getmodel(spectra, bvec) + rep.x[-1]

##            axis = axar[0]
##            if ik ==0: axis.plot(k, ph, 'k', label='Halo', lw=2)
##            axis.plot(k, model, 'C%d--'%ik, lw=2, alpha=0.7)
##            axis.set_xlabel('k (h/Mpc)', fontsize=12)
##            axis.set_ylabel('$P$', fontsize=12)
##            axis.legend(fontsize=12)
##            axis.loglog()
##            axis.grid()
##
            axis = axar[j]
            axis.plot(k, model/ph, 'C%d-'%ik, label='k=%0.2f'%kmax, lw=2)
            axis.axvline(kmax, color='C%d'%ik, lw=0.5, ls="--")
            axis.set_xlabel('k (h/Mpc)', fontsize=12)
            #axis.set_ylabel('$P$', fontsize=12)
            if j == 0: axis.legend(fontsize=12)
            axis.set_ylim(0.9, 1.1)
            axis.semilogx()
            axis.grid(which='both', lw=0.5)
            axis.set_title('%0.2e'%numd[j])

        #plt.suptitle(['%0.2e'%i for i in rep.x])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('figs/%s/S%d-fit-z%03d-R%d.png'%(subf, seed, zz*100, Rsm))




