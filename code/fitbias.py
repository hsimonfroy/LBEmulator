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
import sys
sys.path.append('./utils/')
import za
import features as ft
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)



def fitbias(ph, spectra, binit=[1, 0, 0, 0], k=None, kmax=None):


    if k is not None and kmax is not None:
        ik = np.where(k > kmax)[0][0]
    else: ik = len(ph)
    tomin = lambda b: sum((ph - tools.getmodel(spectra, [1] + list(b)))[:ik]**2)
    rep = minimize(tomin, binit, method='Nelder-Mead', options={'maxfev':10000})
    return rep



if __name__=="__main__":



    subf = '/cm_lowres-5stepB1/'
    
    bs, nc = 1024, 512
    dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%d-%d-9100-fixed/'%(bs, nc)
    dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100/'%(bs, nc)
    #dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100-fixed/'%(bs, nc)
    aa = 0.3333
    zz = 1/aa-1
    Rsm = 0

    
    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
    rank = pm.comm.rank
    #grid = pm.mesh_coordinates()*bs/nc
    
    hcat = BigFileCatalog(dpath+  '/fastpm_%0.4f/LL-0.200/'%aa)

    hpos = hcat['Position']
    #hmass = print('Mass : ', rank, hcat['Mass'][-1].compute())
    hlay = pm.decompose(hpos)
    hmesh = pm.paint(hpos, layout=hlay)
    hmesh /= hmesh.cmean()
    ph = FFTPower(hmesh, mode='1d').power
    k, ph = ph['k'],  ph['power']


    if rank == 0:

        for Rsm in [0]:
            for zadisp in [True, False]:


                header = '1, b1, b2, bg, bk'
                if zadisp: spectra = np.loadtxt('./output/%s/spectraza-%04d-%04d-%04d-R%d.txt'%(subf, aa*10000, bs, nc, Rsm)).T
                else: spectra = np.loadtxt('./output/%s/spectra-%04d-%04d-%04d-R%d.txt'%(subf, aa*10000, bs, nc, Rsm)).T
                k, spectra = spectra[0], spectra[1:]
                header = header.split(',')
                iv = len(header)

                fig, axar = plt.subplots(1, 2, figsize=(8, 4))
                for ik, kmax in enumerate([0.1, 0.3, 0.5, 0.8]):

                    bvec = [1, 1, 1, 1, 1]

                    rep = fitbias(ph, spectra, k=k, kmax=kmax)
                    bvec = [1] + list(rep.x)

                    model = tools.getmodel(spectra, bvec)

                    axis = axar[0]
                    if ik ==0: axis.plot(k, ph, 'k', label='Halo')
                    axis.plot(k, model, 'C%d--'%ik)
                    axis.set_xlabel('k (h/Mpc)', fontsize=12)
                    axis.set_ylabel('$P$', fontsize=12)
                    axis.legend(fontsize=12)
                    axis.loglog()
                    #axis.set_title(rep.x)
                    axis.grid()

                    axis = axar[1]
                    axis.plot(k, model/ph, 'C%d--'%ik, label='k=%0.2f'%kmax)
                    axis.axvline(kmax, color='C%d'%ik, lw=0.5)
                    axis.set_xlabel('k (h/Mpc)', fontsize=12)
                    axis.set_ylabel('$P$', fontsize=12)
                    axis.legend(fontsize=12)
                    axis.set_ylim(0.75, 1.25)
                    axis.semilogx()
                    plt.suptitle(rep.x)
                    axis.grid(which='both', lw=0.5)

                    plt.tight_layout()
                if zadisp: plt.savefig('figs/%s/fitkmax-za-%04d-%04d-%04d-R%d.png'%(subf, aa*1e4, bs, nc, Rsm))
                else: plt.savefig('figs/%s/fitkmax-%04d-%04d-%04d-R%d.png'%(subf, aa*1e4, bs, nc, Rsm))




