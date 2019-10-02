import warnings
warnings.filterwarnings("ignore")

import numpy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from pmesh.pm import ParticleMesh
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from nbodykit.lab import BigFileMesh, BigFileCatalog, FFTPower, ArrayCatalog, FieldMesh
from nbodykit.cosmology import Planck15, EHPower, Cosmology

import tools
import os, sys
sys.path.append('./utils/')
import za
import features as ft
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)

import time

#########################################



def fitbias(ph, spectra, binit=[1, 0, 0, 0], k=None, kmax=None):


    if k is not None and kmax is not None:
        ik = np.where(k > kmax)[0][0]
    else: ik = len(ph)
    tomin = lambda b: sum((ph - tools.getmodel(spectra, [1] + list(b)))[:ik]**2)
    rep = minimize(tomin, binit, method='Nelder-Mead', options={'maxfev':10000})
    return rep

if __name__=="__main__":



    bs, nc = 3200, 2048

    for seed in range(9200, 9210, 1):
        aa = 1.0000
        zz = 1/aa-1
        Rsm = 0

        dpath = '/global/cscratch1/sd/yfeng1/m3127/desi/6144-%d-40eae2464/'%(seed)
        lpath = '/global/cscratch1/sd/chmodi/m3127/crowcanyon/N%d-T0/S%d/'%(nc, seed)
        ofolder = lpath + '/spectra/'
        try: os.makedirs(ofolder)
        except : pass

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f8')
        rank = pm.comm.rank

        
        hcat = BigFileCatalog(dpath+  '/fastpm_%0.4f/LL-0.200/'%aa)

        print(rank, 'files read')


        #print('Mass : ', rank, hcat['Length'][-1].compute()*hcat.attrs['M0']*1e10)

        numd = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
        num = [int(bs**3 * i) for i in numd]
        
        for i in range(len(num)):

            if rank == 0: print(numd[i])
            cat = hcat.gslice(start=0, stop=num[i])

            hlay = pm.decompose(cat['Position'])
            hmesh = pm.paint(cat['Position'], layout=hlay)
            hmesh /= hmesh.cmean()
            
            ph = FFTPower(hmesh, mode='1d').power
            k, ph = ph['k'],  ph['power']

            np.savetxt(ofolder + '/ph-%05d.txt'%(numd[i]*1e5), np.vstack([k, ph]).T.real, header='k ph', fmt='%0.4e')
