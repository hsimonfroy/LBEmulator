#
# Computes halo power spectra, and the halo-matter cross-spectrum.
#
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

    bs, nc = 1536, 2048
    seed   = 9200

    #for seed in range(9200, 9210, 1):
    for aa in [0.2500,0.3333,0.4000,0.5000,0.6667,1.0000]:
        zz = 1/aa-1
        iz = int(100*zz)
        Rsm = 0

        lpath = '/global/cscratch1/sd/chmodi/m3127/lbemulator/N%d-T40-B2/S%d/'%(nc, seed)
        opath = '/global/cscratch1/sd/mwhite/LagEmu/N%d-T40-B2/S%d/'%(nc, seed)
        ofolder = opath + '/spectra/'
        try: os.makedirs(ofolder)
        except : pass

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f8')
        rank = pm.comm.rank

        # Load Matter Mesh
        dyn = BigFileCatalog(lpath +  '/fastpm_%0.4f/1'%aa)
        fpos = dyn['Position'].compute()
        mlay = pm.decompose(fpos)
        mmesh = pm.paint(fpos, layout=mlay)
        mmesh /= mmesh.cmean()
        
        hcat = BigFileCatalog(lpath+  '/fastpm_%0.4f/LL-0.200/'%aa)
        print(rank, 'files read')

        #print('Mass : ', rank, hcat['Length'][-1].compute()*hcat.attrs['M0']*1e10)
        hmass = hcat['Length'].compute()*hcat.attrs['M0']*1e10

        lgMmin = [12.0,12.5,13.0]
        lgMmax = [12.5,13.0,13.5]
        
        for i in range(len(lgMmin)):

            if rank == 0: print(lgMmin[i],lgMmax[i])
            cat = hcat[(hmass>10.**lgMmin[i])&(hmass<10.**lgMmax[i])]
            #cat = hcat.gslice(start=0, stop=num[i])

            hlay  = pm.decompose(cat['Position'])
            hmesh = pm.paint(cat['Position'], layout=hlay)
            hmesh /= hmesh.cmean()
            
            # First compute the halo auto-power.
            ph = FFTPower(hmesh, mode='1d').power
            k, ph = ph['k'],  ph['power']

            outfn = ofolder+"ph_{:05.2f}_{:05.2f}_z{:03d}.txt".\
                    format(lgMmin[i],lgMmax[i],iz)
            hdr   = "Real space halo power spectrum:  "+\
                    "k ph, z={:f}, {:f}<lgM<{:f}".format(zz,lgMmin[i],lgMmax[i])
            np.savetxt(outfn,np.vstack([k, ph]).T.real,header=hdr,fmt='%15.5e')

            # Now compute the halo-mass cross-power.
            ph = FFTPower(mmesh, second=hmesh, mode='1d').power
            k, ph = ph['k'],  ph['power'].real

            outfn = ofolder+"px_{:05.2f}_{:05.2f}_z{:03d}.txt".\
                    format(lgMmin[i],lgMmax[i],iz)
            hdr   = "Real space h-m cross-spectrum:  "+\
                    "k ph, z={:f}, {:f}<lgM<{:f}".format(zz,lgMmin[i],lgMmax[i])
            np.savetxt(outfn,np.vstack([k, ph]).T.real,header=hdr,fmt='%15.5e')
            #
