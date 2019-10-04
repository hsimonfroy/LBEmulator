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
import os, sys
sys.path.append('./utils/')
import za
import features as ft
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)

import time

if __name__=="__main__":


    #seed = 9200
    subf = '/cm_lowres-20stepB1/'
    try: os.makedirs('./output/%s'%subf)
    except : pass
    try: os.makedirs('./figs/%s'%subf)
    except : pass
    
    bs, nc = 1024, 512
    lnMmin = 13.5; Mmin = 10**lnMmin
    
    #dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%d-%d-9100-fixed/'%(bs, nc)
    #dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100/'%(bs, nc)
    galpath = '/global/cscratch1/sd/sfschen/lagrangian_emulator/data/hod/cm_lowres/20stepT-B1/1024-512-9100/'
    cendir = 'cencat-Mmin-12.5-M1f-20.0-alpha-0p8-subvol'
    satdir = 'satcat-Mmin-12.5-M1f-20.0-alpha-0p8-subvol'
    
    dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/20stepT-B1/%d-%d-9100/'%(bs, nc)
    #dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100-fixed/'%(bs, nc)

    aas = [0.5,]
    
    for aa in aas:
        zz = 1/aa-1
        
        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f8')
        rank = pm.comm.rank
        dyn = BigFileCatalog(dpath +  '/fastpm_%0.4f/1'%aa)
        
        # Load Matter Mesh
        fpos = dyn['Position'].compute()
        mlay = pm.decompose(fpos)
        mmesh = pm.paint(fpos, layout=mlay)
        mmesh /= mmesh.cmean()

        # Load halo mesh
        cencat = BigFileCatalog(galpath+  '/fastpm_%0.4f/'%(aa) + cendir)
        satcat = BigFileCatalog(galpath+  '/fastpm_%0.4f/'%(aa) + satdir)    
        
        hpos = np.concatenate((cencat['Position'],satcat['Position']))
        hlay = pm.decompose(hpos)
        hmesh = pm.paint(hpos, layout=hlay)
        hmesh /= hmesh.cmean()

        phm = FFTPower(mmesh, second=hmesh, mode='1d').power
        k, phm = phm['k'],  phm['power'].real
        
        phh = FFTPower(hmesh, mode='1d').power
        k, phh = phh['k'],  phh['power'].real

        np.savetxt('./output/%s/pks-%04d-%04d-%04d-gal.txt'%(subf, aa*10000, bs, nc), np.vstack([k, phh, phm]).T.real, header='k pk/ ', fmt='%0.4e')




