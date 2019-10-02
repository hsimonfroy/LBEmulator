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



if __name__=="__main__":



    bs, nc = 3200, 1024

    for seed in range(9200, 9210, 1):
        aa = 1.0000
        zz = 1/aa-1
        dgrow = cosmo.scale_independent_growth_factor(zz)
        Rsm = 0

        dpath = '/global/cscratch1/sd/yfeng1/m3127/desi/6144-%d-40eae2464/'%(seed)
        ldpath = '/global/cscratch1/sd/chmodi/m3127/crowcanyon/N%d-T0/S%d/'%(2048, seed)
        lpath = '/global/cscratch1/sd/chmodi/m3127/crowcanyon/N%d-T0/S%d/'%(nc, seed)
        ofolder = lpath + '/spectra/'
        try: os.makedirs(ofolder)
        except : pass

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f8')
        rank = pm.comm.rank

        header = '1,b1,b2,bg,bk'
        names = header.split(',')
        #
        dyn = BigFileCatalog(dpath +  '/fastpm_%0.4f/1'%aa)
        fpos = dyn['Position'].compute()
        idd = dyn['ID']
        attrs = dyn.attrs
        play = pm.decompose(fpos)
        
        #
        wts = BigFileCatalog(ldpath +  '/lagweights/')
        wts['1'] = wts['b1']*0 + 1.
        grid = wts['InitPosition']
        iddg = wts['ID']

        print('asserting')
        np.allclose(idd.compute(), iddg.compute())
        print('read fields')

        disp = grid.compute() - fpos
        mask = abs(disp) > bs/2.
        disp[mask] = (bs - abs(disp[mask]))*-np.sign(disp[mask])
        print(rank, ' Max disp: ', disp.max())
        print(rank, ' Std disp: ', disp.std(axis=0))

        eul_fields = []
        for i in range(len(names)):
            eul_fields.append(pm.paint(fpos, mass=wts[names[i]], layout=play))
            print(rank, names[i], eul_fields[-1].cmean())
            #if abs(eul_fields[-1].cmean()) > 0.1: eul_fields[-1]  /= eul_fields[-1].cmean()

        del dyn, fpos, idd, iddg, wts
        print('got fields')
            
        k, spectra = tools.getspectra(eul_fields)
        np.savetxt(ofolder + '/spectra2-z%03d-R%d.txt'%(zz*100, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')

