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



    bs, nc = 3200, 2048

    for seed in range(9200, 9210, 1):
        aa = 1.0000
        zz = 1/aa-1
        dgrow = cosmo.scale_independent_growth_factor(zz)
        Rsm = 0

        dpath = '/global/cscratch1/sd/yfeng1/m3127/desi/6144-%d-40eae2464/'%(seed)
        lpath = '/global/cscratch1/sd/chmodi/m3127/crowcanyon/N%d-T0/S%d/'%(nc, seed)
        ofolder = lpath + '/spectra/'
        try: os.makedirs(ofolder)
        except : pass

        #pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f4')
        #rank = pm.comm.rank


        dyn = BigFileCatalog(dpath +  '/fastpm_%0.4f/1'%aa)

        #
        fpos = dyn['Position'].compute()
        idd = dyn['ID'].compute()
        attrs = dyn.attrs

        grid = tools.getqfromid(idd, attrs, nc)
        print('grid computed')

        del dyn, idd


        header = '1,b1,b2,bg,bk'
        names = header.split(',')


        for i in range(len(names)):
            ff = BigFileMesh(lpath+ '/lag', names[i]).paint()
            pm = ff.pm
            rank = pm.comm.rank
            glay, play = pm.decompose(grid), pm.decompose(fpos)
            wts = ff.readout(grid, layout = glay, resampler='nearest')
            print(rank, ' got weights')
            x = FieldMesh(pm.paint(fpos, mass=wts, layout=play))
            x.save(lpath + 'eul-z%03d-R%d'%(zz*100, Rsm), dataset=names[i], mode='real')
            del pm, ff, wts, x

