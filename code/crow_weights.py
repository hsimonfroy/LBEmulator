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
    ncf = 6144
    
    for seed in range(9202, 9210, 1):
        aa = 1.0000
        zz = 1/aa-1
        dgrow = cosmo.scale_independent_growth_factor(zz)
        Rsm = 0

        dpath = '/global/cscratch1/sd/yfeng1/m3127/desi/6144-%d-40eae2464/'%(seed)
        lpath = '/global/cscratch1/sd/chmodi/m3127/crowcanyon/N%d-T0/S%d/'%(nc, seed)
        ofolder = lpath + '/spectra/'
        try: os.makedirs(ofolder)
        except : pass

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f4')
        rank = pm.comm.rank


        dyn = BigFileCatalog(dpath +  '/fastpm_%0.4f/1'%aa)

        #
        fpos = dyn['Position'].compute()
        idd = dyn['ID'].compute()
        attrs = dyn.attrs

        grid = tools.getqfromid(idd, attrs, ncf)
        if rank == 0: print('grid computed')

        cat = ArrayCatalog({'ID': idd, 'InitPosition': grid}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
        #cat.save(lpath + 'initpos', tosave)
        
        
        #cat.attrs = dyn.attrs


        #header = '1,b1,b2,bg,bk'
        header = 'b1,b2,bg,bk'
        names = header.split(',')

        #cat.save(lpath + 'dynamic/1', ('ID', 'InitPosition', 'Position'))

        tosave = ['ID', 'InitPosition'] + names
        if rank == 0: print(tosave)
        
        for i in range(len(names)):
            ff = BigFileMesh(lpath+ '/lag', names[i]).paint()
            pm = ff.pm
            rank = pm.comm.rank
            glay, play = pm.decompose(grid), pm.decompose(fpos)
            wts = ff.readout(grid, layout = glay, resampler='nearest')
            cat[names[i]] = wts
            #x = FieldMesh(pm.paint(fpos, mass=wts, layout=play))
            #x.save(lpath + 'eul-z%03d-R%d'%(zz*100, Rsm), dataset=names[i], mode='real')
            del pm, ff, wts
        if rank == 0: print(rank, ' got weights')

        cat.save(lpath + 'lagweights/', tosave)
        
