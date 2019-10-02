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

    for seed in range(9204, 9210, 1):
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
        #grid = pm.mesh_coordinates()*bs/nc
        lin = BigFileMesh(lpath+ '/linear', 'LinearDensityK').paint()
        lin -= lin.cmean()

        print(rank, 'lin field read')
        header = '1,b1,b2,bg,bk'
        names = header.split(',')
        lag_fields = tools.getlagfields(pm, lin, R=Rsm) # use the linear field at the desired redshift
        
        print(rank, 'lag field created')

        for i, ff in enumerate(lag_fields):
            x = FieldMesh(ff)
            x.save(lpath + 'lag', dataset=names[i], mode='real')



##        ###
##        dyn = BigFileCatalog(dpath +  '/fastpm_%0.4f/1'%aa)
##
##        print(rank, 'files read')
##
##        #
##        fpos = dyn['Position'].compute()
##        idd = dyn['ID'].compute()
##        attrs = dyn.attrs
##
##        grid = tools.getqfromid(idd, attrs, nc)
##        print(rank, 'grid computed')
##        
###        cat = ArrayCatalog({'ID': idd, 'InitPosition': grid, 'Position' : fpos}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
###        cat.attrs = dyn.attrs
###        cat.save(lpath + 'dynamic/1', ('ID', 'InitPosition', 'Position'))
###        print(rank, 'cat saved')
###
##
###        flay = pm.decompose(fpos)
###        fmesh = pm.paint(fpos, layout=flay)
###        fmesh /= fmesh.cmean()
##
###        x = FieldMesh(fmesh)
###        x.save(lpath + 'meshes', dataset='final', mode='real')
##
###        del x, flay, fpos, idd
##        
##        if pm.comm.rank == 0:
##            print('dynamic model created')
##
##
##        header = '1,b1,b2,bg,bk'
##        names = header.split(',')
##
##        print(rank, 'lag field created')
##
##
##        for Rsm in [0]:
##            for zadisp in [False, True]:
##
##                #
##                #fpos = dyn['Position']
##
##                #if zadisp : fpos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
##                glay, play = pm.decompose(grid), pm.decompose(fpos)
##
##                for i, ff in enumerate(lag_fields):
##                    print(rank, i)
##                    x = FieldMesh(pm.paint(fpos, mass=ff.readout(grid, layout = glay, resampler='nearest'), layout=play))
##                    if zadisp : x.save(lpath + 'za-z%03d-R%d'%(zz*100, Rsm), dataset=names[i], mode='real')
##                    else : x.save(lpath + 'eul-z%03d-R%d'%(zz*100, Rsm), dataset=names[i], mode='real')
##                    del x
##                    
##
####                k, spectra = tools.getspectra(eul_fields)
####                #k, spectra = tools.getspectra(lag_fields)
####
####                if zadisp: np.savetxt(ofolder + '/spectraza-z%03d-R%d.txt'%(zz*100, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
####                else: np.savetxt(ofolder + '/spectra-z%03d-R%d.txt'%(zz*100, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
####                
####
####
####
####
##
