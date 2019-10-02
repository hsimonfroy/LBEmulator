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

        print(rank, 'lin field read')

        hcat = BigFileCatalog(dpath+  '/fastpm_%0.4f/LL-0.200/'%aa)

        print(rank, 'files read')


        hpos = hcat['Position']
        print('Mass : ', rank, hcat['Length'][-1].compute()*hcat.attrs['M0']*1e10)
        hlay = pm.decompose(hpos)
        hmesh = pm.paint(hpos, layout=hlay)
        #hmesh /= hmesh.cmean()

        ph = FFTPower(hmesh, mode='1d').power
        k, ph = ph['k'],  ph['power']

        np.savetxt('ph-crow-%d.txt'%seed, np.vstack([k, ph]).T.real, fmt='%0.4e')
##        
##        x = FieldMesh(hmesh)
##        x.save(lpath + 'meshes', dataset='hmesh', mode='real')
##
##        flay = pm.decompose(fpos)
##        fmesh = pm.paint(fpos, layout=flay)
##        fmesh /= fmesh.cmean()
##
##        x = FieldMesh(fmesh)
##        x.save(lpath + 'meshes', dataset='final', mode='real')
##
##        #ph = FFTPower(hmesh, mode='1d').power
##        #k, ph = ph['k'],  ph['power']
##
##        if pm.comm.rank == 0:
##            print('dynamic model created')
##
##
##        for Rsm in [0]:
##            for zadisp in [False, True]:
##
##                #
##                fpos = dyn['Position'].compute()
##
##                dgrow = cosmo.scale_independent_growth_factor(zz)
##                if zadisp : fpos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
##                dlay = pm.decompose(fpos)
##                disp = grid - fpos
##                mask = abs(disp) > bs/2.
##                disp[mask] = (bs - abs(disp[mask]))*-np.sign(disp[mask])
##                print(rank, ' Max disp: ', disp.max())
##                print(rank, ' Std disp: ', disp.std(axis=0))
##
##
##                ph = FFTPower(hmesh, mode='1d').power
##                k, ph = ph['k'],  ph['power']
##
##                lag_fields = tools.getlagfields(pm, lin*dgrow, R=Rsm) # use the linear field at the desired redshift
##                eul_fields = tools.geteulfields(pm, lag_fields, fpos, grid)
##                k, spectra = tools.getspectra(eul_fields)
##                #k, spectra = tools.getspectra(lag_fields)
##
##                header = '1, b1, b2, bg, bk'
##                if zadisp: np.savetxt(ofolder + '/spectraza-z%03d-R%d.txt'%(zz*100, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
##                else: np.savetxt(ofolder + '/spectra-z%03d-R%d.txt'%(zz*100, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
##                header = header.split(',')
##


    ##            #
    ##            mmass = [5e13, 1e13, 5e12, 1e12, 5e11, 1e11, 5e10, 1e10]
    ##            hcat['Mass'] = hcat['Length']*hcat.attrs['M0']*1e10
    ##            
    ##            cat = fofcat.gslice(0, 1)['Mass'].compute()*1e10
    ##
    ##            hpos = hcat['Position']
    ##            print('Mass : ', rank, hcat['Length'][-1].compute()*hcat.attrs['M0']*1e10)
    ##            hlay = pm.decompose(hpos)
    ##            hmesh = pm.paint(hpos, layout=hlay)
    ##            hmesh /= hmesh.cmean()
    ##
