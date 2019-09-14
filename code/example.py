
import numpy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from pmesh.pm import ParticleMesh
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from nbodykit.lab import BigFileMesh, BigFileCatalog, FFTPower
from nbodykit.cosmology import Planck15, EHPower, Cosmology

import sys
sys.path.append('./utils/')
import za
import features as ft
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)


#########################################


def getlagfields(pm, basemesh, R=0, smoothd0=True, kernel='gauss'):
    '''generate 5 lagrangian fields - 1, \delta, \delta^2, \s^2, \nabla\delta'''

    if R>0:
        base = ft.smooth(basemesh, R, kernel)
    else:
        base = basemesh.copy()

    one = (base*0 + 1).copy()
    d2 = 1.*base**2
    d2 -= d2.cmean()
    s2 = ft.shear(pm, base)
    s2 -= s2.cmean()
    lap = ft.laplace(pm, base)

    #do we smooth the field with b1?
    if smoothd0: d0 = base.copy()
    else: d0 = basemesh.copy()

    return one, d0, d2, s2, lap



def geteulfields(pm, basevec, pos, grid, doed=False):
    '''generate eulerian fields for all lag fields in basevec by painting them at eulerian position 'pos' '''
    
    glay, play = pm.decompose(grid), pm.decompose(pos)
    toret = []
    for base in basevec:
        toret.append(pm.paint(pos, mass=base.readout(grid, layout = glay, resampler='nearest'), layout=play))

    return toret


def getspectra(basevec):
    '''get spectra of all combinations of fields'''
    spec = []
    iv = len(basevec)

    for i in range(iv):
        for j in range(i, iv):
            spec.append(FFTPower(basevec[i], second=basevec[j], mode='1d').power)

    k = spec[0]['k']
    for i, ip in enumerate(spec):
        spec[i] = spec[i]['power']

    return k, spec



def getmodel(spec, bvec):
    '''eval the model by multiplying with bias param. Expected bvec to have 1 at first position'''
    
    iv = len(bvec)

    model = np.zeros_like(spec[0])
    counter = 0
    for i in range(iv):
        for j in range(i, iv):
            model += bvec[i]*bvec[j]*spec[counter]
            counter += 1

    return model



         


if __name__=="__main__":


    bs, nc = 1024, 256
    dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/256-256-9100-fixed/'
    aa = 0.3333
    zz = 1/aa-1

    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
    rank = pm.comm.rank
    grid = pm.mesh_coordinates()*bs/nc
    
    lin = BigFileMesh(dpath + '/linear', 'LinearDensityK').paint()
    dyn = BigFileCatalog(dpath + '/fastpm_%0.4f/1'%(aa))
    fpos = dyn['Position']
    #dgrow = cosmo.scale_independent_growth_factor(zz)
    #zapos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
    dlay = pm.decompose(fpos)
    
    hcat = BigFileCatalog(dpath + '/fastpm_%0.4f/LL-0.200'%(aa))
    hpos = hcat['Position']
    hlay = pm.decompose(hpos)
    hmesh = pm.paint(hpos, layout=hlay)
    hmesh /= hmesh.cmean()
    
    ph = FFTPower(hmesh, mode='1d').power
    k, ph = ph['k'],  ph['power']


    lag_fields = getlagfields(pm, lin, R=1)
    eul_fields = geteulfields(pm, lag_fields, fpos, grid)
    k, spectra = getspectra(eul_fields)

    header = '1, b1, b2, bg, bk'
    bvec = [1, 1, 1, 1, 1]
    model = getmodel(spectra, bvec)
    

    plt.figure()
    for i, ip in enumerate(spectra):
        plt.plot(k, ip)
    plt.plot(k, ph, 'k', label='Halo')
    plt.plot(k, model, 'k--', label='Model')
    plt.xlabel('k (h/Mpc)', fontsize=12)
    plt.ylabel('$P_{ab}$', fontsize=12)
    plt.legend(fontsize=12)
    plt.loglog()
    plt.savefig('example.png')
    
