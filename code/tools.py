import warnings
warnings.filterwarnings("ignore")

import numpy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from pmesh.pm import ParticleMesh
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from nbodykit.lab import BigFileMesh, BigFileCatalog, FFTPower

import sys
sys.path.append('./utils/')
import za
import features as ft


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
    lap -= lap.cmean()

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
            model = model + bvec[i]*bvec[j]*spec[counter]
            counter += 1

    return model


def fitbias(ph, spectra, binit=[1, 0, 0, 0], k=None, kmax=None):


    if k is not None and kmax is not None:
        ik = np.where(k > kmax)[0][0]
    else: ik = len(ph)
    tomin = lambda b: sum((ph - getmodel(spectra, [1] + list(b)))[:ik]**2)
    rep = minimize(tomin, binit, method='Nelder-Mead', options={'maxfev':10000})
    return rep


def getqfromid_cfastpm(idd, attrs, nc):
    
    strides, scale, shift = attrs['q.strides'], attrs['q.scale'], attrs['q.shift']

    qq, qqt = np.zeros((idd.size, 3)), np.zeros((idd.size, 3))
    
    for i in range(idd.size):
        j = idd[i] %nc**3
        for d in range(3):
            qqt[i, d] = int(j/strides[d])
            j -= qqt[i, d]*strides[d]
        for d in range(3):
            qq[i, d] = qqt[i, d]*scale[d]
            qq[i, d] += shift[d]

    return qq



def getqfromid(idd, attrs, nc):
    
    strides, scale, shift = attrs['q.strides'], attrs['q.scale'], attrs['q.shift']

    qq = np.zeros((idd.size, 3))
    j = idd %nc**3
    for d in range(3):
        qq[:, d] = j//strides[d]
        j = j - qq[:, d]*strides[d]
    return qq*scale + shift


