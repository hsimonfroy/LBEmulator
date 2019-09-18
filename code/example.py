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


def fitbias(ph, spectra, binit=[1, 0, 0, 0], k=None, kmax=None):


    if k is not None and kmax is not None:
        ik = np.where(k > kmax)[0][0]
    else: ik = len(ph)
    tomin = lambda b: sum((ph - getmodel(spectra, [1] + list(b)))[:ik]**2)
    rep = minimize(tomin, binit, method='Nelder-Mead', options={'maxfev':10000})
    return rep



if __name__=="__main__":


    bs, nc = 1000, 512
    bs, nc = 400, 256
    #dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%d-%d-9100-fixed/'%(bs, nc)
    dpath = '/global/cscratch1/sd/chmodi/cosmo4d/data/z00/L%04d_N%04d_S0100_40step/'%(bs, nc)
    aa = 1.0000
    zz = 1/aa-1
    Rsm = 0
    zadisp = True
    
    for Rsm in [0, 2]:
        for zadisp in [True, False]:
            pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
            rank = pm.comm.rank
            #grid = pm.mesh_coordinates()*bs/nc

            lin = BigFileMesh(dpath + '/mesh', 's').paint()
            dyn = BigFileCatalog(dpath + '/dynamic/1')
            hcat = BigFileCatalog(dpath + '/FOF/')
            #
            grid = dyn['InitPosition'].compute()
            fpos = dyn['Position'].compute()
            print(rank, (grid-fpos).std(axis=0))

            dgrow = cosmo.scale_independent_growth_factor(zz)
            if zadisp : fpos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
            dlay = pm.decompose(fpos)

            hpos = hcat['CMPosition']
            print('Mass : ', rank, hcat['Mass'][-1].compute())
            hlay = pm.decompose(hpos)
            hmesh = pm.paint(hpos, layout=hlay)
            hmesh /= hmesh.cmean()

            ph = FFTPower(hmesh, mode='1d').power
            k, ph = ph['k'],  ph['power']

            lag_fields = getlagfields(pm, lin, R=Rsm)
            eul_fields = geteulfields(pm, lag_fields, fpos, grid)
            k, spectra = getspectra(eul_fields)

            header = '1, b1, b2, bg, bk'
            if zadisp: np.savetxt('./output/spectraza-%04d-%04d-R%d.txt'%(bs, nc, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
            else: np.savetxt('./output/spectra-%04d-%04d-R%d.txt'%(bs, nc, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
            header = header.split(',')
            bvec = [1, 1, 1, 1, 1]
            model = getmodel(spectra, bvec)
            iv = len(header)

            fig, ax = plt.subplots(1, iv, figsize=(15, 4), sharex=True)
            counter = 0

            for i in range(iv):
                for j in range(i, iv):
                    ax[i].plot(k, spectra[counter], '-C%d'%j, label=header[j])
                    ax[i].plot(k, -spectra[counter], '--C%d'%j)
                    counter += 1
                ax[i].set_title(header[i])

            for axis in ax:
                axis.plot(k, ph, 'k', label='Halo')
                axis.plot(k, model, 'k--', label='Model')
                axis.set_xlabel('k (h/Mpc)', fontsize=12)
                ax[0].set_ylabel('$P_{ab}$', fontsize=12)
                axis.legend(fontsize=12)
                axis.loglog()
            plt.tight_layout()
            if zadisp: plt.savefig('figs/exampleza-%04d-%04d-R%d.png'%(bs, nc, Rsm))
            else: plt.savefig('figs/example-%04d-%04d-R%d.png'%(bs, nc, Rsm))



            if rank == 0:
                rep = fitbias(ph, spectra, k=k)
                bvec = [1] + list(rep.x)

                model = getmodel(spectra, bvec)
                iv = len(header)

                fig, axar = plt.subplots(1, 2, figsize=(8, 4))
                axis = axar[0]
                axis.plot(k, ph, 'k', label='Halo')
                axis.plot(k, model, 'r--', label='Model')
                axis.set_xlabel('k (h/Mpc)', fontsize=12)
                axis.set_ylabel('$P$', fontsize=12)
                axis.legend(fontsize=12)
                axis.loglog()
                #axis.set_title(rep.x)
                axis.grid()

                axis = axar[1]
                axis.plot(k, model/ph, 'k', label='Halo')
                axis.set_xlabel('k (h/Mpc)', fontsize=12)
                axis.set_ylabel('$P$', fontsize=12)
                axis.legend(fontsize=12)
                axis.semilogx()
                plt.suptitle(rep.x)
                axis.grid(which='both', lw=0.5)

                plt.tight_layout()
                if zadisp: plt.savefig('figs/examplefitza-%04d-%04d-R%d.png'%(bs, nc, Rsm))
                else: plt.savefig('figs/examplefit-%04d-%04d-R%d.png'%(bs, nc, Rsm))



