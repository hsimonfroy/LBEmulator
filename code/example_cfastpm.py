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

#########################################



def fitbias(ph, spectra, binit=[1, 0, 0, 0], k=None, kmax=None):


    if k is not None and kmax is not None:
        ik = np.where(k > kmax)[0][0]
    else: ik = len(ph)
    tomin = lambda b: sum((ph - tools.getmodel(spectra, [1] + list(b)))[:ik]**2)
    rep = minimize(tomin, binit, method='Nelder-Mead', options={'maxfev':10000})
    return rep

if __name__=="__main__":


    #seed = 9200
    subf = '/cm_lowres-20stepB1/'
    try: os.makedirs('./output/%s'%subf)
    except : pass
    try: os.makedirs('./figs/%s'%subf)
    except : pass
    
    bs, nc = 1024, 512
    #dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%d-%d-9100-fixed/'%(bs, nc)
    #dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100/'%(bs, nc)
    dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/20stepT-B1/%d-%d-9100/'%(bs, nc)
    #dpath = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100-fixed/'%(bs, nc)

    aa = 0.5000
    zz = 1/aa-1
    Rsm = 0

    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f8')
    rank = pm.comm.rank
    #grid = pm.mesh_coordinates()*bs/nc
    lin = BigFileMesh(dpath+ '/linear', 'LinearDensityK').paint()
    lin -= lin.cmean()
    
    dyn = BigFileCatalog(dpath +  '/fastpm_%0.4f/1'%aa)
    hcat = BigFileCatalog(dpath+  '/fastpm_%0.4f/LL-0.200/'%aa)
    #
    fpos = dyn['Position'].compute()
    idd = dyn['ID'].compute()
    attrs = dyn.attrs
    

    grid = tools.getqfromid(idd, attrs, nc)

    for Rsm in [0, 2]:
        for zadisp in [True, False]:

            #
            fpos = dyn['Position'].compute()

            dgrow = cosmo.scale_independent_growth_factor(zz)
            if zadisp : fpos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
            dlay = pm.decompose(fpos)
            disp = grid - fpos
            mask = abs(disp) > bs/2.
            disp[mask] = (bs - abs(disp[mask]))*-np.sign(disp[mask])
            print(rank, ' Max disp: ', disp.max())
            print(rank, ' Std disp: ', disp.std(axis=0))

            #
            hpos = hcat['Position']
            print('Mass : ', rank, hcat['Length'][-1].compute()*hcat.attrs['M0']*1e10)
            hlay = pm.decompose(hpos)
            hmesh = pm.paint(hpos, layout=hlay)
            hmesh /= hmesh.cmean()

            ph = FFTPower(hmesh, mode='1d').power
            k, ph = ph['k'],  ph['power']

            lag_fields = tools.getlagfields(pm, lin*dgrow, R=Rsm) # use the linear field at the desired redshift
            eul_fields = tools.geteulfields(pm, lag_fields, fpos, grid)
            k, spectra = tools.getspectra(eul_fields)
            #k, spectra = tools.getspectra(lag_fields)

            header = '1, b1, b2, bg, bk'
            if zadisp: np.savetxt('./output/%s/spectraza-%04d-%04d-%04d-R%d.txt'%(subf, aa*10000, bs, nc, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
            else: np.savetxt('./output/%s/spectra-%04d-%04d-%04d-R%d.txt'%(subf, aa*10000, bs, nc, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
            header = header.split(',')
            bvec = [1, 1, 1, 1, 1]
            model = tools.getmodel(spectra, bvec)
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
            if zadisp: plt.savefig('figs/%s/exampleza-%04d-%04d-%04d-R%d.png'%(subf, aa*10000, bs, nc, Rsm))
            else: plt.savefig('figs/%s/example-%04d-%04d-%04d-R%d.png'%(subf, aa*10000, bs, nc, Rsm))



            if rank == 0:
                rep = fitbias(ph, spectra, k=k)
                bvec = [1] + list(rep.x)

                model = tools.getmodel(spectra, bvec)
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
                if zadisp: plt.savefig('figs/%s/examplefitza-%04d-%04d-%04d-R%d.png'%(subf, aa*10000, bs, nc, Rsm))
                else: plt.savefig('figs/%s/examplefit-%04d-%04d-%04d-R%d.png'%(subf, aa*10000, bs, nc, Rsm))




