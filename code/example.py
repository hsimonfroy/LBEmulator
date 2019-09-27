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
import sys
sys.path.append('./utils/')
import za
import features as ft
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)



def fitbias(ph, spectra, binit=[1, 0, 0, 0], k=None, kmax=None):


    if k is not None and kmax is not None:
        ik = np.where(k > kmax)[0][0]
    else: ik = len(ph)
    tomin = lambda b: sum((ph - tools.getmodel(spectra, [1] + list(b)))[:ik]**2)
    rep = minimize(tomin, binit, method='Nelder-Mead', options={'maxfev':10000})
    return rep



if __name__=="__main__":


    bs, nc = 400, 256
    dpath = '/global/cscratch1/sd/chmodi/cosmo4d/data/z00/L%04d_N%04d_S0100_40step/'%(bs, nc)
    aa = 1.0000
    zz = 1/aa-1
    Rsm = 0
    zadisp = True
    
    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
    rank = pm.comm.rank
    #grid = pm.mesh_coordinates()*bs/nc
    
    lin = BigFileMesh(dpath + '/mesh', 's').paint()
    print(lin.cmean())
    dyn = BigFileCatalog(dpath + '/dynamic/1')
    hcat = BigFileCatalog(dpath + '/FOF/')

    for Rsm in [0, 2]:
        for zadisp in [True, False]:
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

            lag_fields = tools.getlagfields(pm, lin, R=Rsm)
            eul_fields = tools.geteulfields(pm, lag_fields, fpos, grid)
            k, spectra = tools.getspectra(eul_fields)

            header = '1, b1, b2, bg, bk'
            if zadisp: np.savetxt('./output/spectraza-%04d-%04d-R%d.txt'%(bs, nc, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
            else: np.savetxt('./output/spectra-%04d-%04d-R%d.txt'%(bs, nc, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')
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
            if zadisp: plt.savefig('figs/exampleza-%04d-%04d-R%d.png'%(bs, nc, Rsm))
            else: plt.savefig('figs/example-%04d-%04d-R%d.png'%(bs, nc, Rsm))



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
                if zadisp: plt.savefig('figs/examplefitza-%04d-%04d-R%d.png'%(bs, nc, Rsm))
                else: plt.savefig('figs/examplefit-%04d-%04d-R%d.png'%(bs, nc, Rsm))




