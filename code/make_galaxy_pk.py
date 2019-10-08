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

from emulator_components import get_filename

import tools
import os, sys
sys.path.append('./utils/')
import za
import features as ft
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)

import time

sc_simpath = '/global/cscratch1/sd/sfschen/cm_crowcanyon_lemu/runs/'
sc_outpath = '/global/cscratch1/sd/sfschen/lagrangian_emulator/data/'

def make_galaxy_pk(scale_factor,nc,seed,bs=1536,T=40,B=2,simpath=sc_simpath,outpath=sc_outpath,Rsm=0):
    
    aa = scale_factor # since particle IDs are ordered only need to load at one redshift
    zz = 1/aa-1
    dgrow = cosmo.scale_independent_growth_factor(zz)

    fname = get_filename(nc,seed,T=T,B=B)
    spath = simpath + fname
    opath = outpath + fname
    galpath = opath + '/hod/'

    try: os.makedirs(opath+'spectra/')
    except : pass
    
    zz = 1/aa-1
        
    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f8')
    rank = pm.comm.rank
    dyn = BigFileCatalog(spath +  '/fastpm_%0.4f/1'%aa)
        
    # Load Matter Mesh
    fpos = dyn['Position'].compute()
    mlay = pm.decompose(fpos)
    mmesh = pm.paint(fpos, layout=mlay)
    mmesh /= mmesh.cmean()

    # Load halo mesh: HOD parameters fixed for now...
    mmin = 10**12.5; m1fac = 20
    cendir ='cencat-aa-%.04f-Mmin-%.1f-M1f-%.1f-alpha-0p8-subvol'%(aa,np.log10(mmin), m1fac)
    satdir ='satcat-aa-%.04f-Mmin-%.1f-M1f-%.1f-alpha-0p8-subvol'%(aa,np.log10(mmin), m1fac)
    
    cencat = BigFileCatalog(galpath + cendir)
    satcat = BigFileCatalog(galpath + satdir)    
        
    hpos = np.concatenate((cencat['Position'],satcat['Position']))
    hlay = pm.decompose(hpos)
    hmesh = pm.paint(hpos, layout=hlay)
    hmesh /= hmesh.cmean()

    phm = FFTPower(mmesh, second=hmesh, mode='1d').power
    k, phm = phm['k'],  phm['power'].real
        
    phh = FFTPower(hmesh, mode='1d').power
    k, phh = phh['k'],  phh['power'].real

    np.savetxt(opath+'spectra/pks-%04d-%04d-%04d-gal.txt'%(aa*10000, bs, nc), np.vstack([k, phh, phm]).T.real, header='k, phh, phm/ ', fmt='%0.4e')



if __name__=="__main__":

    make_galaxy_pk(0.5,2048,9202)




