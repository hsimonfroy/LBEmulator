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

#### Define default directories ####
sc_simpath = '/global/cscratch1/sd/sfschen/cm_crowcanyon_lemu/runs/'
sc_outpath = '/global/cscratch1/sd/sfschen/lagrangian_emulator/data/spectra/'

#### Now define the actual functions

def get_filename(nc,seed,T=40,B=2):
    # Our simulation naming convention. Should probably have included box size but oh well.
    return '/N%d-T%d-B%d/S%d/'%(nc,T,B,seed)

def make_lagfields(nc,seed,bs=1536,T=40,B=2,simpath=sc_simpath,outpath=sc_outpath, Rsm=0):

    fname = get_filename(nc,seed,T=T,B=B)
    spath = simpath + fname
    opath = outpath + fname

    try: os.makedirs(opath)
    except : pass

    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f4')
    rank = pm.comm.rank
        
    lin = BigFileMesh(spath + '/linear', 'LinearDensityK').paint()
    lin -= lin.cmean()

    print(rank, 'lin field read')
    header = '1,b1,b2,bg,bk'
    names = header.split(',')
    lag_fields = tools.getlagfields(pm, lin, R=Rsm) # use the linear field at the desired redshift
        
    print(rank, 'lag field created')

    for i, ff in enumerate(lag_fields):
        x = FieldMesh(ff)
        x.save(opath + 'lag', dataset=names[i], mode='real')

def get_lagweights(nc,seed,bs=1536,T=40,B=2,simpath=sc_simpath,outpath=sc_outpath, dyn=None):
    # Note that dyn is the particle catalog, which we reload if not given
    
    aa = 1.0000 # since particle IDs are ordered only need to load at one redshift
    zz = 1/aa-1

    fname = get_filename(nc,seed,T=T,B=B)
    spath = simpath + fname
    opath = outpath + fname

    try: os.makedirs(opath)
    except : pass

    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f4')
    rank = pm.comm.rank

    if dyn is None:
        particle_file = spath +  '/fastpm_%0.4f/1'%aa
        print("Loading particle data from: " + particle_file)
        dyn = BigFileCatalog(particle_file)

    #
    fpos = dyn['Position'].compute()
    idd = dyn['ID'].compute()
    attrs = dyn.attrs

    grid = tools.getqfromid(idd, attrs, nc)
    if rank == 0: print('grid computed')

    cat = ArrayCatalog({'ID': idd, 'InitPosition': grid}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
        
    header = '1,b1,b2,bg,bk'    
    names = header.split(',')

    tosave = ['ID', 'InitPosition'] + names
    if rank == 0: print(tosave)
        
    for i in range(len(names)):
        ff = BigFileMesh(opath+ '/lag', names[i]).paint()
        pm = ff.pm
        rank = pm.comm.rank
        glay, play = pm.decompose(grid), pm.decompose(fpos)
        wts = ff.readout(grid, layout = glay, resampler='nearest')
        cat[names[i]] = wts
                
        del pm, ff, wts
        
    if rank == 0: print(rank, ' got weights')

    cat.save(opath + 'lagweights/', tosave)
    
def make_component_spectra(scale_factor,nc,seed,bs=1536,T=40,B=2,simpath=sc_simpath,outpath=sc_outpath,Rsm=0):
    
    aa = scale_factor # since particle IDs are ordered only need to load at one redshift
    zz = 1/aa-1
    dgrow = cosmo.scale_independent_growth_factor(zz)

    fname = get_filename(nc,seed,T=T,B=B)
    spath = simpath + fname
    opath = outpath + fname

    try: os.makedirs(opath)
    except : pass

    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc], dtype='f8')
    rank = pm.comm.rank

    header = '1,b1,b2,bg,bk'
    names = header.split(',')
    dfacs = [1, dgrow, dgrow**2, dgrow**2, dgrow]
    #
    dyn = BigFileCatalog(spath +  '/fastpm_%0.4f/1'%aa)
    fpos = dyn['Position'].compute()
    idd = dyn['ID']
    attrs = dyn.attrs
    play = pm.decompose(fpos)
        
    #
    wts = BigFileCatalog(opath +  '/lagweights/')
        
    grid = wts['InitPosition']
    iddg = wts['ID']

    print('asserting')
    np.allclose(idd.compute(), iddg.compute())
    print('read fields')

    disp = grid.compute() - fpos
    mask = abs(disp) > bs/2.
    disp[mask] = (bs - abs(disp[mask]))*-np.sign(disp[mask])
    print(rank, ' Max disp: ', disp.max())
    print(rank, ' Std disp: ', disp.std(axis=0))

    eul_fields = []
    for i in range(len(names)):
        eul_fields.append(pm.paint(fpos, mass=dfacs[i]*wts[names[i]], layout=play))
        print(rank, names[i], eul_fields[-1].cmean())
            

    del dyn, fpos, idd, iddg, wts
    print('got fields')
            
    k, spectra = tools.getspectra(eul_fields)
    np.savetxt(opath + '/spectra-z%03d-R%d.txt'%(zz*100, Rsm), np.vstack([k, spectra]).T.real, header='k / '+header, fmt='%0.4e')