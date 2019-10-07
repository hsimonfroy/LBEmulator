import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, ArrayCatalog
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.transform import HaloRadius, HaloVelocityDispersion
from nbodykit.cosmology.cosmology import Cosmology
import os

from emulator_components import get_filename

import hod

import sys            # 
from time import time

def atoz(a): return 1/a - 1
def ztoa(z): return 1/(z+1)

cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
zzfiles = [round(atoz(aa), 2) for aa in aafiles]



sc_simpath = '/global/cscratch1/sd/sfschen/cm_crowcanyon_lemu/runs/'
sc_outpath = '/global/cscratch1/sd/sfschen/lagrangian_emulator/data/'


def make_galcat(aa, nc,seed,bs=1536,T=40,B=2, mmin=10**12.5, m1=20*10**12.5, alpha=0.9, censuff=None, satsuff=None,simpath=sc_simpath,outpath=sc_outpath):
    
    '''Assign 0s to 
    '''
    
    fname = get_filename(nc,seed,T=T,B=B)
    spath = simpath + fname
    opath = outpath + fname + 'hod/'

    try: os.makedirs(opath)
    except : pass
    
    
    zz = atoz(aa)
    #halocat = readincatalog(aa)
    halocat = BigFileCatalog(spath + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
    rank = halocat.comm.rank

    halocat.attrs['BoxSize'] = np.broadcast_to(halocat.attrs['BoxSize'], 3)

    ghid = halocat.Index.compute()
    halocat['GlobalIndex'] = ghid
    mp = halocat.attrs['MassTable'][1]*1e10
    halocat['Mass'] = halocat['Length'] * mp
    halocat['Position'] = halocat['Position']%bs # Wrapping positions assuming periodic boundary conditions
    rank = halocat.comm.rank
    
    halocat = halocat.to_subvolumes()

    if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%zz)

    hmass = halocat['Mass'].compute()
    hpos = halocat['Position'].compute()
    hvel = halocat['Velocity'].compute()
    rvir = HaloRadius(hmass, cosmo, 1/aa-1).compute()/aa
    vdisp = HaloVelocityDispersion(hmass, cosmo, 1/aa-1).compute()
    ghid = halocat['GlobalIndex'].compute()

    # Select halos that have galaxies
    rands = np.random.uniform(size=len(hmass))
    ncen = hod.ncen(mh=hmass,mcutc=mmin,sigma=0.2)
    ws = (rands < ncen)
    
    hmass = hmass[ws]
    hpos = hpos[ws]
    hvel = hvel[ws]
    rvir = rvir[ws]
    vdisp = vdisp[ws]
    ghid = ghid[ws]
    
    print('In rank = %d, Catalog size = '%rank, hmass.size)
    #Do hod    
    start = time()
    ncen = np.ones_like(hmass)
    #nsat = hod.nsat_martin(msat = mmin, mh=hmass, m1f=m1f, alpha=alpha).astype(int)
    nsat = hod.nsat_zheng(mh=hmass, m0=mmin, m1=20*mmin, alpha=alpha).astype(int)
    
    #Centrals
    cpos, cvel, gchid, chid = hpos, hvel, ghid, np.arange(ncen.size)
    spos, svel, shid = hod.mksat(nsat, pos=hpos, vel=hvel, 
                                 vdisp=vdisp, conc=7, rvir=rvir, vsat=0.5, seed=seed)
    gshid = ghid[shid]

    print('In rank = %d, Time taken = '%rank, time()-start)
    print('In rank = %d, Number of centrals & satellites = '%rank, ncen.sum(), nsat.sum())
    print('In rank = %d, Satellite occupancy: Max and mean = '%rank, nsat.max(), nsat.mean())
    #
    #Save
    cencat = ArrayCatalog({'Position':cpos, 'GlobalID':gchid, 
                           'Nsat':nsat}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])
    minid, maxid = cencat['GlobalID'].compute().min(), cencat['GlobalID'].compute().max() 
    if minid < 0 or maxid < 0:
        print('before ', rank, minid, maxid)
    cencat = cencat.sort('GlobalID')
    minid, maxid = cencat['GlobalID'].compute().min(), cencat['GlobalID'].compute().max() 
    if minid < 0 or maxid < 0:
        print('after ', rank, minid, maxid)

    if censuff is not None:
        colsave = [cols for cols in cencat.columns]
        print("Saving centrals.")
        cencat.save(opath+'cencat'+censuff, colsave)
    

    #satcat = ArrayCatalog({'Position':spos, 'Velocity':svel, 'Velocity_HI':svelh1, 'Mass':smass,  
                         #  'GlobalID':gshid, 'HaloMass':hmass[shid]}, 
                         # BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])
    satcat = ArrayCatalog({'Position':spos,  
                           'GlobalID':gshid}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])
    minid, maxid = satcat['GlobalID'].compute().min(), satcat['GlobalID'].compute().max() 
    if minid < 0 or maxid < 0:
        print('before ', rank, minid, maxid)
    satcat = satcat.sort('GlobalID')
    minid, maxid = satcat['GlobalID'].compute().min(), satcat['GlobalID'].compute().max() 
    if minid < 0 or maxid < 0:
        print('after ', rank, minid, maxid)

    if satsuff is not None:
        colsave = [cols for cols in satcat.columns]
        print("Saving sats.")
        satcat.save(opath+'satcat'+satsuff, colsave)

#

if __name__=="__main__":

    nc, seed = 2048, 9202
    
    for aa in [1.0, 0.5,]:

        #sat hod : N = ((M_h-\kappa*mcut)/m1)**alpha
        zz = 1/aa-1

        mmin = 10**12.5
        m1fac = 20; m1 = mmin * m1fac
        alpha = 0.9

        censuff ='-aa-%.04f-Mmin-%.1f-M1f-%.1f-alpha-0p8-subvol'%(aa,np.log10(mmin), m1fac)
        satsuff ='-aa-%.04f-Mmin-%.1f-M1f-%.1f-alpha-0p8-subvol'%(aa,np.log10(mmin), m1fac)

        make_galcat(aa, nc, seed, censuff=censuff, satsuff=satsuff)

    



