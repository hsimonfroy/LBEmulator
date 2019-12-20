#!/usr/bin/env python3
#
#
import numpy as np
import matplotlib.pyplot as plt


def make_plot():
    """Does all of the work."""
    fig,ax = plt.subplots(1,2,figsize=(8.5,3.5))
    #
    # First show how non-linear things are.
    #
    iz,chi,dchi=100,2300.,847. # dz=+/-0.25 around z=1.
    zz  = 0.01*iz
    kmax= 0.6
    lmax= kmax * chi
    for Mmin,Mmax,sn in zip([12.00,13.00],[12.50,13.50],[437.1,3404.]):
        fn = "data/ph_{:05.2f}_{:05.2f}_z{:03d}.txt".format(Mmin,Mmax,iz)
        dd = np.loadtxt(fn)
        d2 = dd[:,0]**3*dd[:,1]/2/np.pi**2
        ell= dd[:,0]*chi
        cl = dd[:,1]/chi**2/dchi
        ww = sn/chi**2/dchi
        ax[0].plot(dd[:,0],d2,'bs',mfc='None',alpha=0.5)
        ax[1].plot(ell,cl,'bs',mfc='None',alpha=0.5)
        #
        fn = "data/pth_z_{:03.1f}_M_{:04.1f}_{:04.1f}.dat".format(zz,Mmin,Mmax)
        tt = np.loadtxt(fn)
        d2 = tt[:,0]**3*tt[:,1]/2/np.pi**2
        ell= tt[:,0]*chi
        cl = tt[:,1]/chi**2/dchi
        ax[0].plot(tt[:,0],d2,'m-',lw=3)
        ax[1].plot(ell,cl,'m-',lw=3)
        ax[1].plot(ell,(cl-ww).clip(1e-10,1e10),'m--',lw=3)
        ax[1].plot([1,1e5],[ww,ww],'m:',lw=3)
        #
        if Mmin<12.5:
            xloc,yloc = 5e-2,0.03
        else:
            xloc,yloc = 3e-2,5
        ax[0].text(xloc,yloc,r'$lgM\in['+'{:4.1f},{:4.1f}]$'.format(Mmin,Mmax))
    #
    ax[0].text(1.2e-2,30,'$z={:.1f}$'.format(zz))
    ax[0].plot([1e-2,1],[1,1],'k:')
    ax[0].plot([kmax,kmax],[1-2,1e2],'k:')
    ax[1].plot([lmax,lmax],[1-2,1e5],'k:')
    #
    ax[0].set_xlim(1e-2,1.0)
    ax[0].set_ylim(1e-2,100.)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$k\quad [h\ {\rm Mpc}^{-1}]$',fontsize=12)
    ax[0].set_ylabel(r'$\Delta^2(k)$',fontsize=12)
    #
    ax[1].set_xlim(10,4000)
    ax[1].set_ylim(2e-8,3e-5)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\ell$',fontsize=12)
    ax[1].set_ylabel(r'$C_\ell$',fontsize=12)
    #
    plt.tight_layout()
    plt.savefig("high_enough_k.png")
    #



if __name__=="__main__":
    make_plot()
