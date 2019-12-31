#!/usr/bin/env python3
#
#
import numpy as np
import matplotlib.pyplot as plt


def make_plot():
    """Does all of the work."""
    fig,ax = plt.subplots(1,2,figsize=(8.5,3.25))
    #
    iz  = 0
    zz  = 0.01*iz
    fn = "../data/pc_z{:03d}_R0.txt".format(iz)
    dd = np.loadtxt(fn)
    dd = np.abs(dd)
    #
    ax[0].plot(dd[:,0],dd[:,1],color='C0',label=r'$(1,1)$')
    ax[0].plot(dd[:,0],dd[:,2],color='C1',label=r'$(1,\delta)$')
    ax[0].plot(dd[:,0],dd[:,3],color='C2',label=r'$(1,\delta^2)$')
    ax[0].plot(dd[:,0],dd[:,4],color='C3',label=r'$(1,s^2)$')
    ax[0].plot(dd[:,0],dd[:,5],color='C4',label=r'$(1,\nabla^2\delta)$')
    #
    ax[1].plot(dd[:,0],dd[:,6],color='C0',label=r'$(\delta,\delta)$')
    ax[1].plot(dd[:,0],dd[:,7],color='C1',label=r'$(\delta,\delta^2)$')
    ax[1].plot(dd[:,0],dd[:,8],color='C2',label=r'$(\delta,s^2)$')
    ax[1].plot(dd[:,0],dd[:,9],color='C3',label=r'$(\delta,\nabla^2\delta)$')
    ax[1].plot(dd[:,0],dd[:,10],color='C4',label=r'$(\delta^2,\delta^2)$')
    #
    ax[0].legend(framealpha=0.5)
    ax[0].set_xlim(1e-2,1.0)
    ax[0].set_ylim(1e1,4e4)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$k\quad [h\ {\rm Mpc}^{-1}]$',fontsize=12)
    ax[0].set_ylabel(r'$|P_{ij}(k)|\quad[h^{-3}{\rm Mpc}^3]$',fontsize=12)
    ax[0].text(0.9,3e4,'$z=0$',ha='right',va='top')
    #
    ax[1].legend(framealpha=0.5)
    ax[1].set_xlim(1e-2,1.0)
    ax[1].set_ylim(1e1,4e4)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_yticklabels([])
    ax[1].set_xlabel(r'$k\quad [h\ {\rm Mpc}^{-1}]$',fontsize=12)
    #
    plt.tight_layout()
    plt.savefig("some_components.png")
    #



if __name__=="__main__":
    make_plot()
