#!/usr/bin/env python3
#
# Python script to copy power spectrum files from disparate
# places into a single directory with a "sensible" naming
# convention.
#

import numpy as np
import shutil
import glob
import os

# Destination directory.
destdir = "/global/cscratch1/sd/mwhite/LagEmu/AllSpectra"

# Directories to search.
dirlist = ["/global/cscratch1/sd/sfschen/lagrangian_emulator"+\
           "/data/N2048-T40-B2/", \
           "/global/cscratch1/sd/mwhite/LagEmu/N2048-T40-B2/"]

# Seeds to look for.
slist   = range(9200,9210)


def copy_halo_spectra():
    """Copies the halo auto- and cross-spectrum files."""
    for idir in dirlist:
        for iseed in slist:
            db = idir + "S{:04d}/spectra".format(iseed)
            if os.path.isdir(db):
                for infn in glob.glob(db+"/ph_*_z???.txt"):
                    outfn = infn.rstrip(".txt")[len(db)+1:]
                    outfn+= "_{:04d}.txt".format(iseed)
                    shutil.copy2(infn,destdir+"/"+outfn)




def copy_galaxy_spectra():
    """Copies the galaxy auto- and cross-spectrum files."""
    for idir in dirlist:
        for iseed in slist:
            db = idir + "S{:04d}/spectra".format(iseed)
            if os.path.isdir(db):
                for infn in glob.glob(db+"/pks-*-1536-2048-gal.txt"):
                    outfn = infn.rstrip(".txt")[len(db)+5:]
                    aa    = float(outfn[:outfn.find('-')]) / 1e4
                    iz    = int( 100*(1.0/aa-1.)+0.01 )
                    outfn = "pg_z{:03d}".format(iz)
                    outfn+= "_{:04d}.txt".format(iseed)
                    shutil.copy2(infn,destdir+"/"+outfn)






def copy_component_spectra():
    """Copies the component spectrum files -- some of the columns are
    very noisy at low k, so we use PT to overwrite those."""
    for idir in dirlist:
        for iseed in slist:
            db = idir + "S{:04d}/spectra".format(iseed)
            if os.path.isdir(db):
                for infn in glob.glob(db+"/spectra-z???-R0.txt"):
                    pk    = np.loadtxt(infn)
                    # Overwrite the "noisy" columns with PT.
                    zz    = float(infn[len(db)+10:len(db)+13])/100.
                    ptfn  = "../theory/"
                    ptfn += "cleft_components_z_{:3.1f}.dat".format(zz)
                    clpt  = np.loadtxt(ptfn)
                    # Replace ( 1,b2), column 3 with column 6 in PT file.
                    ww       = np.nonzero( pk[:,0]<0.08 )[0]
                    pk[ww,3] = np.interp(pk[ww,0],clpt[:,0],clpt[:, 6])
                    # Replace (b1,b2), column 7 with column 7 in PT file.
                    ww       = np.nonzero( pk[:,0]<0.08 )[0]
                    pk[ww,7] = np.interp(pk[ww,0],clpt[:,0],clpt[:, 7])
                    # Replace ( 1,bs), column 4 with column 9 in PT file.
                    ww       = np.nonzero( pk[:,0]<0.06 )[0]
                    pk[ww,4] = np.interp(pk[ww,0],clpt[:,0],clpt[:, 9]/2)
                    # Replace (b1,bs), column 4 with column 9 in PT file.
                    ww       = np.nonzero( pk[:,0]<0.07 )[0]
                    pk[ww,8] = np.interp(pk[ww,0],clpt[:,0],clpt[:,10]/2)
                    # Now write the modified data to file.
                    outfn = infn.rstrip(".txt")[len(db)+1:]
                    outfn+= "_{:04d}.txt".format(iseed)
                    outfn = outfn.replace("-","_")
                    outfn = outfn.replace("spectra","pc")
                    #
                    fout  = open(destdir+"/"+outfn,"w")
                    fout.write("# Component spectra for z={:f}.\n".format(zz))
                    for i in range(pk.shape[0]):
                        outstr = "{:15.5e}".format(pk[i,0])
                        for j in range(1,pk.shape[1]):
                            outstr += " {:15.5e}".format(pk[i,j])
                        fout.write(outstr+"\n")
                    fout.close()
                    #shutil.copy2(infn,destdir+"/"+outfn)




if __name__=="__main__":
    copy_halo_spectra()
    copy_galaxy_spectra()
    copy_component_spectra()
    #
