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
dirlist = ["/global/cscratch1/sd/mwhite/LagEmu/N2048-T40-B2/",
           "/global/cscratch1/sd/sfschen/lagrangian_emulator"+\
           "/data/N2048-T40-B2/"]
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




def copy_component_spectra():
    """Copies the component spectrum files."""
    for idir in dirlist:
        for iseed in slist:
            db = idir + "S{:04d}/spectra".format(iseed)
            if os.path.isdir(db):
                for infn in glob.glob(db+"/spectra-z???-R0.txt"):
                    outfn = infn.rstrip(".txt")[len(db)+1:]
                    outfn+= "_{:04d}.txt".format(iseed)
                    outfn = outfn.replace("-","_")
                    outfn = outfn.replace("spectra","pc")
                    shutil.copy2(infn,destdir+"/"+outfn)




if __name__=="__main__":
    copy_halo_spectra()
    copy_component_spectra()
    #
