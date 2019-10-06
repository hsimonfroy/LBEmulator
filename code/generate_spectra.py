from emulator_components import*

sc_simpath = '/global/cscratch1/sd/sfschen/cm_crowcanyon_lemu/runs/'
sc_outpath = '/global/cscratch1/sd/sfschen/lagrangian_emulator/data/spectra/'
bs, nc = 1536, 2048


for seed in np.arange(9200, 9202, 2):
    
    make_lagfields(nc,seed,bs=bs,T=40,B=2,simpath=sc_simpath,outpath=sc_outpath, Rsm=0)
    get_lagweights(nc,seed,bs=bs,T=40,B=2,simpath=sc_simpath,outpath=sc_outpath)
    
    # Make components for two redshifts
    make_component_spectra(1,nc,seed,bs=bs,T=40,B=2,simpath=sc_simpath,outpath=sc_outpath,Rsm=0)
    make_component_spectra(0.5,nc,seed,bs=bs,T=40,B=2,simpath=sc_simpath,outpath=sc_outpath,Rsm=0)