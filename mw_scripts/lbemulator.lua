-- parameter file
------ Size of the simulation -------- 

-- For Testing

print(#args)
if #args ~= 5 then
    print("usage: fastpm highres.lua nc random_seed output_prefix B N")
    os.exit(1)
end

nc = tonumber(args[1])
random_seed = tonumber(args[2])
prefix = args[3]
B = tonumber(args[4])   -- Particle Mesh grid pm_nc_factor*nc per dimension in the beginning
N = tonumber(args[5])   -- Number of timesteps



boxsize = 1536
time_step = linspace(0.1, 1, N)
output_redshifts= {3.0, 2.0, 1.5, 1.0, 0.5, 0.0}  -- redshifts of output
--output_redshifts= {0.0}  -- redshifts of output

lc_amin = 1 / (2.2 + 1)
lc_amax = 0.995

-- Cosmology --
omega_m = 0.309167
h       = math.sqrt((0.022470 + 0.119230) / omega_m)

-- Start with a power spectrum file
-- Initial power spectrum: k P(k) in Mpc/h units
-- Must be compatible with the Cosmology parameter
read_powerspectrum= "/global/project/projectdirs/m3127/cosmology/PLANCK18BAO/pk_Planck2018BAO_matterpower_z000.dat"
-- remove_cosmic_variance = true

-------- Approximation Method ---------------
force_mode = "fastpm"
-- force_mode = "cola"

pm_nc_factor = B            -- Particle Mesh grid pm_nc_factor*nc per dimension in the beginning

np_alloc_factor= 2.7      -- Amount of memory allocated for particle



-------- Output ---------------

-- Dark matter particle outputs (all particles)

write_snapshot= prefix .. "/fastpm" 
write_lineark= prefix .. "/linear" 
particle_fraction = 1.0
write_fof     = prefix .. "/fastpm" 
write_powerspectrum = prefix .. '/powerspec'
fof_linkinglength = 0.200
fof_nmin = 8
