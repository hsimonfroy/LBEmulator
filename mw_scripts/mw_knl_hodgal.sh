#!/bin/bash
#SBATCH -J HOD
#SBATCH -t 00:10:00
#SBATCH -N 16
#SBATCH -p debug
#SBATCH -C knl
#SBATCH -o HodGal.out
#SBATCH -e HodGal.err
#SBATCH -L cscratch1
#SBATCH -A m3404
#
module unload darshan
module unload python
#
module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/5.3.0
#
source /global/common/software/m3035/conda-activate.sh 3.7
export OMP_NUM_THREADS=4
#
bcast-pip -U --no-deps https://github.com/bccp/nbodykit/archive/master.zip
echo 'Finally starting'
#
time srun -N ${SLURM_NNODES} --ntasks-per-node 48 -c 4 \
	python -u hodgal.py
#
