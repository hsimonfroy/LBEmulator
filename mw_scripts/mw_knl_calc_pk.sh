#!/bin/bash
#SBATCH -J Pk
#SBATCH -t 00:20:00
#SBATCH -N 32
#SBATCH -p debug
#SBATCH -C knl
#SBATCH -o CalcPk.out
#SBATCH -e CalcPk.err
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
	python -u mw_calc_pk2.py
#
