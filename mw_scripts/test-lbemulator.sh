#! /bin/bash
#SBATCH -J Test
#SBATCH -t 0:30:00
#SBATCH -N 2
#SBATCH -p debug
#SBATCH -C knl
#SBATCH -o Test.out
#SBATCH -e Test.err
#SBATCH -A m3404
#
#DW jobdw capacity=10000GB access_mode=striped type=scratch
#DW stage_out source=$DW_JOB_STRIPED/ destination=/global/cscratch1/sd/mwhite/LagEmu/ type=directory
#
export ATP_ENABLED=1
export OMP_NUM_THREADS=4
export MPICH_ALLREDUCE_BLK_SIZE=$((4096*1024*2))
export MPICH_GNI_MALLOC_FALLBACK=enabled
#export MPICH_GNI_MBOXES_PER_BLOCK=$((${SLURM_NNODES} * 64))
#export MPICH_GNI_MAX_VSHORT_MSG_SIZE=64
#
#module load craype-hugepages2M
#
nc=256
B=2
N=10
for seed in {9200..9200..1}; do
    echo $seed
    output="$DW_JOB_STRIPED/N$nc-T$N-B$B/S$seed/"
    srun -N ${SLURM_NNODES} --ntasks-per-node 48 -c 4 \
      /project/projectdirs/m3127/codes/40eae2464/src/fastpm \
      lbemulator.lua  $nc $seed $output $B $N
done
