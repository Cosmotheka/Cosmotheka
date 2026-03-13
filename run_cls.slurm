#!/bin/bash
#SBATCH --job-name=run_cls
#SBATCH --partition=berg-priority          # or berg, normal, redwood
#SBATCH --nodes=3
#SBATCH --ntasks=3                # total number of MPI tasks 
#SBATCH --cpus-per-task=28        # cores/job_per_node
#SBATCH --output=log/slurm-%j.out 
#SBATCH --mem-per-cpu=7G

# Set up environment
export OMP_NUM_THREADS=28         # threads to match cpus-per-task
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cosmotheka
# PYTHON=$HOME/mambaforge/envs/cosmotheka/bin/python

# Input
# input="input/master_DESILRG_tSZ_ns1024.yml"
# input="input/master_DESILRG_CIB_ns1024.yml"
input="input/master_DESILRG_DESY3wl_CMBkappa_ns4096.yml"
# input="input/master_DESILRG_PlanckCMB18_ns512.yml"

# Log file
# output_path=$(grep '^output:' "$input" | awk -F': ' '{print $2}' | tr -d "'\"")
# timestamp=$(date +%Y%m%d_%H%M)
# logname=${output_path}/log/run_cls_${timestamp}.log

# Make sure log directory exists
# mkdir -p "$(dirname "$logname")"

# Run the job
# $PYTHON run_cls_mpi.py "$input" to_sacc > "$logname" 2>&1
mpirun python run_cls_mpi.py "$input" to_sacc --not_save_cw
# mpirun $PYTHON run_cls_mpi.py "$input" to_sacc --override_yaml