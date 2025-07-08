#!/bin/bash
#SBATCH --job-name=run_cls
#SBATCH --partition=berg          # or berg, normal, redwood
#SBATCH --nodes=2
#SBATCH --ntasks=2                # total number of MPI tasks 
#SBATCH --cpus-per-task=28        # cores/job_per_node
#SBATCH --output=log/slurm-%j.out 

# Set up environment
export OMP_NUM_THREADS=28        # threads to match cpus-per-task
PYTHON=$HOME/mambaforge/envs/cosmotheka/bin/python

# Input
input="input/master_DESILRG_tSZ_ns1024.yml"

# Log file
# output_path=$(grep '^output:' "$input" | awk -F': ' '{print $2}' | tr -d "'\"")
# timestamp=$(date +%Y%m%d_%H%M)
# logname=${output_path}/log/run_cls_${timestamp}.log

# Make sure log directory exists
# mkdir -p "$(dirname "$logname")"

# Run the job
# $PYTHON run_cls_mpi.py "$input" to_sacc > "$logname" 2>&1
mpirun $PYTHON run_cls_mpi.py "$input" to_sacc 