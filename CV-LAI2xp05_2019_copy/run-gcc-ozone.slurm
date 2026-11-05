#!/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --job-name=CV_1
#SBATCH --mem=58G

###############################################################################
### If you are running a nested-grid simulation at fine resolution, you
### will likely need to request additional memory, cores, and time.
###
### -c           : Requests this many cores
### -N           : Requests a single node
### --mem        : Requests this amount of memory in GB
### -p           : Requests these partitions where the job can run
### -t           : Requests time for the job (days-hours:minutes)
###  --exclusive : Reserves entire nodes (i.e. to prevent backfilling jobs)
###############################################################################

source /home/C837642639/gcc-env.sh

ulimit -c unlimited   # coredumpsize
ulimit -u 50000       # maxproc
ulimit -v unlimited   # vmemoryuse
ulimit -s unlimited   # stacksize

export OMP_STACKSIZE=500m

# Set the proper # of threads for OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run GEOS_Chem.  The "time" command will return CPU and wall times.
# Stdout and stderr will be directed to the "GC.log" log file
# (you can change the log file name below if you wish)
srun -c $OMP_NUM_THREADS time -p ./gcclassic >> GC.log

# Exit normally
exit 0

