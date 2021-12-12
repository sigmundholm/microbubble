#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=7-0:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000
########## EDIT JOB NAME #######################
#SBATCH --job-name="hello_test"
################################################
#SBATCH --output=slurm-run.txt
#SBATCH --mail-user=sigmund.holm@gmail.com
#SBATCH --mail-type=ALL
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo
echo "This is the current directory: $PWD"
echo ""
START_TIME="$(date '+%F, %H:%M:%S')"
echo "Start time: $START_TIME"
echo 

module purge
module load foss/2020b
module load CMake/3.18.4-GCCcore-10.2.0.lua
module load deal.II/sigmund-local
module load cutfem/sigmund-local
module list
echo
echo "Variables:"
echo " - DEAL_II_DIR = $DEAL_II_DIR"
echo " - CUTFEM_DIR = $CUTFEM_DIR"

PROJECT_DIR="/cluster/work/sigmuneh/microbubble"
HOMEDIR="/cluster/home/sigmuneh/output"

# Get the current commit message and hash.
echo
git log -1
echo

cd $WORKDIR
rm *.vtk
rm *.csv
# Remove all txt files except the output file.
ls | grep txt | grep -v  slurm-run.txt | xargs rm
make -j10

########################################################
# EDIT THE SCRIPT NAME AND ACTUAL SCRIPT 
########################################################
SCRIPT_NAME="moving-sphere"
# Run the job on only one processor for now
mpirun -np 1 ./fsi-moving-sphere
########################################################

# Write out the current commit message again.
echo
git log -1
echo

END_TIME="$(date '+%F, %H:%M:%S')"
echo
echo "End time: $END_TIME"
echo

# Save the results for later in the home folder.
TODAY="$(date +%F)"
NOW="$(date '+%H-%M-%S')"
mkdir -p $HOMEDIR/$TODAY/$NOW-$SCRIPT_NAME
cp -a $WORKDIR/. $HOMEDIR/$TODAY/$NOW-$SCRIPT_NAME
