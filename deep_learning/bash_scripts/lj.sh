#!/bin/bash

#SBATCH -N 1          		# Total number of nodes requested 
#SBATCH --ntasks-per-node 3     # Number of tasks per node 
#SBATCH --cpus-per-task 8
#SBATCH -o logs/job.%j.out  	# Name of stdout file (%j expands to jobId)
#SBATCH -e logs/err.%j.out	# Name of stderr file (%j expands to jobId)
#SCATCH -J learnsolnmap		# Job name
#SBATCH -p gpu-a100  		# Submission queue
#SBATCH -A ASC23019 		# Project name (allocation)
#SBATCH -t 48:00:00		# Max run time 

#SBATCH --mail-user=rfang@utexas.edu	# Desired email address
#SBATCH --mail-type=all		# Send email at begin and end of job 

LJ_DATA_DIR1=$WORK/solnmap_data/lj/rhmc-H0/dt2e-4_h5e-8/Nchains100_Njumps2000/sigma1e-2/Dt5e-4
LJ_DATA_DIR2=$WORK/solnmap_data/lj/f-rhmc-H0/dt5e-5_h5e-8/Nchains100_Njumps400_Nsteps4/sigma1e-2/Dt5e-4
LJ_DATA_DIR3=$WORK/solnmap_data/lj/rhmc/dt2e-4_h5e-8/Nchains100_Njumps2000/beta4e21/Dt5e-4
LJ_DATA_DIR4=$WORK/solnmap_data/lj/rhmc-H0/dt2e-4_h5e-8/Nchains100_Njumps2000/sigma1e-2/Dt1e-4
LJ_DATA_DIR5=$WORK/solnmap_data/lj/rhmc-H0/dt2e-4_h5e-8/Nchains100_Njumps2000/sigma1e-2/Dt1e-3

module load python3
module load cuda/12.0 

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
#export NCCL_SOCKET_IFNAME=^docker0,lo

python3 ../train_solnmap.py --group lennardjones --Delta_t 1e-4 --data_dir $LJ_DATA_DIR4  --h2h_model HamiltonianReversibleNetwork --h2h_layer_sizes 28 2000 2000 28 --sequence_weights 1 1 1 1 1 --num_epochs 1000 --batch_size 100 --lr 1e-3 --S_strength 0 --gpus 1  &
python3 ../train_solnmap.py --group lennardjones --Delta_t 1e-4 --data_dir $LJ_DATA_DIR4  --h2h_model HamiltonianReversibleNetwork --h2h_layer_sizes 28 2000 2000 28 --sequence_weights 1 1 1 1 1 --num_epochs 1000 --batch_size 100 --lr 1e-4 --S_strength 0 --gpus 1  &
python3 ../train_solnmap.py --group lennardjones --Delta_t 1e-4 --data_dir $LJ_DATA_DIR4  --h2h_model ResMLP --h2h_layer_sizes 28 1000 1000 1000 1000 28 --sequence_weights 1 1 1 1 1 --num_epochs 1000 --lr 1e-3 --S_strength 0 --gpus 0  &

wait 
