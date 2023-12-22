#!/bin/bash

#SBATCH -N 1          		# Total number of nodes requested 
#SBATCH --ntasks-per-node 3     # Number of tasks per node 
#SBATCH --cpus-per-task 8
#SBATCH -o logs/job.%j.out  	# Name of stdout file (%j expands to jobId)
#SBATCH -e logs/err.%j.out	# Name of stderr file (%j expands to jobId)
#SCATCH -J learnsolnmap		# Job name
#SBATCH -p gpu-a100		# Submission queue
#SBATCH -A ASC23019 		# Project name (allocation)
#SBATCH -t 48:00:00		# Max run time 

#SBATCH --mail-user=rfang@utexas.edu	# Desired email address
#SBATCH --mail-type=all		# Send email at begin and end of job 

FPU_DATA_DIR1=$WORK/solnmap_data/fpu/omega300/f-rhmc-H0/dt1e-1_h5e-6_Nchains100_Njumps400_Nsteps4_sigma1e-1/Dt1e-1
FPU_DATA_DIR2=$WORK/solnmap_data/fpu/omega300/f-rhmc-H0/dt1e-1_h5e-6_Nchains100_Njumps400_Nsteps4_sigma1e-1/Dt1e0
FPU_DATA_DIR3=$WORK/solnmap_data/fpu/omega300/rhmc-H0/css4_dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/css4_Dt1e-1_h5e-6
FPU_DATA_DIR4=$WORK/solnmap_data/fpu/omega300/rhmc-H0/css4_dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/css4_Dt1e0_h5e-6
FPU_DATA_DIR5=$WORK/solnmap_data/fpu/omega300/rhmc-H0/css4_dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/ma5_Dt1e0_h1.5259e-5
FPU_DATA_DIR6=$WORK/solnmap_data/fpu/omega300/hmc-H0/VelocityVerlet_dt4e-1_h5e-3_Nchains100_Njumps2000_sigma1e-1/Dt1e-1
FPU_DATA_DIR7=$WORK/solnmap_data/fpu/omega10/rhmc-H0/dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/Dt1e-2
FPU_DATA_DIR8=$WORK/solnmap_data/fpu/omega50/rhmc-H0/ma5_dt2.5e-1_h1.5259e-5_Nchains100_Njumps2000_sigma1e-1/ma5_Dt1e0_h1.5259e-5/

module load python3
module load cuda/12.0 

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
#export NCCL_SOCKET_IFNAME=^docker0,lo

#python3 train_solnmap.py --group fpu --omega 50 --Delta_t 1e0 --data_dir $FPU_DATA_DIR8  --h2h_model ResMLP --h2h_layer_sizes 12 1000 1000 1000 1000 12 --sequence_weights 1 1 1 1 1 --num_epochs 10000 --batch_size 100 --lr 1e-4  --gpus 0 --resume_from_ckpt ./solutionmap/429m8f1s/checkpoints/last.ckpt &
#python3 train_solnmap.py --group fpu --omega 50 --Delta_t 1e0 --data_dir $FPU_DATA_DIR8  --h2h_model ResMLP --h2h_layer_sizes 12 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 12 --sequence_weights 1 1 1 1 1 --num_epochs 5000 --batch_size 100 --lr 5e-4 --gpus 1 --resume_from_ckpt ./solutionmap/vgmzqfgh/checkpoints/last.ckpt &
#python3 train_solnmap.py --group fpu --Delta_t 1e0 --data_dir $FPU_DATA_DIR5  --h2h_model ResMLP --h2h_layer_sizes 12 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 12 --sequence_weights 1 1 1 1 1 --num_epochs 5000 --lr 5e-4 --steps_per_cycle 200000 --V_strength 0  --gpus 0 --resume_from_ckpt ./solutionmap/mgrw0exy/checkpoints/last.ckpt &
#python3 train_solnmap.py --group fpu --Delta_t 1e0 --data_dir $FPU_DATA_DIR5  --h2h_model ResMLP --h2h_layer_sizes 12 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 12 --sequence_weights 1 1 1 1 1 --num_epochs 5000 --lr 5e-4 --steps_per_cycle 200000 --V_strength 0  --gpus 1 --resume_from_ckpt ./solutionmap/668irhmp/checkpoints/last.ckpt &
# python3 train_solnmap.py --group fpu --omega 50 --Delta_t 1e0 --data_dir $FPU_DATA_DIR8  --h2h_model HamiltonianReversibleNetwork --h2h_layer_sizes 12 2000 2000 2000 2000 12 --sequence_weights 1 1 1 1 1 --num_epochs 1000  --lr 5e-4  --gpus 1 &
# python3 train_solnmap.py --group fpu --Delta_t 1e0 --data_dir $FPU_DATA_DIR5  --h2h_model ResMLP --h2h_layer_sizes 12 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 12 --sequence_weights 1 1 1 1 1 --num_epochs 1000 --lr 5e-4 --steps_per_cycle 100000 --V_strength 0  --gpus 1  --resume_from_ckpt /work/08170/rfang/maverick2/learnsolnmap/solutionmap/9343ycw5/checkpoints/last.ckpt &
python3 train_solnmap.py --group fpu --Delta_t 1e0 --data_dir $FPU_DATA_DIR5  --h2h_model ResMLP --h2h_layer_sizes 12 1000 1000 1000 1000 12 --sequence_weights 1 1 1 1 1 --num_epochs 10000 --batch_size 100 --lr 1e-4 --weight_init xavier_uniform --Comm_strength 0.  --gpus 0  --resume_from_ckpt ./solutionmap/zi05y8c3/checkpoints/last.ckpt &
python3 train_solnmap.py --group fpu --Delta_t 1e0 --data_dir $FPU_DATA_DIR5  --h2h_model ResMLP --h2h_layer_sizes 12 1000 1000 1000 1000 12 --sequence_weights 1 1 1 1 1 --num_epochs 10000 --batch_size 100 --lr 1e-4 --weight_init xavier_uniform --Comm_strength 0.01  --gpus 1 --resume_from_ckpt ./solutionmap/zyhdfka1/checkpoints/last.ckpt &
python3 train_solnmap.py --group fpu --Delta_t 1e0 --data_dir $FPU_DATA_DIR5  --h2h_model ResMLP --h2h_layer_sizes 12 1000 1000 1000 1000 12 --sequence_weights 1 1 1 1 1 --num_epochs 10000 --batch_size 100 --lr 1e-4 --weight_init xavier_uniform --Comm_strength 0.1  --gpus 2 --resume_from_ckpt ./solutionmap/9gswy5sq/checkpoints/last.ckpt &
#python3 train_solnmap.py --group fpu --Delta_t 1e0 --data_dir $FPU_DATA_DIR5  --h2h_model ResMLP3 --h2h_layer_sizes 12 1000 1000 1000 1000 12 --sequence_weights 1 1 1 1 1 --num_epochs 1000 --batch_size 100 --lr 1e-4  --weight_init xavier_uniform --gpus 0  &
#python3 train_solnmap.py --group fpu --Delta_t 1e0 --data_dir $FPU_DATA_DIR5  --h2h_model ResMLP3 --h2h_layer_sizes 12 1000 1000 1000 1000 12 --sequence_weights 1 1 1 1 1 --num_epochs 1000 --batch_size 100 --lr 5e-4  --weight_init xavier_uniform --gpus 1  &
#python3 train_solnmap.py --group fpu --Delta_t 1e0 --data_dir $FPU_DATA_DIR5  --h2h_model ResMLP3 --h2h_layer_sizes 12 1000 1000 1000 1000 12 --sequence_weights 1 1 1 1 1 --num_epochs 1000 --batch_size 100 --lr 2e-4  --weight_init xavier_uniform --gpus 2  &
wait