#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=laion5b
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=%x_%j.out
#SBATCH --comment altdiffusion 
#SBATCH --open-mode=append
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=0-02:00:00

# module load openmpi
# module load cuda/11.6

# activate conda env
# source activate $1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# copy from openclip
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=info

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0


export PYTHONPATH=$PYTHONPATH:/fsx/zacliu/AltTools/Altdiffusion/src;
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;

# run script from above
# srun --comment altdiffusion --cpu_bind=v --accel-bind=gn python3 -u /fsx/zacliu/AltTools/Altdiffusion/src/scripts/train_hpc.py > /fsx/zacliu/AltTools/Altdiffusion/src/laion5b.txt 2>&1
srun --comment altdiffusion --partition=g40 --job-name=test_multi --nodes=8 --gres=gpu:8 --ntasks-per-node=2 --cpus-per-gpu=12 \
python3 -u /fsx/zacliu/AltTools/Altdiffusion/src/scripts/train_hpc.py > /fsx/zacliu/AltTools/Altdiffusion/src/laion5b.txt 2>&1