#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --comment altdiffusion 
#SBATCH --partition=g40
#SBATCH --job-name=aesthetic
#SBATCH --nodes=8            # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)

# activate conda env
source /fsx/zacliu/altclip_env/bin/activate
export PYTHONPATH=$PYTHONPATH:/fsx/zacliu/AltTools/Altdiffusion/src

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# python3 -u /admin/home-zacliu/yfl/codebase/AltTools/Altdiffusion/src/scripts/train_hpc.py --MASTER_ADDR $SLURM_STEP_NODELIST --MASTER_PORT 12802 --WORLD_SIZE $SLURM_NTASKS --NODE_RANK $SLURM_PROCID
srun python -u /fsx/zacliu/AltTools/Altdiffusion/src/scripts/train_hpc.py > /fsx/zacliu/AltTools/Altdiffusion/ckpt/laion_aethetics_all_512_xformer_ema_cfg/log.txt 2>&1
# python3 -u /admin/home-zacliu/yfl/codebase/AltTools/Altdiffusion/src/scripts/train_hpc.py