#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output="run_sparse.out"
#SBATCH --nodelist=ault25

AUTO_REQUEUE=1
if [[ "$1" == "--no-requeue" ]]; then
    AUTO_REQUEUE=0
    shift
fi

source ~/.bashrc
pyact Thesis
if [[ $AUTO_REQUEUE -eq 1 ]]; then
    trap 'echo "SIGTERM received"; scontrol requeue ${SLURM_JOB_ID}; exit 15' 15
fi
# Run your program here
# torchrun --standalone --nproc_per_node=8 ~/sparselinear/nanoGPT/train.py config/train_gpt2.py
# python ~/sparselinear/nanoGPT/train.py

torchrun --standalone --nproc_per_node=4 ~/sparselinear/sparsegpt/opt_finetune.py "$@"
# torchrun --standalone --nproc_per_node=4 ~/sparselinear/sparsegpt/opt_finetune.py $1
# python ~/sparselinear/sparsegpt/opt_finetune.py
pop_queue