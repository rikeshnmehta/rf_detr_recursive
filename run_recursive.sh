#!/bin/bash
#SBATCH --job-name=rf_rec_1ep
#SBATCH --output=work_dir/recursive_logs_%j.out
#SBATCH --error=work_dir/recursive_logs_%j.err
#SBATCH --partition=gpu          
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2        
#SBATCH --cpus-per-task=8        
#SBATCH --mem=64G                
#SBATCH --time=04:00:00        

module load CUDA/12.6.2 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Explicitly activate your local virtual environment
source .venv/bin/activate

python -u train_recursive.py