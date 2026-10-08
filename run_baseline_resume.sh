#!/bin/bash
#SBATCH --job-name=rf_base
#SBATCH --output=work_dir/baseline_logs_%j.out
#SBATCH --error=work_dir/baseline_logs_%j.err
#SBATCH --partition=gpu          
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2        
#SBATCH --cpus-per-task=8        
#SBATCH --mem=64G                
#SBATCH --time=2-00:00:00        

module load CUDA/12.6.2 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u train_baseline_resume.py