#!/bin/bash
#SBATCH --job-name=assignment3_stage2
#SBATCH --output=/work/fpb170/assignment3/logs/stage2_%j.out
#SBATCH --error=/work/fpb170/assignment3/logs/stage2_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fpb170@utsa.edu

echo "============================================"
echo "Assignment 3 - Stage 2 Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================"

mkdir -p /work/fpb170/assignment3/logs
module load anaconda3
conda activate /work/fpb170/.conda/envs/llm

source ~/.bashrc

cd /work/fpb170/assignment3

if [ ! -d "outputs/stage1/final" ]; then
    echo "ERROR: Stage 1 checkpoint not found!"
    exit 1
fi

nvidia-smi
echo "Starting Stage 2 training..."
python train_stage2.py

echo "============================================"
echo "Stage 2 Training Complete"
echo "End: $(date)"
echo "============================================"
