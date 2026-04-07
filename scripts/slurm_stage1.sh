#!/bin/bash
#SBATCH --job-name=assignment3_stage1
#SBATCH --output=/work/fpb170/assignment3/logs/stage1_%j.out
#SBATCH --error=/work/fpb170/assignment3/logs/stage1_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fpb170@utsa.edu

echo "============================================"
echo "Assignment 3 - Stage 1 Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================"

mkdir -p /work/fpb170/assignment3/logs
module load anaconda3
conda activate /work/fpb170/.conda/envs/llm

# Set environment variables from ~/.bashrc
source ~/.bashrc

cd /work/fpb170/assignment3
nvidia-smi

echo "Starting Stage 1 training..."
python train_stage1.py

echo "============================================"
echo "Stage 1 Training Complete"
echo "End: $(date)"
echo "============================================"
