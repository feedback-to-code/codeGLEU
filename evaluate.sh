#!/bin/sh
#SBATCH -A herbrich-student
#SBATCH --job-name=feedback2code
#SBATCH --partition magic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --gpus=0
#SBATCH --time=9:0:0

conda activate f2c
python evaluate.py
